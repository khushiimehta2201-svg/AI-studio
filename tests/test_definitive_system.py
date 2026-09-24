from __future__ import annotations

import json
import os
import tempfile
import unittest
import uuid
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
os.environ["AUTH_REQUIRED"]="false"
os.environ["OLLAMA_HOST"]="http://127.0.0.1:11435"

import sys
sys.path.insert(0,str(ROOT))

from fastapi.testclient import TestClient
from app.main import app, STORE, JOBS
from app.services.document_understanding import extract_instruction_blocks, is_metadata_page, split_compound_action
from app.services.pdf_service import process_pdf
from app.services.model_adapters import create_clients
from app.services.production_planner import build_tutorial_plan
from app.services.qa_service import validate_plan, validate_rendered_artifacts
from app.services.targeting import ground_action, target_queries
from app.services.video_service import render_all_sections


class FakeText:
    class Spec: model="fake-text"
    spec=Spec()
    def generate_json(self,prompt):
        if 'Extract only actionable' in prompt:
            return {"items":[{"step_number":None,"action":"Open the Item browser.","context":"The Item browser is used to manage items."}]}
        payload=json.loads(prompt.split("\n\n",1)[1])
        return {"narrations":[{"id":x["id"],"narration":"Please select the " + x["action"].rstrip(".") + " button."} for x in payload]}


class FakeVision:
    class Spec: model="fake-vision"
    spec=Spec()
    def __init__(self, candidates): self.candidates=candidates
    def generate_json(self,prompt,image_b64):
        if 'Verify this target candidate' in prompt:
            return {"confirmed":True,"reason":"Candidate is inside the target control."}
        return {"candidates":self.candidates}


def make_wav(path:Path,seconds=0.25):
    rate=24000;frames=int(rate*seconds)
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(np.zeros(frames,dtype=np.int16).tobytes())


class DefinitiveSystemTests(unittest.TestCase):
    def test_numbered_explanation_containing_click_is_not_action(self):
        page={"page":1,"text":"1. The user can click the toolbar to understand the workspace.\n2. Click Create to begin."}
        blocks=extract_instruction_blocks(page)
        actions=[b["action"] for b in blocks if b.get("action")]
        self.assertEqual(actions,["2. Click Create to begin."])

    def test_repeated_logo_is_not_treated_as_screenshot(self):
        import fitz
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);logo=d/"logo.png";Image.new("RGB",(180,80),(10,10,10)).save(logo);pdf=d/"logo.pdf"
            doc=fitz.open()
            for _ in range(3):
                page=doc.new_page();page.insert_image(fitz.Rect(20,20,200,100),filename=str(logo));page.insert_text((60,180),"Creating an item")
            doc.save(str(pdf));doc.close()
            data=process_pdf(str(pdf),d)
            embedded=[a for a in data["screenshots"] if not a.get("is_full_page")]
            self.assertTrue(embedded)
            self.assertTrue(all(a.get("visual_role") not in {"screenshot_candidate","ui_screenshot"} for a in embedded))

    def test_document_structure_and_source_numbering(self):
        toc={"page":2,"heading":"Table of Contents","text":"Contents\n1 Overview ........ 2\n2 Create Item ........ 4\n3 Save Item ........ 5"}
        self.assertTrue(is_metadata_page(toc));self.assertEqual(extract_instruction_blocks(toc),[])
        page={"page":7,"heading":"Create Item","text":"4.2.1 Click Create\nmenu."}
        block=extract_instruction_blocks(page)[0]
        self.assertEqual(block["step_number"],"4.2.1");self.assertEqual(block["action"],"4.2.1 Click Create menu.")
        self.assertEqual(split_compound_action("Click Create then select Part"),["Click Create","select Part"])

    def test_keyboard_is_not_mouse_target(self):
        self.assertEqual(target_queries("Press Enter"),[])

    def test_highlight_only_is_review_required(self):
        with tempfile.TemporaryDirectory() as td:
            img=Path(td)/"shot.png";Image.new("RGB",(800,600),"white").save(img)
            page={"page":1,"width":800,"height":600}
            shot={"page":1,"path":str(img),"visual_role":"screenshot_candidate","annotation_boxes":[[.4,.4,.6,.5]]}
            result=ground_action("Click Create",page,[shot],None)
            self.assertEqual(result["status"],"review_required");self.assertTrue(result["review_required"])

    def test_duplicate_target_can_remain_unresolved(self):
        with tempfile.TemporaryDirectory() as td:
            img=Path(td)/"shot.png";Image.new("RGB",(800,600),"white").save(img)
            page={"page":1,"width":800,"height":600}
            shot={"page":1,"path":str(img),"visual_role":"screenshot_candidate","annotation_boxes":[]}
            with patch('app.services.targeting._ocr_candidates',return_value=[
                {"point":[.2,.2],"bounding_box":[.1,.1,.3,.3],"target_name":"Create","confidence":.93},
                {"point":[.7,.7],"bounding_box":[.6,.6,.8,.8],"target_name":"Create","confidence":.93},
            ]):
                result=ground_action("Click Create",page,[shot],None)
            self.assertEqual(result["status"],"unresolved")

    def test_vision_grounding_and_verification(self):
        with tempfile.TemporaryDirectory() as td:
            img=Path(td)/"shot.png";Image.new("RGB",(800,600),"white").save(img)
            page={"page":1,"width":800,"height":600}
            shot={"page":1,"path":str(img),"visual_role":"screenshot_candidate","annotation_boxes":[]}
            vision=FakeVision([{"target_name":"Create","bounding_box":[.7,.7,.85,.82],"click_point":[.77,.76],"confidence":.94,"evidence":"Create button visible"}])
            result=ground_action("Click Create",page,[shot],vision)
            self.assertEqual(result["status"],"verified");self.assertEqual(result["source"],"vision_verified")

    def test_unordered_semantic_page_uses_current_text_model(self):
        with tempfile.TemporaryDirectory() as td:
            img=Path(td)/"shot.png";Image.new("RGB",(800,600),"white").save(img)
            data={"job_dir":td,"title":"Item Management","pages":[{"page":1,"heading":"Working with Items","width":800,"height":600,"text":"The Item browser is used to manage items. Use the toolbar to access item operations and create a new item."}],"screenshots":[{"page":1,"path":str(img),"visual_role":"screenshot_candidate","annotation_boxes":[[.7,.7,.85,.82]],"is_full_page":False}]}
            with patch('app.services.production_planner.create_clients',return_value=(FakeText(),FakeVision([]))):
                plan=build_tutorial_plan(data)
            self.assertEqual(plan["schema_version"],4);self.assertGreater(plan["action_count"],0)

    def test_same_page_source_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            img=Path(td)/"shot.png";Image.new("RGB",(800,600),"white").save(img)
            data={"job_dir":td,"title":"Order Test","pages":[{"page":1,"heading":"Section","width":800,"height":600,"text":"synthetic"}],"screenshots":[{"page":1,"path":str(img),"visual_role":"screenshot_candidate","annotation_boxes":[]}]}
            blocks=[
                {"action":"Click Create","context":"","step_number":"1","bbox":None},
                {"action":None,"context":"This explains why creation is required for the workflow to continue correctly.","step_number":None,"bbox":None},
                {"action":"Select Part","context":"","step_number":"2","bbox":None},
            ]
            with patch('app.services.production_planner.create_clients',return_value=(FakeText(),FakeVision([{"target_name":"Create","bounding_box":[.7,.7,.9,.9],"click_point":[.8,.8],"confidence":.95,"evidence":"button"}]))), patch('app.services.production_planner.extract_instruction_blocks',return_value=blocks):
                plan=build_tutorial_plan(data)
            kinds_titles=[(s["kind"],s["title"]) for s in plan["steps"]]
            self.assertEqual(kinds_titles[0][1],"Click Create")
            self.assertEqual(kinds_titles[1][1],"This explains why creation is required for the workflow to continue correctly.")
            self.assertEqual(kinds_titles[2][1],"Select Part")

    def test_required_action_marked_no_target_is_not_publishable(self):
        plan={"schema_version":4,"steps":[{"scene_id":"scene_no_target","kind":"action","source_page":1,"narration":"Please click the Create button.","interaction":"click","target_status":"not_applicable","target_review_required":False,"cursor_enabled":False,"cursor":None}]}
        qa=validate_plan(plan)
        self.assertFalse(qa["publishable"])
        self.assertTrue(any(x["code"]=="REQUIRED_TARGET_MISSING" for x in qa["errors"]))

    def test_verified_target_without_point_is_not_publishable(self):
        plan={"schema_version":4,"steps":[{"scene_id":"scene_1","kind":"action","source_page":1,"narration":"Please click the Create button.","interaction":"click","target_status":"verified","cursor_enabled":True,"cursor":None}]}
        qa=validate_plan(plan)
        self.assertFalse(qa["publishable"])
        self.assertTrue(any(x["code"]=="VERIFIED_TARGET_MISSING_POINT" for x in qa["errors"]))

    def test_model_swap_is_configuration_only(self):
        with patch.dict(os.environ,{"AI_TEXT_MODEL":"better-text","AI_VISION_MODEL":"better-vision"},clear=False):
            text,vision=create_clients()
            self.assertEqual(text.spec.model,"better-text");self.assertEqual(vision.spec.model,"better-vision")

    def test_drag_render_has_real_drag_interval(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);img=d/"ui.png";Image.new("RGB",(900,650),(235,235,235)).save(img);audio=d/"scene.wav";make_wav(audio,0.35)
            plan={"schema_version":4,"steps":[{"id":1,"scene_id":"scene_drag","kind":"action","title":"Drag Row One to Row Two","source_page":1,"page_width":900,"page_height":650,"interaction":"drag","target_status":"verified","cursor_enabled":True,"cursor":[.75,.75],"cursor_path":[[.25,.25],[.75,.75]],"cursor_bounding_box":[.70,.70,.80,.80],"screenshot":str(img),"caption_text":"Drag Row One to Row Two.","narration":"Please drag Row One to Row Two now."}],"sections":[]}
            rendered=render_all_sections(plan,[{"index":0,"scene_id":"scene_drag","path":str(audio),"duration":.35}],str(d))
            t=rendered["dialogue_timeline"][0]
            self.assertEqual(t["interaction"],"drag");self.assertIsNotNone(t["interaction_end_time"]);self.assertGreater(t["interaction_end_time"],t["interaction_time"])

    def test_renderer_and_portal_timeline_contract(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);img=d/"ui.png";Image.new("RGB",(800,600),(235,235,235)).save(img);audio=d/"scene.wav";make_wav(audio)
            plan={"schema_version":4,"steps":[{"id":1,"scene_id":"scene_00001","kind":"action","title":"Click Create","source_page":1,"page_width":800,"page_height":600,"interaction":"click","target_status":"verified","cursor_enabled":True,"cursor":[.7,.7],"cursor_bounding_box":[.65,.65,.75,.75],"screenshot":str(img),"caption_text":"Click Create.","narration":"Click Create.","dialogue":{"current_action":"Click Create","brief":"Please select the Create button."}}],"sections":[]}
            audio_data=[{"index":0,"scene_id":"scene_00001","path":str(audio),"duration":.25}]
            rendered=render_all_sections(plan,audio_data,str(d))
            self.assertTrue((d/"tutorial.mp4").exists());self.assertTrue(rendered["dialogue_timeline"]);self.assertEqual(rendered["dialogue_timeline"][0]["scene_id"],"scene_00001")
            (d/"plan.json").write_text(json.dumps(rendered),encoding='utf-8')
            qa=validate_rendered_artifacts(str(d),rendered)
            self.assertTrue(qa["publishable"],qa)

    def test_plan_qa_and_publication_gate(self):
        plan={"schema_version":4,"steps":[{"scene_id":"scene_00001","kind":"action","source_page":1,"source_step_number":"1","narration":"Click Create.","interaction":"click","target_status":"review_required","target_review_required":True,"cursor_enabled":False}]}
        qa=validate_plan(plan);self.assertFalse(qa["publishable"])

    def test_auth_role_boundary(self):
        import app.main as main
        old=main.AUTH_REQUIRED;main.AUTH_REQUIRED=True
        old_env={k:os.environ.get(k) for k in ("TRAINER_USERNAME","TRAINER_PASSWORD","TRAINEE_USERNAME","TRAINEE_PASSWORD")}
        os.environ.update({"TRAINER_USERNAME":"trainer_test","TRAINER_PASSWORD":"trainer_pw","TRAINEE_USERNAME":"trainee_test","TRAINEE_PASSWORD":"trainee_pw"})
        try:
            client=TestClient(main.app,follow_redirects=False)
            self.assertEqual(client.get('/trainer').status_code,307)
            r=client.post('/login',json={"username":"trainer_test","password":"trainer_pw"});self.assertEqual(r.status_code,200)
            self.assertEqual(client.get('/trainer').status_code,200)
            self.assertEqual(client.get('/trainee').status_code,307)
        finally:
            main.AUTH_REQUIRED=old
            for k,v in old_env.items():
                if v is None: os.environ.pop(k,None)
                else: os.environ[k]=v

    def test_api_health_and_secure_media_boundary(self):
        client=TestClient(app)
        r=client.get('/health');self.assertEqual(r.status_code,200);self.assertEqual(r.json()["schema_version"],4)
        # No job means no arbitrary file exposure.
        r=client.get('/media/trainer/not-a-job/source.pdf');self.assertEqual(r.status_code,404)

    def test_real_pdf_ingestion_to_scene_to_render_pipeline(self):
        import fitz
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); pdf=d/"demo.pdf"
            img_path=d/"ui.png"
            from PIL import ImageDraw
            img=Image.new("RGB",(1000,700),"white");draw=ImageDraw.Draw(img);draw.rectangle((720,560,900,620),outline="red",width=6);draw.text((760,580),"Create",fill="black");img.save(img_path)
            doc=fitz.open();p0=doc.new_page();p0.insert_text((70,80),"Table of Contents\n1 Overview ........ 2\n2 Create Item ........ 3");p1=doc.new_page();p1.insert_text((70,80),"4.2 Creating an Item\n4.2.1 Click Create");p1.insert_image(fitz.Rect(60,120,540,470),filename=str(img_path));doc.save(str(pdf));doc.close()
            data=process_pdf(str(pdf),d);data["job_dir"]=str(d);data["title"]="Demo"
            with patch('app.services.production_planner.create_clients',return_value=(FakeText(),FakeVision([{"target_name":"Create","bounding_box":[.70,.80,.90,.90],"click_point":[.80,.85],"confidence":.95,"evidence":"Create button visible"}]))):
                plan=build_tutorial_plan(data)
            self.assertEqual(plan["schema_version"],4);self.assertGreater(plan["scene_count"],0)
            plan["steps"][0]["target_review_required"]=False
            audio=d/"scene.wav";make_wav(audio)
            audio_data=[]
            for i,scene in enumerate(plan["steps"]):
                audio_data.append({"index":i,"scene_id":scene["scene_id"],"path":str(audio),"duration":.25})
            rendered=render_all_sections(plan,audio_data,str(d));(d/"plan.json").write_text(json.dumps(rendered),encoding="utf-8")
            self.assertTrue((d/"tutorial.mp4").exists());self.assertEqual(len(rendered["dialogue_timeline"]),len(plan["steps"]))

    def test_scanned_page_falls_back_to_ocr_text(self):
        import fitz
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);pdf=d/"scan.pdf"
            doc=fitz.open();p=doc.new_page();p.insert_text((60,80),"")
            # Keep the page image-only; OCR supplies the actual instruction text.
            doc.save(str(pdf));doc.close()
            with patch('app.services.pdf_service._page_ocr',return_value="Click Create to open the item dialog."):
                data=process_pdf(str(pdf),d)
            self.assertEqual(data["pages"][0]["text_source"],"ocr")
            self.assertIn("Click Create",data["pages"][0]["text"])

    def test_drag_action_gets_two_verified_points(self):
        with tempfile.TemporaryDirectory() as td:
            img=Path(td)/"drag.png";Image.new("RGB",(800,600),"white").save(img)
            page={"page":1,"width":800,"height":600}
            shot={"page":1,"path":str(img),"visual_role":"screenshot_candidate","annotation_boxes":[]}
            with patch('app.services.targeting._ocr_candidates',side_effect=lambda q,p: [
                {"point":[.20,.30],"bounding_box":[.15,.25,.25,.35],"target_name":q,"confidence":.95,"source":"ocr"}
                ] if q.lower() in {"row one","row two"} else []):
                result=ground_action("Drag Row One to Row Two",page,[shot],None)
            self.assertEqual(result["status"],"verified")
            self.assertEqual(len(result["points"]),2)

    def test_run_job_end_to_end_orchestration(self):
        import fitz, shutil
        import app.main as main
        jid=f"itest_e2e_{uuid.uuid4().hex[:8]}";job=ROOT/"jobs"/jid;job.mkdir(parents=True,exist_ok=True)
        try:
            pdf=job/"source.pdf"; img_path=job/"ui.png"
            from PIL import ImageDraw
            img=Image.new("RGB",(900,650),"white");draw=ImageDraw.Draw(img);draw.rectangle((650,500,820,570),outline="red",width=6);draw.text((690,525),"Create",fill="black");img.save(img_path)
            doc=fitz.open();p=doc.new_page();p.insert_text((60,80),"Creating an item\n1. Click Create");p.insert_image(fitz.Rect(80,120,580,420),filename=str(img_path));doc.save(str(pdf));doc.close()
            wav=job/"tts.wav";make_wav(wav,0.25)
            main.STORE.create_job(jid,"source.pdf")
            fake_audio=[{"index":0,"scene_id":"scene_00001","path":str(wav),"duration":.25}]
            with patch('app.services.production_planner.create_clients',return_value=(FakeText(),FakeVision([{"target_name":"Create","bounding_box":[.65,.75,.95,.95],"click_point":[.80,.85],"confidence":.95,"evidence":"Create button visible"}]))), \
                 patch('app.main.generate_narration',return_value=fake_audio):
                main.run_job(jid,pdf)
            row=main.STORE.get_job(jid);self.assertEqual(row["status"],"completed")
            plan=main.STORE.get_plan(jid);self.assertEqual(plan["schema_version"],4)
            self.assertTrue((job/"tutorial.mp4").exists());self.assertTrue((job/"tutorial.vtt").exists())
            self.assertIn("post_render",plan["qa"])
        finally:
            try:shutil.rmtree(job)
            except Exception:pass
            try:
                with main.STORE.connect() as c:
                    c.execute("DELETE FROM scene_reviews WHERE job_id=?",(jid,));c.execute("DELETE FROM prerequisites WHERE job_id=?",(jid,));c.execute("DELETE FROM jobs WHERE id=?",(jid,))
            except Exception:pass

    def test_frontend_contract_matches_watch_page(self):
        watch=(ROOT/"app/templates/watch.html").read_text(encoding="utf-8")
        js=(ROOT/"app/static/portal.js").read_text(encoding="utf-8")
        for element in ("tutorialVideo","title","description","topicIntro","currentAction","currentBrief","currentMeta","sceneList","prereqList"):
            self.assertIn(f'id="{element}"',watch)
        self.assertIn("currentSceneIndex",js);self.assertIn("scenes",js);self.assertNotIn("video.currentTime / video.duration",js)

    def test_trainer_review_and_publish_gate(self):
        import app.main as main
        jid=f"itest_review_{uuid.uuid4().hex[:8]}";job=ROOT/"jobs"/jid;job.mkdir(parents=True,exist_ok=True)
        img=job/"shot.png";Image.new("RGB",(640,480),"white").save(img);audio=job/"audio.wav";make_wav(audio)
        plan={"schema_version":4,"title":"Review Test","description":"Test","steps":[{"id":1,"scene_id":"scene_00001","kind":"action","title":"Click Create","source_page":1,"page_width":640,"page_height":480,"source_step_number":"1","action":"Click Create","action_type":"Click","target_status":"review_required","target_review_required":True,"cursor":[.5,.5],"cursor_bounding_box":[.45,.45,.55,.55],"cursor_enabled":False,"cursor_source":"highlight_only","cursor_confidence":.6,"interaction":"click","screenshot":str(img),"narration":"Please select the Create button.","tts_narration":"Please select the Create button.","caption_text":"Please select the Create button.","dialogue":{"current_action":"Click Create","brief":"Please select the Create button."}}],"sections":[],"qa":{"pre_render":{"publishable":False},"publishable":False},"requires_trainer_review":True}
        (job/"plan.json").write_text(json.dumps(plan),encoding="utf-8");(job/"audio.json").write_text(json.dumps([{"index":0,"scene_id":"scene_00001","path":str(audio),"duration":.25}]),encoding="utf-8")
        main.STORE.create_job(jid,"review.pdf");main.STORE.set_plan(jid,plan)
        client=TestClient(main.app)
        r=client.get(f"/api/trainer/videos/{jid}/review");self.assertEqual(r.status_code,200)
        r=client.post(f"/api/trainer/publish/{jid}",json={"published":True});self.assertEqual(r.status_code,409)
        r=client.post(f"/api/trainer/scenes/{jid}/scene_00001/review",json={"decision":"set_target","point":[.62,.63]});self.assertEqual(r.status_code,200)
        plan2=main.STORE.get_plan(jid);self.assertEqual(plan2["steps"][0]["target_status"],"verified");self.assertTrue(plan2["qa"]["publishable"])
        r=client.post(f"/api/trainer/publish/{jid}",json={"published":True});self.assertEqual(r.status_code,200)
        self.assertEqual(main.STORE.get_job(jid)["publication_status"],"published")

    def test_api_generate_persists_job_without_waiting_for_worker(self):
        client=TestClient(app)
        fake_pdf=b'%PDF-1.4 test'
        with patch('app.main.run_job'):
            r=client.post('/generate',files={'file':('sample.pdf',fake_pdf,'application/pdf')})
        self.assertEqual(r.status_code,200)
        jid=r.json()["job_id"];row=STORE.get_job(jid);self.assertIsNotNone(row);self.assertEqual(row["status"],"queued")
        try:
            import shutil
            shutil.rmtree(ROOT/"jobs"/jid)
            with STORE.connect() as c:c.execute("DELETE FROM jobs WHERE id=?",(jid,))
        except Exception:pass


if __name__=='__main__':unittest.main()
