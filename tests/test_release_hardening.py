from __future__ import annotations
import json, os, tempfile, unittest, uuid, wave
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('AUTH_REQUIRED','false')
os.environ.setdefault('OLLAMA_HOST','http://127.0.0.1:11435')

from app.main import app, STORE, JOBS
from app.services.pdf_service import process_pdf
from app.services.qa_service import validate_plan
from app.services.targeting import ground_action
import app.main as main


def make_wav(path:Path,seconds=.25):
    rate=24000; frames=int(rate*seconds)
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(np.zeros(frames,dtype=np.int16).tobytes())


class ReleaseHardeningTests(unittest.TestCase):
    def test_unresolved_target_does_not_abort_generation(self):
        jid=f"hard_unresolved_{uuid.uuid4().hex[:8]}"; job=JOBS/jid; job.mkdir(parents=True,exist_ok=True)
        pdf=job/'source.pdf'; pdf.write_bytes(b'%PDF-test')
        audio=job/'tts.wav'; make_wav(audio)
        plan={"schema_version":4,"title":"Unresolved","description":"Test","steps":[{
            "id":1,"scene_id":"scene_00001","kind":"action","title":"Click Create","source_page":1,
            "source_step_number":"1","page_width":0,"page_height":0,"interaction":"click",
            "target_status":"unresolved","target_review_required":True,"cursor_enabled":False,"cursor":None,
            "cursor_path":[],"narration":"Click Create now.","tts_narration":"Click Create now.","caption_text":"Click Create now.",
            "action":"Click Create","screenshot":None,"source_context":""}],"sections":[]}
        main.STORE.create_job(jid,'source.pdf')
        try:
            with patch('app.main.process_pdf',return_value={"pages":[],"screenshots":[]}), \
                 patch('app.main.build_tutorial_plan',return_value=plan), \
                 patch('app.main.generate_narration',return_value=[{"index":0,"scene_id":"scene_00001","path":str(audio),"duration":.25}]):
                main.run_job(jid,pdf)
            row=main.STORE.get_job(jid)
            self.assertEqual(row['status'],'completed',row)
            saved=main.STORE.get_plan(jid)
            self.assertTrue(saved['requires_trainer_review'])
            self.assertFalse(saved['qa']['publishable'])
            self.assertTrue((job/'tutorial.mp4').exists())
        finally:
            import shutil
            shutil.rmtree(job,ignore_errors=True)
            with main.STORE.connect() as c:
                c.execute('DELETE FROM jobs WHERE id=?',(jid,)); c.execute('DELETE FROM prerequisites WHERE job_id=?',(jid,)); c.execute('DELETE FROM scene_reviews WHERE job_id=?',(jid,))

    def test_mark_no_target_keeps_required_scene_blocked(self):
        jid=f"hard_notarget_{uuid.uuid4().hex[:8]}";job=JOBS/jid;job.mkdir(parents=True,exist_ok=True)
        img=job/'shot.png';Image.new('RGB',(640,480),'white').save(img); audio=job/'audio.wav';make_wav(audio)
        plan={"schema_version":4,"title":"No Target","description":"Test","steps":[{
            "id":1,"scene_id":"scene_00001","kind":"action","title":"Click Create","source_page":1,"page_width":640,"page_height":480,
            "interaction":"click","target_status":"review_required","target_review_required":True,"cursor_enabled":False,"cursor":None,
            "cursor_path":[],"narration":"Click Create now.","tts_narration":"Click Create now.","caption_text":"Click Create now.","screenshot":str(img),"source_context":""}],"sections":[]}
        (job/'plan.json').write_text(json.dumps(plan),encoding='utf8');(job/'audio.json').write_text(json.dumps([{"index":0,"scene_id":"scene_00001","path":str(audio),"duration":.25}]),encoding='utf8')
        main.STORE.create_job(jid,'x.pdf');main.STORE.set_plan(jid,plan)
        try:
            client=TestClient(app)
            r=client.post(f'/api/trainer/scenes/{jid}/scene_00001/review',json={'decision':'mark_no_target'})
            self.assertEqual(r.status_code,200,r.text)
            saved=main.STORE.get_plan(jid)
            self.assertEqual(saved['steps'][0]['target_status'],'unresolved')
            self.assertTrue(saved['steps'][0]['target_review_required'])
            self.assertFalse(saved['qa']['publishable'])
        finally:
            import shutil
            shutil.rmtree(job,ignore_errors=True)
            with main.STORE.connect() as c:
                c.execute('DELETE FROM jobs WHERE id=?',(jid,));c.execute('DELETE FROM prerequisites WHERE job_id=?',(jid,));c.execute('DELETE FROM scene_reviews WHERE job_id=?',(jid,))

    def test_prerequisite_rejects_unsafe_scheme(self):
        jid=f"hard_url_{uuid.uuid4().hex[:8]}";main.STORE.create_job(jid,'x.pdf')
        try:
            client=TestClient(app)
            r=client.post(f'/api/trainer/prerequisites/{jid}',json={'prerequisites':['javascript:alert(1)','https://example.com/video']})
            self.assertEqual(r.status_code,400)
            self.assertIn('invalid',r.json())
        finally:
            with main.STORE.connect() as c:
                c.execute('DELETE FROM jobs WHERE id=?',(jid,));c.execute('DELETE FROM prerequisites WHERE job_id=?',(jid,))

    def test_short_valid_action_narration_is_accepted(self):
        plan={"schema_version":4,"steps":[{"scene_id":"scene_1","kind":"action","source_page":1,"narration":"Click Create.","interaction":"click","target_status":"verified","cursor_enabled":True,"cursor":[.5,.5]}]}
        self.assertTrue(validate_plan(plan)['publishable'])

    def test_full_page_ui_is_classified_and_source_text_is_filtered_from_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); shot=d/'page.png';Image.new('RGB',(1000,700),'white').save(shot)
            data={'pages':[],'screenshots':[]}
            import fitz
            pdf=d/'x.pdf';doc=fitz.open();p=doc.new_page();p.insert_text((50,60),'Application workspace File Edit Save Click Create');doc.save(str(pdf));doc.close()
            parsed=process_pdf(str(pdf),d)
            self.assertEqual(parsed['pages'][0]['visual_page_role'],'ui_page')
            page={'page':1,'width':1000,'height':700,'visual_page_role':'ui_page','action_bbox':[20,20,500,100]}
            real=[
                {'point':[.30,.08],'bounding_box':[.10,.02,.50,.14],'target_name':'Click Create','confidence':.99},
                {'point':[.80,.75],'bounding_box':[.74,.70,.88,.80],'target_name':'Create','confidence':.93},
            ]
            with patch('app.services.targeting._ocr_candidates',return_value=real):
                result=ground_action('Click Create',page,[{'page':1,'path':str(shot),'visual_role':'full_page','is_full_page':True}],None)
            self.assertEqual(result['status'],'verified')
            self.assertAlmostEqual(result['point'][0],.80,places=3)


    def test_partial_explicit_procedure_gets_semantic_repair_without_losing_explicit_action(self):
        from app.services.production_planner import build_tutorial_plan
        class CountingText:
            class Spec: model='counting-text'
            spec=Spec()
            def __init__(self): self.extract_calls=0
            def generate_json(self,prompt):
                if 'Extract only actionable' in prompt:
                    self.extract_calls+=1
                    return {'items':[{'step_number':'2','action':'Open Settings','context':'Open the settings panel before continuing.'}]}
                payload=json.loads(prompt.split('\n\n',1)[1])
                return {'narrations':[{'id':x['id'],'narration':'Please follow this action now.'} for x in payload]}
        class EmptyVision:
            class Spec: model='vision'
            spec=Spec()
            def generate_json(self,prompt,image_b64): return {'candidates':[]}
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); img=d/'shot.png'; Image.new('RGB',(700,500),'white').save(img)
            text=CountingText()
            data={'job_dir':str(d),'title':'Repair','pages':[{'page':1,'heading':'Procedure','width':700,'height':500,'text':'1. Click Create.\n2. The application then requires configuration.\n3. Select Apply after the configuration is complete.'}], 'screenshots':[{'page':1,'path':str(img),'visual_role':'screenshot_candidate','annotation_boxes':[]}] }
            with patch('app.services.production_planner.create_clients',return_value=(text,EmptyVision())):
                plan=build_tutorial_plan(data)
            actions=[s['action'] for s in plan['steps'] if s['kind']=='action']
            self.assertIn('1. Click Create.',actions)
            self.assertIn('Open Settings',actions)
            self.assertTrue(text.extract_calls>=1)

    def test_requirements_include_tesseract_binding(self):
        req=(ROOT/'requirements.txt').read_text(encoding='utf8')
        self.assertIn('pytesseract',req)

if __name__=='__main__':unittest.main()
