"""Pluggable TTS layer with local-first defaults.

Supported:
- sapi: Windows development/local fallback
- kokoro: local neural TTS
- azure: optional production cloud TTS
"""
from __future__ import annotations

import os
import re
import subprocess
import wave
import html
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2]/".env")

PROVIDER=os.getenv("TTS_PROVIDER","sapi").strip().lower()
_KOKORO_PIPELINES: dict[tuple[str,str], Any] = {}


def _clean(text:str)->str:
    text=re.sub(r"https?://\S+|www\.\S+"," ",str(text or ""),flags=re.I)
    text=re.sub(r"\s+"," ",text).strip()
    return text[:1200] or "Continue with the next step."


def wav_duration(path:str)->float:
    try:
        with wave.open(path,"rb") as w:return w.getnframes()/max(1,float(w.getframerate()))
    except Exception:return 0.0


def _sapi(text:str,out:Path)->float:
    clean=_clean(text).replace("'"," ").replace('"',' ').replace("`"," ")
    out.parent.mkdir(parents=True,exist_ok=True)
    safe=str(out.resolve()).replace("'","''")
    script=f"""
Add-Type -AssemblyName System.Speech
$s=New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate=0
$s.Volume=100
$s.SetOutputToWaveFile('{safe}')
$s.Speak('{clean}')
$s.SetOutputToNull()
$s.Dispose()
"""
    r=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-ExecutionPolicy","Bypass","-Command",script],capture_output=True,text=True,timeout=45)
    if r.returncode!=0:raise RuntimeError(r.stderr.strip() or "SAPI failed")
    d=wav_duration(str(out))
    if d<=0.05:raise RuntimeError("SAPI produced empty audio")
    return d


def _kokoro(text:str,out:Path,voice:str)->float:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline
    lang=os.getenv("KOKORO_LANGUAGE","a")
    key=(lang,voice)
    pipe=_KOKORO_PIPELINES.get(key)
    if pipe is None:
        pipe=KPipeline(lang_code=lang)
        _KOKORO_PIPELINES[key]=pipe
    chunks=[audio for _,_,audio in pipe(_clean(text),voice=voice)]
    if not chunks:raise RuntimeError("Kokoro returned no audio")
    audio=np.concatenate(chunks) if len(chunks)>1 else chunks[0]
    sf.write(str(out),audio,24000,subtype="PCM_16")
    return wav_duration(str(out))


def _azure(text:str,out:Path,language:str="en-us")->float:
    import requests
    key=os.getenv("AZURE_SPEECH_KEY","").strip(); region=os.getenv("AZURE_SPEECH_REGION","").strip(); voice=os.getenv("AZURE_SPEECH_VOICE","hi-IN-SwaraNeural").strip()
    if not key or not region:raise RuntimeError("Azure Speech credentials are not configured")
    xml_lang="hi-IN" if str(language).lower() in {"hi","hi-in","hindi","hinglish"} else "en-US"
    ssml=f'''<speak version="1.0" xml:lang="{xml_lang}"><voice name="{voice}">{html.escape(_clean(text))}</voice></speak>'''
    url=f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    r=requests.post(url,headers={"Ocp-Apim-Subscription-Key":key,"Content-Type":"application/ssml+xml","X-Microsoft-OutputFormat":"riff-24khz-16bit-mono-pcm"},data=ssml.encode("utf-8"),timeout=60)
    r.raise_for_status(); out.write_bytes(r.content); d=wav_duration(str(out))
    if d<=0.05:raise RuntimeError("Azure returned empty audio")
    return d


def generate_narration(plan:Dict[str,Any],job_dir:str,progress_callback=None,language:str="en-us",voice:str="af_heart")->List[Dict[str,Any]]:
    out=Path(job_dir)/"audio"; out.mkdir(parents=True,exist_ok=True)
    steps=plan.get("steps",[]); result=[]; total=len(steps)
    actual_provider=PROVIDER
    for i,step in enumerate(steps):
        text=_clean(str(step.get("tts_narration") or step.get("narration") or "Continue with the next step."))
        path=out/f"scene_{i+1:05d}.wav"
        if progress_callback:progress_callback(i,total,f"Synthesizing narration {i+1} of {total}")
        try:
            if actual_provider=="azure":duration=_azure(text,path,language)
            elif actual_provider=="kokoro":duration=_kokoro(text,path,voice)
            elif actual_provider=="sapi":duration=_sapi(text,path)
            else:raise RuntimeError(f"Unsupported TTS_PROVIDER={actual_provider}")
        except Exception as exc:
            # Explicit provider never silently changes in production. SAPI is only
            # a development fallback when AUTO_LOCAL_FALLBACK=true.
            if os.getenv("AUTO_LOCAL_FALLBACK","false").lower() in {"1","true","yes","on"} and actual_provider!="sapi":
                duration=_sapi(text,path); actual_provider="sapi"
            else:
                raise RuntimeError(f"TTS failed for scene {i+1}: {exc}") from exc
        result.append({"index":i,"scene_id":step.get("scene_id"),"text":text,"path":str(path.resolve()),"duration":duration,"provider":actual_provider})
    if progress_callback:progress_callback(total,total,"Narration audio ready")
    return result
