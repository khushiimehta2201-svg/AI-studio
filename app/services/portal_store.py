"""Persistent SQLite state for jobs, publication and trainer review."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

SCHEMA="""
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, filename TEXT NOT NULL, status TEXT NOT NULL,
 progress INTEGER NOT NULL DEFAULT 0, message TEXT NOT NULL DEFAULT '', error TEXT,
 created_at REAL NOT NULL, updated_at REAL NOT NULL,
 publication_status TEXT NOT NULL DEFAULT 'draft', plan_json TEXT
);
CREATE TABLE IF NOT EXISTS prerequisites (
 job_id TEXT PRIMARY KEY, urls_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS scene_reviews (
 job_id TEXT NOT NULL, scene_id TEXT NOT NULL, status TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
 PRIMARY KEY(job_id, scene_id)
);
"""

class PortalStore:
    def __init__(self,path:str|Path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.init()
    @contextmanager
    def connect(self)->Iterator[sqlite3.Connection]:
        c=sqlite3.connect(str(self.path),timeout=10);c.row_factory=sqlite3.Row
        c.execute("PRAGMA busy_timeout=10000")
        c.execute("PRAGMA journal_mode=WAL")
        try:
            yield c;c.commit()
        finally:c.close()
    def init(self):
        with self.connect() as c:c.executescript(SCHEMA)
    def create_job(self,job_id:str,filename:str):
        import time
        now=time.time()
        with self.connect() as c:c.execute("INSERT INTO jobs(id,filename,status,created_at,updated_at) VALUES(?,?,?,?,?)",(job_id,filename,"queued",now,now))
    def update_job(self,job_id:str,**fields:Any):
        import time
        fields["updated_at"]=time.time();cols=[];vals=[]
        for k,v in fields.items():cols.append(f"{k}=?");vals.append(v)
        vals.append(job_id)
        with self.connect() as c:c.execute(f"UPDATE jobs SET {', '.join(cols)} WHERE id=?",vals)
    def get_job(self,job_id:str)->Optional[Dict[str,Any]]:
        with self.connect() as c:
            r=c.execute("SELECT * FROM jobs WHERE id=?",(job_id,)).fetchone();return dict(r) if r else None
    def list_jobs(self)->List[Dict[str,Any]]:
        with self.connect() as c:return [dict(r) for r in c.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()]
    def set_plan(self,job_id:str,plan:Dict[str,Any]):self.update_job(job_id,plan_json=json.dumps(plan,ensure_ascii=False))
    def get_plan(self,job_id:str)->Optional[Dict[str,Any]]:
        row=self.get_job(job_id)
        if not row or not row.get("plan_json"):return None
        try:v=json.loads(row["plan_json"]);return v if isinstance(v,dict) else None
        except Exception:return None
    def set_publication(self,job_id:str,status:str):self.update_job(job_id,publication_status=status)
    def set_prerequisites(self,job_id:str,urls:List[str]):
        with self.connect() as c:c.execute("INSERT INTO prerequisites(job_id,urls_json) VALUES(?,?) ON CONFLICT(job_id) DO UPDATE SET urls_json=excluded.urls_json",(job_id,json.dumps(urls,ensure_ascii=False)))
    def prerequisites(self,job_id:str)->List[str]:
        with self.connect() as c:r=c.execute("SELECT urls_json FROM prerequisites WHERE job_id=?",(job_id,)).fetchone()
        if not r:return []
        try:v=json.loads(r[0]);return v if isinstance(v,list) else []
        except Exception:return []
    def review(self,job_id:str,scene_id:str,status:str,note:str=""):
        with self.connect() as c:c.execute("INSERT INTO scene_reviews(job_id,scene_id,status,note) VALUES(?,?,?,?) ON CONFLICT(job_id,scene_id) DO UPDATE SET status=excluded.status,note=excluded.note",(job_id,scene_id,status,note))
    def reviews(self,job_id:str)->Dict[str,Dict[str,str]]:
        with self.connect() as c:rows=c.execute("SELECT scene_id,status,note FROM scene_reviews WHERE job_id=?",(job_id,)).fetchall()
        return {r["scene_id"]:{"status":r["status"],"note":r["note"]} for r in rows}
