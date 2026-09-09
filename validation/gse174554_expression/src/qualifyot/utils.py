from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd

def sha256(path: Path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')

def norm_name(s): return re.sub(r'[^a-z0-9]+','',str(s).lower())

def find_col(df: pd.DataFrame, tokens, required=True):
    cols=list(df.columns); nn={c:norm_name(c) for c in cols}
    for token in tokens:
        t=norm_name(token)
        for c in cols:
            if nn[c]==t: return c
    for token in tokens:
        t=norm_name(token)
        for c in cols:
            if t and t in nn[c]: return c
    if required: raise KeyError(f"cannot infer column from {tokens}; columns={cols[:80]}")
    return None

def normalize_rows(x):
    x=np.asarray(x,float); x=np.maximum(x,0); s=x.sum(1,keepdims=True); return x/np.where(s>0,s,1)

def patient_equal(values, patients):
    values=np.asarray(values,float); patients=np.asarray(patients).astype(str)
    return float(np.mean([values[patients==p].mean() for p in np.unique(patients)]))
