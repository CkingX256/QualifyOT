from __future__ import annotations
import time
from pathlib import Path
import requests
from .utils import sha256, write_json

def _download_one(url,dest,timeout=60,retries=4):
    dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True); part=dest.with_suffix(dest.suffix+'.part')
    for attempt in range(retries):
        start=part.stat().st_size if part.exists() else 0
        headers={'User-Agent':'QualifyOT-research'}
        if start: headers['Range']=f'bytes={start}-'
        try:
            with requests.get(url,stream=True,timeout=(15,timeout),headers=headers,allow_redirects=True) as r:
                r.raise_for_status()
                mode='ab' if start and r.status_code==206 else 'wb'
                if mode=='wb' and part.exists(): part.unlink()
                with open(part,mode) as f:
                    for chunk in r.iter_content(1024*1024):
                        if chunk: f.write(chunk)
            part.replace(dest)
            return {'url':url,'bytes':dest.stat().st_size,'sha256':sha256(dest),'status':'SUCCESS'}
        except Exception as e:
            if attempt==retries-1: raise
            time.sleep(min(2**attempt,8))

def ensure_file(spec, root: Path):
    dest=root/spec['filename']
    if dest.exists() and dest.stat().st_size>0:
        return dest, {'status':'EXISTING','bytes':dest.stat().st_size,'sha256':sha256(dest)}
    errors=[]
    for key in ('url','fallback_url'):
        if spec.get(key):
            try: return dest,_download_one(spec[key],dest)
            except Exception as e: errors.append(f'{key}: {e!r}')
    raise RuntimeError('download failed: '+' | '.join(errors))

def download_dataset(accession,cfg,root:Path):
    out=root/accession; out.mkdir(parents=True,exist_ok=True); manifest={'accession':accession,'files':{}}
    paths={}
    for name,spec in cfg.get('files',{}).items():
        p,meta=ensure_file(spec,out); paths[name]=p; manifest['files'][name]={**meta,'filename':p.name}
    write_json(out/'DOWNLOAD_MANIFEST.json',manifest)
    return paths
