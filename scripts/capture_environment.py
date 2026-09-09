from __future__ import annotations
import json, os, platform, subprocess, sys
from pathlib import Path

def cmd(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"ERROR: {e}"

out={
    "python":sys.version,
    "executable":sys.executable,
    "platform":platform.platform(),
    "machine":platform.machine(),
    "processor":platform.processor(),
    "cpu_count":os.cpu_count(),
    "pip_freeze":cmd([sys.executable,"-m","pip","freeze"]),
}
try:
    import torch
    out["torch"]={"version":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda_version":torch.version.cuda}
except Exception as e:
    out["torch"]={"error":repr(e)}
Path("results/environment.json").parent.mkdir(parents=True,exist_ok=True)
Path("results/environment.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in out.items() if k!="pip_freeze"},indent=2))
