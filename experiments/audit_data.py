from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]
for f in sorted((ROOT/'data'/'processed_pairs').glob('*.csv')):
    p=pd.read_csv(f)
    src=[c for c in p if c.startswith('source__')]; tgt=[c for c in p if c.startswith('target__')]
    ok=bool(src and tgt and len(src)==len(tgt) and np.isfinite(p[src+tgt].to_numpy(float)).all() and np.allclose(p[src].sum(1),1,atol=1e-6) and np.allclose(p[tgt].sum(1),1,atol=1e-6))
    print(f.name,'rows=',len(p),'patients=',p.patient_id.astype(str).nunique(),'states=',len(src),'simplex_ok=',ok)
