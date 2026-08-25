from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import find_col
from .ontology import canonical_state, STATE_ORDER

def _read_auto(path, **kw):
    # GEO supplementary text is not guaranteed to be UTF-8.  In particular,
    # older GEO exports can contain cp1252 micro/degree symbols.  Try strict
    # decoders first and only fall back to latin1, which is byte-total.
    last=None
    for enc in ('utf-8-sig','utf-8','cp1252','latin1'):
        try:
            df=pd.read_csv(path,sep='\t',comment='#',encoding=enc,**kw)
            if df.shape[1]>1:
                return df
        except Exception as e:
            last=e
    # Some GEO metadata exports (notably GSE236581) are not TSV despite the
    # .txt.gz suffix.  They are whitespace-delimited, quote-aware tables with
    # the cell barcode/rownames emitted as an unlabeled leading field.  The
    # pandas Python regex parser does not reliably honor quote stripping here;
    # use the C engine so quoted headers/values are parsed cleanly.
    for enc in ('utf-8-sig','utf-8','cp1252','latin1'):
        try:
            df=pd.read_csv(path,sep=r'\s+',engine='c',quotechar='"',comment='#',encoding=enc,**kw)
            if df.shape[1]>1:
                # Defensive cleanup for heterogeneous GEO exports.
                df.columns=[str(c).strip().strip('"') for c in df.columns]
                for c in df.select_dtypes(include='object').columns:
                    df[c]=df[c].map(lambda x: x.strip().strip('"') if isinstance(x,str) else x)
                return df
        except Exception as e:
            last=e
    raise RuntimeError(f'cannot parse {path}: {last}')

def _tumor_mask(df, tissue_col=None, ident_col=None):
    # GSE236581 uses compact tissue codes in some releases (T/N/B) and full
    # names in others.  Do not require the literal word "Tumor".
    mask=pd.Series(False,index=df.index)
    if tissue_col is not None:
        s=df[tissue_col].astype(str).str.strip().str.lower()
        mask |= s.isin({'t','tumor','tumour','primary tumor','primary tumour','primary_tumor'})
        mask |= s.str.contains(r'(?:^|[^a-z])tumou?r(?:[^a-z]|$)',regex=True,na=False)
    if ident_col is not None:
        # Ident examples on GEO include P01-T-I, P01-T1-II, P01-T2-II.
        # Deliberately exclude TN, N and B compartments.
        ids=df[ident_col].astype(str)
        mask |= ids.str.contains(r'(?i)(?:^|-)T(?:[0-9]+)?-(?:I|II|III|IV)$',regex=True,na=False)
    return mask

def _infer_prepost_column(df):
    # Choose the column whose values most consistently look like Pre_P# / Post_P#.
    best=None; best_score=-1
    for c in df.columns:
        s=df[c].astype(str)
        score=float(s.str.contains(r'(?i)(?:pre|post)[_-]?P0*[0-9]+',regex=True,na=False).mean())
        if score>best_score:
            best,best_score=c,score
    return best if best_score>0 else None

def _extract_patient(text):
    # GEO sample identifiers commonly use underscores (e.g. su001_pre), for which\b
    # is not a valid boundary because '_' is a regex word character.
    m=re.search(r'(?i)(?:^|[^a-z0-9])(su|p)0*([0-9]{1,3})(?=$|[^0-9])',str(text))
    if not m: return None
    prefix=m.group(1).lower()
    n=int(m.group(2))
    return f'su{n:03d}' if prefix=='su' else f'P{n:02d}'

def _extract_time(text):
    s=str(text)
    low=s.lower()
    if 'pre' in low or 'untreated' in low: return 'pre'
    if 'post' in low or 'treated' in low and 'untreated' not in low: return 'post'
    # roman treatment levels from sample names / metadata
    for x in ('IV','III','II','I'):
        if re.search(rf'(?:^|[-_ ]){x}(?:$|[-_ ])',s): return x
    return None

def _time_rank(x):
    return {'pre':0,'I':0,'II':1,'III':2,'IV':3,'post':1}.get(str(x),999)

def _aggregate_cells(df,patient_col,time_col,state_col,min_cells=100,cohort=''):
    z=df[[patient_col,time_col,state_col]].copy(); z.columns=['patient_id','time','state']; z=z.dropna()
    z['patient_id']=z.patient_id.astype(str); z['time']=z.time.astype(str); z['state']=z.state.astype(str)
    cnt=z.groupby(['patient_id','time','state']).size().unstack(fill_value=0)
    for s in STATE_ORDER:
        if s not in cnt: cnt[s]=0
    cnt=cnt[STATE_ORDER]; cnt['total_cells']=cnt.sum(1); cnt=cnt[cnt.total_cells>=min_cells].reset_index()
    rows=[]
    for patient,g in cnt.groupby('patient_id'):
        g=g.copy(); g['rank']=g.time.map(_time_rank); g=g[g['rank']<999].sort_values(['rank','time'])
        # aggregate duplicate biological timepoint rows if any
        gg=g.groupby(['patient_id','time','rank'],as_index=False)[STATE_ORDER+['total_cells']].sum().sort_values('rank')
        for a,b in zip(range(len(gg)-1),range(1,len(gg))):
            r0,r1=gg.iloc[a],gg.iloc[b]
            src=r0[STATE_ORDER].to_numpy(float); tgt=r1[STATE_ORDER].to_numpy(float)
            if src.sum()==0 or tgt.sum()==0: continue
            src/=src.sum(); tgt/=tgt.sum()
            row={'cohort':cohort,'patient_id':patient,'pair_id':f'{patient}:{r0.time}->{r1.time}','source_time':r0.time,'target_time':r1.time,'source_cells':int(r0.total_cells),'target_cells':int(r1.total_cells)}
            for j,s in enumerate(STATE_ORDER): row[f'source__{s}']=src[j]; row[f'target__{s}']=tgt[j]
            rows.append(row)
    pair_cols=['cohort','patient_id','pair_id','source_time','target_time','source_cells','target_cells']+[f'source__{s}' for s in STATE_ORDER]+[f'target__{s}' for s in STATE_ORDER]
    return pd.DataFrame(rows,columns=pair_cols),cnt

def gse236581(paths,min_cells=100):
    df=_read_auto(paths['metadata'])
    label=find_col(df,['SubCellType','CellType','celltype','MajorCellType','Ident'])
    major=find_col(df,['MajorCellType'],required=False)
    ident=find_col(df,['Ident','orig.ident','sample','Sample'],required=False) or df.columns[0]
    tissue=find_col(df,['Tissue','tissue','Origin','site'],required=False)
    treatment=find_col(df,['Treatment','treatment','timepoint','Time','stage'],required=False)
    patient=find_col(df,['Patient','patient_id','Individual','subject'],required=False)
    z=df.copy()
    if major:
        z=z[z[major].astype(str).str.strip().str.upper().isin({'T','T CELL','T CELLS'}) | z[major].astype(str).str.upper().str.startswith('T CELL')]
    # Accept both compact Tissue='T' and full Tissue='Tumor', with Ident as a
    # second independent source of truth for the tumor compartment.
    tm=_tumor_mask(z,tissue,ident)
    z=z[tm].copy()

    # Ident on the official GEO record encodes patient and treatment stage
    # (e.g. P01-T-I -> P01, I).  Prefer that pre-specified identifier when it is
    # parseable, and use explicit metadata columns only as fallback.
    id_patient=z[ident].map(_extract_patient)
    explicit_patient=z[patient].astype(str) if patient else pd.Series([None]*len(z),index=z.index,dtype='object')
    z['_patient']=id_patient.where(id_patient.notna(),explicit_patient)
    id_time=z[ident].map(_extract_time)
    explicit_time=(z[treatment].astype(str).map(lambda x:_extract_time(x) or str(x).strip()) if treatment else pd.Series([None]*len(z),index=z.index,dtype='object'))
    z['_time']=id_time.where(id_time.notna(),explicit_time)

    mapped=z[label].map(canonical_state); unmapped=mapped.isna()
    z['_state']=mapped.fillna('Other_T')
    pairs_all,counts=_aggregate_cells(z,'_patient','_time','_state',min_cells,'GSE236581')

    # Primary longitudinal protocol: only consecutive treatment stages.  This
    # prevents a missing intermediate visit (e.g. I -> III) from being treated
    # as if it were biologically equivalent to I -> II.  Non-consecutive
    # next-observed transitions belong in a sensitivity analysis, not primary.
    allowed={('I','II'),('II','III'),('III','IV')}
    if len(pairs_all):
        keep=[(str(a),str(b)) in allowed for a,b in zip(pairs_all.source_time,pairs_all.target_time)]
        pairs=pairs_all.loc[keep].reset_index(drop=True)
    else:
        pairs=pairs_all
    transition_counts=(pairs.groupby(['source_time','target_time']).size().to_dict() if len(pairs) else {})

    audit={'rows_raw':len(df),'rows_eligible':len(z),'label_col':label,'major_col':major,'ident_col':ident,'tissue_col':tissue,'time_col':treatment,'patient_col':patient,
           'tumor_filter_note':'accepts Tissue T/Tumor and Ident Pxx-T[digits]-stage; excludes N/B/TN',
           'mapping_primary_fraction':float((~unmapped).mean()) if len(z) else 0.0,'other_t_fallback_fraction':float(unmapped.mean()) if len(z) else 0.0,
           'patients_after_filter':int(z['_patient'].dropna().nunique()) if len(z) else 0,
           'time_values_after_filter':sorted(map(str,z['_time'].dropna().unique().tolist())) if len(z) else [],
           'primary_transition_rule':'strict consecutive stages only: I->II, II->III, III->IV',
           'next_observed_pairs_before_primary_filter':int(len(pairs_all)),
           'transition_counts':{f'{a}->{b}':int(v) for (a,b),v in transition_counts.items()},
           'patients_pairs':int(pairs['patient_id'].nunique()) if len(pairs) else 0,'pairs':len(pairs)}
    return pairs,counts,audit

def gse123813(paths,min_cells=100):
    df=_read_auto(paths['metadata'])
    # Metadata variants are handled by name inference; if phenotype column is absent, use cluster-like label.
    label=find_col(df,['cell.type','cell_type','celltype','phenotype','cluster','Cluster','annotation','Ident'],required=False)
    sample=find_col(df,['sample','Sample','sample_id','orig.ident','patient'],required=False)
    if sample is None: sample=df.columns[0]
    if label is None:
        raise KeyError(f'GSE123813 metadata has no recognizable phenotype/cluster column: {list(df.columns)}')
    z=df.copy(); z['_patient']=z[sample].map(_extract_patient); z['_time']=z[sample].map(_extract_time)
    mapped=z[label].map(canonical_state); unmapped=mapped.isna(); z['_state']=mapped.fillna('Other_T')
    # If patient/time are explicit, prefer them.
    pc=find_col(df,['patient','patient_id'],required=False); tc=find_col(df,['treatment','timepoint','time'],required=False)
    if pc: z['_patient']=z[pc].astype(str)
    if tc:
        explicit=z[tc].astype(str).map(lambda x:_extract_time(x) or x)
        z['_time']=explicit
    pairs,counts=_aggregate_cells(z,'_patient','_time','_state',min_cells,'GSE123813')
    audit={'rows_raw':len(df),'label_col':label,'sample_col':sample,'patient_col':pc,'time_col':tc,'mapping_primary_fraction':float((~unmapped).mean()) if len(z) else 0.0,'other_t_fallback_fraction':float(unmapped.mean()) if len(z) else 0.0,'patients_pairs':pairs.patient_id.nunique() if len(pairs) else 0,'pairs':len(pairs)}
    return pairs,counts,audit

def gse120575(paths,min_cells=100):
    # The GEO patient-ID export contains cp1252 symbols (e.g. micro/degree) in
    # the protocol tail.  pandas' Python parser may decode beyond nrows, so UTF-8
    # can fail even though the cell rows themselves are valid.  cp1252 is the
    # native-safe decoder for this official export; latin1 is a last fallback.
    last=None; meta=None; used_encoding=None
    for enc in ('cp1252','latin1','utf-8-sig','utf-8'):
        try:
            meta=pd.read_csv(paths['patient_metadata'],sep='\t',skiprows=19,nrows=16291,engine='python',encoding=enc)
            if meta.shape[1]>1:
                used_encoding=enc; break
        except Exception as e:
            last=e; meta=None
    if meta is None:
        raise RuntimeError(f'cannot parse GSE120575 patient metadata: {last}')
    clu=_read_auto(paths['cluster_info'])
    cell_meta=find_col(meta,['title','cell','Cell.Name','sample title'])
    cell_clu=find_col(clu,['Cell.Name','cell','title'])
    cluster=find_col(clu,['Cluster.number','cluster'])

    # Avoid relying on a misspelled/unstable GEO column name. Infer the column
    # containing values such as Pre_P1 and Post_P1 directly from its contents.
    lesion=_infer_prepost_column(meta)
    if lesion is None:
        lesion=find_col(meta,['patient.ID','patinet.ID','lesion','sample'],required=False)
    if lesion is None:
        raise KeyError(f'GSE120575: cannot identify Pre/Post patient column; columns={list(meta.columns)[:80]}')

    cluster_map={6:'Exhausted',7:'Regulatory',8:'Effector',9:'Exhausted',10:'Memory',11:'Exhausted'}
    z=meta[[cell_meta,lesion]].copy().merge(clu[[cell_clu,cluster]],left_on=cell_meta,right_on=cell_clu,how='inner')
    z['_patient']=z[lesion].map(_extract_patient)
    z['_time']=z[lesion].map(_extract_time)
    z['_state']=pd.to_numeric(z[cluster],errors='coerce').map(cluster_map)
    z=z.dropna(subset=['_patient','_time','_state'])
    pairs,counts=_aggregate_cells(z,'_patient','_time','_state',min_cells,'GSE120575')
    audit={'rows_meta':len(meta),'rows_cluster':len(clu),'rows_merged_mapped':len(z),'encoding':used_encoding,
           'cell_id_col':cell_meta,'prepost_col':lesion,'cluster_col':cluster,
           'patients_pairs':int(pairs['patient_id'].nunique()) if len(pairs) else 0,'pairs':len(pairs),'cluster_mapping':cluster_map}
    return pairs,counts,audit

def load_pair_csv(path,cohort):
    df=pd.read_csv(path)
    # Support legacy state names by constructing broad canonical mapping.
    source_cols=[c for c in df if c.startswith('source__')]; target_cols=[c for c in df if c.startswith('target__')]
    old_states=[c.split('__',1)[1] for c in source_cols]
    rows=[]
    for _,r in df.iterrows():
        sm={s:0.0 for s in STATE_ORDER}; tm={s:0.0 for s in STATE_ORDER}
        for os in old_states:
            cs=canonical_state(os) or 'Other_T'; sm[cs]+=float(r.get('source__'+os,0)); tm[cs]+=float(r.get('target__'+os,0))
        ss=sum(sm.values()); tt=sum(tm.values())
        if ss<=0 or tt<=0: continue
        row={'cohort':cohort,'patient_id':str(r['patient_id']),'pair_id':str(r.get('pair_id',r.name)),'source_time':r.get('source_time','src'),'target_time':r.get('target_time','tgt'),'source_cells':r.get('source_total_cells',r.get('source_cells',np.nan)),'target_cells':r.get('target_total_cells',r.get('target_cells',np.nan))}
        for s in STATE_ORDER: row['source__'+s]=sm[s]/ss; row['target__'+s]=tm[s]/tt
        rows.append(row)
    return pd.DataFrame(rows)
