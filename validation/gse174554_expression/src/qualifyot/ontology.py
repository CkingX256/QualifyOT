from __future__ import annotations
import re
import pandas as pd

STATE_ORDER=['Memory','Effector','Exhausted','Regulatory','Other_T']

# Frozen, outcome-blind mappings for author-provided GSE236581 T-cell labels.
# Only labels whose author annotation itself clearly identifies a broad state are
# assigned to one of the four mechanistic bins. Specialized/innate-like/cycling
# populations intentionally fall through to Other_T rather than being forced.
_GSE236581_EXACT = {
    # Memory / naive-like
    'c05_cd4_tcm_gpr183':'Memory',
    'c03_cd4_tn_nr4a2':'Memory',
    'c18_cd8_tcm_anxa1':'Memory',
    'c16_cd8_tn_sell':'Memory',
    'c04_cd4_tcm_anxa1':'Memory',
    'c01_cd4_tn_ccr7':'Memory',
    'c17_cd8_tcm_gpr183':'Memory',
    'c02_cd4_tn_sell':'Memory',
    'c15_cd8_tn_ccr7':'Memory',
    # Effector / resident-like
    'c20_cd8_tem_gzmk':'Effector',
    'c22_cd8_trm_hspa1b':'Effector',
    'c06_cd4_trm_hspa1a':'Effector',
    'c21_cd8_trm_xcl1':'Effector',
    'c19_cd8_tem_cmc1':'Effector',
    'c24_cd8_temra_cx3cr1':'Effector',
    'c25_cd8_temra_tyrobp':'Effector',
    'c10_cd4_temra_gzmb':'Effector',
    # Exhausted
    'c23_cd8_tex_layn':'Exhausted',
    # Regulatory
    'c13_cd4_treg_tnfrsf9':'Regulatory',
    'c12_cd4_treg_klrb1':'Regulatory',
    'c11_cd4_treg_foxp3':'Regulatory',
}

def canonical_state(label):
    s=str(label).strip().lower().replace('−','-')
    if s in _GSE236581_EXACT:
        return _GSE236581_EXACT[s]
    # specific regulatory first
    if re.search(r'\btreg\b|regulatory|foxp3|cd4[_ -]?tr',s): return 'Regulatory'
    if re.search(r'terminal.*exhaust|exhaust|\btex\b|dysfunc|pd-?1.*high|cd8.*ex',s): return 'Exhausted'
    if re.search(r'effector|cytotox|tem\b|activated|\bact\b|nk.?like',s): return 'Effector'
    if re.search(r'memory|naive|stem|tpex|progenitor|tcf7|tcf1|central|cd4[_ -]?t\b|cd8[_ -]?memory',s): return 'Memory'
    if re.search(r'\bt\b|lymph|cd4|cd8|tcell|t cell',s): return 'Other_T'
    return None

def map_labels(labels):
    return pd.Series([canonical_state(x) for x in labels],index=getattr(labels,'index',None),dtype='object')
