from __future__ import annotations
import hashlib, math
import numpy as np, pandas as pd
from dataclasses import dataclass

class GraphRefinementCandidate:
    """Training-only finite-library GraphFlow selector.

    The graph is chosen using patient-level CV *inside fit()* only. Held-out rows
    supplied to predict() never enter graph choice. This object is compatible
    with QualifyOT's generic nested LOPO engine.
    """
    def __init__(self, states, graph_library, model_mod, base_name='GraphRefine', alpha=1.0,
                 selector='one_se', cv_folds=3, all_pairs=None, shared=None):
        self.states=list(states); self.graph_library=[tuple(tuple(e) for e in g) for g in graph_library]
        self.model_mod=model_mod; self.alpha=float(alpha); self.selector=selector; self.cv_folds=int(cv_folds)
        self.name=f'{base_name}_{selector}_K{cv_folds}'
        self._model=None; self.selected_edges=None; self.selection_table=None
        if shared is None:
            shared={'flow':{}, 'log':[]}
        self.shared=shared
        if all_pairs is not None:
            self._prime_flow_cache(all_pairs)
    def fresh(self):
        return GraphRefinementCandidate(self.states,self.graph_library,self.model_mod,
            base_name=self.name.split('_')[0],alpha=self.alpha,selector=self.selector,
            cv_folds=self.cv_folds,all_pairs=None,shared=self.shared)
    def state_names(self): return list(self.states)
    def validate_predictions(self,pred,n_rows,atol=1e-7):
        pred=np.asarray(pred,float)
        if pred.shape!=(n_rows,len(self.states)): raise ValueError((pred.shape,n_rows,len(self.states)))
        if not np.isfinite(pred).all(): raise ValueError('nonfinite')
        if (pred < -atol).any() or (np.abs(pred.sum(1)-1)>atol).any(): raise ValueError('not simplex')
        return pred
    def _gkey(self,g): return '|'.join(f'{a}->{b}' for a,b in g)
    def _prime_flow_cache(self,pairs):
        mm=self.model_mod
        _,S,T=mm.feature_matrix(pairs,states=self.states)
        ids=pairs.pair_id.astype(str).tolist()
        for g in self.graph_library:
            key=self._gkey(g); B=mm.incidence(self.states,g)
            for pid,s,t in zip(ids,S,T):
                ck=(key,pid)
                if ck not in self.shared['flow']:
                    self.shared['flow'][ck]=mm.residual_flow(s,t,Bmat=B)[0]
    def _flows(self,pairs,g):
        mm=self.model_mod; key=self._gkey(g); B=mm.incidence(self.states,g)
        _,S,T=mm.feature_matrix(pairs,states=self.states)
        out=[]
        for pid,s,t in zip(pairs.pair_id.astype(str),S,T):
            ck=(key,pid)
            if ck not in self.shared['flow']:
                self.shared['flow'][ck]=mm.residual_flow(s,t,Bmat=B)[0]
            out.append(self.shared['flow'][ck])
        return np.vstack(out)
    def _fold_ids(self, patients):
        u=sorted(set(map(str,patients)))
        k=min(self.cv_folds,max(2,len(u)//3)) if len(u)>=6 else max(2,min(self.cv_folds,len(u)))
        # deterministic balanced assignment via stable hash ordering
        uh=sorted(u,key=lambda x: hashlib.sha256(x.encode()).hexdigest())
        return {p:j%k for j,p in enumerate(uh)},k
    def fit(self,pairs):
        mm=self.model_mod; pairs=pairs.reset_index(drop=True).copy()
        X,S,T=mm.feature_matrix(pairs,states=self.states); P=pairs.patient_id.astype(str).to_numpy()
        if len(np.unique(P))<4:
            # small fallback: use simplest graph, deterministic
            g=min(self.graph_library,key=lambda z:(len(z),self._gkey(z)))
            B=mm.incidence(self.states,g); F=self._flows(pairs,g)
            self._model=mm.fit_graphflow(X,S,T,alpha=self.alpha,flow_labels=F,Bmat=B,edges=g,patients=P,patient_balanced=True)
            self.selected_edges=g; return self
        fmap,k=self._fold_ids(P)
        rows=[]
        for g in self.graph_library:
            B=mm.incidence(self.states,g); F=self._flows(pairs,g); losses=[]; lpats=[]
            for f in range(k):
                te=np.array([fmap[x]==f for x in P]); tr=~te
                if not te.any() or len(np.unique(P[tr]))<2: continue
                gm=mm.fit_graphflow(X[tr],S[tr],T[tr],alpha=self.alpha,flow_labels=F[tr],Bmat=B,edges=g,patients=P[tr],patient_balanced=True)
                pred=gm.predict(X[te],S[te]); losses.extend(mm.mae(T[te],pred)); lpats.extend(P[te])
            losses=np.asarray(losses,float); lpats=np.asarray(lpats,str)
            # each patient usually one row; preserve patient-equal definition for generality
            per=[]
            for p in np.unique(lpats): per.append(float(losses[lpats==p].mean()))
            per=np.asarray(per,float); mean=float(per.mean()); se=float(per.std(ddof=1)/math.sqrt(len(per))) if len(per)>1 else float('inf')
            rows.append({'edges':g,'graph':self._gkey(g),'n_edges':len(g),'cv_risk':mean,'cv_se':se,'per_patient':per})
        best=min(rows,key=lambda r:(r['cv_risk'],r['n_edges'],r['graph']))
        if self.selector=='best':
            chosen=best
        elif self.selector=='one_se':
            threshold=best['cv_risk']+best['cv_se']
            eligible=[r for r in rows if r['cv_risk']<=threshold+1e-15]
            chosen=min(eligible,key=lambda r:(r['n_edges'],r['cv_risk'],r['graph']))
        elif self.selector=='half_se':
            threshold=best['cv_risk']+0.5*best['cv_se']
            eligible=[r for r in rows if r['cv_risk']<=threshold+1e-15]
            chosen=min(eligible,key=lambda r:(r['n_edges'],r['cv_risk'],r['graph']))
        else: raise KeyError(self.selector)
        g=chosen['edges']; B=mm.incidence(self.states,g); F=self._flows(pairs,g)
        self._model=mm.fit_graphflow(X,S,T,alpha=self.alpha,flow_labels=F,Bmat=B,edges=g,patients=P,patient_balanced=True)
        self.selected_edges=g
        self.selection_table=pd.DataFrame([{k:v for k,v in r.items() if k!='per_patient'} for r in rows]).sort_values(['cv_risk','n_edges'])
        self.shared['log'].append({'n_patients':len(np.unique(P)),'patients':tuple(sorted(np.unique(P))), 'selector':self.selector,
                                   'selected_graph':self._gkey(g),'selected_n_edges':len(g),'best_graph':best['graph'],
                                   'best_risk':best['cv_risk'],'selected_risk':chosen['cv_risk'],'best_se':best['cv_se']})
        return self
    def predict(self,pairs):
        if self._model is None: raise RuntimeError('not fitted')
        X,S,_=self.model_mod.feature_matrix(pairs,states=self.states)
        return self.validate_predictions(self._model.predict(X,S),len(pairs))
