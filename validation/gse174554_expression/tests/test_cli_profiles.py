import json
import pandas as pd
from qualifyot.cli import main


def test_cli_profile_patient(tmp_path,capsys):
    p=tmp_path/'effects.csv'
    pd.DataFrame({'patient_id':[f'p{i}' for i in range(8)],'PUC_contribution':[.01,.02,.03,.02,.01,.03,.02,.025],
                  'NPI_contribution':[.005,.01,.02,.01,.005,.015,.01,.012]}).to_csv(p,index=False)
    assert main(['profile-patient',str(p),'--bootstrap','200','--draws','500','--seed','4'])==0
    out=json.loads(capsys.readouterr().out)
    assert out['schema']=='qualifyot-patient-profile' and out['n_patients']==8


def test_cli_graph_profile(tmp_path,capsys):
    p=tmp_path/'loss.csv'
    pd.DataFrame({'patient_id':[f'p{i}' for i in range(12)],'g0':[.1]*12,'g1':[.05]*12,'g2':[.15]*12}).to_csv(p,index=False)
    assert main(['graph-profile',str(p),'--models','g0,g1,g2','--bootstrap','200','--draws','500','--seed','5'])==0
    out=json.loads(capsys.readouterr().out)
    assert out['schema']=='qualifyot-graph-profile'
    assert out['confidence_set']['members']==['g1']
