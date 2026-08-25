from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def write_qualification_report(result, path, title='QualifyOT patient-level qualification report'):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=16*mm,bottomMargin=16*mm)
    styles=getSampleStyleSheet(); story=[Paragraph(title,styles['Title']),Spacer(1,5*mm)]
    decision='QUALIFIED FOR RETENTION' if result['qualified'] else 'NOT QUALIFIED FOR RETENTION'
    story.append(Paragraph(f"<b>Decision:</b> {decision}",styles['Heading2']))
    story.append(Paragraph('This report evaluates predictive evidence. It is not a clinical safety certificate, causal claim or regulatory approval.',styles['BodyText']))
    story.append(Spacer(1,4*mm))
    data=[['Quantity','Estimate','95% interval / detail'],
          ['Candidate',result['candidate'],''],
          ['Physical patients',str(result['patients']),f"{result['pairs']} longitudinal pairs"],
          ['PDR',f"{result['PDR']:.4g}",f"[{result['PDR_lo']:.4g}, {result['PDR_hi']:.4g}]"],
          ['PUC',f"{result['PUC']:.4g}",f"[{result['PUC_lo']:.4g}, {result['PUC_hi']:.4g}]"],
          ['NPI',f"{result['NPI']:.4g}",f"[{result['NPI_lo']:.4g}, {result['NPI_hi']:.4g}]"],
          ['Positive-weight folds',f"{100*result['positive_weight_fold_fraction']:.1f}%",'operational stability diagnostic'],
          ['Reference risk',f"{result['reference_risk']:.5f}",'patient-equal MAE'],
          ['Qualified-predictor risk',f"{result['qualified_risk']:.5f}",'patient-equal MAE']]
    t=Table(data,colWidths=[52*mm,48*mm,70*mm])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAEFF5')),('TEXTCOLOR',(0,0),(-1,0),colors.black),('GRID',(0,0),(-1,-1),0.4,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F8F8F8')])]))
    story += [t,Spacer(1,5*mm),Paragraph('Interpretation',styles['Heading2']),Paragraph('A candidate is retained only when the pre-specified deviation, direct-utility, mixing-weight stability and net-improvement conditions are all met. Failure to qualify does not establish that a biological mechanism is absent; it means that the available patient-level predictive evidence is insufficient for retention under this protocol.',styles['BodyText'])]
    doc.build(story); return path
