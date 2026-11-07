"""Check the reported 128-case comparison using bundled numerical results."""
import csv
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'results'
means={}
for name in ['main_em_dps','main_no_dps']:
 rows=list(csv.DictReader((root/(name+'_dps_best_by_case.csv')).open()))
 assert len(rows)==128
 means[name]=sum(float(r['fdtd_mse']) for r in rows)/128
 print(name,'cases=',len(rows),'mean waveform MSE=',means[name])
print('Reduction (%)=',100*(1-means['main_em_dps']/means['main_no_dps']))
