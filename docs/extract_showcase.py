from pathlib import Path
import argparse, json, hashlib, sys
p=argparse.ArgumentParser();p.add_argument('--source', type=Path, required=True);a=p.parse_args()
S=a.source.resolve(); D=Path(__file__).resolve().parent; R=D.parent
D.mkdir(exist_ok=True)
import numpy as np
import pandas as pd
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def source(rel):return {'path':rel,'sha256':digest(S/rel)}
def write(data):
 import subprocess
 data['dataset_url']='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html'
 data['source_code_commit']=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
 data['extractor_sha256']=digest(Path(__file__))
 data['sources']=sources
 data['source_repository']='https://github.com/HarshSaand/'+R.name
 data['extraction']='python docs/extract_showcase.py --source /path/to/reproduced/project'
 (D/'output-example.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n')
sys.path.insert(0,str(R));from risklens import covariance,min_variance
f='data/returns.csv';df=pd.read_csv(S/f,index_col=0,parse_dates=True);i=np.flatnonzero(df.index>=pd.Timestamp('2019-01-01'))[0];hist=df.iloc[i-252:i];cov,_=covariance(hist.to_numpy(),'ewma',.99);w=min_variance(cov);sigma=float(np.sqrt(w@cov@w));allocation=sorted([dict(industry=n,weight=float(v)) for n,v in zip(df.columns,w)],key=lambda r:-r['weight']);sources=[source(f),source('outputs/evaluation.json')]
assert abs(w.sum()-1)<1e-8 and w.max()<=.15+1e-8
rows=[[str(i+1),r['industry'],f"{r['weight']:.2%}"] for i,r in enumerate(allocation[:10])]
write(dict(title='RiskLens',subtitle='A dated, constrained portfolio allocation',eyebrow='ACTUAL OPTIMIZER OUTPUT',context=f"{df.index[i].date()} · EWMA 0.99 · 252 prior trading days · 15% cap",columns=['Rank','Industry research basket','Allocation'],rows=rows,raw=dict(date=str(df.index[i].date()),history_start=str(hist.index[0].date()),history_end=str(hist.index[-1].date()),weights=allocation,sum_weights=float(w.sum()),max_weight=float(w.max()),daily_sigma=sigma),note='Ten largest weights shown; the JSON contains all 48. Weights sum to one and respect the cap. Historical industry research baskets are not executable instruments; this is not an investment recommendation.',input='252-day historical 48-industry return matrix',output='48 constrained minimum-variance allocation weights'))
