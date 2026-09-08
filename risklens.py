"""Historical research-portfolio risk evaluation, not an executable strategy."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import argparse
import hashlib
import io
import json
import platform
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm,chi2
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
URL='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/48_Industry_Portfolios_daily_CSV.zip'
SOURCE_SHA='d7701a576a75b2e1630de546ef4ff614cb54edf62bba52f503a975d50e287b60'
WINDOW=252
CAP=.15
STRESS={'covid_2020':['2020-02-19','2020-04-30'],
        'inflation_2022':['2022-01-01','2022-12-31'],
        'april_2025':['2025-04-01','2025-04-30']}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def parse_csv(text):
    lines=text.splitlines()
    start=next(i for i,line in enumerate(lines) if 'Average Value Weighted Returns -- Daily' in line)+1
    stop=start+1
    while stop<len(lines) and lines[stop].strip() and lines[stop].split(',')[0].strip().isdigit():
        stop+=1
    frame=pd.read_csv(io.StringIO('\n'.join(lines[start:stop])))
    dates=pd.to_datetime(frame.iloc[:,0].astype(str),format='%Y%m%d')
    frame=frame.iloc[:,1:].apply(pd.to_numeric)
    frame.columns=frame.columns.str.strip()
    frame.index=pd.DatetimeIndex(dates,name='date')
    if len(frame.columns)!=48 or frame.index.duplicated().any():
        raise ValueError('Expected 48 industries and unique dates')
    frame=frame.loc['1990-01-01':'2025-12-31'].replace([-99.99,-999.],np.nan)
    missing=int(frame.isna().any(axis=1).sum())
    frame=frame.dropna()/100.
    if not np.isfinite(frame).all().all() or (frame<=-1).any().any():
        raise ValueError('Invalid simple returns')
    return frame.sort_index(),missing

def prepare(accept_new=False):
    (ROOT/'data').mkdir(exist_ok=True);(ROOT/'outputs').mkdir(exist_ok=True)
    path=ROOT/'data/48_Industry_Portfolios_daily_CSV.zip'
    if not path.exists():
        response=requests.get(URL,timeout=120);response.raise_for_status()
        temporary=path.with_suffix('.partial');temporary.write_bytes(response.content);temporary.rename(path)
    digest=sha256(path)
    if digest!=SOURCE_SHA and not accept_new:
        raise ValueError('Source vintage differs from published run. Retain original local ZIP or explicitly --accept-new-vintage for a NEW experiment.')
    with zipfile.ZipFile(path) as archive:
        member=next(n for n in archive.namelist() if n.lower().endswith('.csv'))
        text=archive.read(member).decode('utf-8-sig')
    frame,missing=parse_csv(text)
    frame.to_csv(ROOT/'data/returns.csv')
    report=dict(url=URL,source_sha256=digest,source_metadata=text.splitlines()[0],
        table='Average Value Weighted Returns -- Daily',source_percent_returns_converted_to_decimal=True,
        first_date=str(frame.index.min().date()),last_date=str(frame.index.max().date()),
        rows=len(frame),industries=frame.columns.tolist(),missing_dates_dropped=missing,
        missing_policy='Drop date containing any -99.99/-999 sentinel; never forward-fill',
        output_sha256=sha256(ROOT/'data/returns.csv'),vintage_matches_published=digest==SOURCE_SHA,
        limitation='Current revised research series; not point-in-time constituents or investable funds')
    (ROOT/'outputs/provenance.json').write_text(json.dumps(report,indent=2))
    return frame

def covariance(x,method,decay=.97):
    if len(x)<2 or not np.isfinite(x).all():raise ValueError('Covariance requires finite historical rows')
    if method=='ledoit_wolf':
        model=LedoitWolf().fit(x);cov=model.covariance_;shrinkage=float(model.shrinkage_)
    elif method=='ewma':
        weights=decay**np.arange(len(x)-1,-1,-1,dtype=float);weights/=weights.sum()
        center=x-weights@x
        cov=(center.T*weights)@center/(1-float(weights@weights));shrinkage=None
    elif method=='sample':cov=np.cov(x,rowvar=False,ddof=1);shrinkage=None
    else:raise ValueError('Unknown covariance estimator')
    cov=(cov+cov.T)/2
    cov+=np.eye(cov.shape[0])*max(float(np.trace(cov)/len(cov))*1e-8,1e-12)
    return cov,shrinkage

def min_variance(cov,cap=CAP):
    n=len(cov)
    if n*cap<1-1e-10:raise ValueError('Infeasible weight cap')
    scale=max(float(np.trace(cov)/n),1e-12)
    matrix=cov/scale
    answer=minimize(lambda w:float(w@matrix@w),np.ones(n)/n,
        jac=lambda w:2*matrix@w,method='SLSQP',bounds=[(0,cap)]*n,
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n)}],
        options={'maxiter':250,'ftol':1e-10})
    if not answer.success:raise RuntimeError('Optimizer failure: '+answer.message)
    w=answer.x
    if abs(w.sum()-1)>1e-8 or w.min()<-1e-8 or w.max()>cap+1e-8:raise ValueError('Constraint violation')
    return w

def drift(weights,returns):
    result=weights*(1+returns)
    if result.sum()<=0:raise ValueError('Portfolio wiped out')
    return result/result.sum()

def replay(frame,start,end,strategy,decay=.97):
    indexes=np.flatnonzero((frame.index>=start)&(frame.index<=end))
    if not len(indexes) or indexes[0]<WINDOW:raise ValueError('Insufficient warm-up')
    x=frame.to_numpy();n=x.shape[1];current=np.ones(n)/n;previous_month=None
    records=[];audits=[];cov=None
    for i in indexes:
        date=frame.index[i];month=date.to_period('M');traded=0.
        if month!=previous_month:
            history=x[i-WINDOW:i]
            if frame.index[i-1]>=date:raise ValueError('Look-ahead in covariance history')
            method=strategy if strategy in ['sample','ewma','ledoit_wolf'] else 'sample'
            cov,shrinkage=covariance(history,method,decay)
            if strategy=='equal_weight':target=np.ones(n)/n
            elif strategy=='inverse_vol':
                target=1/np.sqrt(np.diag(cov));target/=target.sum()
            else:target=min_variance(cov)
            traded=float(np.abs(target-current).sum());current=target
            eigen=np.linalg.eigvalsh(cov)
            audits.append(dict(date=str(date.date()),history_start=str(frame.index[i-WINDOW].date()),
                history_end=str(frame.index[i-1].date()),history_rows=WINDOW,
                one_way_turnover=traded/2,max_weight=float(current.max()),
                covariance_condition=float(eigen[-1]/eigen[0]),min_eigenvalue=float(eigen[0]),
                shrinkage=shrinkage))
            previous_month=month
        gross=float(current@x[i])
        sigma=float(np.sqrt(current@cov@current))
        records.append(dict(date=date,gross_return=gross,traded_notional=traded,
            one_way_turnover=traded/2,sigma=sigma,var95=float(norm.ppf(.95)*sigma),
            var99=float(norm.ppf(.99)*sigma)))
        current=drift(current,x[i])
    return pd.DataFrame(records).set_index('date'),audits

def drawdown(returns):
    wealth=np.r_[1.,np.cumprod(1+returns)]
    return float(np.min(wealth/np.maximum.accumulate(wealth)-1))

def var_audit(returns,var,level):
    exceptions=int((returns < -var).sum());n=len(returns);expected=1-level
    empirical=exceptions/n
    def binomial_log(p):
        if p==0:return 0. if exceptions==0 else -np.inf
        if p==1:return 0. if exceptions==n else -np.inf
        return exceptions*np.log(p)+(n-exceptions)*np.log1p(-p)
    lr=max(0.,2*(binomial_log(empirical)-binomial_log(expected)))
    return dict(exceptions=exceptions,observations=n,exception_rate=empirical,
        nominal_exception_rate=expected,kupiec_lr=lr,kupiec_pvalue=float(chi2.sf(lr,1)))

def summarize(result):
    r=result.gross_return.to_numpy();years=len(r)/252
    summary=dict(days=len(r),annualized_volatility=float(np.std(r,ddof=1)*np.sqrt(252)),
        max_drawdown=drawdown(r),annual_one_way_turnover=float(result.one_way_turnover.sum()/years),
        var95=var_audit(r,result.var95.to_numpy(),.95),var99=var_audit(r,result.var99.to_numpy(),.99),cost_sensitivity={})
    for bps in [0,5,10,25]:
        net=(1-result.traded_notional.to_numpy()*bps/10000)*(1+r)-1
        summary['cost_sensitivity'][str(bps)]=dict(annualized_return=float(np.prod(1+net)**(1/years)-1),
            annualized_volatility=float(np.std(net,ddof=1)*np.sqrt(252)),max_drawdown=drawdown(net))
    return summary

def run():
    started=time.time();out=ROOT/'outputs';out.mkdir(exist_ok=True)
    frame=pd.read_csv(ROOT/'data/returns.csv',index_col=0,parse_dates=True)
    # Only development returns participate in EWMA hyperparameter selection.
    development={}
    for decay in [.94,.97,.99]:
        result,_=replay(frame,'2013-01-01','2018-12-31','ewma',decay)
        development[str(decay)]=summarize(result)
    chosen=min(development,key=lambda k:development[k]['annualized_volatility'])
    report=dict(protocol='Fixed 252-day history and monthly rebalance; EWMA decay selected on 2013-2018 volatility only; frozen 2019-2025 evaluation',
        data_sha256=sha256(ROOT/'data/returns.csv'),historical_data_period=['1990-01-01','2012-12-31'],
        development_period=['2013-01-01','2018-12-31'],test_period=['2019-01-01','2025-12-31'],
        window=WINDOW,weight_cap=CAP,selected_ewma_decay=float(chosen),development_grid=development,
        cost_definition='bps charged on sum(abs(target minus drifted pre-trade weights)); exact multiplicative wealth deduction before daily return',
        risk_definition='Zero-mean Gaussian 95/99% one-day VaR; covariance frozen between monthly updates, daily drifted weights; gross returns evaluated',
        initial_state='Equal weights before first day of each evaluation period; first rebalance turnover is charged',
        stress_windows=STRESS,test={},covariance_audits={},stress_results={},
        runtime=dict(python=platform.python_version(),platform=platform.platform(),threads=2),
        limitations=['Research portfolios are not executable instruments or point-in-time constituents',
            'Revised historical CRSP snapshot and 2025 methodology transition limit investability claims',
            'Hypothetical costs exclude spreads, financing, internal constituent turnover and market impact',
            'Gaussian VaR with monthly-frozen covariance may fail in crises; Kupiec test assumes idealized observations',
            'Stress windows are historical illustrations, not independent confirmation or regulatory validation'])
    streams={}
    for strategy in ['equal_weight','inverse_vol','sample','ewma','ledoit_wolf']:
        result,audits=replay(frame,'2019-01-01','2025-12-31',strategy,float(chosen))
        streams[strategy]=result
        report['test'][strategy]=summarize(result)
        report['covariance_audits'][strategy]=audits
        report['stress_results'][strategy]={name:summarize(result.loc[a:b]) for name,(a,b) in STRESS.items()}
        result.to_csv(ROOT/f'data/{strategy}_test.csv')
        print(strategy,report['test'][strategy]['annualized_volatility'],flush=True)
    report['elapsed_seconds']=time.time()-started
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for strategy,result in streams.items():
        rolling=result.gross_return.rolling(63).std()*np.sqrt(252)*100
        axes[0].plot(result.index,rolling,label=strategy,linewidth=1)
        wealth=(1+result.gross_return).cumprod()
        # Include initial wealth=1 in drawdown peaks, matching summary computation.
        peaks=np.maximum.accumulate(np.r_[1.,wealth.to_numpy()])[1:]
        axes[1].plot(result.index,(wealth/peaks-1)*100,label=strategy,linewidth=1)
    axes[0].set(title='Held-out rolling risk (63 trading days)',ylabel='Annualized volatility (%)')
    axes[1].set(title='Held-out drawdown: hypothetical research baskets',ylabel='Drawdown (%)')
    for ax in axes:ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=30)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'risk-comparison.png',dpi=160)
    print(json.dumps({k:{a:v[a] for a in ['annualized_volatility','max_drawdown','annual_one_way_turnover']} for k,v in report['test'].items()},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--accept-new-vintage',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=2):
        prepare(args.accept_new_vintage) if args.command=='prepare' else run()
