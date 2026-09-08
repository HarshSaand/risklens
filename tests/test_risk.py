import numpy as np
import pandas as pd
import pytest
from risklens import covariance,min_variance,drift,drawdown,var_audit,replay

def test_covariance_psd_and_symmetric():
    x=np.random.default_rng(1).normal(size=(300,10))*.01
    for method in ['sample','ewma','ledoit_wolf']:
        cov,_=covariance(x,method)
        assert np.allclose(cov,cov.T)
        assert np.linalg.eigvalsh(cov).min()>0

def test_optimizer_constraints():
    w=min_variance(np.diag(np.arange(1,11)),cap=.2)
    assert abs(w.sum()-1)<1e-8 and w.min()>=-1e-8 and w.max()<=.2+1e-8

def test_infeasible_cap_rejected():
    with pytest.raises(ValueError):min_variance(np.eye(3),cap=.2)

def test_holdings_drift():
    assert np.allclose(drift(np.array([.5,.5]),np.array([.1,0.])),[.55/1.05,.5/1.05])

def test_drawdown_includes_starting_capital():
    assert np.isclose(drawdown(np.array([-.2,0.])), -.2)

def test_var_exception_count():
    result=var_audit(np.array([-.1,.01,-.02]),np.full(3,.03),.95)
    assert result['exceptions']==1 and result['observations']==3

def test_future_returns_do_not_change_today_risk_or_turnover():
    rng=np.random.default_rng(2);dates=pd.bdate_range('2011-01-01',periods=400)
    frame=pd.DataFrame(rng.normal(0,.01,(400,10)),index=dates)
    a,audit=replay(frame,str(dates[280].date()),str(dates[310].date()),'sample')
    changed=frame.copy();changed.loc[dates[290]:]*=3
    b,_=replay(changed,str(dates[280].date()),str(dates[310].date()),'sample')
    assert np.allclose(a.loc[:dates[290],'sigma'],b.loc[:dates[290],'sigma'])
    assert np.allclose(a.loc[:dates[290],'traded_notional'],b.loc[:dates[290],'traded_notional'])
    assert all(item['history_end']<item['date'] for item in audit)
