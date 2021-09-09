from stattest.helper import to_1darray
from .helper import *
import pymc3 as pm
import numpy as np

def test_sample(mean0, cov0, cov, obs, size):
    dim=len(mean0)
    obs=to_ndarray(obs)
    with pm.Model() as model:
        p=pm.MvNormal('p', mu=mean0, cov=cov0, shape=len(mean0))
        y=pm.MvNormal('y', mu=p, cov=cov, observed=obs)
        trace=pm.sample(int(size/4))
        np.savetxt('test_samp.csv', trace['p'], delimiter=',')
    return open('test_samp.csv')