from math import dist
import pandas as pd
import numpy as np

from .python_code.inference import *

def beta_bernoulli(posterior, obs, parameters):
    posterior =  pd.read_csv(posterior, header=None)
    posterior=np.asarray(posterior).flatten()
    posterior=posterior[np.logical_not(np.isnan(posterior))]

    obs=pd.read_csv(obs, header=None)
    obs=np.asarray(obs).flatten()
    obs=obs[np.logical_not(np.isnan(obs))]
    N=len(posterior)

    return test_cdf(posterior, obs, parameters, distribution_name='beta_bernoulli'), benchmark(obs, parameters, 'beta_bernoulli', N=N)


