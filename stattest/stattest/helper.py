import pandas as pd
import numpy as np

from .python_code.inference import *

def beta_bernoulli(posterior, obs, parameters):
    posterior =  pd.read_csv(posterior, header=None)
    posterior=np.asarray(posterior).flatten()
    posterior=posterior[np.logical_not(np.isnan(posterior))]

    obs=pd.read_csv(obs, header=None)
    obs=obs[np.logical_not(np.isnan(np.asarray(obs).flatten()))]

    return kstest(posterior, obs, parameters, 'beta_bernoulli')

    
