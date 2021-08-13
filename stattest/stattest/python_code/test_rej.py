#testing the inference algorithm based on
#rejection sampling method

from sampling_algorithms.rejection import supx, acc_rej_samp
import numpy as np
import scipy.stats as stats
from inference import beta_bernoulli, compare, kstest_cdf
from stat_test import kstest

#if __name__=='__main__':
if True:
    N=1000
    #print (True)
    param=(5,2)
    obs=[1,1,1,0,1,1,1]
    newparam=beta_bernoulli(param, obs)
    
    def func(x):
        r=1
        for i in obs:
            r*=stats.bernoulli.pmf(k=i, p=x)
        return r
    
    S=supx(func, (0,1))
    g_pdf=lambda x: stats.beta.pdf(x, *param)
    g_samp=lambda x: stats.beta.rvs(*param, size=x)
    AR=acc_rej_samp(func, g_pdf=g_pdf, g_samp=g_samp, supX=S, N=N)
    
    exact=stats.beta(*newparam)
    
    cdf=exact.cdf
    exactsample=exact.rvs(N*10)
    print (kstest_cdf(AR, obs, param, 'beta_bernoulli', plotp=True))
    #print (compare(posterior=AR, obs=obs, parameters=param, distribution_name='beta_bernoulli', factor=10))
    
    
    
    