import scipy.stats as stats
from stat_test import ecdf_x, reorder
import numpy as np
import pandas as pd
#from inference import import_sample

def ecdf_x_ndim(x, dist, weights):
    #x is tuple or list, dist and weights are dataframe
    dim=len(x)
    n=len(dist)
    res=1
    for i in range(dim):
        sample1=np.asarray(dist.loc[:,i])
        weights1=np.asarray(weights.loc[:,i])
        res*=ecdf_x(x[i], reorder(sample1, weights1))
    return res

def mutlivariate_kstest(samples,cdf, args, weights=[]):
    #samples and weights are pandas dataframes
    if len(weights)==0:
        weights=pd.DataFrame(np.ones(samples.shape))
    D= np.max([abs(ecdf_x_ndim(np.asarray(i[1]), samples, weights)-cdf(np.asarray(i[1]), *args)) for i in samples.iterrows()])
    N=len(samples)
    p=stats.kstwo.sf(D, N)
    return D,p


#apply ks test individually to each dimension
def kstest_ndim(samples, cdfs, args):
    N,dim=samples.shape
    Dstats=[]
    pval=[]
    for i, j, k in zip(samples, cdfs, args):
        d, p= stats.kstest(i,j, args=k)
        Dstats.append(d)
        pval.append(p)
    pval=np.sort(pval)
    pval[0]=pval[0]*dim
    D=np.max(Dstats)
    p_=stats.kstwo.sf(D, N)
    return D,p_