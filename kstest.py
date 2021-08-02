import scipy.stats as stats
from stat_test import ecdf_x, reorder
import numpy as np

def ecdf_x_ndim(x, dist):
    dim=len(x)
    n=len(dist)
    res=1
    for i in range(dim):
        sample=dist[:,i]
        res*=ecdf_x(x[i], *reorder(sample))
    return res

def maxD(samples, cdf, args):
    return np.max([abs(ecdf_x_ndim(i, samples)-cdf(i, *args)) for i in samples])

#apply ks test individually to each dimension
def kstest_ndim(samples, cdfs, args):
    N,dim=samples.shape
    Dstat=[]
    pval=[]
    
    for i, j, k in zip(samples, cdfs, args):
        d, p= stats.kstest(i,j, args=k)
        Dstats.append(d)
        pval.append(p)
    pval=np.sort(pval)
    pval[0]=pval[0]*dim
    D=np.max(Dstat)
    p_=stats.kstwo.sf(D, N)
    return D,p_