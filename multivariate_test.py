import scipy.stats as stats
from stat_test import ecdf_x, reorder, kstest
import numpy as np
import pandas as pd
from inference import CustomError

def ecdf_x_ndim(x, dist, weights):
    #x is tuple or list, dist and weights are dataframe
    dim=len(x)
    n=len(dist)
    res=1
    for i in range(dim):
        sample1=np.asarray(dist.loc[:,i])
        weights1=np.asarray(weights.loc[:,i])
        res*=ecdf_x(x[i], *reorder(sample1, weights1))
    return res


        

def mutlivariate_kstest(samples,cdf, args=(), weights=[]):
    #samples and weights are pandas dataframes
    if type(samples)==np.ndarray:
        samples=pd.DataFrame(samples, columns=None)
    if type(weights)==np.ndarray:
        weights=pd.DataFrame(weights, columns=None)
    if len(weights)==0:
        weights=pd.DataFrame(np.ones(samples.shape))
    D= np.max([abs(ecdf_x_ndim(np.asarray(i[1]), samples, weights)-cdf(np.asarray(i[1]), *args)) for i in samples.iterrows()])
    N=len(samples)
    p=stats.kstwo.sf(D, N)
    return D,p


#apply ks test individually to each dimension
def kstest_ndim(samples, cdfs, weights=[]):
    #samples and weights are np array
    N,dim=samples.shape
    if len(cdfs)!=dim:
        raise CustomError('dimension of cdf does not match array')
    Dstats=[]
    pval=[]
    if weights==[]:
        weights=np.ones(N*dim).reshape(N,dim)
    for i, j, w in zip(np.transpose(samples), cdfs, np.transpose(weights)):
        d, p= kstest(sample=i, cdf=j, weights=w)
        Dstats.append(d)
        pval.append(p)
    pval=np.sort(pval)
    pval[0]=pval[0]*dim
    D=np.max(Dstats)
    p_=stats.kstwo.sf(D, N)
    print ('adjusted p', p_*dim)
    return D,p_*dim


#testing multimdimensional ks test with multivariate normal distribution
def test_kstest_multivar_norm(distribution=[],  size=50, mean=[], cov=[]):
    if mean==[] and cov==[] and distribution==[]:
        mean=[0]*4
        cov=np.identity(4)
        sample=stats.multivariate_normal.rvs(mean=mean, cov=cov, size=size)
    if mean==[] and cov==[] and distribution !=[]:
        n,dim=distribution.shape
        mean=[0]*dim
        cov=np.identity(dim)

    if mean==[] and cov!=[] and distribution==[]:
        mean=[0]*len(cov)
        sample=stats.multivariate_normal.rvs(mean=mean, cov=cov, size=size)
    if mean==[] and cov!=[] and distribution!=[]:
        n,dim = distribution.shape
        if len(cov)!=dim:
            cov=np.identity(dim)
        else:
            mean=[0]*dim


    if mean!=[] and cov==[] and distribution==[]:
        cov=np.identity(len(mean))
        sample=stats.multivariate_normal.rvs(mean=mean, cov=cov, size=size)
    if mean!=[] and cov==[] and distribution!=[]:
        n,dim=distribution.shape
        if len(mean)!=dim:
            mean=[0]*dim
        cov=np.identity(dim)

    if mean!=[] and cov!=[] and distribution==[]:
        if len(cov)!=len(mean):
            cov=np.identity(len(mean))
        sample=stats.multivariate_normal.rvs(mean=mean, cov=cov, size=size)
    if mean!=[] and cov!=[] and distribution!=[]:
        n,dim=distribution.shape
        if len(mean)!=dim:
            mean=[0]*dim
        if len(cov)!=dim:
            cov=np.identity(dim)

    try:
        sample
    except NameError:
        sample=np.asarray(distribution)

    cdf_overall= lambda x: stats.multivariate_normal.cdf(x, mean=mean, cov=cov)
    cdfs=[(lambda x: stats.norm.cdf(x, loc=mean[i], scale=cov[i][i])) for i in range(len(mean))]

    kstest1=mutlivariate_kstest(sample, cdf_overall)
    kstest2=kstest_ndim(sample, cdfs)

    return kstest1, kstest2


if __name__=='__main__':
    print (test_kstest_multivar_norm(size=10))