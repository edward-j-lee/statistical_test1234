import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from sampling_algorithms.importance import generate_weighted1
from sampling_algorithms.rejection import sample
import scipy.stats as stats
from scipy.stats._hypotests import _cdf_cvm
cdf=stats.beta.cdf
args=(2,3)

def ecdf_cdf(sample, weights, cdf, args):
    s_w=np.stack((sample, weights), axis=1)
    s_w=s_w[s_w[:, 0].argsort()]
    sample=s_w[:,0]
    weights=s_w[:,1]
    total=np.sum(weights)
    ecdfs=(np.cumsum(weights))/total
    cdfs=cdf(sample, *args)
    return ecdfs, cdfs
    
def kstest(sample, cdf, args, weights=[]):
    N=len(sample)
    if len(weights)==0:
        weights=[1]*N
    ecdf, cdf = ecdf_cdf(sample, weights, cdf, args)
    d=np.abs(np.diff((ecdf, cdf))[0])
    D=np.max(d)
    return D, stats.kstwo.sf(D, N)


def chisquare(sample, cdf, args, bins=100, range_=None, weights=[]):
    N=len(sample)
    if len(weights)==0:
        weights=np.asarray([1]*N)
    if range_==None:
        t=np.histogram(sample, weights=weights, bins=bins)
    else:
        t=np.histogram(sample, weights=weights, bins=bins, range=range_)
    sam=t[0]
    ran=t[1][1:]
    expected=np.diff(np.vectorize((lambda x: cdf(x, *args)))(ran))*N
    #cisqstat=(expected-sam)**2/expected
    return (stats.chisquare(sam, expected))
"""
not working
def cramertest(sample, cdf, args, weights=[]):
    N=len(sample)
    if len(weights)==0:
        weights=[1]*N
    ecdf, cdf= ecdf_cdf(sample, weights, cdf, args)
    tstat=((1/3)*(cdf[0]**3))-((1/3)*((cdf[-1]-1)**3))+ np.sum([(((cdf[i+1]-ecdf[i])**3)-((cdf[i]-ecdf[i])**3)) for i in range(0,N-1)])
    tstat*=N
    #tstat=(1/(12*(N))+(np.sum([(cdf[i]-((2*(i+1)-1)/(2*N)))**2 for i in range(0,N)])))
    return tstat
"""


def plot_p(sampler, cdf, args, sample_size=50, p_size=1000, test=kstest):
    pval=[]
    for i in range(p_size):
        sample=sampler(sample_size)
        if type(sample)==tuple:
            sample, weights =sample
        else:
            weights=[]
        p=test(sample, cdf, args, weights)[1]
        pval.append(p)
    plt.hist(pval, bins=100, density=True)

#example
plot_p(generate_weighted1, cdf, (2,3), sample_size=50, p_size=1000, test=kstest)



print (stats.cramervonmises(sample, stats.beta.cdf, args=(2,3)))
#print ('cramer', cramertest(sample, stats.beta.cdf, args=(2,3)))




def cramervonmises(rvs, cdf, args=(), weights=[]):
    if isinstance(cdf, str):
        cdf = getattr(distributions, cdf).cdf

    vals = (np.asarray(rvs))

    if vals.size <= 1:
        raise ValueError('The sample must contain at least two observations.')
    if vals.ndim > 1:
        raise ValueError('The sample must be one-dimensional.')

    n = len(vals)
    #cdfvals = cdf(vals, *args)
    if not weights:
        weights=[1]*n
    ecdf, cdfvals = ecdf_cdf(vals,weights, cdf, args)
    #u = (2*np.arange(1, n+1) - 1)/(2*n)
    #ecdf=np.arange(1,n+1)/n
    u=ecdf - (1/(2*n))
    w = 1/(12*n) + np.sum((u - cdfvals)**2)

    # avoid small negative values that can occur due to the approximation
    p = max(0, 1. - _cdf_cvm(w, n))

    return (w, p)

print ('cramer2', cramervonmises(sample, stats.beta.cdf, args=(2,3)))
