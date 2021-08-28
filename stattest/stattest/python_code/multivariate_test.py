import scipy.stats as stats
from .stat_test import ecdf_x, reorder, kstest, plt_to_base64_encoded_image
import numpy as np
import pandas as pd
from .inference import CustomError
from scipy.stats.stats import KstestResult
import matplotlib.pyplot as plt

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

    dstats=[abs(ecdf_x_ndim(np.asarray(i[1]), samples, weights)-cdf(np.asarray(i[1]), *args)) for i in samples.iterrows()]
    D= np.max(dstats)
    N=len(samples)
    p=stats.kstwo.sf(D, N)
    p = np.clip(p, 0, 1)
    return KstestResult(D,p)

def plot_p_multivar(samples, cdf, weights, allplots=True):
    N=len(samples)
    if len(weights)!=N:
        weights=pd.DataFrame(np.ones(samples.shape))
    if True:
        critical_p_val=0.01
        cdf
        pval=[]
        passed_count=0
        k=N/1000
        if k>=1000:
            a=0
            while a+1000<=N:#here
                p=mutlivariate_kstest(samples.loc[a:a+1000], cdf, weights=weights.loc[a:a+1000])[1]
                pval.append(p)
                a+=1000
                if p>critical_p_val:
                    passed_count+=1
            if a<N:
                p=mutlivariate_kstest(samples.loc[a:], cdf, weights=weights.loc[a:])[1]
                pval.append(p)
                if p>critical_p_val:
                    passed_count+=1
            perc_passed=(passed_count/len(pval))*100
            if allplots:
                plt.hist(pval, bins=100, density=True)
                plt.title('p values of samples divided into non overalapping groups')
        else:
            print ('sample size is small so calculating p values with overlapping samples')
            listofindices=np.arange(0,N,1)
            for i in range(1000):
                partialindices=np.random.choice(listofindices, int(N/2), replace=False)
                partialsample=samples.loc[partialindices]
                partialweights=weights.loc[partialindices]
                p=mutlivariate_kstest(partialsample, cdf, weights=partialweights)[1]
                pval.append(p)
                if p>critical_p_val:
                    passed_count+=1
            perc_passed=(passed_count/len(pval))*100
            if allplots:
                plt.hist(pval, bins=100, density=True, color='g')
                plt.title('distribution of p value with samples divided into overlapping groups')
        if allplots:
            text=str(perc_passed)+'%'+' of p values has passed'
            plt.text(1,1, text, horizontalalignment="center", verticalalignment="center")
            plot = plt_to_base64_encoded_image()
        else:
            plot=None
    return perc_passed, plot

def multivar_kstest1(samples, cdf,args, weights, allplots):
    if type(samples)==np.ndarray:
        samples=pd.DataFrame(samples, header=None, columns=None)
    if type(weights)==np.ndarray:
        weights=pd.DataFrame(weights, header=None, columns=None)
    if len(weights)==0:
        weights=pd.DataFrame(np.ones(samples.shape))
    
    perc, percplot =plot_p_multivar(samples, lambda x: cdf(x, *args), weights, allplots)
    ksresult= mutlivariate_kstest(samples, cdf, args, weights)
    return perc, ksresult, percplot

#apply ks test individually to each dimension
#cdfs could be list of cdfs or list of samples for two sample test
def multivar_kstest2(samples, cdfs, weights=[], allplots=True, title=None):
    #samples and weights are np array
    N,dim=samples.shape
    if len(cdfs)!=dim:
        raise CustomError('dimension of cdf does not match array')
    Dstats=[]
    pval=[]
    if weights==[]:
        weights=np.ones(N*dim).reshape(N,dim)
    if not callable(cdfs[0]):
        for i, j, w in zip(np.transpose(samples), np.transpose(cdfs), np.transpose(weights)):
            d, p= kstest(sample=i, cdf=j, weights=w)
            Dstats.append(d)
            pval.append(p)
        pval=np.sort(pval)
        if allplots:
            plt.hist(pval)
            if title==None:
                plt.title("p values of KS test in each dimensions")
            else:
                plt.title("p value of KS test in each dimension for estimated mean")
            plotp_each_dim=plt_to_base64_encoded_image()
        else:
            plotp_each_dim=None
        

        D=np.max(Dstats)
        p_=stats.kstwo.sf(D, N)
        p_ = np.clip(p_, 0, 1)
        
        return KstestResult( D,p_*dim), plotp_each_dim
    else:
        for i, j, w in zip(np.transpose(samples), cdfs, np.transpose(weights)):
            d, p= kstest(sample=i, cdf=j, weights=w)
            Dstats.append(d)
            pval.append(p)
        pval=np.sort(pval)
        if allplots:
            plt.hist(pval)
            plt.title("p values of KS test in each dimensions")
            plotp_each_dim=plt_to_base64_encoded_image()
        else:
            plotp_each_dim=None
            
        pval[0]=pval[0]*dim
        D=np.max(Dstats)
        p_=stats.kstwo.sf(D, N)
        p_ = np.clip(p_, 0, 1)
        print ('adjusted p', p_*dim)
        return KstestResult(D,p_*dim), plotp_each_dim


#testing multimdimensional ks test with multivariate normal distribution
def test_kstest_multivar_norm(distribution=[], weights=[], size=50, mean=[], cov=[], allplots=True):
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
    
    perc, result1, percplot =multivar_kstest1(sample, cdf_overall, weights=weights, allplots=allplots)
    result2, pplot =multivar_kstest2(sample, cdfs, weights=weights, allplots=allplots)
    if allplots:
        return (perc, result1, result2), [percplot, pplot]
    else:
        return perc, result1, result2

if __name__=='__main__':
    print (test_kstest_multivar_norm(size=10))