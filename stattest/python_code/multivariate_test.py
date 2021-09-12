import scipy.stats as stats
from .stat_test import ecdf_x, reorder, kstest, plt_to_base64_encoded_image
import numpy as np
import pandas as pd
from .inference import CustomError, plot_p
from scipy.stats.stats import KstestResult
import matplotlib.pyplot as plt


#apply ks test individually to each dimension (second version of multivar ks test)
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
        dimensions=np.arange(dim)+1
        if allplots:
            plt.scatter(dimensions, pval, color='b', label='p value')
            plt.axhline(0.01, label='critical value')
            if title==None:
                plt.title("p values and d stats of KS test in each dimensions")
            else:
                plt.title("p value and d stats of KS test in each dimension for estimated mean")
            plt.ylabel("p value")
            plt.xlabel("dimension")
            plt.legend()
            plotp_each_dim=plt_to_base64_encoded_image()
        else:
            plotp_each_dim=None
        perc= (np.sum([pval>=0.01])/len(pval))*100
        D=np.max(Dstats)
        p_=stats.kstwo.sf(D, N)
        p_ = np.clip(p_, 0, 1)
        new_p=np.clip(p_*dim, 0,1)

        return perc, KstestResult( D, new_p), plotp_each_dim
    else:
        for i, j, w in zip(np.transpose(samples), cdfs, np.transpose(weights)):
            d, p= kstest(sample=i, cdf=j, weights=w)
            Dstats.append(d)
            pval.append(p)
        dimensions=np.arange(dim)+1
        if allplots:
            plt.scatter(dimensions, pval)
            plt.axhline(0.01, label='critical value')
            plt.ylabel("p value")
            plt.title('plotting p values against dimensions')
            plt.xlabel("dimension")
            plt.legend()
            plotp_each_dim=plt_to_base64_encoded_image()
        else:
            plotp_each_dim=None
        pval=np.asarray(pval)
        perc= (np.sum([pval>=0.01])/len(pval))*100
        D=np.max(Dstats)
        p_=stats.kstwo.sf(D, N)
        p_ = np.clip(p_, 0, 1)
        new_p=np.clip(p_*dim, 0,1)
        return perc, KstestResult(D,new_p), plotp_each_dim


#tests whehter a given sample comes from multivariate normal distribution
#  with given mean vector and covariance
def test_kstest_multivar_norm(distribution, mean, cov, weights=[],allplots=True):
    #checks if any of mean, cov or distribution is empty and generates some data for the purpose of unit testing
    #they should all be given in pracitice whenever this function is called
    n,dim=distribution.shape
    sample=np.asarray(distribution)
    if len(weights)==0:
        weights=np.ones(n*dim).reshape(n,dim)
    cdfs=[(stats.norm(loc=mean[i], scale=np.sqrt(cov[i][i])).cdf ) for i in range(dim)]
    pdfs=[stats.norm(loc=mean[i], scale=np.sqrt(cov[i][i])).pdf  for i in range(dim)]


    # result1, percplot =multivar_kstest1(sample, cdf_overall, weights=weights, allplots=allplots)
    perc, result2, pplot =multivar_kstest2(sample, cdfs, weights=weights, allplots=allplots)
    if allplots:
        #histograms are plotted against exact pdf for first 10 dimensions
        final_plots=[pplot]
        for i in range(5):
            if i>=len(pdfs):
                break #if the dimension is less then 10 then break the loop
                      #and go straight to return
            xpoints=np.linspace(min(sample[:,i]), max(sample[:,i]), 10000)
            plt.hist(sample[:,i], weights=weights[:,i],  density=True, bins=100, color='b', label='user sample')
            plt.plot(xpoints, np.vectorize(pdfs[i])(xpoints), color='r', label='exact pdf')
            plt.title('comparing the sample in the '+ str(i+1)+ 'th dimension')
            final_plots.append(plt_to_base64_encoded_image())
        return (perc, result2), final_plots
    else:
        return perc, result2

