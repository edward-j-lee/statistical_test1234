listofpriors=['Beta']

import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import pickle
from stat_test import kstest, reorder, ecdf_x,ecdf_cdf

class CustomError(Exception):
    pass

#one dimension

#following series of functions functions
#take prior parameters and observations (as array)
#generates sample from exact distribution
#N is number of data points of the user generated posterior
#p> 0.01 means pass keep track with increasing posterior size
# examples of multivariate inf problems that work
#generate nice plots with number of samples changed for diff inf algorithm

critical_p_val=0.01

def beta_bernoulli(parameters, obs):
   s=np.sum(obs)
   n=len(obs)
   alpha_new=parameters[0]+s
   beta_new=parameters[1]+n-s
   #stats.beta.rvs(a=alpha_new, b=beta_new, size=factor*N)
   return alpha_new, beta_new
 

def gamma_poisson(parameters, obs):
    s=np.sum(obs)
    n=len(obs)
    alpha_new=parameters[0]+s
    beta_new=parameters[1]+n
    #stats.gamma.rvs(a=alpha_new, scale=1/beta_new, size=factor*N)
    return alpha_new, beta_new

def normal_known_var(parameters, obs):
    prior_mean,prior_std, likelihood_std=parameters
    μ0, τ0, τ = prior_mean, 1/prior_std, 1/likelihood_std
    n=len(obs)
    τ_new=τ0+n*τ
    x_bar=np.mean(obs)
    μ_new=(n*τ*x_bar+τ0*μ0)/(n*τ+τ0)
    std_new=np.sqrt(1/τ_new)
    # (stats.norm.rvs(loc=μ_new, scale=std_new, size=N*factor))
    return μ_new, std_new


def normal_known_mu(parameters, obs):
#inverse gamma
    alpha, beta, mu = parameters
    n=len(obs)
    alpha_new=alpha+n/2
    beta_new=beta+(np.sum([(i-mu)**2 for i in obs]))/2
    # stats.invgamma.rvs(a=alpha_new, scale=beta_new, size=factor*N)
    return alpha_new, beta_new


#takes a name of a csv file that contains one dimensional data
#and returns a 1 dimensional np array 
def import_sample(name):
    name+='.csv'
    df=pd.read_csv(name, header=None)
    Series=np.asarray(df).flatten()
    Series=Series[np.logical_not(np.isnan(Series))]
    return Series #np array


distributions={'beta_bernoulli': beta_bernoulli, 'gamma_poisson':gamma_poisson, 'normal_known_var':normal_known_var, 'normal_known_mu':normal_known_mu,}

dist_func={'beta_bernoulli': stats.beta, 'gamma_poisson': stats.gamma, 'normal_known_var':stats.norm, 'normal_known_mu':stats.invgamma}
biv_dist=['normal_unknown_mu_std']

def plot_p(posterior, exactsample_or_cdf, weights=[], plotp=True):
    N=len(posterior)
    posterior=np.asarray(posterior)
    if len(weights)==0:
        weights=[1]*N
    weights=np.asarray(weights)
    if True:
        exact=exactsample_or_cdf
        pval=[]
        passed_count=0
        k=N/1000
        if k>=1000:
            print ('more than million sample size')
            print ('calculating p values with non overalpping sample')
            a=0
            while a+1000<=N:#########here line 94
                p=kstest(posterior[a:a+1000], exact, weights=weights[a:a+1000])[1]
                pval.append(p)
                a+=1000
                if p>critical_p_val:
                    passed_count+=1
            if a<N:
                p=kstest(posterior[a:], exact, weights=weights[a:])[1]
                pval.append(p)
                if p>critical_p_val:
                    passed_count+=1
            perc_passed=(passed_count/len(pval))*100
            if plotp:
                plt.hist(pval, bins=100, density=True)
                plt.title('p values of samples divided into non overalapping groups')
        else:
            print ('sample size is small so calculating p values with overlapping samples')
            posterior, weights = reorder(posterior, weights)
            listofindices=np.arange(0,N,1)
            for i in range(1000):
                partialindices=np.random.choice(listofindices, int(N/2), replace=False)
                partialsample=posterior[partialindices]
                partialweights=weights[partialindices]
                p=kstest(partialsample, cdf=exact, weights=partialweights)[1]
                pval.append(p)
                if p>critical_p_val:
                    passed_count+=1
            perc_passed=(passed_count/len(pval))*100
            if plotp:
                plt.hist(pval, bins=100, density=True, color='g')
                plt.title('distribution of p value with samples divided into overlapping groups')
        if plotp:
            text=str(perc_passed)+'%'+' of p values has passed'
            plt.annotate(text, (0.8, 0.9))
            plt.show()    
    return perc_passed

def compare(posterior, obs, parameters, distribution_name, weights=[], plot=True, plotp=False, factor=10):
    if distribution_name in biv_dist:
        raise CustomError('inference problem must be one dimensional')
    N=len(posterior)
    inf_prob=distributions[distribution_name]
    print ('generating exact sample', factor*N)
    newparam=inf_prob(parameters, obs)
    generator=dist_func[distribution_name].rvs
    exact=generator(*newparam, size=factor*N)
    weights=[1]*N
    if plot==True:
        points=np.histogram(exact, bins=100, density=True)
        plt.hist(posterior, weights=weights,bins=100, density=True, color='b', label='estimated posterior')
        plt.plot(points[1][:-1], points[0], color='r', label='sample from exact posterior')
        plt.legend(loc="upper right")
        plt.title('comparing estimated posterior with samples drawn from exact distribution')
        plt.show()
    perc_passed= plot_p(posterior, exact, weights=weights, plotp=plotp)
    return perc_passed, stats.kstest(posterior, exact)

def kstest_cdf(posterior, obs, parameters, distribution_name, weights=[], plot=True, plotp=True):
    N=len(posterior)
    newparam=distributions[distribution_name](parameters, obs)
    cdf =lambda x: dist_func[distribution_name].cdf(x, *newparam)
    pdf= lambda x: dist_func[distribution_name].pdf(x, *newparam)
    if plot:
        range_=min(posterior), max(posterior)
        xpoints=np.linspace(*range_, N*10)
        ypoints=np.vectorize(pdf)(xpoints)
        
        #plotting pdf
        plt.plot(xpoints, ypoints, color='r', label='exact pdf')
        plt.hist(posterior, weights=weights, bins=100, color='b', density=True, label='estimated posterior')
        plt.legend(loc="upper right")
        plt.title('comparing estimated posterior with exact pdf')
        plt.show()
        
        #plotting cdf
        sample1=reorder(posterior, weights)
        res=ecdf_cdf(*sample1, cdf=cdf)
        plt.plot(sample1[0], res[0], color='b', label='empirical cdf')
        plt.plot(sample1[0],res[1], color='r', label='exact cdf' )
        plt.legend(loc="upper right")
        plt.title('comparing ecdf of estimated posterior with exact cdf')
        plt.show()
    
    perc_passed= plot_p(posterior, cdf, plotp=plotp)
    
    return perc_passed, stats.kstest(posterior, cdf)


sample_size=[1000,5000,10000,50000,100000,500000,1000000]
passed=[]
overallp=[]
if __name__=='__main__':
    print ('hello world')
    #obs=import_sample('obs2')
    posterior=import_sample('posterior2')
    #print (kstest_cdf(posterior, obs=import_sample('obs2'), parameters=(2,3), distribution_name='beta_bernoulli', plot=True, plotp=True))
    #print (compare(posterior, obs, (2,3),'beta_bernoulli', plot=True,plotp=True, factor=100))
    y_obs=[0]*8+[1]*2
    dictionary=dict()
    """
    for i in sample_size:
        with pm.Model() as model:
            prior=pm.Beta(name='prior', alpha=2, beta=3)
            likelihood=pm.Bernoulli(name='bern', p=prior, observed=y_obs)
            trace=pm.sample(int(i/4))
        print ('comparing sample size', i)
        K=kstest_cdf(posterior=trace['prior'], obs=y_obs, parameters=(2,3), distribution_name= 'beta_bernoulli', plot=False, plotp=False)
        dictionary[i]=(K[0], K[1][1])
        passedratios.append(K[0])
        pvalues.append(K[1][1])
        print ('for sample size', i, ', ',K[0], '% has passed')
    output = open('pymc_posterior_inference_betabernoulli_2_3_with_varying_sample_size', 'wb')
    pickle.dump(dictionary, output)
    output.close()
    """
    
    k='pymc_posterior_inference_betabernoulli_2_3_with_varying_sample_size'
    file=open(k, 'rb')
    b = pickle.load(file)
    xpoints=np.sort(np.asarray(list(b.keys())))
    for i in xpoints:
        overallp.append(b[i][1])
        passed.append(b[i][0]/100)
    plt.plot(xpoints, passed, color='r', label='passed ratio')
    plt.plot(xpoints, overallp, color='b', label='overall p value')
    plt.axhline(y=0.01, label='critical value')
    plt.semilogx()
    plt.legend(loc="upper right")
    plt.xlabel='number of samples'
    plt.title('p values of beta_bernoulli inference problem given different sample size')
    plt.show()


    
        