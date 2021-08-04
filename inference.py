listofpriors=['Beta']

import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#one dimension

#following series of functions functions
#take prior parameters and observations (as array)
#generates sample from exact distribution
#N is number of data points of the user generated posterior
#p> 0.01 means pass keep track with increasing posterior size
# examples of multivariate inf problems that work
#generate nice plots with number of samples changed for diff inf algorithm

critical_p_val=0.01

def beta_bernoulli(parameters, obs, N, factor=10):
   s=np.sum(obs)
   n=len(obs)
   alpha_new=parameters[0]+s
   beta_new=parameters[1]+n-s
   return stats.beta.rvs(a=alpha_new, b=beta_new, size=factor*N)
 

def gamma_poisson(parameters, obs, N, factor=10):
    s=np.sum(obs)
    n=len(obs)
    alpha_new=parameters[0]+s
    beta_new=parameters[1]+n
    return stats.gamma.rvs(a=alpha_new, scale=1/beta_new, size=factor*N)

def normal_known_var(parameters, obs, N, factor=10):
    prior_mean,prior_std, likelihood_std=parameters
    μ0, τ0, τ = prior_mean, 1/prior_std, 1/likelihood_std
    n=len(obs)
    τ_new=τ0+n*τ
    x_bar=np.mean(obs)
    μ_new=(n*τ*x_bar+τ0*μ0)/(n*τ+τ0)
    std_new=np.sqrt(1/τ_new)
    return (stats.norm.rvs(loc=μ_new, scale=std_new, size=N*factor))


def normal_known_mu(parameters, obs, N, factor=10):
#inverse gamma
    alpha, beta, mu = parameters
    n=len(obs)
    alpha_new=alpha+n/2
    beta_new=beta+(np.sum([(i-mu)**2 for i in obs]))/2
    return stats.invgamma.rvs(a=alpha_new, scale=beta_new, size=factor*N)


def normal_unknown_mu_std(parameters, obs, N, factor=10):
    mu0, nu, alpha, beta= parameters
    n=len(obs)
    x_bar=np.mean(obs)
    mu_new=(nu*mu0+n*x_bar)/(nu+n)
    nu_new  = nu+n
    alpha_new=alpha+n/2
    beta_new=beta+(1/2)*np.sum([(xi-x_bar)**2 for xi in obs])+((n*nu)/(nu+n))*((x_bar-mu0)**2/2)
    posterior_var=stats.invgamma.rvs(a=alpha_new, scale=beta_new, size=N*factor)
    posterior_mean=stats.norm.rvs(loc=mu_new, scale=posterior_var/nu_new, size=N*factor)
    return np.stack((posterior_mean, np.sqrt(posterior_var)), axis=1)

#takes a name of a csv file that contains data
#and returns a 1 dimensional np array 

def import_sample(name):
    name+='.csv'
    df=pd.read_csv(name, header=None)
    Series=np.asarray(df).flatten()
    Series=Series[np.logical_not(np.isnan(Series))]
    return Series #np array

distributions={'beta_bernoulli': beta_bernoulli, 'gamma_poisson':gamma_poisson, 'normal_known_var':normal_known_var, 'normal_known_mu':normal_known_mu,
'normal_unknown_mu_std':normal_unknown_mu_std}

def compare(posterior, obs, parameters, distribution_name, plot=True, plotp=False, factor=10):
    N=len(posterior)
    generator=distributions[distribution_name]
    exact=generator(parameters, obs=obs, N=N, factor=factor)
    if plot==True:
        points=np.histogram(exact, bins=100, density=True)
        plt.hist(posterior, bins=100, density=True, color='b', label='estimated posterior')
        plt.plot(points[1][:-1], points[0], color='r', label='sample from exact posterior')
        plt.legend(loc="upper right")
        plt.show()
    if plotp==True:
        pval=[]
        passed_count=0
        k=N/1000
        if k>=1000:
            a=0
            while a+1000<=N:
                p=stats.kstest(posterior[a:a+1000], exact)[1]
                pval.append(p)
                a+=1000
                if p>critical_p_val:
                    passed_count+=1
            if a<N:
                p=stats.kstest(posterior[a:], exact)[1]
                pval.append(p)
                if p>critical_p_val:
                    passed_count+=1
            fig,ax=plt.subplots()
            perc_passed=(passed_count/len(pval))*100
            ax.hist(pval, bins=100, density=True, label=str(perc_passed)+'%'+' of p values has passed')
        else:
            for i in range(1000):
                partialsample=np.random.choice(posterior, int(N/2))
                p=stats.kstest(partialsample, exact)[1]
                pval.append(p)
                if p>critical_p_val:
                    passed_count+=1
            perc_passed=(passed_count/len(pval))*100
            fig, ax=plt.subplots()
            ax.hist(pval, bins=100, density=True, color='g',label=str(perc_passed)+'%'+' of p values has passed')
        ax.set_title('distribution of p value')
        plt.show()
    return stats.ks_2samp(posterior, exact)

n=0
sample_size=[]
for j in range(3,6):
    for i in range(10**j,10**(j+1),10**j):
        sample_size.append(i)
sample_size+=[10**6]

if __name__=='__main__':
    print ('hello world')
    obs=import_sample('obs2')
    posterior=import_sample('posterior2')
    print (compare(posterior, obs, (2,3),'beta_bernoulli', plot=True,plotp=True, factor=1000))
    #import pymc3 as pm
    """
    y_obs=[0]*8+[1]*2
    for i in sample_size:
        pm.Model() as model:
            prior=pm.Beta(name='prior', alpha=2, beta=3)
            likelihood=pm.Bernoulli(name='bern', p=prior, observed=y_obs)
            trace=pm.sample(i)
    """    