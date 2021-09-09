listofpriors=['Beta']

import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import pickle
from .stat_test import all_tests, kstest, reorder, ecdf_x,ecdf_cdf, plt_to_base64_encoded_image
from .sampling_algorithms.importance import imp_sampling_w
from .sampling_algorithms.MCMC import MCMC_sampling_inf
from .sampling_algorithms.rejection import acc_rej_samp


class CustomError(Exception):
    pass

#one dimension

#following series of functions functions
#take prior parameters and observations (as array)
#generates parameters for posteriro

#p> 0.01 means pass keep track with increasing posterior size
critical_p_val=0.01


def beta_bernoulli(parameters, obs):
   s=np.sum(obs)
   n=len(obs)
   alpha_new=parameters[0]+s
   beta_new=parameters[1]+n-s
   #stats.beta.rvs(a=alpha_new, b=beta_new, size=factor*N)
   return alpha_new, beta_new


def gamma_poisson(parameters, obs):
    alpha, scale=parameters
    beta=1/scale
    s=np.sum(obs)
    n=len(obs)
    alpha_new=alpha+s
    beta_new=beta+n
    #stats.gamma.rvs(a=alpha_new, scale=1/beta_new, size=factor*N)
    scale_new=1/beta_new
    return alpha_new, scale_new

def normal_known_var(parameters, obs):
    prior_mean,prior_std, likelihood_std=parameters
    μ0, τ0, τ = prior_mean, 1/prior_std, 1/(likelihood_std**2)
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


#maching problem name to functions
distributions={'beta_bernoulli': beta_bernoulli, 'gamma_poisson':gamma_poisson, 'normal_known_var':normal_known_var, 'normal_known_mu':normal_known_mu,}
prior_likelihood={'normal_known_mu': (stats.invgamma, stats.norm), 'normal_known_var':(stats.norm, stats.norm), 'gamma_poisson': (stats.gamma, stats.poisson), 'beta_bernoulli': (stats.beta, stats.bernoulli)}

#seperate list for prior functiions
dist_func={}
for i in prior_likelihood: 
    dist_func[i]=prior_likelihood[i][0]


#given posterior and either exact sample or cdf, and optional weights,
# divdies the poseterior sample into multiple parts and repeats the ks tests
# returns percentage of p values passed
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
            while a+1000<=N:#here
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
            plt.text(1,1, text, horizontalalignment="center", verticalalignment="center")
            plot = plt_to_base64_encoded_image()
        else:
            plot=None
    return perc_passed, plot

#does ks test of the posterior distribution with samples generated from exact posterior
def compare(posterior, obs, parameters, distribution_name, weights=[], plot=True, plotp=False, factor=10):

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
    return perc_passed, kstest(posterior, exact, weights=weights)

#does multiple tests on posterior with exact cdf
#optionally plots pdf and cdf
def test_cdf(posterior, obs, parameters, distribution_name, weights=[], plot=True, plotp=True):
    all_plots = []
    N=len(posterior)
    print ('comparing with exact cdf')
    newparam=distributions[distribution_name](parameters, obs)
    if distribution_name=="gamma_poisson": #gamma poisson has to be seperated because of position of its parameters are different
        F=stats.gamma(a=newparam[0], scale=newparam[1])
    else:
        F=dist_func[distribution_name](*newparam)
    cdf =lambda x: F.cdf(x)
    pdf= lambda x: F.pdf(x)
    if len(weights)==0:
        weights=[1]*N
    if plot:
        range_=min(posterior), max(posterior)
        xpoints=np.linspace(*range_, N*10)
        ypoints=np.vectorize(pdf)(xpoints)
        
        #plotting pdf
        plt.plot(xpoints, ypoints, color='r', label='exact pdf')
        plt.hist(posterior, weights=weights, bins=100, color='b', density=True, label='estimated posterior')
        plt.legend(loc="upper right")
        plt.title('comparing estimated posterior with exact pdf')
        all_plots.append(plt_to_base64_encoded_image())
        

        #plotting cdf
        sample1=reorder(posterior, weights)
        res=ecdf_cdf(*sample1, cdf=cdf)
        plt.plot(sample1[0], res[0], color='b', label='empirical cdf')
        plt.plot(sample1[0],res[1], color='r', label='exact cdf' )
        plt.legend(loc="upper right")
        plt.title('comparing ecdf of estimated posterior with exact cdf')
        all_plots.append(plt_to_base64_encoded_image())

    perc_passed, plot = plot_p(posterior, cdf, plotp=plotp, weights=weights)
    all_plots.append(plot)
    test_result, plot2=all_tests(posterior, F, weights=weights)
    all_plots.append(plot2)
    #text='Dstat: '+ str(ksresult[0] )+ " pvalue: " + str(ksresult[1])
    #plt.text(0.2,0.5, text)
    
    return perc_passed, test_result, all_plots


#benchmark inference with pymc
def benchmark(obs, parameters, distribution_name, N):
    N=int(N/4)
    if distribution_name=='beta_bernoulli':
        a,b=parameters
        with pm.Model() as model:
            θ=pm.Beta('θ', alpha=a, beta=b)
            y=pm.Bernoulli('y', p=θ, observed=obs)
            trace=pm.sample(N)
        F=stats.beta(*beta_bernoulli(parameters, obs))
        return all_tests(trace['θ'], F)[0]
    if distribution_name=='gamma_poisson':
        a,scale=parameters
        with pm.Model() as model:
            θ=pm.Gamma('θ', alpha=a, beta=1/scale)
            y=pm.Poisson('y', mu=θ, observed=obs)
            trace=pm.sample(N)
        F=stats.gamma(*gamma_poisson(parameters, obs))
        return all_tests(trace['θ'], F)[0]
    if distribution_name=='normal_known_var':
        mu0, std0, std=parameters
        with pm.Model() as model:
            mean=pm.Normal('mean', mu=mu0, sigma=std0)
            y=pm.Normal('y', mu=mean, sigma=std, observed=obs)
            trace=pm.sample(N)
        F=stats.norm(*normal_known_var(parameters, obs))
        return all_tests(trace['mean'], F)[0]
    if distribution_name=='normal_known_mu':
        a,b, mu=parameters
        with pm.Model() as model:
            var=pm.InverseGamma('var', allpha=a, beta=b)
            y=pm.Normal('y', mu=mu, sigma=np.sqrt(var), observed=obs) ##########not sure
            trace=pm.sample(N)
        F=stats.invgamma(*normal_known_mu(parameters, obs))
        return all_tests(trace['var'], F)[0]

#inference with other algorithms
#name of inference algorithm is passed as parameters
def benchmark2(obs, parameters, distribution_name, N, inference):
    prior, likelihood=prior_likelihood[distribution_name]
    #seperates the prior parameters from likelihood parameters
    def sep_param(param, name):
        if name=='normal_known_var' or name=='normal_known_mu':
            return [param[0], param[1]], [param[-1]]
        else:
            return param, []
    prior_param, likeli_param = sep_param(parameters, distribution_name)
    if distribution_name=='gamma_poisson': #gamma poisson has to be seperated because of issue
            #related to positining of parameters with the scipy.gamma method
        prior_=stats.gamma(a=prior_param[0], scale=prior_param[1])
        prior_samp=lambda x: prior_.rvs(size=x)
        prior_pdf=lambda x: prior_.pdf(x)
    else:
        prior_samp=lambda x: prior.rvs(size=x, *prior_param)
        prior_pdf=lambda x: prior.pdf(x, *prior_param)

    def Likelihood_prior1(x): #product of likelihood and prior (=posterior without normalizing constant - evidence)
            return np.prod(likelihood.pdf(obs, x, *likeli_param)) * prior_pdf(x)
    newparam=distributions[distribution_name](parameters, obs)

    def Likelihood_prior2(x): # the above function is used for continuous distribution
        #discrete must be seperated since discrete distribuiton in scipy have 'pmf' instead of
        #'.pdf' method
        return np.prod(likelihood.pmf(obs, x))*prior_pdf(x)

    F=prior(*newparam)

    if inference=='rejection':
        if distribution_name=='beta_bernoulli':
            res=acc_rej_samp(func=Likelihood_prior2, g_pdf=prior_pdf, g_samp=prior_samp, N=N, ran=(0,1)) 
            #since in beta bernoulli x is between 0 and 1
        elif distribution_name=='gamma_poisson':
            res=acc_rej_samp(func=Likelihood_prior2, g_pdf=prior_pdf, g_samp=prior_samp, N=N) 
            F=prior(a=newparam[0], scale=newparam[1]) 
        else:
            res=acc_rej_samp(func=Likelihood_prior1, g_pdf=prior_pdf, g_samp=prior_samp, N=N)
        return all_tests(res, F)[0]
    elif inference=='importance':
        if distribution_name=='beta_bernoulli' or distribution_name=='gamma_poisson':
            res=imp_sampling_w(Likelihood_prior2, N, prior_samp, prior_pdf)
            if distribution_name=='gamma_poisson':
                F=prior(a=newparam[0], scale=newparam[1])
        else:
            res=imp_sampling_w(Likelihood_prior1, N, prior_samp, prior_pdf)
        return all_tests(res[0], F, weights=res[1])[0]
    elif inference=='mcmc':
        if distribution_name=='beta_bernoulli' or distribution_name=='gamma_poisson':
            res=MCMC_sampling_inf(Likelihood_prior2, size=N)
            if distribution_name=='gamma_poisson':
                F=prior(a=newparam[0], scale=newparam[1])
        else:
            res=MCMC_sampling_inf(Likelihood_prior1, size=N)
        return all_tests(res, F)[0]

#takes information necessary to do bayesian inference 
#(observatiosn, parameters, prbolem name and sample size)
#along with the name of inference algorithm to use (pymc, rejection, importance and MCMC)
#returns the test result of the inference algorithm in given inference problem
def any_benchmark(obs, parameters, distribution_name, N, algorithm):
    if algorithm=='pymc3':
        return benchmark(obs, parameters, distribution_name, N)
    else:
        return benchmark2(obs, parameters, distribution_name, N, inference=algorithm)


        
