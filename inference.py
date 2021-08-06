listofpriors=['Beta']

import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import pickle
from stat_test import reorder, ecdf_x,ecdf_cdf

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


def normal_unknown_mu_std(parameters, obs):
    mu0, nu, alpha, beta= parameters
    n=len(obs)
    x_bar=np.mean(obs)
    mu_new=(nu*mu0+n*x_bar)/(nu+n)
    nu_new  = nu+n
    alpha_new=alpha+n/2
    beta_new=beta+(1/2)*np.sum([(xi-x_bar)**2 for xi in obs])+((n*nu)/(nu+n))*((x_bar-mu0)**2/2)
    return mu_new, nu_new, alpha_new, beta_new


#takes a name of a csv file that contains data
#and returns a 1 dimensional np array 

def import_sample(name):
    name+='.csv'
    df=pd.read_csv(name, header=None)
    Series=np.asarray(df).flatten()
    Series=Series[np.logical_not(np.isnan(Series))]
    return Series #np array

class normal_inv_gamma:
    def __init__(self):
        pass
    @classmethod
    def rvs(self, mu_new, nu_new,alpha_new, beta_new, size):
        posterior_var=stats.invgamma.rvs(a=alpha_new, scale=beta_new, size=size)
        posterior_mean=stats.norm.rvs(loc=mu_new, scale=np.sqrt(posterior_var/nu_new), size=size)
        return np.stack((posterior_mean, np.sqrt(posterior_var)), axis=1) 
    @classmethod
    def cdf(self, val, mu_new, nu_new, alpha_new, beta_new):
        IG=stats.invgamma.cdf(val[1]**2,a=alpha_new, scale=beta_new)
        N=stats.norm.cdf(val[0], loc=mu_new, scale=val[1]**2/nu_new)
        return N*IG    
    @classmethod
    def pdf(self, val, mu, nu, alpha, beta):
        IG=stats.invgamma.pdf(val[1]**2, a=alpha, b=beta)
        N=stats.norm.pdf(val[0], loc=mu, scale=val[1]**2/nu)
        return N*IG

distributions={'beta_bernoulli': beta_bernoulli, 'gamma_poisson':gamma_poisson, 'normal_known_var':normal_known_var, 'normal_known_mu':normal_known_mu,
'normal_unknown_mu_std':normal_unknown_mu_std}

dist_func={'beta_bernoulli': stats.beta, 'gamma_poisson': stats.gamma, 'normal_known_var':stats.norm, 'normal_known_mu':stats.invgamma,
         'normal_unknown_mu_std':normal_inv_gamma}
biv_dist=['normal_unknown_mu_std']

def plot_p(posterior, exactsample_or_cdf, plotp=True):
    if True:
        N=len(posterior)
        exact=exactsample_or_cdf
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
            perc_passed=(passed_count/len(pval))*100
            if plotp:
                fig,ax=plt.subplots()
                ax.hist(pval, bins=100, density=True)
        else:
            for i in range(1000):
                partialsample=np.random.choice(posterior, int(N/2))
                p=stats.kstest(partialsample, exact)[1]
                pval.append(p)
                if p>critical_p_val:
                    passed_count+=1
            perc_passed=(passed_count/len(pval))*100
            if plotp:
                fig, ax=plt.subplots()
                ax.hist(pval, bins=100, density=True, color='g')
        if plotp:
            ax.set_title('distribution of p value')
            text=str(perc_passed)+'%'+' of p values has passed'
            plt.annotate(text, (0.8, 0.9))
            plt.show()    
    return perc_passed

def compare(posterior, obs, parameters, distribution_name, plot=True, plotp=False, factor=10):
    if distribution_name in biv_dist:
        raise CustomError('inference problem must be one dimensional')
    N=len(posterior)
    inf_prob=distributions[distribution_name]
    print ('generating exact sample', factor*N)
    newparam=inf_prob(parameters, obs)
    generator=dist_func[distribution_name].rvs
    exact=generator(*newparam, size=factor*N)
    if plot==True:
        points=np.histogram(exact, bins=100, density=True)
        plt.hist(posterior, bins=100, density=True, color='b', label='estimated posterior')
        plt.plot(points[1][:-1], points[0], color='r', label='sample from exact posterior')
        plt.legend(loc="upper right")
        plt.title('comparing estimated posterior with samples drawn from exact distribution')
        plt.show()
    perc_passed= plot_p(posterior, exact, plotp=plotp)
    return perc_passed, stats.kstest(posterior, exact)

def kstest_cdf(posterior, obs, parameters, distribution_name, plot=True, plotp=True):
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
        plt.hist(posterior, bins=100, color='b', density=True, label='estimated posterior')
        plt.legend(loc="upper right")
        plt.title('comparing estimated posterior with exact pdf')
        plt.show()
        
        #plotting cdf
        sample1=reorder(posterior)
        res=ecdf_cdf(*sample1, cdf=cdf)
        plt.plot(sample1[0], res[0], color='b', label='empirical cdf')
        plt.plot(sample1[0],res[1], color='r', label='exact cdf' )
        plt.legend(loc="upper right")
        plt.title('comparing ecdf of estimated posterior with exact cdf')
        plt.show()
    
    perc_passed= plot_p(posterior, cdf, plotp=plotp)
    
    return perc_passed, stats.kstest(posterior, cdf)

def test_normal_two_unknowns(posterior_mean, posterior_var, obs, parameters, distribution_name, plot=True, plotp=True):
    if distribution_name !='normal_unknown_mu_std':
        raise CustomError('wrong inference problem')
    N=len(posterior_mean)
    
    if N!=len(posterior_var):
        raise CustomError('estimated mean and variance is of different length')
    mu2,nu2, a2,b2=normal_unknown_mu_std(parameters, obs)
    print ('newparam', mu2, nu2, a2, b2)
    exact_mean=stats.norm(loc=mu2, scale=np.sqrt(np.mean(posterior_var)/nu2))
    exact_var=stats.invgamma(a=a2, scale=b2)
    
    if plot==True:
        
        xpoints_var=np.linspace(min(posterior_var), max(posterior_var), N*10)
        ypoints_var=np.vectorize(exact_var.pdf)(xpoints_var)
        plt.plot(xpoints_var, ypoints_var, color='r', label='exact distribution')
        plt.hist(posterior_var, bins=100, density=True, color='b', label='estimated distribution')
        plt.legend(loc="upper right")
        plt.title('comparison of estimated and exact posterior distribution of variance')
        plt.show()

        xpoints_mean=np.linspace(min(posterior_mean), max(posterior_mean), N*10)
        ypoints_mean=np.vectorize(exact_mean.pdf)(xpoints_mean)
        #ypoints_mean=np.vectorize(lambda x: stats.norm.pdf(x, loc=mu2, scale=np.sqrt(np.mean(posterior_var)/nu2)))(xpoints_mean)
        plt.plot(xpoints_mean, ypoints_mean, color='r', label='exact distribution')
        plt.hist(posterior_mean, bins=100, density=True, color='b', label='estimated distribution')
        plt.legend(loc="upper right")
        plt.title('comparison of estimated and exact posterior distribution of mean')
        plt.show()

    perc_passed_mean=plot_p(posterior_mean, exact_mean.cdf, plotp=plotp)
    perc_passed_var=plot_p(posterior_var, exact_var.cdf, plotp=plotp)
    print ('returning percentage passed for mean, variance, overall p value for mean, variance in this order')
    return perc_passed_mean, perc_passed_var, stats.kstest(posterior_mean, exact_mean.cdf)[1], stats.kstest(posterior_var, exact_var.cdf)[1]


obs=[1,1,1,0,1,1,0,0,1,1,1,1]

def testing_test_function_normal(size, obs, parameters=(1,2,3,4)):
    mu,nu,alpha,beta=parameters
    with pm.Model() as model:
        var=pm.InverseGamma('var', alpha=alpha, beta=beta) #prior sigma - inv gamma
        μ=pm.Normal('μ', mu=mu, sigma=np.sqrt(var/nu))
        y=pm.Normal('y', mu=μ, sigma=np.sqrt(var), observed=obs) #likelihood - normal with mean μ and std σ
        trace=pm.sample(size)
    posterior_mean=trace['μ']
    posterior_var=trace['var']
    return test_normal_two_unknowns(posterior_mean, posterior_var, obs, parameters, 'normal_unknown_mu_std', True, True)




sample_size=[1000,5000,10000,50000,100000,500000,1000000]
passed=[]
overallp=[]
if __name__=='__main__':
    print ('hello world')
    #obs=import_sample('obs2')
    posterior=import_sample('posterior2')
    #print (kstest_cdf(posterior, obs, (2,3), 'beta_bernoulli', plot=True, plotp=True))
    #print (compare(posterior, obs, (2,3),'beta_bernoulli', plot=True,plotp=True, factor=100))
    print(testing_test_function_normal(size=2000, obs=obs))
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
    """

    
        