import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import pymc3 as pm
from inference import plot_p, CustomError

def import_sample_ndim(name):
    name+='.csv'
    df=pd.read_csv(name, header=None)
    return np.asarray(df)

def normal_unknown_mu_std(parameters, obs):
    mu0, nu, alpha, beta= parameters
    n=len(obs)
    x_bar=np.mean(obs)
    mu_new=(nu*mu0+n*x_bar)/(nu+n)
    nu_new  = nu+n
    alpha_new=alpha+n/2
    beta_new=beta+(1/2)*np.sum([(xi-x_bar)**2 for xi in obs])+((n*nu)/(nu+n))*((x_bar-mu0)**2/2)
    return mu_new, nu_new, alpha_new, beta_new

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
if __name__=='__main__':
    print(testing_test_function_normal(size=2000, obs=obs))