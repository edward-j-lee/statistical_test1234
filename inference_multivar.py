import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import pymc3 as pm
from inference import plot_p, CustomError
from multivariate_test import test_kstest_multivar_norm

def import_sample_ndim(name):
    name+='.csv'
    df=pd.read_csv(name, header=None)
    return np.asarray(df)
#1
def normal_unknown_mu_std(parameters, obs):
    mu0, nu, alpha, beta= parameters
    n=len(obs)
    x_bar=np.mean(obs)
    mu_new=(nu*mu0+n*x_bar)/(nu+n)
    nu_new  = nu+n
    alpha_new=alpha+n/2
    beta_new=beta+(1/2)*np.sum([(xi-x_bar)**2 for xi in obs])+((n*nu)/(nu+n))*((x_bar-mu0)**2/2)
    return mu_new, nu_new, alpha_new, beta_new

#2
def multivar_norm_known_cov(mu0, cov0, cov, obs):
    n=len(obs)
    inv_prior_cov=np.linalg.inv(cov0)
    inv_likeli_cov=np.linalg.inv(cov)
    samplemean=np.mean(obs, axis=0)
    cov_new= np.linalg.inv(inv_prior_cov+np.multiply(n, inv_likeli_cov ))
    
    first=np.matmul(inv_prior_cov, mu0)
    n_sig_inv=np.multiply(n, inv_likeli_cov)
    second=np.matmul(n_sig_inv, samplemean)
    multiplying_term=np.add(first, second)
    
    mean_new=np.matmul(cov_new, multiplying_term)
    
    return mean_new, cov_new

multivar_dist={'multivar_norm_known_cov':multivar_norm_known_cov}

#3
#multivarite normal with known mean vector and unknown cov
#inverse Wishart
def multiver_norm_known_mu(v, phi, mu, obs):
    n=len(obs)
    v_new= n+v
    
    s=[np.outer(np.subtract(xi,mu), np.subtract(xi,mu)) for xi in obs]
    C=s[0]
    for c in s[1:]:
        C=np.add(C, c)
    
    phi_new=np.add(phi, C)
    
    return v_new, phi_new
    
#4
def multivar_norm_inv_wishart(parameters, obs):
    x1, x2, x3, x4 = parameters
    xbar=np.mean(obs, axis=0)
    n=len(obs)
    first_new=np.add(np.multiply(x2, x1), np.multiply(n,xbar))/(x2+n)
    second_new=x2+n
    third_new=x3+n
    C_= [np.outer(np.subtract(xi,xbar),np.subtract(xi,xbar)) for xi in obs]
    C=C_[0]
    for c in C_[1:]:
        C=np.add(C, c)
    
    term4_a=np.add(x4, C)
    term4_b1=(x2*n)/(x2+n)
    term4_b2=np.outer(np.subtract(xbar, x1), np.subtract(xbar, x1))
    term4_b=np.multiply(term4_b1, term4_b2)
    term4= np.add(term4_a, term4_b)
    
    return first_new, second_new, third_new, term4


def test_normal_two_unknowns(posterior_mean, posterior_var, obs, parameters, distribution_name, plot=True, plotp=True):
    if distribution_name !='normal_unknown_mu_std':
        raise CustomError('wrong inference problem')
    N=len(posterior_mean)
    
    if N!=len(posterior_var):
        raise CustomError('estimated mean and variance is of different length')
    mu2,nu2, a2,b2=normal_unknown_mu_std(parameters, obs)
    print ('newparam', mu2, nu2, a2, b2)
    exact_var=stats.invgamma(a=a2, scale=b2)
    exact_mean=stats.norm(loc=mu2, scale=np.sqrt(exact_var.mean()/nu2))
    
    
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


def test_multivar_norm_known_cov(obs=[], size=2500, mean0=[0,0,0], cov0=np.identity(3), cov=np.identity(3)):
    if len(mean0)!=len(cov0):
        raise CustomError('dimension of prior cov does not match prior mean')
    if len(cov0)!=len(cov):
        raise CustomError('dimension of prior cov must equal to likelihood cov')
    dim=len(mean0)
    if obs==[]:
        obs=np.random.randn(30).reshape(10,3)
    N=len(obs)
    with pm.Model() as model:
        mean=pm.MvNormal('mean', mu=mean0, cov=cov0, shape=dim)
        likelihood=pm.MvNormal('y', mu=mean, cov=np.multiply(1/N,cov), observed=obs)
        trace=pm.sample(size)
    newparam=multivar_norm_known_cov(mean0, cov0, cov, obs)
    return test_kstest_multivar_norm(distribution=trace['mean'], mean=newparam[0], cov=newparam[1])

def multivarnorm_unknown_cov(posterior_cov, obs, parameters, mu):
    n, dim=np.asarray(posterior_cov).shape
    newparam= multiver_norm_known_mu(*parameters, mu=mu, obs=obs)
    exact_cov=stats.invwishart.rvs(*newparam, size=n*100)
    
    Dvals_cov=[]
    for i in dim:
        for j in dim:
            d=stats.kstest(posterior_cov[:,i,j], exact_cov[:,i,j])[0]
            Dvals_cov.append(d)
    d_cov=max(Dvals_cov)
    p_cov=stats.kstwo.sf(d_cov,n)
    
    return d_cov, p_cov
#given parameters, returns samples of multivariate mean and cov
def multivarnorm_two_unknowns_sample(mu, k, v, phi, size=100):
    if len(phi)!=len(mu):
        raise CustomError('dimension of mean must equat to dimension of cov')
    if v<len(phi):
        raise CustomError('df must be greater than dimension of phi')
    cov=stats.invwishart.rvs(df=v, scale=phi, size=size)
    mean=stats.multivariate_normal.rvs(size=size, mean=mu, cov=np.multiply(1/k,cov))
    return mean, cov


#tests multivariate distirbutions for unknown mean and cov
#given parameters for Normal-inverse-Wishart distribution,
#and estimated posterior distributions, 
#returns the result of ks test with respect to the exact sample
   
def compare_NIW_exact_sample(posterior_mean, posterior_var, obs, parameters):
    n=len(posterior_mean)
    newparam=multivar_norm_inv_wishart(parameters, obs)
    exact_mean, exact_cov=multivarnorm_two_unknowns_sample(*newparam, size=n*100)
    dim=len(np.transpose(exact_mean))
    Dvals_mean=[]
    for i in dim:
       d=stats.kstest(np.transpose(posterior_mean)[i], exact_mean[i])[0]
       Dvals_mean.append(d)
    d_mean=max(Dvals_mean)
    p_mean=stats.kstwo.sf(d_mean, n)
    
    Dvals_cov=[]
    for i in dim:
        for j in dim:
            d=stats.kstest(posterior_var[:,i,j], exact_cov[:,i,j])[0]
            Dvals_cov.append(d)
    d_cov=max(Dvals_cov)
    p_cov=stats.kstwo.sf(d_cov,n)
    return p_mean, p_cov
    
    
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
    #pass
    #testing normal with two unknown
    print(testing_test_function_normal(size=2000, obs=obs))
    
    #testing multivar norm with known cov
    #print (test_multivar_norm_known_cov())
    
    #testing multvar norm 
    