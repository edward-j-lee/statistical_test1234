import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import pymc3 as pm
from .inference import plot_p, CustomError, kstest
from .multivariate_test import test_kstest_multivar_norm, multivar_kstest2
from .stat_test import kstest, plt_to_base64_encoded_image, all_tests
from scipy.stats.stats import KstestResult

def import_sample_ndim(name):
    df=pd.read_csv(name, header=None)
    return np.asarray(df)

#following four functions take prior/likelihood parameters, observation and returns the updated 
#posterior parameters

#1 univariate normal with unknown mean and std
# prior: normal (mean), inverse gamma (var), likelihood: normal
def normal_unknown_mu_std(parameters, obs):
    mu0, nu, alpha, beta= parameters
    n=len(obs)
    x_bar=np.mean(obs)
    mu_new=(nu*mu0+n*x_bar)/(nu+n)
    nu_new  = nu+n
    alpha_new=alpha+n/2
    beta_new=beta+(1/2)*np.sum([(xi-x_bar)**2 for xi in obs])+((n*nu)/(nu+n))*((x_bar-mu0)**2/2)
    return mu_new, nu_new, alpha_new, beta_new

#2 Normal with known cov
#prior: normal , liikelihood: normal
def multivar_norm_known_cov(parameters, obs):
    mu0, cov0, cov= parameters
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


#3
#multivarite normal with known mean vector and unknown cov
#prior: inverse Wishart, likelihood: multivariate normal
def multivar_norm_known_mu(parameters, obs):
    v, psi, mu= parameters
    n=len(obs)
    v_new= n+v
    
    s=[np.outer(np.subtract(xi,mu), np.subtract(xi,mu)) for xi in obs]
    C=s[0]
    for c in s[1:]:
        C=np.add(C, c)

    psi_new=np.add(psi, C)
    
    return v_new, psi_new

#4 multivariate norm with unknwon mean and cov
# prior: multivariate normal (mean), inverse wishart (cov), likelihood (normal)
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

#takes a sample for mean and for standar deviation, observations, paramters, (and optionsl weights)
#and returns % of p values passd for mean, for variance, test result for mean and for variance in that order
#optional plot=True will plot  the pdf of mean/var against exact curve

def test_normal_two_unknowns(posterior_mean, posterior_var, obs, parameters, mean_weights=[], var_weights=[], plot=True):
    N=len(posterior_mean)
    
    if N!=len(posterior_var):
        raise CustomError('estimated mean and variance is of different length')
    mu2,nu2, a2,b2=normal_unknown_mu_std(parameters, obs)
    print ('newparam', mu2, nu2, a2, b2)
    exact_var=stats.invgamma(a=a2, scale=b2)
    exact_mean=stats.norm(loc=mu2, scale=np.sqrt(exact_var.mean()/nu2))
    
    all_plots=[]

    if len(mean_weights)==0:
        mean_weights=[1]*N
    if len(var_weights)==0:
        var_weights=[1]*N
    
    if plot==True:

        xpoints_var=np.linspace(min(posterior_var), max(posterior_var), N*10)
        ypoints_var=np.vectorize(exact_var.pdf)(xpoints_var)
        plt.plot(xpoints_var, ypoints_var, color='r', label='exact distribution')
        plt.hist(posterior_var, weights=var_weights, bins=100, density=True, color='b', label='estimated distribution')
        plt.legend(loc="upper right")
        plt.title('comparison of estimated and exact posterior distribution of variance')
        all_plots.append(plt_to_base64_encoded_image())

        xpoints_mean=np.linspace(min(posterior_mean), max(posterior_mean), N*10)
        ypoints_mean=np.vectorize(exact_mean.pdf)(xpoints_mean)
        #ypoints_mean=np.vectorize(lambda x: stats.norm.pdf(x, loc=mu2, scale=np.sqrt(np.mean(posterior_var)/nu2)))(xpoints_mean)
        plt.plot(xpoints_mean, ypoints_mean, color='r', label='exact distribution')
        plt.hist(posterior_mean, weights=mean_weights, bins=100, density=True, color='b', label='estimated distribution')
        plt.legend(loc="upper right")
        plt.title('comparison of estimated and exact posterior distribution of mean')
        all_plots.append(plt_to_base64_encoded_image())
        
    perc_passed_mean, p_plot1 =plot_p(posterior_mean, exact_mean.cdf, weights=mean_weights, plotp=plot)
    perc_passed_var, p_plot2=plot_p(posterior_var, exact_var.cdf, weights=var_weights, plotp=plot)
    all_plots+=[p_plot1, p_plot2]
    if plot:
        return perc_passed_mean, perc_passed_var, all_tests(posterior_mean, F=exact_mean, weights=mean_weights, tup=False), all_tests(posterior_var, F=exact_var, weights=var_weights, tup=False), all_plots
    else:
        return perc_passed_mean, perc_passed_var, all_tests(posterior_mean, F=exact_mean, weights=mean_weights, tup=False), all_tests(posterior_var, F=exact_var, weights=var_weights, tup=False)

#tests the above function by generating samples for mean and variance and then passes them to above function
def testing_test_function_normal(size, obs, parameters=(1,2,3,4)):
    mu,nu,alpha,beta=parameters
    with pm.Model() as model:
        var=pm.InverseGamma('var', alpha=alpha, beta=beta) #prior var - inv gamma
        μ=pm.Normal('μ', mu=mu, sigma=np.sqrt(var/nu)) #prior mean
        y=pm.Normal('y', mu=μ, sigma=np.sqrt(var), observed=obs) #likelihood - normal with mean μ and std σ
        trace=pm.sample(int(size/4))
    posterior_mean=trace['μ']
    posterior_var=trace['var']
    return test_normal_two_unknowns(posterior_mean, posterior_var, obs, parameters, plot=False)


#generates posterior distribution for multivariate normal using pymc's inference algorithm
#and passes the result to a function that tests it 
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
        trace=pm.sample(int(size/4))
    newparam=multivar_norm_known_cov(mean0, cov0, cov, obs)
    return test_kstest_multivar_norm(distribution=trace['mean'], mean=newparam[0], cov=newparam[1], allplots=False)

#given sample of covariance, exact sample (one from exact distribution) and optional weights of the same form
#and goes through each dimension of the covariance matrix and does ks test individually 
#and return the largest d value and correspoinding p
def test_cov(posterior_cov, exact_cov, weights):
    n =len(np.asarray(posterior_cov))
    dim=len(posterior_cov[0])
    if dim!=len(np.asarray(exact_cov)[0]):
        raise CustomError("dimesnion of sample cov and exact (sample) cov does not match")
    Dvals_cov=[]
    pvals_cov=[]
    for i in dim:
        for j in dim:
            d=  kstest(posterior_cov[:,i,j], exact_cov[:,i,j], weights=weights[:,i,j])[0]
            p=stats.kstwo.sf(d, n)
            Dvals_cov.append(d)
            pvals_cov.append(p)
    d_cov=max(Dvals_cov)
    p_cov=stats.kstwo.sf(d_cov,n)
    p_cov=np.clip(p_cov, 0,1)
    
    plt.plt(pvals_cov)
    plt.title("p values accross each dimension of the covariance")
    pplot_cov=plt_to_base64_encoded_image()
    
    return KstestResult(d_cov, p_cov), pplot_cov


### gets posterior of cov, observations and parameters as inputs
# generte exact sample from analytically calculated posterior
#and passes the posterior and exact sample to the above fuction
def multivarnorm_unknown_cov(posterior_cov, obs, parameters, weights=[]):
    n =len(np.asarray(posterior_cov))
    newparam= multivar_norm_known_mu(*parameters, obs=obs)
    exact_cov=stats.invwishart.rvs(*newparam, size=n*100)
    return test_cov(posterior_cov, exact_cov, weights)

def multivar_norm_known_mu_benchmark(parameters,obs, size):
    with pm.Model() as model:
        #cannot write benchmark function for inverse wishart 
        #no inv wishart function in pymc3
        pass


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
   
def compare_NIW_exact_sample(posterior_mean, posterior_cov, obs, parameters,mean_weights=[], cov_weights=[]):
    n=len(posterior_mean)
    newparam=multivar_norm_inv_wishart(parameters, obs)
    
    if True:
        exact_mean, exact_cov=multivarnorm_two_unknowns_sample(*newparam, size=n*100)
    
        ks_mean, pplot_mean= multivar_kstest2(posterior_mean, cdfs=exact_mean, weights=mean_weights, allplots=True, title="mean")
        
        ks_cov, pplot_cov = test_cov(posterior_cov, exact_cov)
    
        return ks_mean, ks_cov, [pplot_mean, pplot_cov]

