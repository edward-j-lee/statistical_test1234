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
    mu0=np.asarray(mu0)
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
    cov_new= np.linalg.inv(np.add(inv_prior_cov,np.multiply(n, inv_likeli_cov )))
    
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
    nu, psi, mu= parameters
    n=len(obs)
    nu_new= n+nu

    s=[np.outer(np.subtract(xi,mu), (np.subtract(xi,mu))) for xi in obs]
    C=s[0]
    
    for c in s[1:]:
        C=np.add(C, c)
    
    psi_new=np.add(psi, C)
    
    return nu_new, psi_new

#4 multivariate norm with unknwon mean and cov
# prior: multivariate normal (mean), inverse wishart (cov), likelihood (normal)
def multivar_norm_inv_wishart(parameters, obs):
    x1, x2, x3, x4 = parameters
    print ('afsd')
    print (obs)
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

#tests the above function by generating samples (benchmark) for mean and variance and then passes them to above function
def testing_test_function_normal(size, obs, parameters=(1,2,3,4)):
    mu,nu,alpha,beta=parameters
    with pm.Model() as model:
        var=pm.InverseGamma('var', alpha=alpha, beta=beta) #prior var - inv gamma
        μ=pm.Normal('μ', mu=mu, sigma=np.sqrt(var/nu)) #prior mean
        y=pm.Normal('y', mu=μ, sigma=np.sqrt(var), observed=obs) #likelihood - normal with mean μ and variance var
        trace=pm.sample(int(size/4))
    posterior_mean=trace['μ']
    posterior_var=trace['var']
    return test_normal_two_unknowns(posterior_mean, posterior_var, obs, parameters, plot=False)


#generates posterior distribution for multivariate normal using pymc's inference algorithm
#and passes the result to a function that tests it 
def test_multivar_norm_known_cov(obs, size, mean0, cov0, cov):
    if len(mean0)!=len(cov0):
        raise CustomError('dimension of prior cov does not match prior mean')
    if len(cov0)!=len(cov):
        raise CustomError('dimension of prior cov must equal to likelihood cov')
    dim=len(mean0)
    N=len(obs)
    with pm.Model() as model:
        mean=pm.MvNormal('mean', mu=mean0, cov=cov0, shape=dim)
        likelihood=pm.MvNormal('y', mu=mean, cov=cov, observed=obs)
        trace=pm.sample(int(size/4))
    newparam=multivar_norm_known_cov((mean0, cov0, cov), obs)
    return test_kstest_multivar_norm(distribution=trace['mean'], mean=newparam[0], cov=newparam[1], allplots=False)

#given sample of covariance, and sample from exact distribution
# and optional weights of the same form
#goes through each dimension of the covariance matrix and does ks test individually 
#and return the largest d value and correspoinding p
def test_cov(posterior_cov, exact_cov, weights):
    n =len(np.asarray(posterior_cov))
    dim=len(posterior_cov[0])
    if dim!=len(np.asarray(exact_cov)[0]):
        raise CustomError("dimesnion of sample cov and exact (sample) cov does not match")
    Dvals_cov=[]
    pvals_cov=[]
    final_cov_plots=[]
    if len(weights)==0:
        weights=np.ones(n*dim*dim).reshape(n,dim,dim)
    for i in range(dim):
        for j in range(dim):
            d=  kstest(posterior_cov[:,i,j], exact_cov[:,i,j], weights=weights[:,i,j])[0]
            p=stats.kstwo.sf(d, n)
            Dvals_cov.append(d)
            pvals_cov.append(p)
    plt.scatter(np.arange(len(pvals_cov))+1, pvals_cov)
    plt.axhline(0.01, color='r', label='critical value')
    plt.title('plot of all the p values (cov)')
    plt.xlabel('dimensions (flattened)')
    plt.ylabel('p value')
    plt.legend()
    final_cov_plots.append(plt_to_base64_encoded_image())

    ij=[]
    for someindex in range(10):
        if len(ij)>=dim**2: # if less than 10 dimensions are plottted but there are no more new dimensions 
            break 
        while True:
            i,j=np.random.choice(np.arange(dim),2,replace=True)
            if (i,j) not in ij: #to check if ith col and jth row has already been plotted
                ij.append((i,j))
                break
            else:
                continue
        sample_to_plot=posterior_cov[:,i,j]
        plt.hist(sample_to_plot, weights=weights[:,i,j], color='b', density=True, bins=100, label='user')
        low, high=np.min(sample_to_plot), np.max(sample_to_plot)
        cond=(exact_cov[:,i,j]<=high) & (exact_cov[:,i,j]>=low) #filter out outliers - those in exact sample
                                                                #that lies outside the user's sample
        newexact=exact_cov[:,i,j][cond]
        plt.hist(newexact, color='r', density=True, histtype='step', bins=100, label='exact')
        plt.legend()
        plt.title('plotting result of '+str(i+1)+'th row and '+str(j+1)+'th column')
        final_cov_plots.append(plt_to_base64_encoded_image())
    d_cov=max(Dvals_cov)
    p_cov=stats.kstwo.sf(d_cov,n)*dim
    p_cov=np.clip(p_cov, 0,1)
    pvals_cov=np.asarray(pvals_cov)
    perc= (np.sum([pvals_cov>=0.01])/len(pvals_cov))*100
    return perc, KstestResult(d_cov, p_cov), final_cov_plots


### gets posterior of cov, observations and parameters as inputs
# generte exact sample from analytically calculated posterior
#and passes the posterior and exact sample to test_cov
def multivarnorm_unknown_cov(posterior_cov, obs, parameters, weights=[]):
    n =len(np.asarray(posterior_cov))
    newparam= multivar_norm_known_mu(parameters, obs=obs)
    if n<=1000:
        exact_cov_size=100000 #if given sample is small, exact sample size is 100k
    else:
        exact_cov_size=10**6 #otherwise it is fixed to 1m
    exact_cov=stats.invwishart.rvs(*newparam, size=exact_cov_size)

    return test_cov(posterior_cov, exact_cov, weights)

def multivar_norm_known_mu_benchmark(parameters,obs, size): 
    with pm.Model() as model:
        #cannot write benchmark function for inverse wishart 
        #no inv wishart function in pymc3
        pass


#given parameters, returns samples of multivariate mean and sample of cov
#following Normal distribution for mean and Inverse Wishart distribution for cov
def multivarnorm_two_unknowns_sample(mu, k, v, phi, size=100):
    if len(phi)!=len(mu):
        raise CustomError('dimension of mean must equat to dimension of cov')
    if v<len(phi):
        raise CustomError('df must be greater than dimension of phi')
    cov=stats.invwishart.rvs(df=v, scale=phi, size=size)
    print ('cov')
    meanlist=[]
    for i in cov:
        mean=stats.multivariate_normal.rvs(size=1, mean=mu, cov=np.multiply(1/k,i))
        meanlist.append(mean)
    mean=np.asarray(meanlist)
    return mean, cov


#tests multivariate distirbutions for unknown mean and cov
#given parameters for Normal-inverse-Wishart distribution,
#and estimated posterior distributions, 
#returns the result of ks test with respect to the exact sample
   
def compare_NIW_exact_sample(posterior_mean, posterior_cov, obs, parameters,mean_weights=[], cov_weights=[]):
    print ('compare niw start')
    n=len(posterior_mean)
    newparam=multivar_norm_inv_wishart(parameters, obs)
    if n<=1000:
        exact_size=10000
    else:
        exact_size=10**6
    exact_mean, exact_cov=multivarnorm_two_unknowns_sample(*newparam, size=exact_size)
    perc_mean, ks_mean, plot_mean= test_mu(posterior_mean, exact_mean, mean_weights)
    perc_cov, ks_cov, plot_cov = test_cov(posterior_cov, exact_cov, weights=cov_weights)
    print (plot_mean, plot_cov)
    return (perc_mean, ks_mean), (perc_cov, ks_cov), plot_mean+plot_cov

#analogous to test_cov but with mean vectors
#testing samples of mu against exact samples
def test_mu(posterior_mu, exact_mu, weights):
    n =len(np.asarray(posterior_mu))
    dim=len(posterior_mu[0])
    if dim!=len(np.asarray(exact_mu)[0]):
        raise CustomError("dimesnion of sample of mu and exact (sample) of mu does not match")
    Dvals_mu=[]
    pvals_mu=[]
    final_mu_plots=[]
    if len(weights)==0:
        weights=np.ones(n*dim).reshape(n,dim)
    for i in range(dim):
        d=  kstest(posterior_mu[:,i], exact_mu[:,i], weights=weights[:,i])[0]
        p=stats.kstwo.sf(d, n)
        Dvals_mu.append(d)
        pvals_mu.append(p)
    pvals_mu=np.asarray(pvals_mu)
    perc= (np.sum([pvals_mu>=0.01])/len(pvals_mu))*100
    plt.scatter(np.arange(dim)+1, pvals_mu)
    plt.axhline(0.01, color='r', label='critical value')
    plt.title('plotting all the p values (mu)')
    plt.xlabel('dimensions')
    plt.ylabel('p value')
    final_mu_plots.append(plt_to_base64_encoded_image())
    i_=[]
    for someindex in range(10):
        if len(i_)>=dim:
            break
        while True:
            i=np.random.choice(np.arange(dim),replace=True)
            if i not in i_:
                i_.append(i)
                break
            else:
                continue
        sample_to_plot=posterior_mu[:,i]
        plt.hist(sample_to_plot, weights=weights[:,i], color='b', density=True, bins=100, label='user')
        low, high=np.min(sample_to_plot), np.max(sample_to_plot)
        cond=(exact_mu[:,i]<=high) & (exact_mu[:,i]>=low)
        newexact=exact_mu[:,i][cond]
        plt.hist(newexact, color='r', density=True, histtype='step', bins=100, label='exact')
        plt.legend()
        plt.title('plotting result of '+str(i+1)+'th dimension')
        final_mu_plots.append(plt_to_base64_encoded_image())

    d_mu=max(Dvals_mu)
    p_mu=stats.kstwo.sf(d_mu,n)
    p_mu=np.clip(p_mu, 0,1)
    return perc, KstestResult(d_mu, p_mu), final_mu_plots
