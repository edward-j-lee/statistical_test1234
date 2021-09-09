from math import dist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

from .python_code.inference import *
from .python_code.inference_multivar import test_multivar_norm_known_cov, test_normal_two_unknowns, testing_test_function_normal, multivar_norm_known_cov, multivarnorm_unknown_cov, compare_NIW_exact_sample
from .python_code.multivariate_test import test_kstest_multivar_norm
from .python_code.stat_test import kstest, plt_to_base64_encoded_image

#converts file to 1 dimensional array
def to_1darray(file):
    res= np.asarray(pd.read_csv(file,header=None)).flatten()
    res=res[np.logical_not(np.isnan(res))]
    return res

#covnerts file to list of n dimensional arrays
def to_ndarray(file):
    res=pd.read_csv(file, header=None)
    return np.asarray(res)

#converts file to list of nxn (matrix) dimensional arrays
def to_nbyn_matrix(file,n):
    res=np.genfromtxt(file, delimiter=',', skip_header=0)
    return np.asarray([r.reshape(n,n) for r in res])

#performs set of tests for one dimensional problem 
def one_dimensional_test(posterior, obs, parameters, dist_name, weights=None, algorithm_name='pymc3'):
    posterior=to_1darray(posterior) 
    obs = to_1darray(obs)
    N=len(posterior)
    try:
        weights= to_1darray(weights)
    except:
        weights=[]
     
    return test_cdf(posterior, obs, parameters, weights=weights, distribution_name=dist_name), any_benchmark(obs, parameters, dist_name, N=N, algorithm=algorithm_name)

#performs ks test on normal inference problem with two unknowns
def two_dimensional_test(post_mean, post_var, mean_w, var_w, obs, parameters):
    mean=to_1darray(post_mean)
    var=to_1darray(post_var)
    obs=to_1darray(obs)
    if mean_w:
        mean_w=to_1darray(mean_w)
    if var_w:
        var_w=to_1darray(var_w)
    N=len(mean)

    return test_normal_two_unknowns(mean, var, obs, parameters, mean_w, var_w, plot=True), testing_test_function_normal(N, obs, parameters)

#performs test with the benchmark on multivaraite normal with known cov
def multivar_norm_known_cov1(posterior, weights, obs, parameters):
    posterior=to_ndarray(posterior)
    obs=to_ndarray(obs)
    if weights:
        weights=to_ndarray(weights)
    N=len(posterior)
    mean0, cov0, cov= parameters
    benchmark=test_multivar_norm_known_cov(obs=obs, size=N, mean0=mean0, cov0=cov0, cov=cov)
    newparam= multivar_norm_known_cov(parameters, obs)
    res, all_plots =test_kstest_multivar_norm(posterior, weights=weights, mean=newparam[0], cov=newparam[1])
    return res, benchmark, all_plots

#performs ks test on multivar normal inference problem with known mu
def multivar_norm_known_mu(posterior, weights, obs, parameters,n):
    posterior=to_nbyn_matrix(posterior, n)

    parameters[1]=to_ndarray(parameters[1]) #converting scale matrix file to array

    parameters[2]=to_1darray(parameters[2]) #converting likelihood mean to array
    if weights:
        weights=to_nbyn_matrix(weights, n)
    obs=to_ndarray(obs)
    N=len(posterior)
    perc, result, plots = multivarnorm_unknown_cov(posterior, obs, parameters, weights)
    # no benchmark
    return perc, result, plots

def multiver_norm_unknown(posterior_mean, posterior_cov, mean_weights, cov_weights, parameters, obs):
    posterior_mean=to_ndarray(posterior_mean)
    dim=len(posterior_mean[0])
    posterior_cov=to_nbyn_matrix(posterior_cov, dim)

    if mean_weights:
        mean_weights=to_ndarray(mean_weights)
    if cov_weights:
        cov_weights=to_nbyn_matrix(cov_weights, dim)
    result_mean, result_cov, all_plots= compare_NIW_exact_sample(posterior_mean, posterior_cov, obs, parameters, mean_weights, cov_weights)
    return result_mean, result_cov, all_plots


#suite of bechmark problems
prior_param_normal =[[0,1],[0,10]]
likelihood_param_normal = [1,10]
obs = [0, 10]

#all parameters, obs tuple for normal
# eight in total - there is two possible value for prior std, two for likehood std, two for obs 
#so 2x2x2=8 problems in total for normal (there are three additional problems for beta bernoulli)
all_param_obs_norm= []
for pr in prior_param_normal:
    for li in likelihood_param_normal:
        for ob in obs:
            k=pr+[li]
            all_param_obs_norm.append((k,[ob]))

beta_param_obs=[([1,12.55], [0]), ([1,12.55], [0.99]), ([1,1], [0]) ]

#list of all problems in ([parameters], [obs]) form
all_problem_list= all_param_obs_norm+beta_param_obs

#runs the set of samples and weights on appropriate kstest
def benchmark_problems(list_posteriors, list_weights):
    newdic={}
    for i, (posterior, weights, param_obs) in enumerate(zip(list_posteriors, list_weights, all_problem_list)):
        i+=1
        post=to_1darray(posterior)
        N=len(post)
        if weights:
            weights=to_1darray(weights)
        else:
            weights=[]

        
        param=param_obs[0]
        observed=param_obs[-1]
        if len(param)==3:
            newparam=normal_known_var(param, observed)
            exact=stats.norm(*newparam)
            ks=kstest(post, exact.cdf, weights=weights)
            perc= plot_p(post, exact.cdf, weights=weights, plotp=False)[0]
        elif len(param)==2:
            newparam=beta_bernoulli(param, observed)
            exact=stats.beta(*newparam)
            ks=kstest(post, exact.cdf, weights=weights)
            perc=plot_p(post, exact.cdf, weights=weights, plotp=False)[0]
        xs=np.linspace(min(post), max(post),10*N )
        ys=np.vectorize(exact.pdf)(xs)
        
        plt.plot(xs, ys, label='exact distribution', color='r')
        
        if len(weights)==0:
            plt.hist(post, density=True, label='user estimated', color='b')
        else:
            plt.hist(post, weights=weights, density=True, label='user estimated', color='b')
        plt.title('problem '+str(i))
        plt.legend()
        newdic["problem_"+str(i)]=[perc, ks],  plt_to_base64_encoded_image()

    return newdic
