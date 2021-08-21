from math import dist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .python_code.inference import *
from .python_code.inference_multivar import test_multivar_norm_known_cov, test_normal_two_unknowns, testing_test_function_normal, multivar_norm_known_cov, multivarnorm_unknown_cov, compare_NIW_exact_sample
from .python_code.multivariate_test import test_kstest_multivar_norm
from .python_code.stat_test import kstest, plt_to_base64_encoded_image


def to_1darray(file):
    res= np.genfromtxt(file, skip_header=0, skip_footer=0).flatten()
    res=res[np.logical_not(np.isnan(res))]
    return res
def to_ndarray(file):
    res=pd.read_csv(file, header=None)
    return np.asarray(res)

def one_dimensional_test(posterior, obs, parameters, dist_name, weights=None):
    posterior=to_1darray(posterior) 
    obs = to_1darray(obs)
    N=len(posterior)
    try:
        weights= to_1darray(weights)
    except:
        weights=[]
    
    return test_cdf(posterior, obs, parameters, weights=weights, distribution_name=dist_name), benchmark(obs, parameters, dist_name, N=N)

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

def multivar_norm_known_cov(posterior, weights, obs, parameters):
    posterior=to_ndarray(posterior)
    obs=to_ndarray(obs)
    if weights:
        weights=to_ndarray(weights)
    N=len(posterior)
    mean0, cov0, cov= parameters
    newparam= multivar_norm_known_cov(parameters, obs)
    benchmark=test_multivar_norm_known_cov(obs=obs, size=N, mean0=mean0, cov0=cov0, cov=cov)
    res, all_plots =test_kstest_multivar_norm(posterior, weights, mean=newparam[0], cov=newparam[1])
    return res, benchmark, all_plots

def multivar_norm_known_mu(posterior, weights, obs, parameters):
    posterior=to_ndarray(posterior)
    if weights:
        weights=to_ndarray(weights)
    N=len(posterior)
    result, plot =multivarnorm_unknown_cov(posterior, obs, parameters, weights)
    # no benchmark
    return result, [plot]

def multiver_norm_unknown(posterior_mean, posterior_cov, mean_weights, cov_weights, parameters, obs):
    posterior_mean=to_ndarray(posterior_mean)
    posterior_cov=to_ndarray(posterior_cov)
    if mean_weights:
        mean_weights=to_ndarray(mean_weights)
    if cov_weights:
        cov_weights=to_ndarray(cov_weights)
    result_mean, result_cov, all_plots= compare_NIW_exact_sample(posterior_mean, posterior_cov, obs, parameters, mean_weights, cov_weights)
    return result_mean, result_cov, all_plots


obs=[[1], [10], [0,-1,1,-0.5,0.8,1.4,2], stats.norm.rvs(size=100, loc=0, scale=2)]
def benchmark_problems(list_posteriors, list_weights, times, param=(0,1,1)):
    newdic={}
    for i, (posterior, weights, t) in enumerate(zip(list_posteriors, list_weights, times)):
        i+=1
        post=to_1darray(posterior)
        N=len(post)
        if weights:
            weights=to_1darray(weights)
        else:
            pass
        #newdic[i]=(post, weights)

        newparam=normal_known_var(param, obs[i])
        exact=stats.norm(*newparam)
        ks=kstest(post, exact.cdf, weights=weights)
        perc, N= plot_p(post, exact.cdf, weights=weights, plotp=False)

        xs=np.linspace(min(post), max(post),10*N )
        ys=np.vectorize(exact.pdf)(xs)
        
        plt.plot(xs, ys, label='exact distribution', color='r')
        
        if len(weights)==0:
            plt.hist(post, density=True, label='user estimated', color='b')
        else:
            plt.hist(post, weights=weights, density=True, label='user estimated', color='b')
        plt.title('problem '+str(i))
        plt.legend()
        newdic["problem_"+str(i)]=[perc, ks,t],  plt_to_base64_encoded_image()

    return newdic
