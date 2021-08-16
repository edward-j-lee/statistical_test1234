from stattest.python_code.inference_multivar import multivarnorm_unknown_cov
from pandas.core.accessor import delegate_names
from theano.printing import PatternPrinter
from stattest.python_code.inference import benchmark
from typing import NamedTuple
from django.forms.forms import Form
from django.http import HttpResponse
from django.http import HttpRequest
from .forms import *
from django.shortcuts import render
from . import helper
import numpy as np

INFERENCE_PROBLEMS = {
    "beta_bernoulli": (BetaBernoulliForm, ["alpha", "beta"], 1),
    "gamma_poisson": (GammaPoissonForm, ["alpha", "beta"], 1),
    "normal_known_var": (NormalKnownVarForm, ["prior_mean", "prior_std", "likelihood_std"], 1),
    "normal_known_mu": (NormalKnownMuForm, ["alpha", "beta", "mu"], 1),
    "normal_two_unknowns": (NormalTwoUnknownsForm, ["mu", "nu", "alpha", "beta"], 2),
    "multivar_norm_known_cov": (MultiVarNormalKnownCov, ["prior_mean", "prior_cov", "likelihood_cov"], 3),
    "multivar_norm_known_mu": (MultiVarNormKnownMu, ["nu", "psi", "likelihood_mu"], 3),
    "multivar_norm_inv_wishart": (MultiVarNormTwoUnknowns, ["prior_mu", "kappa", "nu", "psi"], 3)}

def inference_problem(request, problem_type):
    if problem_type not in INFERENCE_PROBLEMS:
        return HttpResponse("Error: No such problem type")

    form = INFERENCE_PROBLEMS[problem_type][0]
    if request.method == "GET":
        return render(request, "form.html", {"form" : form()})

    elif request.method == "POST": 
        form = form(request.POST, request.FILES)
        if not form.is_valid():
            return HttpResponse("Form is not valid")



        if INFERENCE_PROBLEMS[problem_type][2]==1:
            parameters = INFERENCE_PROBLEMS[problem_type][1]

            parameters = [form.cleaned_data[parameter] for parameter in parameters]
            #weights 
            if "weights" in request.FILES:
                test_results, benchmark_results = helper.one_dimensional_test(posterior=request.FILES["test_file_posterior"], obs=request.FILES["test_file_obs"], parameters=parameters, dist_name=problem_type, weights=request.FILES["weights"])
            else:
                test_results, benchmark_results = helper.one_dimensional_test(posterior=request.FILES["test_file_posterior"], obs=request.FILES["test_file_obs"], parameters=parameters, dist_name=problem_type, weights=None)
            
            all_plots=[i.decode() for i in test_results[2]]
            newdic={}
            for i in test_results[1].keys():
                newdic[i]=test_results[1][i],benchmark_results[i] 
            return render(request, "results.html", {"form" : BetaBernoulliForm(), "perc_passed" : test_results[0], "text_results" : newdic, "plots": all_plots})
                                                                    #betabernoulliiform????

        elif INFERENCE_PROBLEMS[problem_type][2] == 2: #speical case
            parameters = INFERENCE_PROBLEMS[problem_type][1]

            parameters = [form.cleaned_data[parameter] for parameter in parameters]

            posterior_mean = request.FILES["mean_posterior"]
            posterior_var= request.FILES["var_posterior"]
            
            if "mean_weights" in request.FILES:
                mean_weights = request.FILES["mean_weights"]
            else:
                mean_weights=[]
            if "var_weights" in request.FILES:
                var_weights= request.FILES["var_weights"]
            else:
                var_weights=[]
            obs=request.FILES["obs"]
            test_result, benchmark_result=helper.two_dimensional_test(posterior_mean, posterior_var, mean_weights, var_weights, obs, parameters)
            all_plots=[i.decode() for i in test_result[4]]
            #need fix send plots and result to results.html

        elif INFERENCE_PROBLEMS[problem_type][2]==3:
            if problem_type=="multivar_norm_known_cov" or problem_type=="multivar_norm_known_mu":
                posterior=request.FILES["posterior"]
                obs=request.FILES["obs"]
                if "weights" in request.FILES:
                    weights=request.FILES["weights"]
                else:
                    weights=[]
                if problem_type=="multivar_norm_known_cov":
                    prior_mean=request.FILES["prior_mean"]
                    prior_cov=request.FILES["prior_cov"]
                    likelihood_cov=request.FILES["likelihood_cov"]

                    prior_mean=helper.to_1darray(prior_mean)
                    prior_cov=helper.to_ndarray(prior_cov)
                    likelihood_cov=helper.to_ndarray(likelihood_cov)

                    parameters=prior_mean, prior_cov, likelihood_cov
                    result, benchmark_result= helper.multivar_norm_known_cov(posterior, weights, obs, parameters)
                elif problem_type=="multivar_norm_known_mu":
                    nu = INFERENCE_PROBLEMS[problem_type][1][0]
                    nu = form.cleaned_data[nu]

                    psi=request.FILES["psi"]
                    psi=helper.to_ndarray(psi)

                    likelihood_mu=request.FILES["likelihood_mu"]
                    likelihood_mu=helper.to_ndarray(likelihood_mu)
                    parameters= nu, psi, likelihood_mu
                    result=helper.multivar_norm_known_mu(posterior, weights, obs, parameters)

                elif problem_type=="multivar_norm_inv_wishart":
                    posterior_mean = request.FILES["posterior_mean"]
                    posterior_cov= request.FILES["posterior_cov"]

                    if "mean_weights" in request.FILES:
                        mean_weights = request.FILES["mean_weights"]
                    else:
                        mean_weights = []
                    if "cov_weights" in request.FILES:
                        cov_weights = request.FILES["cov_weights"]
                    else:
                        cov_weights=[]

                    obs=request.FILES["obs"]

                    #parameters
                    prior_mu=request.FILES["prior_mu"]
                    prior_mu=helper.to_1darray(prior_mu)

                    kappa=INFERENCE_PROBLEMS[problem_type][1][1]
                    kappa = form.cleaned_data[kappa]
                    nu = INFERENCE_PROBLEMS[problem_type][1][2]
                    nu=form.cleaned_data[nu]

                    psi=request.FILES["psi"]
                    psi=helper.to_ndarray(psi)

                    parameters=prior_mu, kappa, nu, psi

                    result=helper.multiver_norm_unknown(posterior_mean, posterior_cov, mean_weights, cov_weights, parameters, obs)


                





