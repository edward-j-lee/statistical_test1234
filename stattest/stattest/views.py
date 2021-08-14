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
    "normal_known_var": (NormalKnownVarForm, ["prior_mean", "prior_std", "likelihood_std"], 1)
}

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

        parameters = INFERENCE_PROBLEMS[problem_type][1]

        parameters = [form.cleaned_data[parameter] for parameter in parameters]


        if INFERENCE_PROBLEMS[problem_type][2] == 2: #speical case
            POSTERIOR = request.FILES[""]


#fix this
        else:


            #weights 
            weights = []
            if "weights" in request.FILES:
                weights = #converts file to array
          #  posterior = convert to array


            test_results, benchmark_results = helper.beta_bernoulli(posterior=request.FILES["test_file_posterior"], obs=request.FILES["test_file_obs"], parameters=parameters)






        all_plots=[i.decode() for i in test_results[2]]

        newdic={}
        for i in test_results[1].keys():
            newdic[i]=test_results[1][i],benchmark_results[i]
        

        return render(request, "results.html", {"form" : BetaBernoulliForm(), "perc_passed" : test_results[0], "text_results" : newdic, "plots": all_plots})





