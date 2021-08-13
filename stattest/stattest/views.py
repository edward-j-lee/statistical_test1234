from typing import NamedTuple
from django.forms.forms import Form
from django.http import HttpResponse
from django.http import HttpRequest
from .forms import BetaBernoulliForm
from django.shortcuts import render
from .helper import beta_bernoulli
import numpy as np

def beta_bernoulli(request):
    if request.method == "GET":
        return render(request, "article.html", {"form" : BetaBernoulliForm()})

    elif request.method == "POST":
        form = BetaBernoulliForm(request.POST, request.FILES)
        if not form.is_valid():
            return HttpResponse("Form is not valid")

        parameters = form.alpha, form.beta

        beta_bernoulli(posterior=request.FILES["test_file_posterior"], obs=request.FILES["test_file_obs"], parameters=parameters)

        return HttpResponse("File is being handled..... ")





