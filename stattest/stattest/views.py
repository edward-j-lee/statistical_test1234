from typing import NamedTuple
from django.forms.forms import Form
from django.http import HttpResponse
from django.http import HttpRequest
from .forms import StatTestForm
from django.shortcuts import render
from .helper import handle_data_file
import numpy as np

def index(request):
    if request.method == "GET":
        return render(request, "article.html", {"form" : StatTestForm()})

    elif request.method == "POST":
        form = StatTestForm(request.POST, request.FILES)
        if not form.is_valid():
            return HttpResponse("Form is not valid")

        handle_data_file(file=request.FILES["test_file"])
        return HttpResponse("File is being handled..... ")





