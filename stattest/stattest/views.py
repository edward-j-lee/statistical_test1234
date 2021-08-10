from typing import NamedTuple
from django.forms.forms import Form
from django.http import HttpResponse
from django.http import HttpRequest
from .forms import StatTestForm
from django.shortcuts import render

def index(request):
    if request.method == "GET":
        return render(request, "article.html", {"form" : StatTestForm()})

    elif request.method == "POST":
        form = StatTestForm(request.POST, request.FILES)
        if not form.is_valid():
            print()
            print(form)
            print()
            return HttpResponse("Form is not valid")
        file  = open(request.FILES['test_file'])
        data = file.read()
        return HttpResponse(data)





