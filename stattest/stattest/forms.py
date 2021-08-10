from django import forms

class StatTestForm(forms.Form):
    test_file = forms.FileField()
    func = forms.CharField(max_length=100)
    param = forms.CharField(max_length=100) 