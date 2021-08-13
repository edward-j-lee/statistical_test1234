from django import forms

class BetaBernoulliForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    test_file_obs = forms.FileField(label="observed")
    test_file_weights = forms.FileField(label="weights", required=False) #optional
    alpha = forms.IntegerField()
    beta= forms.IntegerField()

class NormalKnownVarForm(forms.Form):
    test_file = forms.FileField()
    prior_mean = forms.IntegerField()
    prior_std = forms.IntegerField()
    likelihood_std =forms.IntegerField()

