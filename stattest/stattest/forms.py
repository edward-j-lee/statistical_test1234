from django import forms

class BetaBernoulliForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    test_file_obs = forms.FileField(label="observed")
    test_file_weights = forms.FileField(label="weights", required=False) #optional
    alpha = forms.IntegerField(label="alpha")
    beta= forms.IntegerField(label="beta")

class GammaPoissonForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    test_file_obs = forms.FileField(label="observed")
    test_file_weights = forms.FileField(label="weights", required=False) #optional
    alpha = forms.IntegerField(label="alpha")
    beta= forms.IntegerField(label="beta")


class NormalKnownVarForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    test_file_obs = forms.FileField(label="observed")
    prior_mean = forms.IntegerField(label="prior mean")
    prior_std = forms.IntegerField(label = "prior std")
    likelihood_std =forms.IntegerField(label = "likelihood std")


class NormalKnownMuForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    test_file_obs = forms.FileField(label="observed")
    test_file_weights = forms.FileField(label="weights", required=False) #optional
    alpha = forms.IntegerField(label="alpha")
    beta= forms.IntegerField(label="beta")
    mu=forms.IntegerField(label="mu")