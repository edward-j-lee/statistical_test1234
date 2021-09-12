from django import forms
from django.forms import ModelForm
#the forms for receiving file and parameters from user

Choices=(('pymc3', 'pymc3 (NUTS)'), ('rejection','accept-reject sampling'), ('importance', 'importance sampling'), ('mcmc', 'mcmc sampling'))

class BetaBernoulliForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    test_file_obs = forms.FileField(label="observed")
    weights = forms.FileField(label="weights", required=False) #optional
    alpha = forms.FloatField(label="alpha")
    beta= forms.FloatField(label="beta")
    inf_algorithm= forms.ChoiceField(label="benchmark inference algoirthm", choices=Choices)

class GammaPoissonForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    test_file_obs = forms.FileField(label="observed")
    weights = forms.FileField(label="weights", required=False) #optional
    alpha = forms.FloatField(label="alpha")
    scale= forms.FloatField(label="scale")
    inf_algorithm= forms.ChoiceField(label="benchmark inference algoirthm", choices=Choices)
class NormalKnownVarForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    weights = forms.FileField(label="weights", required=False) #optional
    test_file_obs = forms.FileField(label="observed")
    prior_mean = forms.FloatField(label="prior mean")
    prior_std = forms.FloatField(label = "prior std")
    likelihood_std =forms.FloatField(label = "likelihood std")
    inf_algorithm= forms.ChoiceField(label="benchmark inference algoirthm", choices=Choices)

class NormalKnownMuForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    test_file_obs = forms.FileField(label="observed")
    weights = forms.FileField(label="weights", required=False) #optional
    alpha = forms.FloatField(label="alpha")
    beta= forms.FloatField(label="beta")
    mu=forms.FloatField(label="mu")
    inf_algorithm= forms.ChoiceField(label="benchmark inference algoirthm", choices=Choices)
#μ,ν,α,β
class NormalTwoUnknownsForm(forms.Form):
    mean_posterior = forms.FileField(label="posterior mean")
    mean_weights = forms.FileField(label="mean weights", required=False) #optional
    var_posterior = forms.FileField(label="posterior var")
    var_weights= forms.FileField(label="variance weights", required=False) #opt
    obs = forms.FileField(label="observed")
    mu= forms.FloatField(label="μ")
    nu = forms.FloatField(label = "ν")
    alpha=forms.FloatField(label="α")
    beta=forms.FloatField(label="β")

class MultiVarNormalKnownCov(forms.Form):
    posterior = forms.FileField(label="posterior")
    weights = forms.FileField(label="weights", required=False) #optional
    obs = forms.FileField(label="observed")
    prior_mean = forms.FileField(label="prior mean vector")
    prior_cov = forms.FileField(label = "prior covariance matrix")
    likelihood_cov =forms.FileField(label = "likelihood covariance matrix")

class MultiVarNormKnownMu(forms.Form):
    posterior = forms.FileField(label="posterior")
    weights = forms.FileField(label="weights", required=False) #optional
    obs = forms.FileField(label="observed") 
    nu=forms.FloatField(label= "ν (degrees of freedom)")
    psi=forms.FileField(label="Ψ (scale matrix, positive definite)")
    likelihood_mu=forms.FileField(label="likelihood mean vector")
    dim=forms.IntegerField(label="dimension of the covariance")

class MultiVarNormTwoUnknowns(forms.Form):
    posterior_mean = forms.FileField(label="posterior mean")
    mean_weights = forms.FileField(label="mean weights", required=False) #optional
    posterior_cov= forms.FileField(label="posterior cov")
    cov_weights= forms.FileField(label="covariance weights", required=False)
    obs = forms.FileField(label="observed")
    #parameters
    prior_mu = forms.FileField(label="prior mean vector")
    kappa = forms.FloatField(label="κ")
    nu=forms.FloatField(label="ν (degree of freedom)")
    psi=forms.FileField(label="Ψ (scale matrix)")


class BlackboxInference(forms.Form):
    sample1=forms.FileField(label="problem 1: normal with prior parameters of (0,1), likelihood std of 1, obs=[0]")
    weights1=forms.FileField(label="weights for problem 1", required=False)
    sample2=forms.FileField(label="problem 2: normal with prior parameters of (0,1), likelihood std of 10, obs=0")
    weights2=forms.FileField(label="weights for problem 2", required=False)
    sample3=forms.FileField(label="problem 3: normal with prior parameters of (0,10) and likelihood std of 1, obs=0")
    weights3=forms.FileField(label="weights for problem 3", required=False)
    sample4=forms.FileField(label="problem 4: normal with prior parameters of (0,10) and likelihood std of 10, obs=0")
    weights4=forms.FileField(label="weights for problem 4", required=False)
    
    sample5=forms.FileField(label="problem 5: normal with prior parameters of (0,1), likelihood std of 1, obs=[10]")
    weights5=forms.FileField(label="weights for problem 5", required=False)
    sample6=forms.FileField(label="problem 6: normal with prior parameters of (0,1), likelihood std of 10, obs=10")
    weights6=forms.FileField(label="weights for problem 6", required=False)
    sample7=forms.FileField(label="problem 7: normal with prior parameters of (0,10) and likelihood std of 1, obs=10")
    weights7=forms.FileField(label="weights for problem 7", required=False)
    sample8=forms.FileField(label="problem 8: normal with prior parameters of (0,10) and likelihood std of 10, obs=10")
    weights8=forms.FileField(label="weights for problem 8", required=False)
    
"""
    sample9=forms.FileField(label="problem 9: Beta Bernoulli with prior parameters of (1,12.55), obs=[0]")
    weights9=forms.FileField(label="weights for problem 9", required=False)
    sample10=forms.FileField(label="problem 10: Beta Bernoulli with prior parameters of (1,12.55), obs=[0.99]")
    weights10=forms.FileField(label="weights for problem 10", required=False)
    sample11=forms.FileField(label="problem 11: Beta Bernoulli with prior parameters of (1,1), obs=[0]")
    weights11=forms.FileField(label="weights for problem 11", required=False)
"""

