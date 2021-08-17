from django import forms

class BetaBernoulliForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    test_file_obs = forms.FileField(label="observed")
    weights = forms.FileField(label="weights", required=False) #optional
    alpha = forms.IntegerField(label="alpha")
    beta= forms.IntegerField(label="beta")

class GammaPoissonForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    test_file_obs = forms.FileField(label="observed")
    weights = forms.FileField(label="weights", required=False) #optional
    alpha = forms.IntegerField(label="alpha")
    beta= forms.IntegerField(label="beta")
    
class NormalKnownVarForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    weights = forms.FileField(label="weights", required=False) #optional
    test_file_obs = forms.FileField(label="observed")
    prior_mean = forms.IntegerField(label="prior mean")
    prior_std = forms.IntegerField(label = "prior std")
    likelihood_std =forms.IntegerField(label = "likelihood std")


class NormalKnownMuForm(forms.Form):
    test_file_posterior = forms.FileField(label="posterior")
    test_file_obs = forms.FileField(label="observed")
    weights = forms.FileField(label="weights", required=False) #optional
    alpha = forms.IntegerField(label="alpha")
    beta= forms.IntegerField(label="beta")
    mu=forms.IntegerField(label="mu")

#μ,ν,α,β
class NormalTwoUnknownsForm(forms.Form):
    mean_posterior = forms.FileField(label="posterior mean")
    mean_weights = forms.FileField(label="mean weights", required=False) #optional
    var_posterior = forms.FileField(label="posterior var")
    var_weights= forms.FileField(label="variance weights", required=False) #opt
    obs = forms.FileField(label="observed")
    mu= forms.IntegerField(label="μ")
    nu = forms.IntegerField(label = "ν")
    alpha=forms.IntegerField(label="α")
    beta=forms.IntegerField(label="β")

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
    nu=forms.IntegerField(label= "ν (degrees of freedom)")
    psi=forms.FileField(label="Ψ (scale matrix, positive definite)")
    likelihood_mu=forms.FileField(label="likelihood mean vector")

class MultiVarNormTwoUnknowns(forms.Form):
    posterior_mean = forms.FileField(label="posterior mean")
    mean_weights = forms.FileField(label="mean weights", required=False) #optional
    posterior_cov= forms.FileField(label="posterior cov")
    cov_weights= forms.FileField(label="covariance weights", required=False)
    obs = forms.FileField(label="observed")
    #parameters
    prior_mu = forms.FileField(label="prior mean vector")
    kappa = forms.IntegerField(label="κ")
    nu=forms.IntegerField(label="ν (degree of freedom)")
    psi=forms.IntegerField(label="Ψ (scale matrix)")

class BlackboxInference(forms.Form):
    sample1=forms.FileField(label="sample for problem 1")
    weights1=forms.FileField(label="weights for problem 1", required=False)
    sample2=forms.FileField(label="sample for problem 2")
    weights2=forms.FileField(label="weights for problem 2", required=False)
    sample3=forms.FileField(label="sample for problem 3")
    weights3=forms.FileField(label="weights for problem 3", required=False)
    sample4=forms.FileField(label="sample for problem 4")
    weights4=forms.FileField(label="weights for problem 4", required=False)