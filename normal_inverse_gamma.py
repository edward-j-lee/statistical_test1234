#normal inverse gamma
import scipy.stats as stats
import numpy as  np
from scipy.special import gamma
from sampling_algorithms.rejection import acc_rej_samp, supx
from normal_invgamma import normal_invgamma

#pdf
def gamma_pdf(x, σ, μ, λ, α, β):
    first=np.sqrt(λ/(2*np.pi*(σ**2)))
    second=(β**α)/gamma(α)
    third=(1/σ**2)**(α+1)
    fourth=np.exp(-(2*β+λ*(x-μ)**2)/(2*σ**2))
    return first*second*third*fourth

g_pdf= lambda x: stats.norm.pdf(x, loc=0, scale=5)
g_samp=lambda x: stats.norm.rvs(loc=0, scale=5,size=x)

x, sig2, loc, variance_scale, shape, scale=np.asarray([12]), np.asarray([4]),1.4,4,2.2, 1.3
print ('pdf1', normal_invgamma.pdf(x=x,sig2=sig2, variance_scale=variance_scale, shape=shape, scale=scale))

x, sig2, loc, variance_scale, shape, scale =12,4,1.4,4,2.2, 1.3

#print('pdf', gamma_pdf(x, (sig2)**0.5, loc, 1/np.sqrt(variance_scale), shape, scale ))

def gamma_pdf2(x, sigma, mu, lam, alpha, beta):
    a=stats.norm.pdf(x, loc=mu, scale=sigma/np.sqrt(lam))
    b=stats.invgamma.pdf(x, a=alpha, scale=beta)
    return a*b
#print('pdf2', gamma_pdf2(x, (sig2), loc, 1/(variance_scale)**2, shape, scale ))


print (normal_invgamma.rvs(1.4,4,2.2, 1.3,size=100))
