import numpy as np
import scipy.stats as stats

def  imp_sampling_w(p, N, q_samp, q_pdf, f=lambda x:x):
    #q - beta dis in question (prior)
    #p - bernoulli (likelihood)
    xvar=(q_samp(N))
    zvar=np.vectorize(lambda x: (p(x))/q_pdf(x))(xvar)
    return xvar, zvar

def generate_weighted1(size, f=lambda x: stats.beta.pdf(x,2,3)):
    g_pdf=lambda x: stats.uniform.pdf(x)
    g_samp=lambda x: stats.uniform.rvs(size=x)
    return imp_sampling_w(f, size, g_samp, g_pdf)    

def generate_weighted2(size, f=lambda x: stats.beta.pdf(x,2,3)):
    g_pdf=lambda x: stats.halfnorm.pdf(x)
    g_samp=lambda x: stats.halfnorm.rvs(size=x)
    return imp_sampling_w(f, size, g_samp, g_pdf)    

imp_sampling_w(lambda z: stats.bernoulli.pdf(z, 2,3), stats.beta.rvs(), stats.beta.pdf)
#prior with mean of 0 #supply obs with 6s and 7s #test if users algorithm can