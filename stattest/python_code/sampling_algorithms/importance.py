import numpy as np
import scipy.stats as stats

def  imp_sampling_w(p, N, q_samp, q_pdf):
    xvar=(q_samp(N))
    zvar=np.vectorize(lambda x: (p(x))/q_pdf(x))(xvar)
    return xvar, zvar

def generate_weighted1(size, p=lambda x: stats.beta.pdf(x,2,3)):
    g_pdf=lambda x: stats.uniform.pdf(x)
    g_samp=lambda x: stats.uniform.rvs(size=x)
    return imp_sampling_w(p, size, g_samp, g_pdf)    

def generate_weighted2(size, p=lambda x: stats.beta.pdf(x,2,3)):
    g_pdf=lambda x: stats.halfnorm.pdf(x)
    g_samp=lambda x: stats.halfnorm.rvs(size=x)
    return imp_sampling_w(p, size, g_samp, g_pdf)    
