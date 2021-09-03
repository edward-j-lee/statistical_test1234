import numpy as np
from scipy import stats
        
def supx(func, xrange=(-20,20)):
    x=np.linspace(*xrange, 1000)
    y=np.vectorize(func)(x)
    d=np.max(y)
    condition=(y==d)
    return x[np.where(condition)[0][0]]

def acc_rej_samp(func, g_pdf, g_samp, supX, N=100):
    #acceptance rejection sampling
    #func is a  pdf without normalizing constant
    M=func(supX)/g_pdf(supX)
    c=np.zeros(1)
    counter=0
    while np.size(c)-1<N:
        uvar=np.random.uniform(low=0, high=1)
        yvar=g_samp(1)
        if uvar<func(yvar)/(M*g_pdf(yvar)):
            c=np.append(c,yvar)
    return c[1:]

target_dist=lambda x: stats.beta.pdf(x,2,3)
def rej_generate1(N, target_dist=target_dist):
    g_pdf=lambda x: stats.uniform.pdf(x)
    g_samp=lambda x: np.random.uniform(low=0,high=1, size=x)
    return acc_rej_samp(target_dist, g_pdf, g_samp, supx(target_dist), N=N)


def rej_generate2(N, target_dist=target_dist):
    g_pdf=lambda x: stats.halfnorm.pdf(x)
    g_samp=lambda x: stats.halfnorm.rvs(size=x)
    return acc_rej_samp(target_dist, g_pdf, g_samp, supx(target_dist), N=N)

def rej_generate3(N, target_dist=target_dist):
    g_pdf= lambda x: stats.norm.pdf(x)
    g_samp=lambda x: np.random.normal(size=x)
    return acc_rej_samp(target_dist, g_pdf, g_samp, supx(target_dist), N=N)
rej1_10k=rej_generate1(10000)
rej2_10k=rej_generate2(10000)
rej3_10k=rej_generate3(10000)
sample=rej_generate1(100)