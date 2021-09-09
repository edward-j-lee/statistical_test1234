import numpy as np
from scipy import stats
        
def calc_M(f, g, xrange=(0,1)):
    x=np.linspace(*xrange, 10000)
    g_x=np.vectorize(g)(x)
    f_x=np.vectorize(f)(x)
    
    cond=(g_x!=0)
    g_x=g_x[cond]
    f_x=f_x[cond]
    
    y=f_x/g_x
    return np.max(y)

def acc_rej_samp(func, g_pdf, g_samp, N=100, ran=()):
    #acceptance rejection sampling
    #func is a  pdf of target dist without normalizing constant
    if len(ran)==0:    # if range is not given then it is calculated by first drawing a large number of samples from g
        ran=g_samp(50000)  #and taking its minimum/maximum
        range_=min(ran)-1, max(ran)+1
    else:
        range_=ran
    M=calc_M(func, g_pdf, range_)

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
    return acc_rej_samp(target_dist, g_pdf, g_samp, N=N)


def rej_generate2(N, target_dist=target_dist):
    g_pdf=lambda x: stats.halfnorm.pdf(x)
    g_samp=lambda x: stats.halfnorm.rvs(size=x)
    return acc_rej_samp(target_dist, g_pdf, g_samp, N=N)

def rej_generate3(N, target_dist=target_dist):
    g_pdf= lambda x: stats.norm.pdf(x)
    g_samp=lambda x: np.random.normal(size=x)
    return acc_rej_samp(target_dist, g_pdf, g_samp,  N=N)
rej1_10k=rej_generate1(10000)
rej2_10k=rej_generate2(10000)
rej3_10k=rej_generate3(10000)
sample=rej_generate1(100)