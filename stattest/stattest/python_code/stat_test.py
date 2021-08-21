import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#from .sampling_algorithms.importance import generate_weighted1
#from .sampling_algorithms.rejection import sample
import scipy.stats as stats
from scipy.stats._hypotests import _cdf_cvm
from scipy.stats.stats import KstestResult
from . import ks_2samp_modified
import base64
import io

cdf=stats.beta.cdf
args=(2,3)


def plt_to_base64_encoded_image():
    io_bytes = io.BytesIO()
    plt.savefig(io_bytes, format="png")
    io_bytes.seek(0)
    plt.clf()
    return base64.b64encode(io_bytes.read())

def reorder(sample, weights=[]):
    #inputs are np arrays
    print ('hello')
    print ('samples: ', sample, 'weights: ', weights) 
    if len(weights)==0:
        weights=np.asarray([1]*len(sample))
    s_w=np.stack((sample, weights), axis=1)
    s_w=s_w[s_w[:, 0].argsort()]
    sample=s_w[:,0]
    weights=s_w[:,1]
    cond=weights>0
    return sample[cond], weights[cond]

def ecdf_cdf(sample, weights, cdf, args=()):
    #sample and weights are assumed to be ordered appropriately
    total=np.sum(weights)
    ecdfs=(np.cumsum(weights))/total
    cdfs=cdf(sample, *args)
    if True:
        return ecdfs, cdfs


def ecdf_x(x, sample, weights):
    #sample and weights are assuemd to be reorderd already
    total=np.sum(weights)
    ecdfs=np.cumsum(weights)/total
    if x<sample[0]:
        return 0
    if x>=sample[-1]:
        return 1
    else:
        ind=np.where(sample>x)[0][0]-1
        return ecdfs[ind]

def all_tests(sample, F, args=(), weights=[], tup=True):
    ksresult=kstest(sample, F.cdf, args, weights)
    chisqtest=chisquare(sample, F.cdf, args, bins=100, range_=None, weights=weights)
    cramer=cramer2(sample, F.cdf, args, weights)
    rsquared, plot2=R_sqaure(sample, F.pdf, weights, plot=True, bins=100)
    MSE_=MSE(sample, F, 10)
    totalresult= {'KS test': ksresult,
    'chi square test': chisqtest,
    'cramer von mises criteron': cramer,
    'R squared': rsquared,
    'Mean Squared Error': MSE_}
    if tup:
        return totalresult, plot2
    else:
        return totalresult
    


def kstest(sample, cdf, args=(), weights=[]):
    N=len(sample)
    if callable(cdf):
        sample, weights= reorder(sample, weights)
        ecdfs, cdfs = ecdf_cdf(sample, weights, cdf, args)
        d=np.abs(np.diff((ecdfs, cdfs), axis=0))
        D=np.max(d)
        p=stats.kstwo.sf(D,N)
        return KstestResult(D, p)
    else:
        exact=cdf
        D,p= ks_2samp_modified.ks_2samp(data1=sample, weights=weights, data2=exact)
        return KstestResult(D,p)
        
#print (kstest( [0,0,1,1,0,0,1], lambda x: stats.beta.cdf(x, 2,3)))


def chisquare(sample, cdf, args=(), bins=100, range_=None, weights=[]):
    N=len(sample)
    if len(weights)==0:
        weights=np.asarray([1]*N)
    if range_==None:
        range_=(np.min(sample), np.max(sample))
    t=np.histogram(sample, weights=weights, bins=bins, range=range_)
    sam=t[0]
    ran=t[1]
    expected=np.diff(np.vectorize((lambda x: cdf(x, *args)))(ran))*N
    #cisqstat=(expected-sam)**2/expected
    return (stats.chisquare(sam, expected))


#plots p values given a sampling algorithm and retunrs percentage
#of pvals greater than critical value (0.01)
def plot_p(sampler, cdf, args, sample_size=50, p_size=1000, test=kstest):
    pval=[]
    perc=0
    for i in range(p_size):
        sample=sampler(sample_size)
        if type(sample)==tuple:
            sample, weights =sample
        else:
            weights=[]
        p=test(sample, cdf, args, weights)[1]
        pval.append(p)
        if p>=0.01:
            perc+=1
    plt.hist(pval, bins=100, density=True)
    return (perc/p_size) *100

#example
#plot_p(generate_weighted1, cdf, (2,3), sample_size=50, p_size=1000, test=kstest)


#https://towardsdatascience.com/integrals-are-fun-illustrated-riemann-stieltjes-integral-b71a6e003072
def derivative(f, a, h=0.01):
    return (f(a + h) - f(a - h))/(2*h)
    #https://towardsdatascience.com/integrals-are-fun-illustrated-riemann-stieltjes-integral-b71a6e003072
def stieltjes_integral(f, g, a, b, n):
    eps = 1e-9
    h = (b - a)/(n + eps)  # width of the rectangle
    dg = lambda x: derivative(g, x, h=1e-8)  # derivative of the integrator function
    result = 0.5*f(a)*dg(a) + sum([f(a + i*h)*dg(a + i*h) for i in range(1, n)]) + 0.5*f(b)*dg(b)
    result *= h
    return result

#attempt to calculate cramer von miser test with RS integral
def cramer_rs(sample, cdf, args=(), weights=[]):
    N=len(sample)
    sample, weights = reorder(sample, weights)
    def F_n(x): #ecdf
        return ecdf_x(x, sample, weights)
    def F_0(x): #cdf
        return cdf(x, *args)
    def Ψ(x):
        return (F_n(x)-F_0(x))**2
    def ith_term(i):
        a=sample[i]
        b=sample[i+1]
        return stieltjes_integral(Ψ, F_0, a, b, 1000)
    C_n= ((1/3)*F_0(sample[0])**3)-(1/3)*(F_0(sample[-1])-1)**3+np.sum([ith_term(i) for i in range(0,N-1)])
    w=N*C_n
    p = max(0, 1. - _cdf_cvm(w, N))
    return w, p
 
def cramer2(sample, cdf, args=(), weights=[]):
    sample, weights= reorder(sample, weights)
    ecdfs, cdfs= ecdf_cdf(sample, weights, cdf, args)
    N=len(sample)
    firstterm=(1/3)*((cdfs[0])**3)
    lastterm=(1/3)*(cdfs[-1]-1)**3
    sigma=np.sum([(cdfs[i+1]-ecdfs[i])**3-(cdfs[i]-ecdfs[i])**3 for i in range(0, N-1)])
    C_n=firstterm+(1/3)*sigma+lastterm
    w=N*C_n
    p = max(0, 1. - _cdf_cvm(w, N))
    return w,p


def R_sqaure(sample, pdf, weights=[], bins=0, plot=False):
    N=len(sample)
    if len(weights)==0:
        weights=[1]*N
    if bins==0:
        bins=int(N/10)
    yval, xval=np.histogram(sample, weights=weights, bins=bins, density=True)
    xval=xval[1:]
    ȳ=np.mean(yval)
    SS_tot=0
    SS_res=0
    es=[]
    for i in range(bins):
        SS_tot+=(yval[i]-ȳ)**2
        e=pdf(xval[i])-yval[i]
        es.append(e)
        SS_res+=e**2
    R_sq=1-(SS_res/SS_tot)
    if plot==True: # plots residual graph
        plt.scatter(xval, es)
        plt.title('Residual graph for R square')
        plot = plt_to_base64_encoded_image()
        return R_sq, plot
    return R_sq

def nmoment(x, n):
    return np.sum((x)**n) / np.size(x)

f=stats.beta(a=2,b=3)

def MSE(sample, f, n):
    res=0
    for i in range(n):
        sample_moment=nmoment(sample, i)
        true_mom=f.moment(n=i)
        res+=(true_mom-sample_moment)**2
    return res/n

if __name__=='__main__':
    pass
    #print ('cramer scipy', stats.cramervonmises(sample, stats.beta.cdf, args=(2,3)))
    #print ('cramer estimated', cramer_rs(sample, stats.beta.cdf, args=(2,3)))
    #print ('cramer new ', cramer2(sample, stats.beta.cdf, (2,3)))

