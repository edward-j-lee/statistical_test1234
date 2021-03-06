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
"""
def _cdf_cvm(x, n=None):
    
    #Calculate the cdf of the Cramér-von Mises statistic for a finite sample
    #size n. If N is None, use the asymptotic cdf (n=inf)

    #See equation 1.8 in Csorgo, S. and Faraway, J. (1996) for finite samples,
    #1.2 for the asymptotic cdf.

    #The function is not expected to be accurate for large values of x, say
    #x > 2, when the cdf is very close to 1 and it might return values > 1
    #in that case, e.g. _cdf_cvm(2.0, 12) = 1.0000027556716846.
    
    x = np.asarray(x)
    if n is None:
        y = _cdf_cvm_inf(x)
    else:
        # support of the test statistic is [12/n, n/3], see 1.1 in [2]
        y = np.zeros_like(x, dtype='float')
        sup = (1./(12*n) < x) & (x < n/3.)
        # note: _psi1_mod does not include the term _cdf_cvm_inf(x) / 12
        # therefore, we need to add it here
        y[sup] = _cdf_cvm_inf(x[sup]) * (1 + 1./(12*n)) + _psi1_mod(x[sup]) / n
        y[x >= n/3] = 1

    if y.ndim == 0:
        return y[()]
    return y

"""

#converts plots to base 64 to display on html page
def plt_to_base64_encoded_image():
    io_bytes = io.BytesIO()
    plt.savefig(io_bytes, format="png")
    io_bytes.seek(0)
    plt.clf()
    return base64.b64encode(io_bytes.read())

#reorders sample and weights (and generates uniform weights if weights is empty)
#filters samples with empty or neglibgible weights
def reorder(sample, weights=[]):
    #inputs are np arrays
    if len(weights)==0:
        weights=np.ones(len(sample))
    s_w=np.stack((sample, weights), axis=1)
    s_w=s_w[s_w[:, 0].argsort()]
    sample=s_w[:,0]
    weights=s_w[:,1]
    cond=weights>0  
    return sample[cond], weights[cond]

#returns list of ecdfs and list of cdfs
def ecdf_cdf(sample, weights, cdf, args=()):
    #sample and weights are assumed to be ordered appropriately (ie already passed through 'reorder' function)
    total=np.sum(weights)
    ecdfs=(np.cumsum(weights))/total
    cdfs=cdf(sample, *args)
    if True:
        return ecdfs, cdfs

#calculates an ecdf for a given sample and wegiths
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

#peforms all statitiscal tests on given sample and weights
# if tup is true it returns the result dictionary and plot in a tuple
def all_tests(sample, F, args=(), weights=[], tup=True):
    sample, weights= reorder(sample, weights)
    ksresult=kstest(sample, F.cdf, args, weights)
    chisqtest=chisquare(sample, F.cdf, args, bins=100, range_=None, weights=weights)
    cramer=cramer2(sample, F.cdf, args, weights)
    rsquared, plot2=R_square(sample, F.pdf, weights, plot=tup, bins=100)
    MSE_=MSE(sample, weights, F.pdf)
    totalresult= {'KS test': ksresult,
    'chi square test': chisqtest,
    'cramer von mises criteron': cramer,
    'R squared': rsquared,
    'Mean Squared Error': MSE_}
    if tup:
        return totalresult, plot2
    else:
        return totalresult
    

#performs ks test to a given sample and weights
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

#performs chisquaretest 
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
def plot_p_from_sampling_algo(sampler, cdf, args, sample_size=50, p_size=1000, test=kstest):
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
 
 #cramer for weighted case using direct formula (see report)
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

#R squared  
def R_square(sample, pdf, weights=[], bins=100, plot=False):
    if len(weights)==0:
        weights=np.ones(len(sample))
    yval, xval=np.histogram(sample, weights=weights, bins=bins, density=True)
    xval=(xval[1:]+xval[:-1])/2
    y_line=np.vectorize(pdf)(xval)
    var_mean=np.sum(np.square(np.subtract(yval, np.mean(yval))))
    obs_error=np.subtract(yval, y_line)
    var_line=np.sum(np.square(obs_error))
    if plot==True:
        plt.scatter(xval, obs_error)
        plt.title('residual plot')
        residual_plot=plt_to_base64_encoded_image()
    else:
        residual_plot=None
    return (var_mean-var_line)/var_mean, residual_plot


#mean squared error
def MSE(sample, weights, f_pdf):
    N=int(len(sample)/10)
    vals= np.histogram(sample, weights=weights, bins=N, density=True)
    xs  = vals[1][1:]
    ys  = vals[0]
    return np.mean([(ys[i]-f_pdf(xs[i]))**2 for i in range(len(ys))])

if __name__=="__main__":
    while True:
        continue