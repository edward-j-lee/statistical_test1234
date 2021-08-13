import numpy as np
from math import gcd
import warnings
from scipy.stats import distributions
from collections import namedtuple

from scipy.stats.stats import _attempt_exact_2kssamp, _compute_prob_outside_square, _compute_prob_inside_method, _count_paths_outside_method
#import scipy.speical as special

KstestResult = namedtuple('KstestResult', ('statistic', 'pvalue'))
def reorder(sample, weights=[]):
    #inputs are np arrays 
    if len(weights)==0:
        weights=np.asarray([1]*len(sample))
    s_w=np.stack((sample, weights), axis=1)
    s_w=s_w[s_w[:, 0].argsort()]
    sample=s_w[:,0]
    weights=s_w[:,1]
    cond=weights>0
    return sample[cond], weights[cond]


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

def ks_2samp(data1, weights, data2, alternative='two-sided', mode='auto'):
    if mode not in ['auto', 'exact', 'asymp']:
        raise ValueError(f'Invalid value for mode: {mode}')
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
       alternative.lower()[0], alternative)
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError(f'Invalid value for alternative: {alternative}')
    MAX_AUTO_N = 10000  # 'auto' will attempt to be exact if n1,n2 <= MAX_AUTO_N
    if np.ma.is_masked(data1):
        data1 = data1.compressed()
    if np.ma.is_masked(data2):
        data2 = data2.compressed()
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    if min(n1, n2) == 0:
        raise ValueError('Data passed to ks_2samp must not be empty')

    data_all = np.concatenate([data1, data2])
    # using searchsorted solves equal data problem
    
    #<<<<<<<<<<<<<<<<<<<<<<<< modified from here
    """
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    cddiffs = cdf1 - cdf2
    """
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<until here
    
    ecdf1=lambda x: ecdf_x(x, *reorder(data1, weights))
    ecdf2=lambda x: ecdf_x(x, *reorder(data2))
    cddiffs=[ecdf1(x)-ecdf2(x) for x in data1]
    
    minS = np.clip(-np.min(cddiffs), 0, 1)  # Ensure sign of minS is not negative.
    maxS = np.max(cddiffs)
    alt2Dvalue = {'less': minS, 'greater': maxS, 'two-sided': max(minS, maxS)}
    d = alt2Dvalue[alternative]

    

    
    g = gcd(n1, n2)
    n1g = n1 // g
    n2g = n2 // g
    prob = -np.inf
    original_mode = mode
    if mode == 'auto':
        mode = 'exact' if max(n1, n2) <= MAX_AUTO_N else 'asymp'
    elif mode == 'exact':
        # If lcm(n1, n2) is too big, switch from exact to asymp
        if n1g >= np.iinfo(np.int_).max / n2g:
            mode = 'asymp'
            warnings.warn(
                f"Exact ks_2samp calculation not possible with samples sizes "
                f"{n1} and {n2}. Switching to 'asymp'.", RuntimeWarning)

    if mode == 'exact':
        success, d, prob = _attempt_exact_2kssamp(n1, n2, g, d, alternative)
        if not success:
            mode = 'asymp'
            if original_mode == 'exact':
                warnings.warn(f"ks_2samp: Exact calculation unsuccessful. "
                              f"Switching to mode={mode}.", RuntimeWarning)

    if mode == 'asymp':
        # The product n1*n2 is large.  Use Smirnov's asymptoptic formula.
        # Ensure float to avoid overflow in multiplication
        # sorted because the one-sided formula is not symmetric in n1, n2
        m, n = sorted([float(n1), float(n2)], reverse=True)
        en = m * n / (m + n)
        if alternative == 'two-sided':
            prob = distributions.kstwo.sf(d, np.round(en))
        else:
            z = np.sqrt(en) * d
            # Use Hodges' suggested approximation Eqn 5.3
            # Requires m to be the larger of (n1, n2)
            expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
            prob = np.exp(expt)

    prob = np.clip(prob, 0, 1)
    return KstestResult(d, prob)