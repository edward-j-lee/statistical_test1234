import numpy as np
import scipy.stats as stats
#https://github.com/ritvikmath/YouTubeVideoCode/blob/main/MCMC%20Experiments.ipynb
def MCMC_sampling(f, size=10000, burn_in=10000):
    samples=[1]
    num_accepted=0
    g=lambda x: (stats.unioform.pdf(x, loc=samples[-1]-1, scale=2))
    for i in range(size+burn_in):
        candidate=np.random.uniform(low=samples[-1]-1, high=samples[-1]+1)
        prob=min(1, f(candidate)/f(samples[-1]))
        if np.random.random()<prob:
            samples.append(candidate)
            num_accepted+=1
        else:
            samples.append(samples[-1])
    #print ('percentage accepted: ', (num_accepted/(size+burn_in))*100)
    return np.array(samples[burn_in+1:size+burn_in])

def MCMC_sampling_inf(f,gen_noise=stats.norm.rvs, size=10000, burn_in=10000):
    samples=[1]
    num_accepted=0
    g=lambda x: (stats.unioform.pdf(x, loc=samples[-1]-1, scale=2))
    for i in range(size+burn_in):
        noise=gen_noise()
        candidate=noise+samples[-1]
        prob=min(1, f(candidate)/f(samples[-1]))
        if np.random.random()<prob:
            samples.append(candidate)
            num_accepted+=1
        else:
            samples.append(samples[-1])
    #print ('percentage accepted: ', (num_accepted/(size+burn_in))*100)
    return np.array(samples[burn_in+1:size+burn_in])

target_dist=lambda x: stats.beta.pdf(x,2,3)
def generate_MCMC(N, f=target_dist):
    return MCMC_sampling(f, size=N)


def MCMC_sampling2(f, size=10000, burn_in=10000):
    samples=[1]
    num_accepted=0
    g=lambda x: (stats.normal.pdf(x, loc=samples[-1], scale=1))
    for i in range(size+burn_in):
        candidate=np.random.normal(loc=samples[-1], scale=1)
        prob=min(1, f(candidate)/f(samples[-1]))
        if np.random.random()<prob:
            samples.append(candidate)
            num_accepted+=1
        else:
            samples.append(samples[-1])
    #print ('percentage accepted: ', (num_accepted/(size+burn_in))*100)
    return np.array(samples[burn_in+1:size+burn_in])

target_dist=lambda x: stats.beta.pdf(x,2,3)
def generate_MCMC(N, f=target_dist):
    return MCMC_sampling(f, size=N)

MCMC_10k=generate_MCMC(10000)