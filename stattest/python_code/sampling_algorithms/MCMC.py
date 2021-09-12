import numpy as np
import scipy.stats as stats

def MCMC_sampling_inf(target, size, gen_noise=stats.norm.rvs, burnin=10000, initial=1):
    sample=[initial]
    for i in range(size+burnin):
        noise=gen_noise()
        last=sample[-1]
        candidate=last+noise
        test=target(candidate)/target(last)
        if test>1:
            sample.append(candidate)
        elif test>stats.uniform.rvs():
            sample.append(candidate)
        else:
            sample.append(last)
    return np.asarray(sample[burnin:])

target_dist=lambda x: stats.beta.pdf(x,2,3)
def generate_MCMC(N, f=target_dist):
    return MCMC_sampling_inf(f, size=N)

def generate_MCMC(N, f=target_dist):
    return MCMC_sampling_inf(f, size=N)

MCMC_10k=generate_MCMC(10000)