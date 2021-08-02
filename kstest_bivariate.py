from kstest import ecdf_x_ndim
import scipy.stats as stats
import numpy as np

Σ=[[1,0],[0,1]]
mu=[0.0000001,0.0000001]
mu2=[100,100]
mu3=[0,0]
#test sample
k=stats.multivariate_normal.rvs(mean=mu3, size=100)
def kstest_bivariate(samples, cdf):
    #G_n is empirical cdf
    #G is cdf
    G_n=lambda x: ecdf_x_ndim(x,samples)
    G=cdf
    N,dim=samples.shape
    def D_plus(u):
        return G_n(u)-G(u)
    def D_minus(u):
        return G(u)-G_n(u)
    def intersectionpoints(u_i):
        n=len(u_i)
        res=[]
        for i in range(n):
            for j in range(n):
                if u_i[j][0]>u_i[i][0] and u_i[j][1]<u_i[j][0]:
                    res.append((u_i[j][0], u_i[j][0]))
        return res
    IntP=intersectionpoints(samples)
    y=samples[:,1]
    x=samples[:,0]
    D_1=np.max([D_plus(samples[i]) for i in range(N)])
    D_2=np.max([D_plus(x) for x in IntP])
    D_3=2/N-np.min([D_plus(x) for x in IntP])
    D_4=1/100-np.min([D_plus((1,y[i])) for i in range(len(y))])
    D_5=1/100-np.min([D_plus((x[i],1)) for i in range(len(x))])
    D_n=np.max([D_1,D_2,D_3,D_4,D_5])
    return D_n, stats.kstwo.sf(D_n, N)
print (kstest_bivariate(k, lambda x: stats.multivariate_normal.cdf(x)))