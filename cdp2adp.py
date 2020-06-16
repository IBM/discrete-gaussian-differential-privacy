#Code for computing approximate differential privacy guarantees
# for discrete Gaussian and, more generally, concentrated DP
# See https://arxiv.org/abs/2004.00010
# - Thomas Steinke dgauss@thomas-steinke.net 2020

import math
import matplotlib.pyplot as plt

import discretegauss as dg

# Begin with some code to compute aDP guarantees for discrete and continuous Gaussians

#compute the smallest delta such that adding discrete N_Z(0,sigma^2) to
#sensitivity-1 attains (eps,delta)-DP
# https://arxiv.org/pdf/2004.00010v3.pdf#page=7 Theorem 7
def dg_delta(sigma2,eps,sens=1,iters=None):
    assert sigma2>0
    assert eps>=0
    assert sens>0
    lower_limit=int(math.floor(eps*sigma2/sens-sens/2))+1
    upper_limit=int(math.floor(eps*sigma2/sens+sens/2))+1
    #If X~discreteGaussian(sigma2), then
    #delta = P[X >= lower_limit] - exp(eps) * P[X >= upper_limit]
    #      = P[lower_limit <= X < upper_limit] - (exp(eps)-1) * P[X >= upper_limit]
    norm_const = dg.normalizing_constant(sigma2) #normalizing constant of discrete gaussian
    expepsm1 = math.expm1(eps)
    #big question: how far to run the infinite series
    #any finite truncation still provides a valid upper bound on delta
    # so don't need to be too paranoid about this being large
    if iters is None: #insert default value
        iters = 1000 + dg.floorsqrt(200*sigma2)
    sum_delta = 0
    for x in range(lower_limit,upper_limit):
        sum_delta = sum_delta + math.exp(-x*x/(2.0*sigma2)) /  norm_const
    for x in range(upper_limit,upper_limit+iters):
        sum_delta = sum_delta - expepsm1*math.exp(-x*x/(2.0*sigma2)) /  norm_const
    return sum_delta
    
#compute the smallest delta such that adding cts N(0,sigma^2) to
#sensitivity-1 attains (eps,delta)-DP
#https://arxiv.org/pdf/1805.06530.pdf
def cg_delta(sigma2,eps):
    #erf(x) = (2/sqrt(pi)) int_0^x exp(-y^2) dy
    #       = 2*P[N(0,1/2) in [0,x]]
    #       = 2*P[N(0,v) in [0,x*sqrt(2*v)]]
    #       = 2*P[N(0,v) in [-inf,x*sqrt(2*v)]]-1
    #       = 2*(1-P[N(0,v) in [x*sqrt(2*v),inf]])-1
    #       = 1-2*P[N(0,v) in [x*sqrt(2*v),inf]]
    #(1-erf(x))/2 = P[N(0,v) in [x*sqrt(2*v),inf]]
    #P[N(0,v) in [t,inf]] = (1-erf(t/sqrt(2*v))/2
    #
    #delta = P[N(0,1)>eps*sigma-1/2*sigma]-e^eps*P[N(0,1)>eps*sigma+1/2*sigma]
    #      = (1-erf((eps*sigma-0.5/sigma)/sqrt(2))/2-e^eps*(1-erf((eps*sigma+0.5/sigma)/sqrt(2)))/2
    sigma = math.sqrt(sigma2)
    a=(eps*sigma-0.5/sigma)/math.sqrt(2)
    b=(eps*sigma+0.5/sigma)/math.sqrt(2)
    return math.erfc(a)/2-math.exp(eps)*math.erfc(b)/2

#*********************************************************************
#Now we move on to concentrated DP
    
#compute delta such that
#rho-CDP implies (eps,delta)-DP
#Note that adding cts or discrete N(0,sigma2) to sens-1 gives rho=1/(2*sigma2)

#start with standard P[privloss>eps] bound via markov
def cdp_delta_standard(rho,eps):
    assert rho>=0
    assert eps>=0
    if rho==0: return 0 #degenerate case
    #https://arxiv.org/pdf/1605.02065.pdf#page=15
    return math.exp(-((eps-rho)**2)/(4*rho))

#Our new bound:
# https://arxiv.org/pdf/2004.00010v3.pdf#page=13
def cdp_delta(rho,eps):
    assert rho>=0
    assert eps>=0
    if rho==0: return 0 #degenerate case

    #search for best alpha
    #Note that any alpha in (1,infty) yields a valid upper bound on delta
    # Thus if this search is slightly "incorrect" it will only result in larger delta (still valid)
    # This code has two "hacks".
    # First the binary search is run for a pre-specificed length.
    # 1000 iterations should be sufficient to converge to a good solution.
    # Second we set a minimum value of alpha to avoid numerical stability issues.
    # Note that the optimal alpha is at least (1+eps/rho)/2. Thus we only hit this constraint
    # when eps<=rho or close to it. This is not an interesting parameter regime, as you will
    # inherently get large delta in this regime.
    amin=1.01 #don't let alpha be too small, due to numerical stability
    amax=(eps+1)/(2*rho)+2
    for i in range(1000): #should be enough iterations
        alpha=(amin+amax)/2
        derivative = (2*alpha-1)*rho-eps+math.log1p(-1.0/alpha)
        if derivative<0:
            amin=alpha
        else:
            amax=alpha
    #now calculate delta
    delta = math.exp((alpha-1)*(alpha*rho-eps)+alpha*math.log1p(-1/alpha)) / (alpha-1.0)
    return min(delta,1.0) #delta<=1 always

#Above we compute delta given rho and eps, now we compute eps instead
#That is we wish to compute the smallest eps such that rho-CDP implies (eps,delta)-DP
def cdp_eps(rho,delta):
    assert rho>=0
    assert delta>0
    if delta>=1 or rho==0: return 0.0 #if delta>=1 or rho=0 then anything goes
    epsmin=0.0 #maintain cdp_delta(rho,eps)>=delta
    epsmax=rho+2*math.sqrt(rho*math.log(1/delta)) #maintain cdp_delta(rho,eps)<=delta
    #to compute epsmax we use the standard bound
    for i in range(1000):
        eps=(epsmin+epsmax)/2
        if cdp_delta(rho,eps)<=delta:
            epsmax=eps
        else:
            epsmin=eps
    return epsmax

#Now we compute rho
#Given (eps,delta) find the smallest rho such that rho-CDP implies (eps,delta)-DP
def cdp_rho(eps,delta):
    assert eps>=0
    assert delta>0
    if delta>=1: return 0.0 #if delta>=1 anything goes
    rhomin=0.0 #maintain cdp_delta(rho,eps)<=delta
    rhomax=eps+1 #maintain cdp_delta(rhomax,eps)>delta
    for i in range(1000):
        rho=(rhomin+rhomax)/2
        if cdp_delta(rho,eps)<=delta:
            rhomin=rho
        else:
            rhomax=rho
    return rhomin

#plot comparison of approx (eps,delta)-DP guarantees implied by rho-CDP or adding N(0,1/2*rho)
#delta in [10^-pdelta,1] is plotted
def plot_epsdelta(rho, epsmax=None, pdelta=None, save=None):
    if pdelta is None:
        pdelta=20
    if epsmax is None:
        epsmax=cdp_eps(rho,0.1**pdelta)
        
    sigma2=1/(2*rho)
    epss=[i*0.01*epsmax for i in range(101)]
    
    delta_cg=[cg_delta(sigma2,eps) for eps in epss]
    delta_dg=[dg_delta(sigma2,eps) for eps in epss]
    
    delta_cdp_standard=[cdp_delta_standard(rho,eps) for eps in epss]  
    delta_cdp_func=[cdp_delta(rho,eps) for eps in epss] 
    
    plt.plot(epss,delta_dg,'b-',label="$\\mathcal{N}_{\\mathbb{Z}}(0,1/2\\rho)$ discrete")
    plt.plot(epss,delta_cg,'r:',label="$\\mathcal{N}(0,1/2\\rho)$ continuous")
    plt.plot(epss,delta_cdp_standard,'y--',label="$\\rho$-CDP standard bound")
    plt.plot(epss,delta_cdp_func,'g-.',label="$\\rho$-CDP upper bound")

    
    plt.xlabel("$\\varepsilon$")
    plt.ylim(0.1**pdelta,1)
    plt.yscale("log")
    plt.ylabel("$\\delta$")
    plt.title("$(\\varepsilon,\\delta)$-Differential Privacy Guarantees for $(\\rho="+str(rho)+")$-CDP")
    plt.legend(loc='lower left')
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
    plt.clf()

if __name__=='__main__':
    print("cdp_delta(rho=0.025,eps=1)="+str(cdp_delta(0.025,1)))
    print("cdp_eps(rho=0.025,delta=10^-6)="+str(cdp_eps(0.025,1e-6)))
    print("cdp_rho(eps=1,delta=10^-6)="+str(cdp_rho(1,1e-6)))
    #plot_epsdelta(0.1, epsmax=10, pdelta=20, save="cdp_epsdelta.png")
    #plot_epsdelta(0.05, epsmax=5, pdelta=20, save="cdp_epsdelta.png")
    #plot_epsdelta(0.001, epsmax=1, pdelta=50, save="cdp_epsdelta.png")
    plot_epsdelta(0.1)
