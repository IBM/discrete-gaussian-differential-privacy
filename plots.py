#This is the code used to create the plots for our paper
# https://arxiv.org/abs/2004.00010
#This code is not meant to be re-usable.
# - Thomas Steinke dgauss@thomas-steinke.net 2020

import math
import matplotlib.pyplot as plt

dpi=300
scal=0.4 #scale size of images. Both should have same height, but different width
        
#compute 1-cdf of discrete gaussian
#returns array ccdf such that ccdf[k] = P[X > k]
#for 0 <= k < maxk
def dg_ccdf(sigma2,maxk,iters=1000):
    assert maxk>=1
    ccdf=[0]*maxk
    sum=0 #sum will be the normalizing constant
    for k in range(iters+maxk,0,-1):
        term = math.exp(-k*k/(2.0*sigma2))
        sum = sum + term
    sum = 1 + 2*sum
    above=0
    for k in range(iters+maxk,-1,-1):
        term = math.exp(-k*k/(2.0*sigma2))
        if k<maxk:
            ccdf[k]=above/sum
        above = above + term
    return ccdf
    
#compute 1-cdf of continuous gaussian
#returns P[N(0,sigma2)>t]
def cg_ccdf(sigma2,t):
    #erf(x) = (2/sqrt(pi)) int_0^x exp(-y^2) dy
    #       = 2*P[N(0,1/2) in [0,x]]
    #       = 2*P[N(0,v) in [0,x*sqrt(2*v)]]
    #       = 2*P[N(0,v) in [-inf,x*sqrt(2*v)]]-1
    #       = 2*(1-P[N(0,v) in [x*sqrt(2*v),inf]])-1
    #       = 1-2*P[N(0,v) in [x*sqrt(2*v),inf]]
    #(1-erf(x))/2 = P[N(0,v) in [x*sqrt(2*v),inf]]
    #P[N(0,v) in [t,inf]] = (1-erf(t/sqrt(2*v))/2
    return math.erfc(t/math.sqrt(2*sigma2))/2.0
    
#plot 1-cdf
def plot_dg_cdf(sigma2,maxk=None,save=None):
    if maxk is None:
        maxk=0
        while maxk*maxk <= 9*sigma2:
            maxk=maxk+1
    
    #plot discrete
    ccdf=dg_ccdf(sigma2,maxk)
    xd=[0]*(2*maxk)
    yd=[0]*(2*maxk)
    for k in range(maxk):
        xd[2*k]=k
        yd[2*k]=ccdf[k]
        xd[2*k+1]=k+1
        yd[2*k+1]=ccdf[k]
    plt.plot(xd,yd,'b-',label="$X=\\mathcal{N}_{\\mathbb{Z}}(0,"+str(sigma2)+")$")
    
    #plot continuous
    xc=[i/100.0 for i in range(100*maxk+1)]
    yc=[cg_ccdf(sigma2,x) for x in xc]
    plt.plot(xc,yc,'r:',label="$X=\\mathcal{N}(0,"+str(sigma2)+")$")
    
    #plot rounded continuous
    xr=[0]*(2*maxk)
    yr=[0]*(2*maxk)
    for k in range(maxk):
        v = cg_ccdf(sigma2,k+0.5)
        assert v >= ccdf[k] #assert discrete gaussian stochastically dominates rounded gaussian.
        #v=P[round(N(0,sigma2))>k]=P[N(0,sigma2)>k+0.5]
        xr[2*k]=k
        yr[2*k]=v
        xr[2*k+1]=k+1
        yr[2*k+1]=v
    xr[0]=0.0
    plt.plot(xr,yr,'g--',label="$X=\\mathsf{round}(\\mathcal{N}(0,"+str(sigma2)+"))$")
    
    plt.xlim(0,maxk-0.5)
    plt.ylim(0,0.5)#min(0.5,max(yd[0],yr[0])+0.01))
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$\\mathbb{P}[X>x]$")
    plt.title("Comparison of Tail Probabilities")
    plt.gcf().set_size_inches(16*scal, 12*scal)

    if save is None:
        plt.show()
    else:
        plt.savefig(save,dpi=dpi)
    plt.clf()
        

# compute the variance of the discrete gaussian
def dg_var(sigma2,iters=10000):
    assert sigma2>0
    assert math.exp(-iters*iters/(2.0*sigma2)) < 1e-12 #if large we won't get accurate results
    denominator=0
    numerator=0
    #sum over x=1,2,...iters
    for i in range(iters):
        x=iters-i #sum from small to large for numerical reasons
        term = math.exp(-x*x/(2.0*sigma2))
        denominator = denominator + term
        numerator = numerator + term*x*x
    #now symmetrize and add the x=0 term
    denominator = 2.0*denominator+1.0
    numerator = 2.0*numerator
    return numerator/denominator

# compute the variance of the discrete laplace
#pmf(x) = exp(-|x|*eps)*(exp(eps)-1)/(exp(eps)+1)
#var = 2 * sum_{x=1}^infty x^2*exp(-x*eps) * (exp(eps)-1)/(exp(eps)+1)
#    = 2 * exp(eps) / (exp(eps)-1)^2
def dl_var(eps, iters=10000):
    assert eps>0

    return 2 * math.exp(eps) / ((math.expm1(eps))**2)

    assert math.exp(iters*eps) < 1e-12
    denominator=0
    numerator=0
    for i in range(iters):
        x=iters-i
        term = math.exp(eps*x)
        denominator = denominator + term
        numerator = numerator + term*x*x
    denominator = denominator*2+1
    numerator = numerator*2
    return numerator/denominator

#compute the variance of the rounded gaussian
def rg_var(sigma2,iters=1000):
    assert sigma2>0
    sum=0
    for x in range(iters,0,-1):
        #compute P[X=x] where X=round(N(0,sigma2))
        p=cg_ccdf(sigma2,x-0.5)-cg_ccdf(sigma2,x+0.5)
        sum=sum+p*x*x
    #double to symmetrize. Can ignore x=0.
    return 2*sum
    
# plot the variance as a function of sigma^2
def plot_dg_var(save=None):
    x=[math.exp(i/100.0-5.0) for i in range(1000)]
    xx=[math.exp(i/100.0-5.0) for i in range(600)] #don't plot full range to avoid overflow issues
    yd=[dg_var(sigma2) for sigma2 in x]
    yr=[rg_var(sigma2) for sigma2 in x]
    ylb=[1.0/(math.exp(1.0/sigma2)-1.0) for sigma2 in x]
    yub=[sigma2*(1.0-4.0*math.pi*math.pi*sigma2/(math.exp(4.0*math.pi*math.pi*sigma2)-1.0)) for sigma2 in xx]
    xxx=[math.exp(-i/100.0)/3.0 for i in range(500)] #don't plot full range to avoid overflow issues
    yub2=[3.0*math.exp(-0.5/sigma2) for sigma2 in xxx]
    plt.plot(x,yd,'b-',label="$X=\\mathcal{N}_{\\mathbb{Z}}(0,\\sigma^2)$")
    plt.plot(x,x,'r:',label="$X=\\mathcal{N}(0,\\sigma^2)$")
    plt.plot(x,yr,'g--',label="$X=\\mathsf{round}(\\mathcal{N}(0,\\sigma^2))$")
    #plt.plot(x,ylb,'y-.',label="lower bound")
    #plt.plot(xx,yub,'y-.',label="upper bound")
    #plt.plot(xxx,yub2,'y-.')
    plt.gca().set_aspect('equal')
    plt.xlim(math.exp(-5),math.exp(3))
    plt.xscale("log")
    plt.xlabel("$\\sigma^2$")
    plt.ylim(math.exp(-10),math.exp(3))
    plt.yscale("log")
    plt.ylabel("$\\mathsf{Var}[X]$")
    plt.title("Comparison of Variance")
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(9*scal, 12*scal)
    if save is None:
        plt.show()
    else:
        plt.savefig(save,dpi=dpi)
    plt.clf()
        
#compute the smallest delta such that adding discrete N(0,sigma^2) to
#sensitivity-1 attains (eps,delta)-DP
def dg_delta(sigma2,eps,iters=100000):
    norm_const = 0 #normalizing costant
    sum_pl = 0 #sum priv loss unnormalized
    sum_delta = 0 #sum max{0,1-exp(eps-priv_loss)}
    for x in range(-iters,iters+1):
        unnorm_prob = math.exp(-x*x/(2.0*sigma2))
        norm_const = norm_const + unnorm_prob
        priv_loss = (x+0.5)/sigma2 #=-x^2/2*sigma2+(x+1)^2/2*sigma2
        sum_pl = sum_pl + priv_loss*unnorm_prob
        if priv_loss>eps:
            sum_delta = sum_delta + unnorm_prob*(1-math.exp(eps-priv_loss))
    #normalize
    sum_pl = sum_pl/norm_const
    sum_delta = sum_delta/norm_const
    #print("sigma2="+str(sigma2)+"\tnorm_const="+str(norm_const)+"\tKL="+str(sum_pl))
    assert -1e-9 <= sum_pl-0.5/sigma2 <= 1e-9 #assert kl divergence is correct
    return sum_delta

#compute analytic upper bound on dg_delta
def dg_delta_ub(sigma2,eps):
    #delta <= P[N(0,sigma2)>floor(eps*sigma2-1/2)]-exp(eps)*(1-1/(1+sqrt(2*pi*sigma2)))*P[N(0,sigma2)>floor(eps*sigma2+1/2)]
    #       = p-(1-1/blah)*q
    #P[N(0,v) in [t,inf]] = (1-erf(t/sqrt(2*v))/2
    p=math.erfc(math.floor(eps*sigma2-0.5)/math.sqrt(2*sigma2))/2
    q=math.erfc(math.floor(eps*sigma2+0.5)/math.sqrt(2*sigma2))/2
    return p-math.exp(eps)*(1-1/(1+math.sqrt(2*math.pi*sigma2)))*q
    
#compute the smallest delta such that adding cts N(0,sigma^2) to
#sensitivity-1 attains (eps,delta)-DP
#https://arxiv.org/pdf/1805.06530.pdf
def cg_delta(sigma2,eps):    #erf(x) = (2/sqrt(pi)) int_0^x exp(-y^2) dy
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

#compute some delta such that
#(1/2*sigma2)-CDP implies (eps,delta)-DP
#This is the standard bound
def cdp_delta_standard(sigma2,eps):
    return math.exp(-(eps-0.5/sigma2)**2 * sigma2 / 2)
    
#compute the smallest delta such that
#(1/2*sigma2)-CDP implies (eps,delta)-DP
#Need to find sharp bound
def cdp_delta_analytic(sigma2,eps):
    #print("sigma2="+str(sigma2)+"\teps="+str(eps))
    assert sigma2>0
    assert eps>=0

    #our bound
    amin=1.001 #need a floor to avoid numerical instability
    amax=sigma2*(eps+1)+10
    for i in range(1000):
        alpha=(amin+amax)/2
        derivative = math.log1p(-1.0/alpha)-eps+(alpha-0.5)/sigma2
        if derivative<0:
            amin=alpha
        else:
            amax=alpha
    alpha=(amin+amax)/2
    return math.exp(alpha*math.log1p(-1/alpha)-(alpha-1)*eps+alpha*(alpha-1)*0.5/sigma2)/(alpha-1.0)

    
#compute the smallest delta such that
#(1/2*sigma2)-CDP implies (eps,delta)-DP
#Has major overflow problems
def cdp_delta_numeric(sigma2,eps,alphamax=22):
    #binary search, fails due to overflow :(
    deltamin=0.0 #maintain that (eps,deltamin)-DP is NOT implied
    deltamax=1.0 #maintain that (eps,deltamax)-DP is implied
    for i in range(50):
        delta=(deltamin+deltamax)/2
        if cdp_delta_viable(sigma2,eps,delta,alphamax=alphamax):
            deltamax=delta
        else:
            deltamin=delta
    return deltamax
    
def fdash(q,alpha,eps,delta):
    etoeps=math.exp(eps)
    val = alpha*etoeps*((etoeps+delta/q)**(alpha-1))
    val = val - (alpha-1)*((etoeps+delta/q)**alpha)
    val = val - alpha*etoeps*(((1-etoeps*q-delta)/(1-q))**(alpha-1))
    val = val + (alpha-1)*(((1-etoeps*q-delta)/(1-q))**alpha)
    return val
def worstq(alpha,eps,delta):
    qmin=0.0
    qmax=(1.0-delta)*math.exp(-eps) #qmax*exp(eps)+delta=1
    for i in range(50):
        q = (qmin+qmax)/2
        if fdash(q,alpha,eps,delta)<0:
            qmin=q
        else:
            qmax=q
    return (qmin+qmax)/2
#tell me if (1/2*sigma2)-CDP implies (eps,delta)-DP True/False
def cdp_delta_viable(sigma2,eps,delta,alphamax=22):
    alphas=[]
    for i in range(1,alphamax*10): alphas.append(1+i*0.1)
    #for i in range(1,50): alphas.append(1+i*0.1)
    for alpha in alphas: #try various alpha values to seek contradiction
        q = worstq(alpha,eps,delta)
        #val = ((math.exp(eps)*q+delta)**alpha) * (q**(1-alpha)) + ((1-math.exp(eps)*q-delta)**alpha) * ((1-q)**(1-alpha))
        val = ((math.exp(eps)+delta/q)**alpha) * q + ((1-((math.exp(eps)-1)*q+delta)/(1-q))**alpha) * (1-q)
        bound = math.exp(alpha*(alpha-1)*0.5/sigma2)
        if val>=bound:
            return True
    return False
    
#plot comparison of approx (eps,delta)-DP guarantees
def plot_epsdelta(sigma2, epsmax=1, pdelta=20, save=None,numeric=False,include_aub=False):
    epss=[i*0.01*epsmax for i in range(101)]
    delta_cg=[cg_delta(sigma2,eps) for eps in epss]
    delta_dg=[dg_delta(sigma2,eps) for eps in epss]
    delta_dg_ub=[dg_delta_ub(sigma2,eps) for eps in epss]
    delta_cdp_s=[cdp_delta_standard(sigma2,eps) for eps in epss] #standard 
    delta_cdp_a=[cdp_delta_analytic(sigma2,eps) for eps in epss] #good analytic
    #delta_cdp_a=[min(cdp_delta_analytic(sigma2,eps),cdp_delta_numeric(sigma2,eps,alphamax=10*int(math.sqrt(sigma2)))) for eps in epss] #min of numeric and analytic
    
    ratios=[delta_dg[i]/delta_cg[i] for i in range(len(epss))]
    ratios_ub=[delta_dg_ub[i]/delta_cg[i] for i in range(len(epss))]
    ratios_a=[delta_cdp_a[i]/delta_cg[i] for i in range(len(epss))]
    
    print(str(min(ratios)-1)+"<=delta_discrete/delta_cts-1<="+str(max(ratios)-1)+"\tsigma^2="+str(sigma2))
    print(str(min(ratios_a)-1)+"<=delta_cdp/delta_cts-1<="+str(max(ratios_a)-1)+"\tanalytic")

    if numeric:
        delta_cdp_n=[cdp_delta_numeric(sigma2,eps,alphamax=10*int(math.sqrt(sigma2))) for eps in epss]
        ratios_n=[delta_cdp_n[i]/delta_cg[i] for i in range(len(epss))]
        print(str(min(ratios_n)-1)+"<=delta_cdp/delta_cts-1<="+str(max(ratios_n)-1)+"\tnumeric")
        ratios_q=[delta_cdp_a[i]/delta_cdp_n[i] for i in range(len(epss))]
        print(str(min(ratios_q)-1)+"<=delta_cdp_analytic/delta_cdp_numeric-1<="+str(max(ratios_q)-1))
        plt.plot(epss,ratios_n,label="cdp/cts numeric")

    plt.plot(epss,ratios,label="discrete/cts")
    #plt.plot(epss,ratios_ub,label="ub/cts")
    #plt.plot(epss,ratios_a,label="cdp/cts analytic")
    plt.title("Comparison of aDP bounds")
    plt.xlabel("$\\varepsilon$")
    plt.ylabel("$\\delta$ ratio")
    #plt.yscale("log")
    plt.legend()
    #plt.ylim(math.sqrt(0.1) ,100)
    plt.savefig("ratios.png")
    plt.clf()
    
    plt.plot(epss,delta_dg,'b-',label="$\\mathcal{N}_{\\mathbb{Z}}(0,\\sigma^2)$ discrete")
    plt.plot(epss,delta_cg,'r:',label="$\\mathcal{N}(0,\\sigma^2)$ continuous")
    if include_aub:
        plt.plot(epss,delta_dg_ub,'g--',label="Analytic upper bound")
    plt.plot(epss,delta_cdp_s,'m-',label="$\\frac{1}{2\\sigma^2}$-CDP upper bound (standard)")
    plt.plot(epss,delta_cdp_a,'y-.',label="$\\frac{1}{2\\sigma^2}$-CDP upper bound (ours)")
    if numeric:
        plt.plot(epss,delta_cdp_n,'c:',label="$\\frac{1}{2\\sigma^2}$-CDP upper bound [ALCKS20]")
    #plt.plot(epss,delta_cdp2,'y-.')
    #plt.gca().set_aspect('equal')
    #plt.xlim(math.exp(-5),math.exp(3))
    #plt.xscale("log")
    plt.xlabel("$\\varepsilon$")
    plt.ylim(0.1**pdelta,1)
    plt.yscale("log")
    plt.ylabel("$\\delta$")
    plt.title("$(\\varepsilon,\\delta)$-Differential Privacy Guarantees ($\\sigma^2="+str(sigma2)+"$)")
    plt.legend(loc='lower left')
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().set_size_inches(12.5*scal, 12*scal)
    if save is None:
        plt.show()
    else:
        plt.savefig(save,dpi=dpi)
    plt.clf()

#optimal DP composition theorem
# k-fold composition of (eps,0)-DP
#Thm 1.4 in Murtagh-Vadhan https://arxiv.org/pdf/1507.03113.pdf
def optcomp(k,eps_i,eps_g):
    vals=[]
    binom=1
    for i in range(k+1):
        if eps_i*(2*i-k)>=eps_g:
            #sum = sum + binom*(math.exp(i*eps_i)-math.exp(eps_g+(k-i)*eps_i))
            #sum = sum + binom*(math.exp(i*eps_i-eps_g-(k-i)*eps_i))-1)*math.exp(eps_g+(k-i)*eps_i)
            vals.append(binom*(math.expm1((2*i-k)*eps_i-eps_g)*math.exp(eps_g+(k-i)*eps_i)))
        #binom = BinomialCoefficient(k,i) = k!/(i! * (k-i)!)
        #update to BinomialCoefficient(k,i+1) = k!/((i+1)! * (k-i-1)!)
        assert (binom*(k-i))%(i+1)==0
        binom = (binom*(k-i))//(i+1)
    
    denom = (1 + math.exp(eps_i))**k
    summ=0
    vals.sort()
    for v in vals: #sum in sorted order for improved numerical stability
        summ = summ+v
    return summ/denom

#plot Laplace/Gauss comparison
#fix k and per-coordinate variance
#plot corresponding eps,delta guarantees
def plot_ed_comp(k,sdev,top_eps=None,pdelta=None,save=None,include_cts=False):
    var=sdev*sdev
    #pick the correct sigma2 to get desired variance
    sigma2_max = var #maintain dg_var(sigma2_max)>=var
    sigma2_min = 0 #maintain dg_var(sigma2_min)<var
    for i in range(1000):
        sigma2 = (sigma2_max+sigma2_min)/2
        if dg_var(sigma2)<var:
            sigma2_min=sigma2
        else:
            sigma2_max=sigma2
    sigma2 = (sigma2_max+sigma2_min)/2

    #pick the correct eps_l to get the desired variance
    eps_min=0 #maintain dl_var(eps_min)>=var
    eps_max=1 #maintain dl_var(eps_max)<var
    while dl_var(eps_max)>var:
        eps_max=eps_max*2
    for i in range(1000):
        eps_l = (eps_min+eps_max)/2
        if dl_var(eps_l)<var:
            eps_max=eps_l
        else:
            eps_min=eps_l
    eps_l = (eps_min+eps_max)/2

    print("Laplace: ("+str(eps_l*k)+",0)-pDP")
    print("Laplace: (1,"+str(optcomp(k,eps_l,1))+")-aDP")
    print("Gaussian: (1,"+str(cdp_delta_analytic(sigma2/k,1))+")-aDP")
    print("Gaussian: ("+str(eps_l*k)+","+str(cdp_delta_analytic(sigma2/k,eps_l*k))+")-aDP")

    if top_eps is None:
        top_eps = math.ceil(eps_l*k)

    #how compute the eps,delta values
    epss = [i*0.01*top_eps for i in range(101)]
    delta_dg = [cdp_delta_analytic(sigma2/k,eps) for eps in epss]
    delta_dl = [optcomp(k,eps_l,eps) for eps in epss]
    if include_cts:
        #continuous laplace variance = 2/eps^2, so eps=sqrt(2/var)
        eps_cl = math.sqrt(2/var)
        delta_cl = [optcomp(k,eps_cl,eps) for eps in epss]
        delta_cg = [cdp_delta_analytic(var/k,eps) for eps in epss]

    #now plot
    plt.plot(epss,delta_dg,'b-',label="discrete Gaussian")
    plt.plot(epss,delta_dl,'m--',label="discrete Laplace")
    if include_cts:
        plt.plot(epss,delta_cg,'r:',label="continuous Gaussian")
        plt.plot(epss,delta_cl,'y-.',label="continuous Laplace")
    plt.xlabel("$\\varepsilon$")
    if pdelta is not None:
        plt.ylim(0.1**pdelta,1)
    plt.yscale("log")
    plt.ylabel("$\\delta$")
    plt.title("Privacy Comparison: $(\\varepsilon,\\delta)$-DP Curve for Answering\n$"+str(k)+"$ Queries, Each with Standard Deviation $"+str(sdev)+"$")
    #plt.title("$(\\varepsilon,\\delta)$-DP for Answering $"+str(k)+"$ Queries,\nEach with Standard Deviation $"+str(sdev)+"$")
    plt.legend(loc='lower left')
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().set_size_inches(12.5*scal, 12*scal)
    if save is None:
        plt.show()
    else:
        plt.savefig(save,dpi=dpi)
    plt.clf()

#plot Laplace/Gauss comparison
#fix eps,delta
#plot k vs per-coordinate variance
def plot_kvar_comp(eps,pdelta,kmax,save=None,include_cts=False):
    assert pdelta>0
    delta=0.1**pdelta
    
    ks = [k for k in range(1,kmax+1)]
    var_dl = []
    var_dg = []
    var_cl = []
    var_cg = []
    for k in ks:
        #discrete Laplace first
        eps_min = 0 #maintain optcomp(k,eps_min,eps)<=delta
        assert optcomp(k,eps_min,eps)<=delta
        eps_max = eps+1 #maintain optcomp(k,eps_max,eps)>=delta
        assert optcomp(k,eps_max,eps)>=delta
        for i in range(1000):
            eps_dl = (eps_min+eps_max)/2
            if optcomp(k,eps_dl,eps)>delta:
                eps_max=eps_dl
            else:
                eps_min=eps_dl
        eps_dl=(eps_min+eps_max)/2
        var_dl.append(dl_var(eps_dl))
        #continuous laplace
        var_cl.append(2.0/(eps_dl*eps_dl))
        #now discrete Gauss
        sigma2_min = 0 #maintain cdp_delta_analytic(sigma2_min/k,eps)>=delta
        sigma2_max = 1 #maintain cdp_delta_analytic(sigma2_max/k,eps)<delta
        while cdp_delta_analytic(sigma2_max/k,eps)>=delta:
            sigma2_max = sigma2_max * 2
        for i in range(1000):
            sigma2_dg = (sigma2_max+sigma2_min)/2
            if cdp_delta_analytic(sigma2_dg/k,eps)<delta:
                sigma2_max = sigma2_dg
            else:
                sigma2_min = sigma2_dg
        sigma2_dg = (sigma2_min + sigma2_max)/2
        var_dg.append(dg_var(sigma2_dg))
        #continuous gaussian
        var_cg.append(sigma2_dg)
            

        #print("k="+str(k)+":\tratio="+str(var_dl[-1]/var_dg[-1]))

    #now plot
    plt.plot(ks,var_dg,'b-',label="discrete Gaussian")# $("+str(eps)+",10^{-"+str(pdelta)+"})$-DP")
    plt.plot(ks,var_dl,'m--',label="discrete Laplace")# $("+str(eps)+",10^{-"+str(pdelta)+"})$-DP")
    if include_cts:
        plt.plot(ks,var_cg,'r:',label="continuous Gaussian")
        plt.plot(ks,var_cl,'y-.',label="continuous Laplace")
    plt.xlabel("Number of queries")
    #if pdelta is not None:
    #    plt.ylim(0.1**pdelta,1)
    #plt.yscale("log")
    plt.ylabel("Noise Variance (per query)")
    plt.title("Utility Comparison Under $("+str(eps)+",10^{-"+str(pdelta)+"})$-DP")
    plt.legend(loc='best')
    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().set_size_inches(12.5*scal, 12*scal)
    if save is None:
        plt.show()
    else:
        plt.savefig(save,dpi=dpi)
    plt.clf()

#delta=10^-pdelta
plot_ed_comp(100,50,top_eps=3,pdelta=50,save="gausslaplace_epsdelta.png",include_cts=False)
plot_kvar_comp(1,6,100,save="gausslaplace_kvar.png",include_cts=False)
plot_epsdelta(100, epsmax=1, pdelta=20, save="dg_epsdelta.png",numeric=False,include_aub=True)
plot_epsdelta(4, epsmax=5, pdelta=20, save="dg_epsdelta2.png",numeric=False,include_aub=True)
plot_dg_cdf(2,save="dg_tail.png")
plot_dg_var(save="dg_var.png")

