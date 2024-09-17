from wolframclient.evaluation import WolframLanguageSession
from tqdm import tqdm
from wolframclient.language import wl,wlexpr
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import sys
import sympy as sympy

def carnahan_starling(eta, d, beta):
    #eta=4*np.pi*rho*(1/3)*d**3
    rho=3*eta/(3*np.pi*d**3)
    return rho*(1+eta+eta**2-eta**3)/(beta*(1-eta)**3)

def boublik_mansoori_CS(etaTot,molar_fraction,size_ratio):
    xi=lambda m:np.pi*etaTot*np.sum(molar_fraction*(2*size_ratio**m))/6
    return (6/np.pi)*((xi(0))/(1-xi(3))+(3*xi(1)*xi(2))/(1-xi(3))**2+((3-xi(3)*xi(2)**3))/(1-xi(3))**3)



def ideal_gas(eta,d,beta):

    rho=3*eta/(3*np.pi*d**3)
    return rho/beta

def overlap_volume(D,d,r):
    return (np.pi/6)*(2*D+2*d+r)*(D+d-r)**2

def monodisperse_correlation_function(r,sigma,phi0):
 
    rstar=(2.0116-1.0647*phi0+0.0538*phi0**2)/(sigma)
   
    gsigma=(((1+phi0+phi0**2-(2/3)*phi0**3-(2/3)*phi0**4)/(1-phi0)**3)-1)/(4*phi0)
    gm=1.0286-0.6095*phi0+3.5781*phi0**2-21.3651*phi0**3+42.6344*phi0**4-33.8485*phi0**5
    alpha=(44.554+79.868*phi0+116.432*phi0**2-44.652*np.exp(2*phi0))/(sigma)
    beta=(-5.022+5.857*phi0+5.089*np.exp(-4*phi0))/(sigma)
    
    d=np.power((2*phi0*(phi0**2-3*phi0-3+np.sqrt(3*(phi0**4-2*phi0**3+phi0**2+6*phi0+3)))),1/3)
    
    alpha0=(2*phi0/(1-phi0))*(-1+d/(4*phi0)-phi0/(2*d))/sigma
    beta0=(2*phi0/(1-phi0))*np.sqrt(3)*(-d/(4*phi0)-phi0/(2*d))/sigma
    mu=(2*phi0/(1-phi0))*(-1-d/(2*phi0)+phi0/d)/sigma
    
    k=(4.674*np.exp(-3.935*phi0)+3.536*np.exp(-56.270*phi0))/sigma
    omega=(-0.682*np.exp(-24.696*phi0)+4.720+4.4450*phi0)/sigma
    
    gamma=np.arctan(-(sigma*(alpha0*(alpha0**2+beta0**2)-mu*(alpha0**2-beta0**2))*(1+0.5*phi0)+(alpha0**2+beta0**2-mu*alpha0)*(1+2*phi0))/(beta0*(sigma*(alpha0**2+beta0**2-mu*alpha0)*(1+0.5*phi0)-mu*(1+2*phi0))))
    
    delta=-omega*rstar-np.arctan((k*rstar+1)/(omega*rstar))
    
    C=(rstar*(gm-1)*np.exp(k*rstar))/(np.cos(omega*rstar+delta))
    B=rstar*(gm-(sigma*gsigma/rstar)*np.exp(mu*(rstar-sigma)))/(np.cos(beta*(rstar-sigma)+gamma)*np.exp(alpha*(rstar-sigma))-np.cos(gamma)*np.exp(mu*(rstar-sigma)))
    A=sigma*gsigma-B*np.cos(gamma)
    
    if r<sigma:
        return 0
    if r<rstar:
        return A*np.exp(mu*(r-sigma))/r+B*np.cos(beta*(r-sigma)+gamma)*np.exp(alpha*(r-sigma))/r
    else:
        return 1+C*np.cos(omega*r+delta)*np.exp(-k*r)/r
   
def laplace_binary_correlation_function(s,r1,r2,eta1,eta2,ij):
    r12=(r1+r2)/2
    xi=eta1*r1**3+eta2*r2**3
    h=36*eta1*eta2*(r2-r1)**2
    l1=12*eta2*((1+0.5*xi)+1.5*eta1*(r2-r1)*r1**2)*r2*s**2+(12*eta2*(1+2*xi)-h*r1)*s+h
    l2=12*eta1*((1+0.5*xi)+1.5*eta2*(r1-r2)*r2**2)*r1*s**2+(12*eta1*(1+2*xi)-h*r2)*s+h
    sf=h+(12*(eta1*eta2)*(1+2*xi)-h*(r1+r2))*s-18*(eta1*r1**2+eta2*r2**2)**2*s**2-6*(eta1*r1**2+eta2*r2**2)*(1-xi)*s**3-((1-xi)**2)*s**4
    d=h-l1*np.exp(s*r1)-l2*np.exp(s*r2)+sf*np.exp(s*(r1+r2))
    if ij==1:
        n=np.sqrt(eta1*eta2)*s**2*np.exp(s*r12)*((0.75*(eta2*r2**3-eta1*r1**3)*(r2-r1)-r12*(1+0.5*xi))*s-1-2*xi)
    else:
        n=s*(h-l2*np.exp(s*r2))
    return n/d

def laplace_correlation_function(s,r1,eta1):
    
    r2=r1
    eta2=0
    h=36*eta1*eta2*(r2-r1)**2
    r12=(r1+r2)/2
    xi=eta1*r1**3+eta2*r2**3

    l1=12*eta2*((1+0.5*xi)+1.5*eta1*(r2-r1)*r1**2)*r2*s**2+(12*eta2*(1+2*xi)-h*r1)*s+h
    l2=12*eta1*((1+0.5*xi)+1.5*eta2*(r1-r2)*r2**2)*r1*s**2+(12*eta1*(1+2*xi)-h*r2)*s+h
    sf=h+(12*(eta1+eta2)*(1+2*xi)-h*(r1+r2))*s-18*(eta1*r1**2+eta2*r2**2)**2*s**2-6*(eta1*r1**2+eta2*r2**2)*(1-xi)*s**3-((1-xi)**2)*s**4

    d=h-l1*np.exp(s*r1)-l2*np.exp(s*r2)+sf*np.exp(s*(r1+r2))
    n11=s*(h-l2*np.exp(s*r2))
    return n11/d

def direct_correlation_function(r,eta1,eta2,r1,r2,ij=True):
    xi=eta1*r1**3+eta2*r2**3
    r12=(r1+r2)/2
    
    g11=((1+0.5*xi)+1.5*eta2*(r1-r2)*r2**2)/(1-xi)**2
    g22=((1+0.5*xi)+1.5*eta1*(r2-r1)*r1**2)/(1-xi)**2
    
    g12=(r2*g11+r1*g22)/(2*r12)
    
    a1=(1+xi+xi**2)/(1-xi)**3
    a2=(1+xi+xi**2)/(1-xi)**3
    
    b1=-6*(eta1*r1**2*g11**2+eta2*r12**2*g12**2)
    b2=-6*(eta2*r2**2*g22**2+eta1*r12**2*g12**2)
    
    b=-6*(eta1*r1*g11+eta2*r2*g22)*r12*g12
    
    d=0.5*(eta1*a1+eta2*a2)
    
    if ij:
        l=0
        x=r-l
        toReturn=a1+(b*x**2+4*l*x**3+d*x**4)/r
    else:
        toReturn=(a1+b1*r+d*r**3)
    
    return toReturn

def mComponentDirectCorrelationFunction(r,eta,R,i,j):
    N=len(r)
    xi=eta1*r1**3+eta2*r2**3*np.sum(eta*r**3)
    Rij=np.zeros(N,N)
    gij=np.zeros(N,N)
    for i in range(N):
        
        for j in rangije(i+1,N):
            Rij[i,j]=(R[i]+R[j])/2
    for i in range(N):
        gii[i,i]=((1+0.5*xi)+1.5*np.sum([eta[j]*(R[i]-R[j])*R[j]**2 for j in range(N) if j != i]))/(1-xi)**2
    for i in range(N):
        for j in range(i+1,N):
            g[i,j]=(R[i]*g[j,j]+R[j]*g[i,i])/(2*Rij[i,j])
    
    
    a1=(1+xi+xi**2)/(1-xi)**3
    a2=(1+xi+xi**2)/(1-xi)**3
    
    b1=-6*(eta1*r1**2*g11**2+eta2*r12**2*g12**2)
    b2=-6*(eta2*r2**2*g22**2+eta1*r12**2*g12**2)
    
    b=-6*(eta1*r1*g11+eta2*r2*g22)*r12*g12
    
    d=0.5*(eta1*a1+eta2*a2)
    
    if ij:
        l=0
        x=r-l
        toReturn=a1+(b*x**2+4*l*x**3+d*x**4)/r
    else:
        toReturn=(a1+b1*r+d*r**3)
    
    return toReturn

def alpha_equation_RFA(alpha,etaTot,molarFraction,sizeRatio):
    nComponents=len(sizeRatio)
    myRange=range(nComponents)
    sigmaij=np.zeros([nComponents,nComponents])
    c=np.zeros(nComponents)
    Aij=np.zeros([4,nComponents,nComponents])
    contactGij=np.zeros([nComponents,nComponents])
    Lij0=np.zeros([nComponents,nComponents])
    Lij1=np.zeros([nComponents,nComponents])
    Lij2=np.zeros([nComponents,nComponents])
    Bij0=np.zeros([nComponents,nComponents])
    Bij1=np.zeros([nComponents,nComponents])
    htilde=np.zeros([nComponents,nComponents])
    for i in myRange:
        c[i]=molarFraction[i]
        sigmaij[i][i]=sizeRatio[i]
    totalRho=(6*etaTot)/(np.pi*np.sum([c[i]*sigmaij[i][i]**3 for i in myRange]))
    rho=totalRho*c           
    for i in myRange:
        for j in range(i+1,nComponents):
            sigmaij[i][j]=(sigmaij[i][i]+sigmaij[j][j])/2
            sigmaij[j][i]=sigmaij[i][j]

    zeta=lambda n:np.sum([rho[i]*sigmaij[i][i]**n for i in myRange])
    greekL=2*np.pi/(1-etaTot)
    greekLPrime=(np.pi**2)*zeta(2)/(1-etaTot)**2

    for i in myRange:
        for j in range(i,nComponents):
            contactGij[i][j]=(greekL+0.5*greekLPrime*(sigmaij[i][i]*sigmaij[j][j])/sigmaij[i][j]+(1/18)*greekLPrime**2*(sigmaij[i][i]**2*sigmaij[j][j]**2)/(greekL*sigmaij[i][j]**2))/(2*np.pi)
            contactGij[j][i]=contactGij[i][j]
    chi=((totalRho)/(1-etaTot)**2+np.pi*zeta(1)*zeta(2)/(1-etaTot)**3+((np.pi**2)/36)*zeta(2)**3*(9-4*etaTot+etaTot**2)/(1-etaTot)**4)
    chi=totalRho/chi
    
    for i in myRange:
        for j in range(i,nComponents):
            Lij2[i][j]=2*np.pi*alpha*sigmaij[i][j]*contactGij[i][j]
            Lij2[j][i]=Lij2[i][j]
            
    for i in range(nComponents):
        for j in range(nComponents):
            Lij0[i][j]=greekL+greekLPrime*sigmaij[j][j]+2*greekLPrime*alpha-greekL*np.sum([rho[k]*sigmaij[k][k]*Lij2[k][j] for k in myRange])
            Lij1[i][j]=greekL*sigmaij[i][j]+0.5*greekLPrime*sigmaij[i][i]*sigmaij[j][j]+(greekL+greekLPrime*sigmaij[i][i])*alpha-0.5*greekL*sigmaij[i][i]*np.sum([rho[k]*sigmaij[k][k]*Lij2[k][j] for k in myRange])
    
    for n in range(4):
        for i in myRange:
            for j in myRange:
                Aij[n][i][j]=(-1)**n*rho[i]*(sigmaij[i][i]**(n+3)*Lij0[i][j]/(np.math.factorial(n+3))-sigmaij[i][i]**(n+2)*Lij1[i][j]/(np.math.factorial(n+2))+sigmaij[i][i]**(n+1)*Lij2[i][j]/(np.math.factorial(n+1)))
               
    #print(Aij)
    for i in myRange:
        for j in myRange:
            Bij0[i][j]=Lij2[i][j]/(2*np.pi)+np.sum([Aij[2][k][j] for k in myRange])-np.sum([sigmaij[i][k]*(alpha*(k==j)-Aij[1][k][j]) for k in myRange])-0.5*np.sum([sigmaij[i][k]**2*((k==j)-Aij[0][k][j]) for k in myRange])
    Hij0=np.matmul(Bij0,(np.linalg.inv(np.eye(nComponents)-Aij[0][:][:])))
    #GARANTITO GIUSTO FINO A QUA 
    
    for i in myRange:
        for j in myRange:
            Bij1[i][j]=np.sum([Aij[3][k][j]for k in myRange])+np.sum([sigmaij[i][k]*Aij[2][k][j]for k in myRange])-np.sum([(0.5*sigmaij[i][k]**2 + Hij0[i][k])*(alpha*(k==j)-Aij[1][k][j]) for k in myRange])-np.sum([((1/6)*sigmaij[i][k]**3 + sigmaij[i][k]*Hij0[i][k])*((k==j)-Aij[0][k][j]) for k in myRange])
        
    Hij1=np.matmul(Bij1,np.linalg.inv(np.eye(nComponents)-Aij[0][:][:]))
    for i in myRange:
        for j in myRange:
            htilde[i][j]=-4*np.pi*np.sqrt(rho[i]*rho[j])*Hij1[i][j]
    inverseChi=0
    for i in myRange:
        for j in myRange:
            inverseChi=inverseChi+np.sqrt(molarFraction[i]*molarFraction[j])*np.linalg.inv(np.eye(nComponents)+htilde)[i][j]
    

    np.set_printoptions(suppress=True)

    #print(htilde)
    return (1/chi)-np.sum(inverseChi)

def alpha_RFA(etaTot, molarFraction,sizeRatio):
    f=lambda x:alpha_equation_RFA(x,etaTot,molarFraction,sizeRatio)
    alpha=scipy.optimize.fsolve(f,0)
    return alpha

def laplace_rfa_symbolical(s,iComponent,jComponent,etaTot,molarFraction,sizeRatio,alpha):
    xx = [sympy.symbols('x%d' % i) for i in range(3)]
    nComponents=len(sizeRatio)
    myRange=range(nComponents)
    sigmaij=sympy.zeros(nComponents,nComponents)
    c=sympy.zeros(nComponents)
    Aij=sympy.zeros(nComponents,nComponents)
    contactGij=sympy.zeros(nComponents,nComponents)
    Lij0=sympy.zeros(nComponents,nComponents)
    Lij1=sympy.zeros(nComponents,nComponents)
    Lij2=sympy.zeros(nComponents,nComponents)
    Lij=sympy.zeros(nComponents,nComponents)
    htilde=sympy.zeros(nComponents,nComponents)

    phi=lambda x,n:(x**(-(n+1)))*(sum([((-x)**m)/sympy.factorial(m) for m in range(0,n+1)])-sympy.exp(-x))

    for i in myRange:
        c[i]=molarFraction[i]
        sigmaij[i,i]=sizeRatio[i]
    totalRho=(6*etaTot)/(sympy.pi*sum([c[i]*sigmaij[i,i]**3 for i in myRange]))
    rho=totalRho*c           
    for i in myRange:
        for j in range(i+1,nComponents):
            sigmaij[i,j]=(sigmaij[i,i]+sigmaij[j,j])/2
            sigmaij[j,i]=sigmaij[i,j]

    zeta=lambda n:sum([rho[i]*sigmaij[i,i]**n for i in myRange])
    greekL=2*sympy.pi/(1-etaTot)
    greekLPrime=(sympy.pi**2)*zeta(2)/(1-etaTot)**2

    for i in myRange:
        for j in range(i,nComponents):
            contactGij[i,j]=(greekL+0.5*greekLPrime*(sigmaij[i,i]*sigmaij[j,j])/sigmaij[i,j]+(1/18)*greekLPrime**2*(sigmaij[i,i]**2*sigmaij[j,j]**2)/(greekL*sigmaij[i,j]**2))/(2*sympy.pi)
            contactGij[j,i]=contactGij[i,j]
    chi=((totalRho)/(1-etaTot)**2+sympy.pi*zeta(1)*zeta(2)/(1-etaTot)**3+((sympy.pi**2)/36)*zeta(2)**3*(9-4*etaTot+etaTot**2)/(1-etaTot)**4)
    chi=totalRho/chi


    for i in myRange:
        for j in myRange:
            Lij2[i,j]=2*sympy.pi*alpha*sigmaij[i,j]*contactGij[i,j]

    for i in myRange:
        for j in myRange:
            Lij0[i,j]=greekL+greekLPrime*sigmaij[j,j]+2*greekLPrime*alpha-greekL*sum([rho[k]*sigmaij[k,k]*Lij2[k,j] for k in myRange])
            Lij1[i,j]=greekL*sigmaij[i,j]+0.5*greekLPrime*sigmaij[i,i]*sigmaij[j,j]+(greekL+greekLPrime*sigmaij[i,i])*alpha-0.5*greekL*sigmaij[i,i]*sum([rho[k]*sigmaij[k,k]*Lij2[k,j] for k in myRange])
    for i in myRange:
        for j in myRange:
            Aij[i,j]=rho[i]*((1 - s*sigmaij[i,i]+0.5*s**2*sigmaij[i,i]**2 - xx[i])/s**3*Lij0[i,j]+(1-s*sigmaij[i,i]-xx[i])/s**2*Lij1[i,j] +(1 - xx[i])/s*Lij2[i,j]);
    
    for i in myRange:
        for j in myRange:
            Lij[i,j]=sympy.simplify(Lij0[i,j]+Lij1[i,j]*s+Lij2[i,j]*s**2)
    invA=(sympy.simplify((1+alpha*s)*sympy.eye(nComponents)-Aij)).inverse_ADJ()
    Gij=1/(2*sympy.pi*s**2)*(Lij*invA)
    return Gij.subs({'x0':sympy.exp(-sigmaij[0,0]*s),'x1':sympy.exp(-sigmaij[1,1]*s), 'x2':sympy.exp(-sigmaij[2,2]*s)})[0,0]

def laplace_rfa(s,etaTot,molarFraction,sizeRatio,alpha,iComponent=0,jComponent=0):
    nComponents=len(sizeRatio)
    myRange=range(nComponents)
    sigmaij=np.zeros([nComponents,nComponents],dtype=np.complex128)
    c=np.zeros(nComponents,dtype=np.complex128)
    Aij=np.zeros([nComponents,nComponents],dtype=np.complex128)
    contactGij=np.zeros([nComponents,nComponents],dtype=np.complex128)
    Lij0=np.zeros([nComponents,nComponents],dtype=np.complex128)
    Lij1=np.zeros([nComponents,nComponents],dtype=np.complex128)
    Lij2=np.zeros([nComponents,nComponents],dtype=np.complex128)
    Lij=np.zeros([nComponents,nComponents],dtype=np.complex128)
    htilde=np.zeros([nComponents,nComponents],dtype=np.complex128)
    
    phi=lambda x,n:(x**(-(n+1)))*(np.sum([((-x)**m)/np.math.factorial(m) for m in range(0,n+1)])-np.exp(-x))
    
    for i in myRange:
        c[i]=molarFraction[i]
        sigmaij[i][i]=sizeRatio[i]
    totalRho=(6*etaTot)/(np.pi*np.sum([c[i]*sigmaij[i][i]**3 for i in myRange]))
    rho=totalRho*c           
    for i in myRange:
        for j in range(i+1,nComponents):
            sigmaij[i][j]=(sigmaij[i][i]+sigmaij[j][j])/2
            sigmaij[j][i]=sigmaij[i][j]

    zeta=lambda n:np.sum([rho[i]*sigmaij[i][i]**n for i in myRange])
    greekL=2*np.pi/(1-etaTot)
    greekLPrime=(np.pi**2)*zeta(2)/(1-etaTot)**2

    for i in myRange:
        for j in range(i,nComponents):
            contactGij[i][j]=(greekL+0.5*greekLPrime*(sigmaij[i][i]*sigmaij[j][j])/sigmaij[i][j]+greekLPrime**2*(sigmaij[i][i]**2*sigmaij[j][j]**2)/(18*greekL*sigmaij[i][j]**2))/(2*np.pi)
            contactGij[j][i]=contactGij[i][j]
    chi=totalRho*((totalRho)/(1-etaTot)**2+np.pi*zeta(1)*zeta(2)/(1-etaTot)**3+((np.pi**2)/36)*zeta(2)**3*(9-4*etaTot+etaTot**2)/(1-etaTot)**4)
    
    
    for i in myRange:
        for j in myRange:
            Lij2[i][j]=2*np.pi*alpha*sigmaij[i][j]*contactGij[i][j]

    for i in myRange:
        for j in myRange:
            Lij0[i][j]=greekL+greekLPrime*sigmaij[j][j]+2*greekLPrime*alpha-greekL*np.sum([rho[k]*sigmaij[k][k]*Lij2[k][j] for k in myRange])
            Lij1[i][j]=greekL*sigmaij[i][j]+0.5*greekLPrime*sigmaij[i][i]*sigmaij[j][j]+(greekL+greekLPrime*sigmaij[i][i])*alpha-0.5*greekL*sigmaij[i][i]*np.sum([rho[k]*sigmaij[k][k]*Lij2[k][j] for k in myRange])
    xx=np.zeros(3,dtype=np.complex128)
    xx[0]=np.exp(-sigmaij[0][0]*s)
    xx[1]=np.exp(-sigmaij[1][1]*s)
    xx[2]=np.exp(-sigmaij[2][2]*s)
    for i in myRange:
        for j in myRange:
            #Aij[i][j]=0
            Aij[i][j]=rho[i]*(phi(sigmaij[i][i]*s,2)*sigmaij[i][i]**3*Lij0[i][j]+phi(sigmaij[i][i]*s,1)*sigmaij[i][i]**2*Lij1[i][j] +phi(sigmaij[i][i]*s,0)*sigmaij[i][i]*Lij2[i][j])
            #Aij[i][j]=rho[i]*((1 - s*sigmaij[i][i]+0.5*s**2*sigmaij[i][i]**2 - xx[i])/s**3*Lij0[i][j]+(1-s*sigmaij[i][i]-xx[i])/s**2*Lij1[i][j] +(1 - xx[i])/s*Lij2[i][j]);
    for i in myRange:
        for j in myRange:
            Lij[i][j]=Lij0[i][j]+Lij1[i][j]*s+Lij2[i][j]*s**2
    invA=np.linalg.inv((1+alpha*s)*np.eye(nComponents)-Aij)
#    return Aij
    '''if s<0.5:
        Bij0[i][j]=Lij2[i][j]/(2*np.pi)+np.sum([Aij[2][k][j] for k in myRange])-np.sum([sigmaij[i][k]*(alpha*(k==j)-Aij[1][k][j]) for k in myRange])-0.5*np.sum([sigmaij[i][k]**2*((k==j)-Aij[0][k][j]) for k in myRange])
        Hij0=np.matmul(Bij0,(np.linalg.inv(np.eye(nComponents)-Aij[0][:][:])))
        Bij1[i][j]=np.sum([Aij[3][k][j]for k in myRange])+np.sum([sigmaij[i][k]*Aij[2][k][j]for k in myRange])-np.sum([(0.5*sigmaij[i][k]**2 + Hij0[i][k])*(alpha*(k==j)-Aij[1][k][j]) for k in myRange])-np.sum([((1/6)*sigmaij[i][k]**3 + sigmaij[i][k]*Hij0[i][k])*((k==j)-Aij[0][k][j]) for k in myRange])
        
        Hij1=np.matmul(Bij1,np.linalg.inv(np.eye(nComponents)-Aij[0][:][:]))
        return 1+Hij0*s+Hij1*s**2'''
    #return(invA[iComponent][jComponent])   
    if s==0:
        return 0
    return (np.exp(-sigmaij[iComponent][jComponent]*s))/(2*np.pi*s**2)*((np.matmul(Lij,invA))[iComponent][jComponent])

'''def rdf_rfa(etaTot,molarFraction,sizeRatio,iComponent=0,jComponent=0,numTerm=4096, meshSize=0.625, gamma=1):
    alpha=alpha_RFA(etaTot,molarFraction,sizeRatio)
    YSDLap=lambda s:laplace_rfa(s,etaTot,molarFraction,sizeRatio,alpha,iComponent=iComponent,jComponent=jComponent)
    x,y=continue_Euler_transform(YSDLap,meshSize,numTerm,gamma)
    r=(sizeRatio[iComponent]+sizeRatio[jComponent])/2
    y=[y[i] if x[i]>r else 0 for i in range(len(x))]
    y[1:numTerm-1]=y[1:numTerm-1]/x[1:numTerm-1]
    return x,y'''

def rdf_rfa(etaTot,molarFraction,sizeRatio,iComponent=0,jComponent=0,numTerm=4096, meshSize=0.625, gamma=1):
    alpha=alpha_RFA(etaTot,molarFraction,sizeRatio)
    YSDLap=lambda s:laplace_rfa(s,etaTot,molarFraction,sizeRatio,alpha,iComponent=iComponent,jComponent=jComponent)
    x,y=continue_Euler_transform(YSDLap,meshSize,numTerm,gamma)
    r=(sizeRatio[iComponent]+sizeRatio[jComponent])/2
    y=[y[i] if x[i]>r else 0 for i in range(len(x))]
    y[1:int(numTerm/2)-1]=y[1:int(numTerm/2)-1]/x[1:int(numTerm/2)-1]
    return x[0:int(numTerm/2)-1],y[0:int(numTerm/2)-1]

def rdf_rfa_mathematica(iComponent,jComponent,xs,etaTot,sizeRatio,molarFraction):
    nco=len(sizeRatio)
    session = WolframLanguageSession()
    session.evaluate(wl.SetDirectory("/home/eugenio/Desktop/tesi/hardSpherePotential/"))
    session.evaluate(wl.Needs("YDS`","./_mathematicaModule.m"))
    session.evaluate(wl.ResetDirectory())
#    alpha=alpha_mathematica(etaTot,sizeRatio,molarFraction)
    alpha=0.2
    g=[session.evaluate(wlexpr(f"RDF[{iComponent},{jComponent},{x},100,{nco},{etaTot},{{{molarFraction[0]},{molarFraction[1]},{molarFraction[2]}}}, {{{sizeRatio[0]},{sizeRatio[1]},{sizeRatio[2]}}},{alpha}]")) for x in xs]
    session.terminate()
    return g


def alpha_mathematica(etaTot,sizeRatio,molarFraction):
    nco=len(sizeRatio)
    session = WolframLanguageSession()
    session.evaluate(wl.SetDirectory("/home/eugenio/Desktop/tesi/hardSpherePotential/"))
    session.evaluate(wl.Needs("YDS`","./_mathematicaModule.m"))
    session.evaluate(wl.ResetDirectory())

    alpha=session.evaluate(wlexpr(f"ALFA[{nco},{etaTot},{{{molarFraction[0]},{molarFraction[1]},{molarFraction[2]}}}, {{{sizeRatio[0]},{sizeRatio[1]},{sizeRatio[2]}}}]"))
    session.terminate()
    return alpha 

def continue_Euler_transform(G,meshSize,numTerm,gamma):
    p=np.sqrt(numTerm/2)
    q=np.sqrt(numTerm/2)
    w=lambda x:0.5*scipy.special.erfc(x/p-q)
    toIfft=scipy.fft.fftshift(np.array([w(np.abs(n*meshSize))*G(gamma+n*meshSize*1j) for n in  range(-int(numTerm/2),int(numTerm/2)-1)]))
    x=np.linspace(0,2*np.pi/(meshSize),numTerm-1)
    #y=numTerm*meshSize*np.exp(gamma*(2*np.pi*scipy.fft.fftshift(np.arange(-int(numTerm/2),int(numTerm/2)-1))/(numTerm*meshSize)))/(2*np.pi)*scipy.fft.ifft(toIfft)
    y=numTerm*meshSize*np.exp(gamma*(2*np.pi*(np.arange(0,numTerm-1))/(numTerm*meshSize)))/(2*np.pi)*scipy.fft.ifft(toIfft)
    return x,y

def second_virial_coefficient_from_rdf(x,g):
    return np.real(2*np.pi*scipy.integrate.trapezoid((1-np.array(g))*x**2,x))


if __name__=="__main__":
    meshSize=np.pi/10
    gamma=0.5
    numTerm=4096
    etaTot=float(sys.argv[1])
    sizeRatio=np.array([float(i) for i in sys.argv[2].split(',')])
    molarFraction=np.array([float(i) for i in sys.argv[3].split(',')])
    '''etaTot=0.5
    sizeRatio=[1,2/3,1/3]
    molarFraction=[0.8,0.1,0.1]'''
    iComponent=0
    jComponent=0
    #alpha=alpha_RFA(etaTot,molarFraction,sizeRatio)
    #YSDLap=lambda s:laplace_rfa(s,etaTot,molarFraction,sizeRatio,alpha,iComponent=iComponent,jComponent=jComponent)
    #x,y=continue_Euler_transform(YSDLap,meshSize,numTerm,gamma)
    x,y = rdf_rfa(etaTot,molarFraction,sizeRatio,iComponent=0,jComponent=0,numTerm=numTerm,meshSize=meshSize,gamma=gamma) 
    plt.plot(x,np.array(np.real(y)), label=f'B2= {second_virial_coefficient_from_rdf(x,y):.3f}')
    plt.grid()
    plt.legend()
    plt.show()
