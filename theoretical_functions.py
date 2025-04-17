from tqdm import tqdm
from wolframclient.language import wl,wlexpr
import datetime
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import sys
import sympy as sympy

def carnahan_starling(eta, d, beta):
    """returns the pressure given by the Carnahan-Starlin equation of state for a gas of hard spheres of radius d, at inverse temperature beta and packing fraction eta"""
    #eta=4*np.pi*rho*(1/3)*d**3
    rho=3*eta/(3*np.pi*d**3)
    return rho*(1+eta+eta**2-eta**3)/(beta*(1-eta)**3)


def boublik_mansoori_CS(eta_tot,molar_fraction,size_ratio):
    """returns the beta*pressure given by the Boublik Mansoori equation of state for polydisperse gas of hard spheres, with radiuses contained in in the np array size_ratio, concetration fraction for each specie contained in the np array molar fraction and with total occupied volum eta_tot """
    xi=lambda m:np.pi*eta_tot*np.sum(molar_fraction*(2*size_ratio**m))/6
    return (6/np.pi)*((xi(0))/(1-xi(3))+(3*xi(1)*xi(2))/(1-xi(3))**2+((3-xi(3)*xi(2)**3))/(1-xi(3))**3)



def ideal_gas(eta,d,beta):

    rho=3*eta/(3*np.pi*d**3)
    return rho/beta

def overlap_volume(D,d,r):
    """returns the volume of the overlap of a sphere of radius D and a sphere of radius d at a distance r"""
    return (np.pi/6)*(2*D+2*d+r)*(D+d-r)**2

def monodisperse_correlation_function(r,sigma,eta_tot):
    """returns the pair correlation function g given by the solution of the OZ equation with the PY closure for a liquid of hard spheres of radius sigma and concentration eta_tot. The function is evalutaed at a distance r"""
    rstar=(2.0116-1.0647*eta_tot+0.0538*eta_tot**2)/(sigma)
   
    gsigma=(((1+eta_tot+eta_tot**2-(2/3)*eta_tot**3-(2/3)*eta_tot**4)/(1-eta_tot)**3)-1)/(4*eta_tot)
    gm=1.0286-0.6095*eta_tot+3.5781*eta_tot**2-21.3651*eta_tot**3+42.6344*eta_tot**4-33.8485*eta_tot**5
    alpha=(44.554+79.868*eta_tot+116.432*eta_tot**2-44.652*np.exp(2*eta_tot))/(sigma)
    beta=(-5.022+5.857*eta_tot+5.089*np.exp(-4*eta_tot))/(sigma)
    
    d=np.power((2*eta_tot*(eta_tot**2-3*eta_tot-3+np.sqrt(3*(eta_tot**4-2*eta_tot**3+eta_tot**2+6*eta_tot+3)))),1/3)
    
    alpha0=(2*eta_tot/(1-eta_tot))*(-1+d/(4*eta_tot)-eta_tot/(2*d))/sigma
    beta0=(2*eta_tot/(1-eta_tot))*np.sqrt(3)*(-d/(4*eta_tot)-eta_tot/(2*d))/sigma
    mu=(2*eta_tot/(1-eta_tot))*(-1-d/(2*eta_tot)+eta_tot/d)/sigma
    
    k=(4.674*np.exp(-3.935*eta_tot)+3.536*np.exp(-56.270*eta_tot))/sigma
    omega=(-0.682*np.exp(-24.696*eta_tot)+4.720+4.4450*eta_tot)/sigma
    
    gamma=np.arctan(-(sigma*(alpha0*(alpha0**2+beta0**2)-mu*(alpha0**2-beta0**2))*(1+0.5*eta_tot)+(alpha0**2+beta0**2-mu*alpha0)*(1+2*eta_tot))/(beta0*(sigma*(alpha0**2+beta0**2-mu*alpha0)*(1+0.5*eta_tot)-mu*(1+2*eta_tot))))
    
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
    """Returns the laplace transform of the radial distribution function for a bidisperse solution of hard spheres, where the excluded volume for the two species of crowders are eta1 and eta2 and their radiuses are r1 and r2.If ij!=1, the functions return the laplace transform of the radial distribution function between specie 1 and specie 2. The function is evaluated at s"""
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
    """Returns the laplace pair correlation function g given by the solution of the OZ equation with the PY closure for a liquid of hard spheres of radius r1 and concentration eta1. The function is evaluted at s"""
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
    """Returns the direct correlation function for a bidisperse solution of hard spheres, where the excluded volume for the two species of crowders are eta1 and eta2 and their radiuses are r1 and r2.If ij!=1, the functions return the laplace transform of the radial distribution function between specie 1 and specie 2. The function is evaluated at s"""
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

'''def mComponentDirectCorrelationFunction(r,eta,R,i,j):
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
'''
def alpha_equation_RFA(alpha,eta_tot,molar_fraction,size_ratio):
    """Given a polydisperse hard-sphere liquids with the concentration fraction for each species contained in the np array molar_fraction, the respective radii contained in size_ratio, and the toatl packing fraction given by eta_tot, finding the value of alpha that makes this function 0 is equivalent to solving the equation for alpha in the RFA approach by Yuste, Santo, de Haro"""
    nComponents=len(size_ratio)
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
        c[i]=molar_fraction[i]
        sigmaij[i][i]=size_ratio[i]
    totalRho=(6*eta_tot)/(np.pi*np.sum([c[i]*sigmaij[i][i]**3 for i in myRange]))
    rho=totalRho*c           
    for i in myRange:
        for j in range(i+1,nComponents):
            sigmaij[i][j]=(sigmaij[i][i]+sigmaij[j][j])/2
            sigmaij[j][i]=sigmaij[i][j]

    zeta=lambda n:np.sum([rho[i]*sigmaij[i][i]**n for i in myRange])
    greekL=2*np.pi/(1-eta_tot)
    greekLPrime=(np.pi**2)*zeta(2)/(1-eta_tot)**2

    for i in myRange:
        for j in range(i,nComponents):
            contactGij[i][j]=(greekL+0.5*greekLPrime*(sigmaij[i][i]*sigmaij[j][j])/sigmaij[i][j]+(1/18)*greekLPrime**2*(sigmaij[i][i]**2*sigmaij[j][j]**2)/(greekL*sigmaij[i][j]**2))/(2*np.pi)
            contactGij[j][i]=contactGij[i][j]
    chi=((totalRho)/(1-eta_tot)**2+np.pi*zeta(1)*zeta(2)/(1-eta_tot)**3+((np.pi**2)/36)*zeta(2)**3*(9-4*eta_tot+eta_tot**2)/(1-eta_tot)**4)
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
            inverseChi=inverseChi+np.sqrt(molar_fraction[i]*molar_fraction[j])*np.linalg.inv(np.eye(nComponents)+htilde)[i][j]
    

    np.set_printoptions(suppress=True)

    #print(htilde)
    return (1/chi)-np.sum(inverseChi)

def alpha_RFA(eta_tot, molar_fraction,size_ratio):
    """returns the value of alpha needed for the RFA approach of the radial distribution function for a liquid of hard spheres with total packing fraction eta_tot, where the concentrations of the species and their radii are contained in molar_fraction and size_ratio respectively"""
    f=lambda x:alpha_equation_RFA(x,eta_tot,molar_fraction,size_ratio)
    alpha=scipy.optimize.fsolve(f,0)
    return alpha

def laplace_RFA_symbolical(s,i_component,j_component,eta_tot,molar_fraction,size_ratio,alpha):
    """Should return the laplace transform of the radial distribution function given by the RFA. Does not work, sympy has problems evaluating it correctly"""
    xx = [sympy.symbols('x%d' % i) for i in range(3)]
    nComponents=len(size_ratio)
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
        c[i]=molar_fraction[i]
        sigmaij[i,i]=size_ratio[i]
    totalRho=(6*eta_tot)/(sympy.pi*sum([c[i]*sigmaij[i,i]**3 for i in myRange]))
    rho=totalRho*c           
    for i in myRange:
        for j in range(i+1,nComponents):
            sigmaij[i,j]=(sigmaij[i,i]+sigmaij[j,j])/2
            sigmaij[j,i]=sigmaij[i,j]

    zeta=lambda n:sum([rho[i]*sigmaij[i,i]**n for i in myRange])
    greekL=2*sympy.pi/(1-eta_tot)
    greekLPrime=(sympy.pi**2)*zeta(2)/(1-eta_tot)**2

    for i in myRange:
        for j in range(i,nComponents):
            contactGij[i,j]=(greekL+0.5*greekLPrime*(sigmaij[i,i]*sigmaij[j,j])/sigmaij[i,j]+(1/18)*greekLPrime**2*(sigmaij[i,i]**2*sigmaij[j,j]**2)/(greekL*sigmaij[i,j]**2))/(2*sympy.pi)
            contactGij[j,i]=contactGij[i,j]
    chi=((totalRho)/(1-eta_tot)**2+sympy.pi*zeta(1)*zeta(2)/(1-eta_tot)**3+((sympy.pi**2)/36)*zeta(2)**3*(9-4*eta_tot+eta_tot**2)/(1-eta_tot)**4)
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

def laplace_RFA(s,eta_tot,molar_fraction,size_ratio,alpha,i_component=0,j_component=0):
    """Returns the laplace transform of the radial distribution function between the 'i_component' component and the 'j_component' component for a polysperse solution of hard spheres with total packing fraction eta_tot, and with the crowders concentration fraction and radii contained in the np arrays molar_fraction and size_ratio. The function is evaluated for at s"""
    nComponents=len(size_ratio)
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
        c[i]=molar_fraction[i]
        sigmaij[i][i]=size_ratio[i]
    totalRho=(6*eta_tot)/(np.pi*np.sum([c[i]*sigmaij[i][i]**3 for i in myRange]))
    rho=totalRho*c           
    for i in myRange:
        for j in range(i+1,nComponents):
            sigmaij[i][j]=(sigmaij[i][i]+sigmaij[j][j])/2
            sigmaij[j][i]=sigmaij[i][j]

    zeta=lambda n:np.sum([rho[i]*sigmaij[i][i]**n for i in myRange])
    greekL=2*np.pi/(1-eta_tot)
    greekLPrime=(np.pi**2)*zeta(2)/(1-eta_tot)**2

    for i in myRange:
        for j in range(i,nComponents):
            contactGij[i][j]=(greekL+0.5*greekLPrime*(sigmaij[i][i]*sigmaij[j][j])/sigmaij[i][j]+greekLPrime**2*(sigmaij[i][i]**2*sigmaij[j][j]**2)/(18*greekL*sigmaij[i][j]**2))/(2*np.pi)
            contactGij[j][i]=contactGij[i][j]
    chi=totalRho*((totalRho)/(1-eta_tot)**2+np.pi*zeta(1)*zeta(2)/(1-eta_tot)**3+((np.pi**2)/36)*zeta(2)**3*(9-4*eta_tot+eta_tot**2)/(1-eta_tot)**4)
    
    
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
    #return(invA[i_component][j_component])   
    if s==0:
        return 0
    return (np.exp(-sigmaij[i_component][j_component]*s))/(2*np.pi*s**2)*((np.matmul(Lij,invA))[i_component][j_component])

'''def rdf_RFA(eta_tot,molar_fraction,size_ratio,i_component=0,j_component=0,num_term=4096, mesh_size=0.625, gamma=1):
    alpha=alpha_RFA(eta_tot,molar_fraction,size_ratio)
    YSDLap=lambda s:laplace_RFA(s,eta_tot,molar_fraction,size_ratio,alpha,i_component=i_component,j_component=j_component)
    x,y=continuous_Euler_transformation(YSDLap,mesh_size,num_term,gamma)
    r=(size_ratio[i_component]+size_ratio[j_component])/2
    y=[y[i] if x[i]>r else 0 for i in range(len(x))]
    y[1:num_term-1]=y[1:num_term-1]/x[1:num_term-1]
    return x,y'''

def rdf_RFA(eta_tot,molar_fraction,size_ratio,i_component=0,j_component=0,num_term=4096, mesh_size=0.625, gamma=1):
    """returns the radial distribution function between the 'i_component' component and the 'j_component' component of a polydisperse hard-spheres liquid given by the RFA method. eta_tot is the total packing fraction, molar_fraction must be a np array containing the concentration fraction for each species, size_ratio must be an array containing the respective size, and num_term, mesh_size and gamma are the parameters used to compute the inverse laplace transform(see continuous_Euler_transformation)"""
    alpha=alpha_RFA(eta_tot,molar_fraction,size_ratio)
    YSDLap=lambda s:laplace_RFA(s,eta_tot,molar_fraction,size_ratio,alpha,i_component=i_component,j_component=j_component)
    x,y=continuous_Euler_transformation(YSDLap,mesh_size,num_term,gamma)
    r=(size_ratio[i_component]+size_ratio[j_component])/2
    y=[y[i] if x[i]>r else 0 for i in range(len(x))]
    y[1:int(num_term/2)-1]=y[1:int(num_term/2)-1]/x[1:int(num_term/2)-1]
    return x[0:int(num_term/2)-1],y[0:int(num_term/2)-1]


def continuous_Euler_transformation(G,mesh_size,num_term,gamma):
    """return the inverse laplace transform of G using the continuous Euler transformation as described in Ouura, RIMS Kokyuroku, 2000"""
    p=np.sqrt(num_term/2)
    q=np.sqrt(num_term/2)
    w=lambda x:0.5*scipy.special.erfc(x/p-q)
    toIfft=scipy.fft.fftshift(np.array([w(np.abs(n*mesh_size))*G(gamma+n*mesh_size*1j) for n in  range(-int(num_term/2),int(num_term/2)-1)]))
    x=np.linspace(0,2*np.pi/(mesh_size),num_term-1)
    #y=num_term*mesh_size*np.exp(gamma*(2*np.pi*scipy.fft.fftshift(np.arange(-int(num_term/2),int(num_term/2)-1))/(num_term*mesh_size)))/(2*np.pi)*scipy.fft.ifft(toIfft)
    y=num_term*mesh_size*np.exp(gamma*(2*np.pi*(np.arange(0,num_term-1))/(num_term*mesh_size)))/(2*np.pi)*scipy.fft.ifft(toIfft)
    return x,y

def second_virial_coefficient_from_rdf(x,g):
    """returns the second virial coefficient given the radial distribution function"""
    return np.real(2*np.pi*scipy.integrate.trapezoid((1-np.array(g))*x**2,x))

def second_virial_coefficient_from_parameters(eta_tot,molar_fraction,size_ratio,i_component=0,j_component=0,num_term=4096, mesh_size=0.625, gamma=1):
    x_rdf,rdf=rdf_RFA(eta_tot,molar_fraction,size_ratio,i_component,j_component,num_term,mesh_size,gamma)
    svc=second_virial_coefficient_from_rdf(x_rdf,rdf)
    return svc

def write_potential_table(eta_tot,molar_fraction,size_ratio,i_component=0,j_component=0,output_file='output.txt', heading=''):
    """writes the potential table suitable for molecular dynamics in LAMMPS"""
    mesh_size=np.pi/10
    gamma=0.5
    num_term=4096

    
    x,y = rdf_RFA(eta_tot,molar_fraction,size_ratio,i_component=i_component,j_component=j_component,num_term=num_term,mesh_size=mesh_size,gamma=gamma) 
    y=np.real([-np.log(y[i])  if y[i]!=0 else np.abs(10*x[i]**(-6)) for i in range(len(y))])
    force=[-(y[i+1]-y[i])/(x[i+1]-x[i]) for i in range(len(x[:-1]))]
    #plt.show()
      
    with open(output_file, 'w') as f:
        f.write(f'#{heading}\n')
        f.write(f'#DATE: {datetime.datetime.now()} UNITS: lj\n')
        f.write(f'#effective potential for hard sphere depletion interaction\n')
        f.write('\n')
        f.write('DEPLETION_INTERACTION\n')
        f.write(f'N {len(force)-1} R {sorted(x)[1]} {max(x[:-1])}\n')
        f.write('\n')
        for i in range(1, len(x)-1):
            potential_to_write  =1e16 if np.isinf(y[i]) else y[i]
            force_to_write      =1e16 if np.isinf(force[i]) else force[i]
            f.write(f'{i} {x[i]} {potential_to_write} {force_to_write}\n')

if __name__=="__main__":
    '''mesh_size=np.pi/10
    gamma=0.5
    num_term=4096
    eta_tot=float(sys.argv[1])
    size_ratio=np.array([float(i) for i in sys.argv[2].split(',')])
    molar_fraction=np.array([float(i) for i in sys.argv[3].split(',')])
    eta_tot=0.5
    size_ratio=[1,2/3,1/3]
    molar_fraction=[0.8,0.1,0.1]
    i_component=0
    j_component=0
    #alpha=alpha_RFA(eta_tot,molar_fraction,size_ratio)
    #YSDLap=lambda s:laplace_RFA(s,eta_tot,molar_fraction,size_ratio,alpha,i_component=i_component,j_component=j_component)
    #x,y=continuous_Euler_transformation(YSDLap,mesh_size,num_term,gamma)
    x,y = rdf_RFA(eta_tot,molar_fraction,size_ratio,i_component=0,j_component=0,num_term=num_term,mesh_size=mesh_size,gamma=gamma) 
    plt.plot(x,np.array(np.real(y)), label=f'B2= {second_virial_coefficient_from_rdf(x,y):.3f}')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig('./figures/second_virial_coefficient.pdf',format='pdf')
    params = {'font.size':'28','legend.fontsize':'x-large','figure.figsize':(10,6),'axes.labelsize':'20','axes.titlesize':'18','xtick.labelsize':'15','ytick.labelsize':'15','legend.fontsize':'20'}
    plt.rcParams.update(params)
    x=np.linspace(0,10,1000)
    y=[monodisperse_correlation_function(i,1,0.45) for i in x]
    plt.plot(x,y)
    plt.ylabel('$g(r)$')
    plt.xlabel('$r$')
    plt.savefig('./figures/monodisperse_PY.png',format='png')
    plt.show()

    print(second_virial_coefficient_from_rdf(x,y))
    x=np.linspace(0,10,1000)
    y=[-np.log(i) if i != 0 else 1000 for i in y]
    plt.plot(x,y)
    plt.ylim([-2,1])
    plt.ylabel('$\psi(r)$')
    plt.xlabel('$r$')
    plt.savefig('./figures/monodisperse_mean_force_potential.png',format='png')
    plt.show()'''

    
    gamma=0.5
    mesh_size=np.pi/10
    num_term=4096
    eta_tot=float(sys.argv[1])
    n_mono=float(sys.argv[2])
    n=float(sys.argv[4])
    N=float(sys.argv[3])
    n_tot=n_mono+n+N
    size_ratio=[1,2/3,1/3]
    molar_fraction=[n_mono/n_tot,N/n_tot,n/n_tot]
    i_component=0
    j_component=0
    x,y = rdf_RFA(eta_tot,molar_fraction,size_ratio,i_component=0,j_component=0,num_term=num_term,mesh_size=mesh_size,gamma=gamma) 
    y2=np.real([-np.log(y[i])  if y[i]!=0 else np.abs(x[i]**(-6)) for i in range(len(y))])
    plt.plot(x,np.array(np.real(y2)), label=f'B2= {second_virial_coefficient_from_rdf(x,y):.3f}')
    plt.ylim([-10,10])
    plt.grid()
    plt.legend()
    plt.show()
    plt.plot(x,np.array(np.real(y)), label=f'B2= {second_virial_coefficient_from_rdf(x,y):.3f}')
    plt.grid()
    plt.legend()
    plt.show()

    #write_potential_table(eta_tot,[n_mono/n_tot,N/n_tot,n/n_tot],size_ratio,0,0,'ciao.txt') 
