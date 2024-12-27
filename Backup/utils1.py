import numpy as np
from numpy import linalg as LA
from random import random
from scipy.stats import unitary_group
import picos as pic
from scipy.spatial import ConvexHull #Medições
#Para o plot:
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize


def G_matrix(n,m):
    ''' 
    Generation of the random matrix from the Ginibre ensemble
    A complex matrix with elements having real and complex part 
    distributed with the normal distribution 
    
    input: dimensions of the Matrix G of size n x m (integers)
    output: array of matrix G of size n x m
    '''

    G = (np.random.randn(n, m) + 1j * np.random.randn(n, m)) / np.sqrt(2)
    return G

def rho_mixed(n):
    '''
    Generation a random mixed density matrix (Bures metric)
    Input: n = dimension of the density matrix (integer)
    Output: array of density matrix 
    '''

    # Create random unitary matrix
    U = unitary_group.rvs(n)
    # Create random Ginibre matrix
    G = G_matrix(n,n)
    # Create identity matrix
    I = np.eye(4)
    # Construct density matrix
    rho = (I+U)@G@(G.conjugate().T)@(I+U.conjugate().T)
    # Normalize density matrix
    rho = rho/(rho.trace())
    return rho

def rho_mixed_HS(n):
    '''
    Generation a random mixed density matrix (Hilbert-Schmidt metric)
    Input: n = dimension of the density matrix (integer)
    Output: array of density matrix 
    '''

    # Create random Ginibre matrix
    G = G_matrix(n,n)
    # Construct density matrix
    rho = G@(G.conjugate().T)
    # Normalize density matrix
    rho = rho/(rho.trace())
    return rho

def Ent_cert(rho):
    '''
    Entanglement certification using PPT criterion
    Input: density matrix
    Output: w, v = eigenvalues of the partial transpose density matrix w
            ppt = 0 if separable, 1 if entangled

    '''

    # Calculate partial transpose
    n = rho.shape
    rho_TA = np.zeros((n[0],n[1]),dtype=np.complex_)

    a = int(n[0]/2)
    b = int(n[1]/2)
    
    rho_TA[:a,:b] = rho[:a,:b]
    rho_TA[a:,b:] = rho[a:,b:]
    rho_TA[a:,:b] = rho[a:,:b].T
    rho_TA[:a,b:] = rho[:a,b:].T
    
    # v - eigenvectors, w - eigenvalues
    w, v = LA.eig(rho_TA)
    
    # PPT Criterion: Are all eigenvalues >=0?
    if all(i >= 0 for i in w):
        # print('Yes: separable state.')
        ppt = 0
    else:
        # print('No: entangled state.')
        ppt = 1
    
    return w,v,ppt


def measurements(n,PLOT=False):
    '''
    Creating polytopes from dichotomic projective measurements
    INPUT: n is the cicle we are in, may have the values: [1,2,3,4]; PLOT:False/True to plot or not the polyhedron
    n == 1: 6 vertices
    n == 2: 18 vertices
    n == 3: 26 vertices
    For n == 4, the SDP becomes too big.
    Output: complex array with the measurements
    '''

    #Create the vertices of the polytope
    vert_p = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])

    vert_s = np.array([[0,1/(np.sqrt(2)),1/(np.sqrt(2))],[0,-1/(np.sqrt(2)),-1/(np.sqrt(2))],
    [0,-1/(np.sqrt(2)),1/(np.sqrt(2))],[0,1/(np.sqrt(2)),-1/(np.sqrt(2))],
    [1/(np.sqrt(2)),0,1/(np.sqrt(2))],[-1/(np.sqrt(2)),0,-1/(np.sqrt(2))],
    [-1/(np.sqrt(2)),0,1/(np.sqrt(2))],[1/(np.sqrt(2)),0,-1/(np.sqrt(2))],
    [1/(np.sqrt(2)),1/(np.sqrt(2)),0],[-1/(np.sqrt(2)),-1/(np.sqrt(2)),0],
    [-1/(np.sqrt(2)),1/(np.sqrt(2)),0],[1/(np.sqrt(2)),-1/(np.sqrt(2)),0]
    ])

    vert_s = np.concatenate((vert_p,vert_s))

    vert_t = np.array([[1/2,1/2,1/(np.sqrt(2))],[-1/2,-1/2,-1/(np.sqrt(2))],
    [-1/2,1/2,1/(np.sqrt(2))],[1/2,-1/2,-1/(np.sqrt(2))],
    [1/2,1/2,-1/(np.sqrt(2))],[-1/2,-1/2,1/(np.sqrt(2))],
    [1/2,-1/2,1/(np.sqrt(2))],[-1/2,1/2,-1/(np.sqrt(2))]
    ])

    vert_t = np.concatenate((vert_s,vert_t))

    vert_q = np.array([[1/2,(np.sqrt(3))/2,0],[-1/2,-(np.sqrt(3))/2,0],
    [(np.sqrt(3))/2,1/2,0],[-(np.sqrt(3))/2,-1/2,0],
    [0,(np.sqrt(3))/2,1/2],[0,-(np.sqrt(3))/2,-1/2],
    [(np.sqrt(3))/2,0,1/2],[-(np.sqrt(3))/2,0,-1/2],
    [-1/2,(np.sqrt(3))/2,0],[1/2,-(np.sqrt(3))/2,0],
    [-(np.sqrt(3))/2,1/2,0],[(np.sqrt(3))/2,-1/2,0],
    [0,-(np.sqrt(3))/2,1/2],[0,(np.sqrt(3))/2,-1/2],
    [(np.sqrt(3))/2,0,-1/2],[-(np.sqrt(3))/2,0,1/2]
    ])

    vert_q = np.concatenate((vert_t,vert_q))

    #Choose which set of vertices we want to use depending on the cycle we are in
    if n == 1:
        vert = vert_p
    elif n == 2:
        vert = vert_s
    elif n == 3:
        vert = vert_t
    elif n == 4:
        vert = vert_q

    #Create the measurements of each vertex
    m_k = vert.shape[0]
    medicoes = np.zeros([m_k,2,2], dtype=complex)

    for i in range(m_k):
        med_00 = (1+vert[i][2])/2
        med_01 = (vert[i][0]-vert[i][1]*1j)/2
        med_10 = (vert[i][0]+vert[i][1]*1j)/2
        med_11 = (1-vert[i][2])/2
    
        medicoes[i] = [[med_00,med_01],[med_10,med_11]]

    #Verify that the sum of each dichotomous measurement is the identity
    #for i in range(int(m_k/2)):
        #print("Sum")
        #print(medicoes[2*i]+medicoes[2*i+1])
    
    #Construct the polyhedron
    hull = ConvexHull(vert)
    #Find the Insphere radius
    r = np.min(np.abs(hull.equations[:, -1]))

    #Plot
    if PLOT == True:
        polys = Poly3DCollection([hull.points[simplex] for simplex in hull.simplices])

        # Good choice of colors: 'deeppink' and 'hotpink'
        polys.set_edgecolor('blue')
        polys.set_linewidth(.8)
        polys.set_facecolor('azure')
        polys.set_alpha(.25)
        
        #Build the insphere
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = r*np.cos(u)*np.sin(v)
        y = r*np.sin(u)*np.sin(v)
        z = r*np.cos(v)
        
        #Build the Bloch sphere
        x_uni = np.cos(u)*np.sin(v)
        y_uni = np.sin(u)*np.sin(v)
        z_uni = np.cos(v)
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")
        #ax = Axes3D(plt.figure())
        #Plot Bloch sphere
        ax.plot_surface(x_uni,y_uni,z_uni,color='lightgray',alpha=.15)
        #Plot insphere
        # Good choice of color: 'lime', alpha=.35
        ax.plot_surface(x,y,z,color='yellow',alpha=.95)
        #Plot polyhedron
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        ax.set_box_aspect([1,1,1])
        ax.set_xticks([-1,0,1])
        ax.set_yticks([-1,0,1])
        ax.set_zticks([-1,0,1])
        ax.view_init(15,75)
        #plt.axis('off')

        ax.add_collection3d(polys)
        #plt.show()
        plt.savefig('poliedro_'+str(m_k)+'.png', transparent=True,dpi=300)
    
    #Return the measurements and the insphere radius
    return medicoes,r

def strategies_LHS(m,k):
    '''
    Creating the deterministic strategies
    INPUT: m = number of measurements; k = number of results
    Output: array with all the deterministic strategies
    '''
    
    #k**m = number of strategies = n_lambdas

    n_lambdas = k**m
    
    #Creating the strategies
    all_est = [np.base_repr(el+n_lambdas,base=k)[-m:] for el in range(n_lambdas)]
    
    all_est = np.array([[int(digit) for digit in el] for el in all_est])

    detp = np.zeros((n_lambdas,k*m))

    for i in range(n_lambdas):
        for j in range(m):
            aux = np.zeros(k)
            aux[all_est[i][j]] = 1
            detp[i][j*k:j*k+k] = np.array(aux)    
            
    #Return the deterministic strategies
    return detp

def SDP_LHS(m,k,rho,rho_sep,eta,detp,medicoes):
    '''
    Creating the SDP to find the Local Hiden State model for the given state
    Input: m - number of measurements (Integer)
           k - number of results per measurement (Integer)
           rho - target state (Array)
           rho_sep - separable state (Array)
           eta - insphere radius of the measurements (Float)
           detp - deterministic strategies (Array)
           medicoes - measurements (Array)
    Output: the problem created, the solution found, the value of q
    '''

    #Creating the problem
    P = pic.Problem()

    #Creating the optimization variables
    q = pic.RealVariable('q')

    chi = pic.HermitianVariable('chi',(4,4))

    sigma = [pic.HermitianVariable('Sigma_lambda[{}]'.format(i),(2,2)) for i in range(k**m)]

    rho_q = rho*q+(1-q)*rho_sep

    rho_eta = eta*chi+(1-eta)*(pic.partial_trace(rho_sep,subsystems=1,dimensions=(2,2)))@(pic.partial_trace(chi,subsystems=0,dimensions=(2,2)))

    est_det = [pic.sum([sigma[j]*detp[j,i] for j in range(k**m)]) for i in range(k*m)]

    est = [(np.kron(medicoes[i],np.eye(2)))*chi for i in range(k*m)]

    #Creating the constraints
    P.add_constraint(q<=1)

    P.add_constraint(q>=0)

    P.add_list_of_constraints([sigma[i]>>0 for i in range(k**m)]) 

    P.add_constraint(rho_q == rho_eta)

    P.add_list_of_constraints([pic.partial_trace(est[i],subsystems=0,dimensions=(2,2))==est_det[i] for i in range(k*m)])

    #Setting the objective
    P.set_objective('max',q)

    P.options['primals'] = None

    #Finding the solution
    solution = P.solve()

    #Return the problem created, the solution found, the value of q
    return P, solution, q

def ent_threshold(rho, rho_sep):
    '''
    Calculating the limit of entanglement
    Input: rho - target state (Array)
           rho_sep - separable state (Array)
    Output: Entanglement threshold
    '''

    r = 0.5
    eps = 0.5
    i = 0

    while eps >= 10**(-15) and r>=0 and r<=1 and i<=10**5:
        rho_ent = r*rho+(1-r)*rho_sep
        w,v, ppt = Ent_cert(rho_ent)

        eps = eps/2

        if ppt == 0:
            r = r + eps
        else:
            r = r - eps

        i += 1

    eta_ent = r

    return eta_ent

def nlol_threshold(dim,c,rho,rho_sep,k_A,m_A,k_B,m_B,rounds,D_max):
    '''
    Calculating the limit of entanglement
    Input: dimension - dimension of the measurements of Alice(Integer)
           c - Bell inequality constants (List of floats)
           rho - target state (Complex array)
           rho_sep - separable state (Complex Array)
           k_A - number of results per measurement for Alice (Integer)
           m_A - number of measurements for Alice (Integer)
           k_B - number of results per measurement for Bob (Integer)
           m_B - number of measurements for Bob (Integer)
           rounds - how many times to run the SewSaw (Integer)
           D_max - threshold for classicality
    Output: Nonlocality threshold
    '''

    r = 0.5
    eps = 0.5
    i = 0

    while eps >= 10**(-2) and r>=0 and r<=1 and i<=10:

        rho_ent = r*rho+(1-r)*rho_sep

        D = SeeSaw(dim,c,rho_ent,k_A,m_A,k_B,m_B,rounds)

        eps = eps/2

        if D <= D_max:
            r = r + eps
        else:
            r = r - eps

        i += 1

    nlol_ent = r

    return nlol_ent


def Haar_random(dimension):
    '''
    Generating random unitary matrix according to Haar measure.
    Ref.: https://arxiv.org/abs/math-ph/0609050v2

    Input: dimension (Integer)
    Output: array of the matrix
    '''
    A = G_matrix(dimension,dimension)
    q, r = np.linalg.qr(A)
    m = np.diagonal(r)
    m = m / np.abs(m)
    return np.multiply(q, m, q)


def proj_meas_random(dimension):
    '''
    Generating random projective measurement according to Haar measure.
    Input: dimension (integer)
    Output: array of the measurement
    '''

    Haar = Haar_random(dimension)

    measurement = np.zeros((dimension,dimension,dimension),dtype=complex)

    for i in range(dimension):
        M = np.zeros((dimension, dimension),dtype=complex)
        M[i][i] = 1
        measurement[i] = Haar@M@Haar.conj().T

    return measurement

def is_measurement(measurement):
    '''
    Checks if the array is a measurement.
    Input: array of measurement.
    Output: True/False
    '''

    k = measurement.shape[0]
    dim = measurement.shape[1]
    soma = np.zeros((dim, dim),dtype=complex)
    for i in range(k):
        soma = measurement[i]+soma
        w, v = np.linalg.eig(measurement[i])
        #Has to be positive semi-definite
        if not (np.all(w>=0)): return False
    
    #Has to sum identity
    is_it = np.allclose(soma, np.eye(dim,dtype=complex), rtol=1.e-7, atol=1.e-8)

    return is_it

def Comp_measu():
    '''
    Generating the computational measurements 00, 11, + and -.
    '''

    #Medição computacional
    KetBra_00 = np.array([[1,0],[0,0]])
    KetBra_11 = np.array([[0,0],[0,1]])

    ket_mais = (np.array([[1],[1]]))/np.sqrt(2)
    ket_menos = (np.array([[1],[-1]]))/np.sqrt(2)

    KetBra_mais = ket_mais@np.transpose(ket_mais)
    KetBra_menos = ket_menos@np.transpose(ket_menos)

    return KetBra_00, KetBra_11, KetBra_mais, KetBra_menos

def max_Measurements(dimension,c,rho,measurements,k,m, side):
    '''
    Inputs:
    dimension - dimension of the measurements (Integer)
    c - Bell inequality constants (List of floats)
    rho - target state (Complex array)
    measurements- Other side measurements (Complex array)
    k - number of results per measurement for this side (Integer)
    m - number of measurements for this side (Integer)
    side - 'Alice' or 'Bob' (String)

    Output:
    P - Problem created
    measurements_new - Measurements created
    D - Value of Bell's inequality 
    solution - Result of the problem 
    '''

    k_o = int(measurements.shape[1])
    m_o = int(measurements.shape[0]/k_o)

    #Creating the problem
    P = pic.Problem()

    #Creating the optimization variables
    measurements_side = [pic.HermitianVariable('Measurements_'+side+'[{}]'.format(i),(dimension,dimension)) for i in range(k*m)]

    #Creating the constraints
    P.add_list_of_constraints([measurements_side[i]>>0 for i in range(k*m)])

    ident = [pic.sum([measurements_side[m*j+i] for i in range(k)]) for j in range(m)]

    P.add_list_of_constraints([ident[j]==np.eye(dimension,dtype='complex') for j in range(m)]) 

    if side=='Alice':
        trace = [pic.trace((measurements_side[i]@measurements[j])*rho) for i in range(m*k) for j in range(m_o*k_o)]
    elif side=='Bob':
        trace = [pic.trace((measurements[i]@measurements_side[j])*rho) for i in range(m_o*k_o) for j in range(m*k)]
    
    
    D = pic.sum([c[i]*trace[i] for i in range(c.shape[0])]).real

    #Setting the objective
    P.set_objective('max',D)

    #Finding the solution
    solution = P.solve()

    return P, np.array(measurements_side), float(D), solution

def random_pure_state(dimension):
    '''
    Generates a random pure quantum state in Haar measure.
    Takes first column of a Haar-random unitary operator.
    Input: dimension (integer)
    Output: density matrix
    '''

    vector = Haar_random(dimension)[:,:1]

    density_matrix = np.array(vector@vector.conj().transpose())

    return density_matrix

def identity_state(dimension):
    '''
    Generates identity state.
    Input: dimension (integer)
    Output: identity state
    '''

    iden = np.eye(dimension)/dimension

    return iden

def increase_dimension(rho,copies):
    '''
    Generates increase in dimension of the density matrix given by outer product.
    Input: rho (array)
           copies - number of copies of the state (Integer)
            (1 copy = rho@rho
             2 copies = rho@rho@rho)
    Output: density matrix 
    '''

    rho_inc = rho

    for i in range(copies):
        rho_inc = np.kron(rho_inc,rho)

    return rho_inc

def SeeSaw(dim,c,rho,k_A,m_A,k_B,m_B,rounds):
    '''
    Inputs:
    dimension - dimension of the measurements of Alice(Integer)
    c - Bell inequality constants (List of floats)
    rho - target state (Complex array)
    measurements- Other side measurements (Complex array)
    k_A - number of results per measurement for Alice (Integer)
    m_A - number of measurements for Alice (Integer)
    k_B - number of results per measurement for Bob (Integer)
    m_B - number of measurements for Bob (Integer)
    rounds - how many times to run the SewSaw (Integer)

    Output: Maximum value of Bell's inequality
    
    '''

    results = np.zeros(rounds)

    for j in range(rounds):
        D_ant = 0
        i=0
        delta = 1
        
        measurements_A = np.zeros((k_A*m_A,dim,dim),dtype = complex)

        for i in range(m_A):

            random_meas = proj_meas_random(dim)

            for k in range(k_A):

                measurements_A[i*k_A+k] = random_meas[k]

        while delta > 10**(-7) and i<10^4:
            P, measurements_B, D, solution = max_Measurements(dim,c,rho,measurements_A,k_B,m_B,'Bob')
            P, measurements_A, D, solution = max_Measurements(dim,c,rho,measurements_B,k_A,m_A,'Alice')

            delta = np.abs(D_ant-D)
            D_ant = D
            i+=1
        results[j] = D

    return results.max()

def pentakis(PLOT=False):
    #INPUT: PLOT:False/True to plot or not the polyhedron

    #Create the vertices of the polytope
    g_r = (1+np.sqrt(5))/2
    vert = np.array([[0,1,g_r],[0,-1,-g_r],
                     [g_r,0,1],[-g_r,0,-1],
                     [1,g_r,0],[-1,-g_r,0],
                     [0,-1,g_r],[0,1,-g_r],
                     [g_r,0,-1],[-g_r,0,1],
                     [-1,g_r,0],[1,-g_r,0],
                     [1,1,1],[-1,-1,-1],
                     [1,1,-1],[-1,-1,1],
                     [1,-1,1],[-1,1,-1],
                     [-1,1,1],[1,-1,-1],
                     [g_r,1/g_r,0],[-g_r,-1/g_r,0],
                     [0,g_r,1/g_r],[0,-g_r,-1/g_r],
                     [1/g_r,0,g_r],[-1/g_r,0,-g_r],
                     [-g_r,1/g_r,0],[g_r,-1/g_r,0],
                     [0,-g_r,1/g_r],[0,g_r,-1/g_r],
                     [1/g_r,0,-g_r],[-1/g_r,0,g_r]])

    vert  = normalize(vert)
    #Create the measurements of each vertex
    m_k = vert.shape[0]
    medicoes = np.zeros([m_k,2,2], dtype=complex)

    for i in range(m_k):
        med_00 = (1+vert[i][2])/2
        med_01 = (vert[i][0]-vert[i][1]*1j)/2
        med_10 = (vert[i][0]+vert[i][1]*1j)/2
        med_11 = (1-vert[i][2])/2
    
        medicoes[i] = [[med_00,med_01],[med_10,med_11]]
    
    #Construct the polyhedron
    hull = ConvexHull(vert)
    #Find the Insphere radius
    r = np.min(np.abs(hull.equations[:, -1]))

    #Plot
    if PLOT == True:
        polys = Poly3DCollection([hull.points[simplex] for simplex in hull.simplices])

        # Good choice of colors: 'deeppink' and 'hotpink'
        polys.set_edgecolor('blue')
        polys.set_linewidth(.8)
        polys.set_facecolor('azure')
        polys.set_alpha(.25)
        
        #Build the insphere
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = r*np.cos(u)*np.sin(v)
        y = r*np.sin(u)*np.sin(v)
        z = r*np.cos(v)
        
        #Build the Bloch sphere
        x_uni = np.cos(u)*np.sin(v)
        y_uni = np.sin(u)*np.sin(v)
        z_uni = np.cos(v)
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")
        #ax = Axes3D(plt.figure())
        #Plot Bloch sphere
        ax.plot_surface(x_uni,y_uni,z_uni,color='lightgray',alpha=.15)
        #Plot insphere
        # Good choice of color: 'lime', alpha=.35
        ax.plot_surface(x,y,z,color='yellow',alpha=.95)
        #Plot polyhedron
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        ax.set_box_aspect([1,1,1])
        ax.set_xticks([-1,0,1])
        ax.set_yticks([-1,0,1])
        ax.set_zticks([-1,0,1])
        ax.view_init(15,75)
        #plt.axis('off')

        ax.add_collection3d(polys)
        #plt.show()
        plt.savefig('poliedro_'+str(m_k)+'.png', transparent=True,dpi=300)
    
    #Return the measurements and the insphere radius
    return medicoes,r

def LHS_threshold(rho_target,rho_sep,politopo):
    '''
    Calculating the locality threshold lower bound by SDP.
    Input: rho_target - target state (Array)
           rho_sep - separable state (Array)
           politopo - 1 to 6 vertices
                      2 to 18 vertices
                      3 to 26 vertices
                      4 to 32 vertices
    Output: Locality threshold
    '''
    # Creating the measurements
    if politopo <= 3:
        medicoes, eta = measurements(politopo)
    else:
        medicoes, eta = pentakis()

    m = int(medicoes.shape[0]/2)

    # Creating the deterministic strategies
    detp = strategies_LHS(m,2)

    # Finding the threshold for locality

    P,solution,q = SDP_LHS(m,2,rho_target,rho_sep,eta,detp,medicoes)

    return q

def G_CHSH():
    '''
    CHSH inequality

    D = \sum_{a,b,x,y}c(a,b,x,y)\text{tr}[(E_{a|x} \otimes F_{b|y})\rho

    D = a_{+,0}xb_{+,0} - a_{+,0}xb_{-,0} + a_{+,0}xb_{+,1} - a_{+,0}xb_{-,1}
        - a_{-,0}xb_{+,0} + a_{-,0}xb_{-,0} - a_{-,0}xb_{+,1} + a_{-,0}xb_{-,1} 
        + a_{+,1}xb_{+,0} - a_{+,1}xb_{-,0} - a_{+,1}xb_{+,1} + a_{+,1}xb_{-,1} 
        - a_{-,1}xb_{+,0} + a_{-,1}xb_{-,0} + a_{-,1}xb_{+,1} - a_{-,1}xb_{-,1}
    '''

    c = np.array([1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,1,-1])
    return c