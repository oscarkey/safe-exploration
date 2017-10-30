# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:13:29 2017

@author: tkoller
"""


from utils_ellipsoid import sum_ellipsoids,ellipsoid_from_rectangle,sum_two_ellipsoids
from utils import compute_remainder_overapproximations, print_ellipsoid
from numpy import sqrt,trace,zeros,diag, eye
#from casadi import *
import warnings
import numpy as np

def onestep_reachability(p_center,gp,k_ff,l_mu,l_sigma,q_shape = None,k_fb = None,
                         c_safety = 1.,verbose = 1,a = None, b = None):
    """ Overapproximate the reachable set of states under affine control law
    
    given a system of the form:
        x_{t+1} = \mathcal{N}(\mu(x_t,u_t), \Sigma(x_t,u_t)),
    where x,\mu \in R^{n_s}, u \in R^{n_u} and \Sigma^{n_s \times n_s} are given bei the gp predictive mean and variance respectively
    we approximate the reachset of a set of inputs x_t \in \epsilon(p,Q)
    describing an ellipsoid with center p and shape matrix Q
    under the control low u_t = Kx_t + k 
    
    Parameters
    ----------
        p_center: n_s x 1 array[float]     
            Center of state ellipsoid        
        gp: SimpleGPModel     
            The gp representing the dynamics            
        k: n_u x 1 array[float]     
            The additive term of the controls
        L_mu: 1d_array of size n_s
            Set of Lipschitz constants on the Gradients of the mean function (per state dimension)
        L_sigm: 1d_array of size n_s
            Set of Lipschitz constants of the predictive variance (per state dimension)
        q_shape: np.ndarray[float], array of shape n_s x n_s, optional
            Shape matrix of state ellipsoid
        K: n_u x n_s array[float], optional     
            The state feedback-matrix for the controls         
        c_safety: float, optional
            The scaling of the semi-axes of the uncertainty matrix 
            corresponding to a level-set of the gaussian pdf.        
        verbose: int
            Verbosity level of the print output            
    Returns:
    -------
        p_new: n_s x 1 array[float]
            Center of the overapproximated next state ellipsoid
        Q_new: np.ndarray[float], array of shape n_s x n_s
            Shape matrix of the overapproximated next state ellipsoid  
    """         
    n_s = np.shape(p_center)[0]
    n_u = np.shape(k_ff)[0]
    
    if a is None:
        a = np.eye(n_s)
        b = np.zeros((n_s,n_u))
        
    if q_shape is None: # the state is a point
        u_p = k_ff
        
        if verbose >0:
            print("\nApplying action:")
            print(u_p)
            
        z_bar = np.vstack((p_center,u_p))
        mu_0, sigm_0 = gp.predict(z_bar.T)
        rkhs_bounds = c_safety * np.sqrt(sigm_0).reshape((n_s,))
        
        print(warnings.warn("Need to verify this!"))
        q_1 = ellipsoid_from_rectangle(rkhs_bounds)
        
        p_lin = np.dot(a,p_center) + np.dot(b,u_p)
        p_1 = p_lin + mu_0.T
        
        if verbose >0:
            print_ellipsoid(p_1,q_1,text="uncertainty first state")
        
        return p_1, q_1
    else: # the state is a (ellipsoid) set
        if verbose > 0:
            print("????")
            print(np.shape(q_shape))
            print_ellipsoid(p_center,q_shape,text="initial uncertainty ellipsoid")
        ## compute the linearization centers
        x_bar = p_center   # center of the state ellipsoid
        u_bar = k_ff   # u_bar = K*(u_bar-u_bar) + k = k
        z_bar = np.vstack((x_bar,u_bar))
        
        if verbose >0:
            print("\nApplying action:")
            print(u_bar)
        ##compute the zero and first order matrices
        mu_0, sigm_0 = gp.predict(z_bar.T)
        
        if verbose > 0:
            print_ellipsoid(mu_0,diag(sigm_0.squeeze()),text="predictive distribution")
            
        jac_mu = gp.predictive_gradients(z_bar.T)
        a_mu = jac_mu[0,:,:n_s]
        b_mu = jac_mu[0,:,n_s:]
         
        ## reach set of the affine terms
        H = a + a_mu + np.dot(b_mu+b,k_fb)
        p_0 = mu_0.T  + np.dot(a,x_bar) + np.dot(b,u_bar)
        
        Q_0 = np.dot(H,np.dot(q_shape,H.T))
        
        if verbose > 0:
            print_ellipsoid(p_0,Q_0,text="linear transformation uncertainty")
        ## computing the box approximate to the lagrange remainder
        
        #lb_mean,ub_mean = compute_bounding_box_lagrangian(q_shape,L_mu,K,k,order = 2,verbose = verbose)
        #lb_sigm,ub_sigm = compute_bounding_box_lagrangian(q_shape,L_sigm,K,k,order = 1,verbose = verbose)
        ub_mean, ub_sigma = compute_remainder_overapproximations(q_shape,k_fb,l_mu,l_sigma)
        
        b_sigma_eps = c_safety*(np.sqrt(sigm_0) + ub_sigma) 
        print(np.shape(b_sigma_eps))
        Q_lagrange_sigm = ellipsoid_from_rectangle(b_sigma_eps.squeeze())   
        p_lagrange_sigm = zeros((n_s,1))
        
        if verbose > 0:
            print_ellipsoid(p_lagrange_sigm,Q_lagrange_sigm,text="overapproximation lagrangian sigma")
    
        print(np.shape(ub_mean))
        Q_lagrange_mu = ellipsoid_from_rectangle(ub_mean)
        p_lagrange_mu = zeros((n_s,1))
        
        if verbose > 0:
            print_ellipsoid(p_lagrange_mu,Q_lagrange_mu,text="overapproximation lagrangian mu")
        
        p_sum_lagrange,Q_sum_lagrange = sum_two_ellipsoids(p_lagrange_sigm,Q_lagrange_sigm,p_lagrange_mu,Q_lagrange_mu)
        
        p_1 , q_1 = sum_two_ellipsoids(p_sum_lagrange,Q_sum_lagrange,p_0,Q_0) 
        
        if verbose > 0:
            print_ellipsoid(p_new,Q_new,text="accumulated uncertainty current step")
            
            print("volume of ellipsoid summed individually")
            print(np.linalg.det(np.linalg.cholesky(q_1)))

        
        return p_1,q_1
        
        
def multistep_reachability(p_0,gp,K,k,L_mu,L_sigm,q_0 = None, c_safety = 1.,verbose = 1,a = None, b= None):
    """ Ellipsoidal overapproximation of a probabilistic safe set after multiple actions
    
    Overapproximate the region containing a pre-specified percentage of the probability
    mass of the system after n actions are applied to the system. The overapproximating
    set is given by an ellipsoid.
    
    Parameters
    ----------
    
    p_0: n_s x 1 array[float]     
            Center of state ellipsoid        
    gp: SimpleGPModel     
        The gp representing the dynamics            
    K: n x n_u x n_s array[float]     
        The state feedback-matrices for the controls at each time step        
    k: n x n_u array[float]     
        The additive term of the controls at each time step
    L_mu: 1d_array of size n_s
        Set of Lipschitz constants on the Gradients of the mean function (per state dimension)
    L_sigm: 1d_array of size n_s
        Set of Lipschitz constants of the predictive variance (per state dimension)
    q_shape: np.ndarray[float], array of shape n_s x n_s, optional
        Shape matrix of the initial state ellipsoid
    c_safety: float, optional
        The scaling of the semi-axes of the uncertainty matrix 
        corresponding to a level-set of the gaussian pdf.        
    verbose: int
        Verbosity level of the print output            
    
    """
    n, n_u, n_s = np.shape(K)
    p_all = np.empty((n,n_s))
    q_all = np.empty((n,n_s,n_s))
    
    ## compute the reachable set in the first time step
    K_0 = K[0]
    k_0 = k[0,:,None]
    p_new,q_new = onestep_reachability(p_0,gp,k_0,L_mu,L_sigm,q_0,None,c_safety,verbose,a,b)
    p_all[0] = p_new.T
    q_all[0] = q_new
    
    ## iteratively compute it for the next steps
    for i in range(1,n):
        p_new,q_new = onestep_reachability(p_new,gp,k[i,:,None],L_mu,L_sigm,q_new,K[i],c_safety,verbose,a,b)
        p_all[i] = p_new.T
        q_all[i] = q_new
        
    return p_new, q_new, p_all, q_all
    
     
def lin_ellipsoid_safety_distance(p_center,q_shape,h_mat,h_vec,c_safety = 1.0):
    """ Compute the distance between eLlipsoid and polytope
    
    Evaluate the distance of an  ellipsoid E(p_center,q_shape), to a polytopic set
    of the form:
        h_mat * x <= h_vec.
        
    Parameters
    ----------
    p_center: n_s x 1 array[float]
        The center of the state ellipsoid
    q_shape: n_s x n_s array[float]
        The shape matrix of the state ellipsoid
    h_mat: m x n_s array[float]
        The shape matrix of the safe polytope (see above)
    h_vec: m x 1 array[float]
        The additive vector of the safe polytope (see above)
        
    Returns
    -------
    d_safety: 1darray[float] of length m
        The distance of the ellipsoid to the polytope. If d < 0 (elementwise),
        the ellipsoid is inside the poltyope (safe), otherwise safety is not guaranteed.
    """
    
    m, n_s = np.shape(h_mat)
    assert np.shape(p_center) == (n_s,1), "p_center has to have shape n_s x 1"
    assert np.shape(q_shape) == (n_s,n_s), "q_shape has to have shape n_s x n_s" 
    assert np.shape(h_vec) == (m,1), "q_shape has to have shape m x 1" 
    
    d_center = np.dot(h_mat,p_center)
    d_shape  = c_safety * np.sqrt(np.sum(np.dot(q_shape,h_mat.T)*h_mat.T,axis = 0)[:,None]) ## MISSING SQRT
    d_safety = d_center + d_shape - h_vec
    
    return d_safety
    