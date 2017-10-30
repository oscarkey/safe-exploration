# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:18:58 2017

@author: tkoller
"""

import numpy as np
import casadi as cas
import casadi.tools as castools

from casadi import SX, mtimes, vertcat, diag, sum2, sum1, sqrt,Function, repmat
from utils_casadi import compute_bounding_box_lagrangian,matrix_norm_2, compute_remainder_overapproximations
from utils_ellipsoid_casadi import sum_two_ellipsoids, ellipsoid_from_rectangle
import warnings

def onestep_reachability(p_center,gp,k_ff,l_mu,l_sigma,
                         q_shape = None,k_fb = None, c_safety = 1.,verbose = 1, a = None, b = None):
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
                 
        k_ff: n_u x 1 array[float]     
            The additive term of the controls
        l_mu: 1d_array of size n_s
            Set of Lipschitz constants on the Gradients of the mean function (per state dimension)
        l_sigma: 1d_array of size n_s
            Set of Lipschitz constants of the predictive variance (per state dimension)
        q_shape: np.ndarray[float], array of shape n_s x n_s, optional
            Shape matrix of state ellipsoid
        k_fb: n_u x n_s array[float], optional     
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
        a = SX.eye(n_s)
        b = SX.zeros(n_s,n_u)
        
    if q_shape is None: # the state is a point
    
        u_p = k_ff
        if verbose >0:
            print("\nApplying action:")
            print(u_p)     
        z_bar = vertcat(p_center,u_p)
        mu_new, pred_var = gp.predict_casadi_symbolic(z_bar.T)
        
        p_lin = mtimes(a,p_center) + mtimes(b,u_p)
        p_1 = p_lin + mu_new.T
        
        rkhs_bound = c_safety*sqrt(pred_var.T)
        q_1 = ellipsoid_from_rectangle(rkhs_bound)

        return p_1, q_1
    else: # the state is a (ellipsoid) set

        ## compute the linearization centers
        x_bar = p_center   # center of the state ellipsoid
        u_bar = k_ff   # u_bar = K*(u_bar-u_bar) + k = k
        z_bar = vertcat(x_bar,u_bar)
        
        ##compute the zero and first order matrices
        mu_0, sigm_0, jac_mu = gp.predict_casadi_symbolic(z_bar.T,True)
                   
        a_mu = jac_mu[:,:n_s]
        b_mu = jac_mu[:,n_s:]
         
        ## reach set of the affine terms
        H = a + a_mu + mtimes(b_mu+b,k_fb)
        p_0 = mu_0.T  + mtimes(a,x_bar) + mtimes(b,u_bar)
        
        Q_0 = mtimes(H,mtimes(q_shape,H.T))
        
        ub_mean,ub_sigma = compute_remainder_overapproximations(q_shape,k_fb,l_mu,l_sigma)
        ## computing the box approximate to the lagrange remainder
        #lb_mean,ub_mean = compute_bounding_box_lagrangian(q_shape,L_mu,K,k,order = 2,verbose = verbose)
        #lb_sigm,ub_sigm = compute_bounding_box_lagrangian(q_shape,L_sigm,K,k,order = 1,verbose = verbose)
        Q_lagrange_mu = ellipsoid_from_rectangle(ub_mean)
        p_lagrange_mu = SX.zeros((n_s,1))        
        
        b_sigma_eps = c_safety*(sqrt(sigm_0.T) + ub_sigma)
        Q_lagrange_sigm = ellipsoid_from_rectangle(b_sigma_eps)   
        p_lagrange_sigm = SX.zeros((n_s,1))
          
        
               
        p_sum_lagrange,Q_sum_lagrange = sum_two_ellipsoids(p_lagrange_sigm,Q_lagrange_sigm,p_lagrange_mu,Q_lagrange_mu)
        
        p_1 , q_1 = sum_two_ellipsoids(p_sum_lagrange,Q_sum_lagrange,p_0,Q_0)      
        
        return p_1,q_1
        

def lin_ellipsoid_safety_distance(p_center,q_shape,h_mat,h_vec,c_safety = 1.0):
    """ Compute symbolically the distance between eLlipsoid and polytope in casadi
    
    Evaluate the distance of an  ellipsoid E(p_center,q_shape), to a polytopic set
    of the form:
        h_mat * x <= h_vec.
        
    Parameters
    ----------
    p_center: n_s x 1 array
        The center of the state ellipsoid
    q_shape: n_s x n_s array
        The shape matrix of the state ellipsoid
    h_mat: m x n_s array:
        The shape matrix of the safe polytope (see above)
    h_vec: m x 1 array
        The additive vector of the safe polytope (see above)
        
    Returns
    -------
    d_safety: 1darray of length m
        The distance of the ellipsoid to the polytope. If d < 0 (elementwise),
        the ellipsoid is inside the poltyope (safe), otherwise safety is not guaranteed.
    """
    d_center = mtimes(h_mat,p_center)
    
    d_shape  = c_safety * sqrt(sum1(mtimes(q_shape,h_mat.T)*h_mat.T)).T 
    d_safety = d_center + d_shape - h_vec
    
    return d_safety


def objective(p_all,q_all,p_target,k_ff_all,wx_cost,wu_cost,q_target=None):
    """ 
    
    """    
    n, n_s = np.shape(p_all)
    c = 0
    #for i in range(n-1):
    #    c += mtimes(mtimes(p_all[i,:]-p_target.T,wx_cost),p_all[i,:].T-p_target)
    #    c += mtimes(mtimes(k_ff_all[i,:],wu_cost),k_ff_all[i,:].T)
    c = mtimes(mtimes(p_all[0,:]-p_target.T,wx_cost),p_all[0,:].T-p_target)    
    return c 
    
if __name__ == "__main__":
    p = SX.sym("p",(3,1))
    q = SX.sym("q",(3,3))
    h_mat_safe = np.hstack((np.eye(3,1),-np.eye(3,1))).T
    h_safe = np.array([0.5,0.5])
    f = Function("f",[p,q],[lin_ellipsoid_safety_distance(p,q,h_mat_safe,h_safe)])
    
    print(f(np.zeros((3,1)),0.25*np.eye(3)))



    
    