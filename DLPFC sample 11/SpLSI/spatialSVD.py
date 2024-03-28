import sys
import numpy as np
from numpy.linalg import norm, svd, solve, qr
import networkx as nx

from SpLSI.utils import *
sys.path.append('./SpLSI/pycvxcluster/src/')
import pycvxcluster.pycvxcluster
# use pycvxcluster from "https://github.com/dx-li/pycvxcluster/tree/main"

from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score


from SpLSI import generate_topic_model as gen_model ### for knn interpolation


from scipy.spatial import Delaunay



def spatialSVD(
        D,
        K, 
        df,
        weights,
        lambd_fixed,
        lamb_start,
        step_size,
        grid_len,
        eps,
        verbose,
        method
):
    X = D.T
    n = X.shape[0]
    p = X.shape[1]
    G, mst, path = generate_mst(df, weights, n)  ### G is the k-nn graph in our case
    srn, fold1, fold2 = get_folds(mst, path, n)
    folds = {0:fold1, 1:fold2}

    lambd_grid = (lamb_start + step_size*np.arange(grid_len)).tolist()   ### equally-spaced grid
    #lambd_grid = (lamb_start*np.power(step_size, np.arange(grid_len))).tolist()  ### exponentially-spaced grid
    lambd_grid.insert(0, 1e-06)

    
    
     

    lambd_errs = {'fold_errors': {}, 'final_errors': []}
    
    U0, L0, V0 = trunc_svd(X, K)
    
    
    for j in folds.keys():
        fold = folds[j]

        ### interpolating method
        X_tilde = interpolate_X_weights(X, mst, folds, j, path, mst, srn, weights)
        #X_tilde = interpolate_X_deg(X, G, folds, j, path, mst, srn, weights, node_deg = 6)



        
        X_j = X[fold,:]  

        errs = []
        
        for lambd in lambd_grid:
            niter = 0
            thres = 1
            U, L, V = trunc_svd(X_tilde, K)
            while thres > eps:
                UUT_old = np.dot(U, U.T)
                VVT_old = np.dot(V, V.T)

                #U, lambd, lambd_errs = update_U_tilde(X, V, G, weights, folds, path, mst, srn, lambd_grid, n, K)
                XV_tilde = np.dot(X_tilde, V)
                
                ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd, verbose=0)
                ssnal.fit(X=XV_tilde, weight_matrix=weights, save_centers=True)
                UL_hat_full = ssnal.centers_.T
                Q, R = qr(UL_hat_full)

                U = Q
                V, L = update_V_L_tilde(X_tilde, U)
    
                
                UUT = np.dot(U, U.T)
                VVT = np.dot(V, V.T)
                thres = np.max([norm(UUT-UUT_old)**2, norm(VVT-VVT_old)**2])
                niter += 1
                
                print(f"Error is {thres}")
            
            print(f"SpatialSVD ran for {niter} steps for lambda = {lambd} and fold {j}.")
            
             
            E = U @ L @ V.T 

            ### Compute reconstruction errors
            err = norm(X_j-E[fold,:])
            errs.append(err) 
            

        lambd_errs['fold_errors'][j] = errs
    
    cv_errs = np.add(lambd_errs['fold_errors'][0],lambd_errs['fold_errors'][1])
    lambd_cv = lambd_grid[np.argmin(cv_errs)]

            

    return U, L, lambd_cv, lambd_errs



#### This function is for computing for a fixed lambd
def spatialSVD2(
        D,
        K, 
        weights,
        lambd,
        eps    
):
    X = D.T
    n = X.shape[0]
    p = X.shape[1]
    
    niter = 0
    thres = 1
    U, L, V = trunc_svd(X, K)
    while thres > eps:
        UUT_old = np.dot(U, U.T)
        VVT_old = np.dot(V, V.T)

        #U, lambd, lambd_errs = update_U_tilde(X, V, G, weights, folds, path, mst, srn, lambd_grid, n, K)
        XV = np.dot(X, V)
                
        ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd, verbose=0)
        ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
        UL_hat_full = ssnal.centers_.T
        Q, R = qr(UL_hat_full)

        U = Q
        V, L = update_V_L_tilde(X, U)
    
                
        UUT = np.dot(U, U.T)
        VVT = np.dot(V, V.T)
        thres = np.max([norm(UUT-UUT_old)**2, norm(VVT-VVT_old)**2])
        niter += 1
                
        print(f"Error is {thres}")  
    return U, V
    
##################


def update_U_tilde_nocv(X, V, weights, lambd):
    XV = np.dot(X, V)
    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd, verbose=0)
    ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)

    U_hat = ssnal.centers_.T
    E = np.dot(U_hat, V.T)
    err = norm(X-E)

    Q, R = qr(U_hat)
    return Q, lambd, err


def update_U_tilde(X, V, G, weights, folds, path, mst, srn, lambd_grid, n, K):
    #UL_best_comb = np.zeros((n,K))
    #lambds_best = []
    lambd_errs = {'fold_errors': {}, 'final_errors': []}
    XV = np.dot(X, V)

    for j in folds.keys():
        fold = folds[j]
        X_tilde = interpolate_X(X, G, folds, j, path, mst, srn)
        # print((X_tilde[fold[j],:]==X[fold[j],:]).sum()) # shouldn't be large
        #assert((X_tilde[fold[j],:]==X[fold[j],:]).sum()<=1)
        XV_tilde = np.dot(X_tilde, V)
        X_j = X[fold,:]

        errs = []
        #best_err = float("inf")
        #UL_best = None
        #lambd_best = 0
        
        for lambd in lambd_grid:
            ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd, verbose=0)
            ssnal.fit(X=XV_tilde, weight_matrix=weights, save_centers=True)
            UL_hat = ssnal.centers_.T
            E = np.dot(UL_hat, V.T)
            err = norm(X_j-E[fold,:])
            errs.append(err)
            #if err < best_err:
                #lambd_best = lambd
                #UL_best = UL_hat
                #best_err = err
        lambd_errs['fold_errors'][j] = errs
        #UL_best_comb[fold,:] = UL_best[fold,:]
        #lambds_best.append(lambd_best)

    cv_errs = np.add(lambd_errs['fold_errors'][0],lambd_errs['fold_errors'][1])
    lambd_cv = lambd_grid[np.argmin(cv_errs)]

    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0)
    ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
    UL_hat_full = ssnal.centers_.T

    Q, R = qr(UL_hat_full)
    return Q, lambd_cv, lambd_errs



def update_V_L_tilde(X, U_tilde):
    V_hat = np.dot(X.T, U_tilde)
    Q, R = qr(V_hat)
    return Q, R







