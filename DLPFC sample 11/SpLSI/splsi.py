import numpy as np
from numpy.linalg import norm, svd, solve, qr
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
import networkx as nx

import cvxpy as cp
from cvxpy import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem

from SpLSI import generate_topic_model as gen_model
from SpLSI.utils import *
from SpLSI.spatialSVD import *
# use pycvxcluster from "https://github.com/dx-li/pycvxcluster/tree/main"


class SpLSI(object):

    def __init__(
            self,
            lambd = None, 
            lamb_start = 0.01,
            step_size = 1.15,
            grid_len = 100,
            eps = 1e-06,
            method = "spatial",
            step = "two-step",
            return_anchor_docs = True,
            verbose = 1
    ):
        """
        Parameters
        -----------

        """
        self.lambd = lambd
        self.lamb_start = lamb_start
        self.step_size = step_size
        self.grid_len = grid_len
        self.eps = eps
        self.method = method
        self.return_anchor_docs = return_anchor_docs
        self.verbose = verbose
        self.step = step
        
    def fit(self, 
            D, 
            K, 
            df, 
            weights
    ):
        if self.method != "spatial":
            self.U, _, _ = trunc_svd(D.T, K)
            #print("Running vanilla SVD...")
        
        else:
            #print("Running spatial SVD...")
            self.U, self.L, self.lambd, self.lambd_errs = spatialSVD(
                                D, 
                                K, 
                                df, 
                                weights,
                                self.lambd,
                                self.lamb_start,
                                self.step_size,
                                self.grid_len,
                                self.eps,
                                self.verbose,
                                self.step
        )
        #print("Running SPOC...")
        #n = D.shape[1]
        #J = []
        #S = self.preprocess_U(self.U, K).T
        #for t in range(K):
            #maxind = np.argmax(norm(S, axis=0))
            #s = np.reshape(S[:, maxind], (K, 1))
            #S1 = (np.eye(K) - np.dot(s, s.T) / norm(s)**2).dot(S)
            #S = S1
            #J.append(maxind) 

        #H_hat = self.U[J, :]
        #self.W_hat = self.get_W_hat(self.U, H_hat, n, K)

        #if self.return_anchor_docs:
            #self.anchor_indices = J

        return self

    def fit2(self, 
            D, 
            K, 
            weights,
            lambd
    ):

        self.U, self.V = spatialSVD2(
                                D, 
                                K, 
                                weights,
                                lambd,
                                self.eps
                                
        )
        return self
        
    @staticmethod
    def preprocess_U(U, K):
        for k in range(K):
            if U[0, k] < 0:
                U[:, k] = -1 * U[:, k]
        return U
    
    @staticmethod
    def get_W_hat(U, H, n, K):
        Theta = Variable((n,K))
        constraints = [
            cp.sum(Theta[i, :]) == 1 for i in range(n)
        ]
        constraints += [
            Theta[i, j] >= 0 for i in range(n)
            for j in range(K)
        ]
        obj = Minimize(cp.norm(U - Theta @ H, 'fro'))
        prob = Problem(obj, constraints)
        prob.solve()
        return np.array(Theta.value)

class SpLSI_two_sides(object):

    def __init__(
            self,
            lambd = None, 
            lamb_start = 0.01,
            step_size = 1.15,
            grid_len = 100,
            eps = 1e-06,
            method = "spatial",
            step = "two-step",
            tau = 0.1,
            return_anchor_docs = True,
            verbose = 1
    ):
        """
        Parameters
        -----------

        """
        self.lambd = lambd
        self.lamb_start = lamb_start
        self.step_size = step_size
        self.grid_len = grid_len
        self.eps = eps
        self.method = method
        self.return_anchor_docs = return_anchor_docs
        self.verbose = verbose
        self.step = step

        self.tau = tau
        
    def fit(self, 
            D, 
            K, 
            df, 
            weights
    ):
        if self.method != "spatial":
            self.U, _, _ = trunc_svd(D.T, K)
            #print("Running vanilla SVD...")
        
        else:
            #print("Running spatial SVD...")
            self.U, self.L, self.lambd, self.lambd_errs = spatialSVD_two_sides(
                                D, 
                                K, 
                                df, 
                                weights,
                                self.lambd,
                                self.lamb_start,
                                self.step_size,
                                self.grid_len,
                                self.eps,
                                self.verbose,
                                self.step,
                                self.tau
        )
        #print("Running SPOC...")
        #n = D.shape[1]
        #J = []
        #S = self.preprocess_U(self.U, K).T
        #for t in range(K):
            #maxind = np.argmax(norm(S, axis=0))
            #s = np.reshape(S[:, maxind], (K, 1))
            #S1 = (np.eye(K) - np.dot(s, s.T) / norm(s)**2).dot(S)
            #S = S1
            #J.append(maxind) 

        #H_hat = self.U[J, :]
        #self.W_hat = self.get_W_hat(self.U, H_hat, n, K)

        #if self.return_anchor_docs:
            #self.anchor_indices = J

        return self

    def fit2(self, 
            D, 
            K, 
            weights,
            lambd
    ):

        self.U, self.V = spatialSVD_two_sides_2(
                                D, 
                                K, 
                                weights,
                                lambd,
                                tau,
                                self.eps,
                                
        )
        return self
        
    @staticmethod
    def preprocess_U(U, K):
        for k in range(K):
            if U[0, k] < 0:
                U[:, k] = -1 * U[:, k]
        return U
    
    @staticmethod
    def get_W_hat(U, H, n, K):
        Theta = Variable((n,K))
        constraints = [
            cp.sum(Theta[i, :]) == 1 for i in range(n)
        ]
        constraints += [
            Theta[i, j] >= 0 for i in range(n)
            for j in range(K)
        ]
        obj = Minimize(cp.norm(U - Theta @ H, 'fro'))
        prob = Problem(obj, constraints)
        prob.solve()
        return np.array(Theta.value)
    
class SpLSI_knn(object):

    def __init__(
            self,
            lambd = None, 
            lamb_start = 0.01,
            step_size = 1.15,
            grid_len = 100,
            eps = 1e-06,
            method = "spatial",
            step = "two-step",
            G_knn = 0,
            weights_knn = 0,
            return_anchor_docs = True,
            verbose = 1
    ):
        """
        Parameters
        -----------

        """
        self.lambd = lambd
        self.lamb_start = lamb_start
        self.step_size = step_size
        self.grid_len = grid_len
        self.eps = eps
        self.method = method
        self.return_anchor_docs = return_anchor_docs
        self.verbose = verbose
        self.step = step

        self.G_knn = G_knn
        self.weights_knn = weights_knn
    def fit(self, 
            D, 
            K, 
            df, 
            weights
    ):
        if self.method != "spatial":
            self.U, _, _ = trunc_svd(D.T, K)
            #print("Running vanilla SVD...")
        
        else:
            #print("Running spatial SVD...")
            self.U, self.L, self.lambd, self.lambd_errs = spatialSVD_knn(
                                D, 
                                K, 
                                df, 
                                weights,
                                self.lambd,
                                self.lamb_start,
                                self.step_size,
                                self.grid_len,
                                self.eps,
                                self.verbose,
                                self.step,
                                self.G_knn,
                                self.weights_knn
        )
        #print("Running SPOC...")
        #n = D.shape[1]
        #J = []
        #S = self.preprocess_U(self.U, K).T
        #for t in range(K):
            #maxind = np.argmax(norm(S, axis=0))
            #s = np.reshape(S[:, maxind], (K, 1))
            #S1 = (np.eye(K) - np.dot(s, s.T) / norm(s)**2).dot(S)
            #S = S1
            #J.append(maxind) 

        #H_hat = self.U[J, :]
        #self.W_hat = self.get_W_hat(self.U, H_hat, n, K)

        #if self.return_anchor_docs:
            #self.anchor_indices = J

        return self

   