import numpy as np
import kernel_kmeans as kkm

def compute_sq_distance_matrix(X):
    n = X.shape[0]
    G = X.dot(X.T)
    di = np.diag(G)
    P = np.outer(di,np.ones(n))
    P = P+P.T
    D = P-2*G # Distance matrix
    return D

def mean_sdist_to_knn(X, sq_distance_matrix=None, params={}):
    # Mean distance of a point to its k-th nearest neighbour (Von Luxburg)
    # The kernel in the paper divides by 2sigma^2, so we transform accordingly    
    n = X.shape[0]
    if 'k' in params:
        k = params['k']
    else:
        k = int(np.log(n)+1) #n. of neighbours to consider
    if sq_distance_matrix is None:
        D = 2*compute_sq_distance_matrix(X)
    else:
        D = 2*sq_distance_matrix

    sdists = np.max(np.partition(D, k, axis=1)[:,:k], axis=1)    
    return np.mean(sdists)
