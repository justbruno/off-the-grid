import numpy as np
import collections

def gaussian_kernel(x,y, sigma=1.):
    d = x-y
    return np.exp(-d.dot(d)/sigma)

def compute_square_distance_matrix(X):
    Xn = np.repeat([np.linalg.norm(X, axis=1)**2], X.shape[0], axis=0)
    W = Xn + Xn.T - 2*X.dot(X.T)    
    return W-np.diag(np.diag(W))

def compute_kernel_matrix(X, sigma=1.):
    W = compute_square_distance_matrix(X)
    W[W<1e-12]=0
    return np.exp(-W/sigma)

def get_minimum_pairwise_distance(X, D=None):
    if D is None:
        Xn = np.repeat([np.linalg.norm(X, axis=1)**2], X.shape[0], axis=0)
        W = Xn + Xn.T - 2*X.dot(X.T)
    else:
        W = np.copy(D)
    for i in range(X.shape[0]):
        W[i,i] = np.inf
    return np.min(W)

def get_percentile_pairwise_distance(X, q=10):
    Xn = np.repeat([np.linalg.norm(X, axis=1)**2], X.shape[0], axis=0)
    W = Xn + Xn.T - 2*X.dot(X.T)
    for i in range(X.shape[0]):
        W[i,i] = np.inf
    return np.percentile(W, q)


def compute_kernel_kmeans_cost(K, cluster_indicators, k=None):
    if k == None:
        k = len(np.unique(cluster_indicators))
    # Number of points per cluster
    Ns = collections.Counter(cluster_indicators)    
    # Array of arrays with the points in each cluster
    clusters = [(cluster_indicators==i).nonzero()[0] for i in range(k)]
    affinities = np.vstack([np.sum(K[clusters[i],:],0)*2./Ns[i] for i in range(k)])        
    tightnesses = np.vstack([np.sum(K[clusters[i],:][:,clusters[i]])/(Ns[i]**2) for i in range(k)])
    return np.sum(np.max(affinities-tightnesses, axis=0))


def compute_nearest_neighbour_cost(D, k, indicators, argsort_D, scale='harmonic', L=None):
    N = D.shape[0]
    if L is None:
        L = N
    total_cost = 0
    v=indicators.reshape((N,1))+1
    Xi = np.where(v/v.T==1, 0, 1) # Indicator match matrix
    Xi = Xi[np.arange(N), argsort_D.T].T # Sort by distance
    if scale == 'harmonic':
        a = np.arange(1,L)**2 # Scaling factor
        C = np.log(L-1) + np.euler_gamma + 1/(2*L-2)
    elif scale == 'geometric':
        a = np.arange(1,L)*k**(np.arange(1,L))  # Scaling factor
        r = 1/k
        C = (1-r**L)/(1-r)-1
    cXi = np.cumsum(Xi,axis=1)[:,1:L]
    cums = cXi/a
    cums[np.isnan(cums)]=0

    Ns = collections.Counter(indicators)    
    Ni = 1/np.array([Ns[indicators[i]] for i in range(N)])
    Ni[np.isinf(Ni)]=1    

    precost = np.sum(Ni.dot(cums))/C
    missing = k-len(np.unique(indicators))
    precost += missing
    
    return precost/k


##################################################
# Our kernel k-means implementation

def kernel_kmeans(k, K, weights=None, cluster_indicators=None, max_iterations=100):
    N = K.shape[0]
    if cluster_indicators is None:
        cluster_indicators = np.random.choice(np.arange(k), N)

    if weights is None:
        weights = np.ones(N)
    pre_ci = np.copy(cluster_indicators)
    converged = False
    iterations = 0

    while not converged:
        new_ci = []
        
        # Number of points per cluster
        Ns = collections.Counter(cluster_indicators)
        
        # Array of arrays with the points in each cluster
        clusters = [(cluster_indicators==i).nonzero()[0] for i in range(k)]

        Ws = [np.sum(weights[clusters[i]]) for i in range(k)]

        affinities = np.vstack([np.sum(K[clusters[i],:],0)*2./Ws[i] if Ws[i]>0 else np.zeros((1,N)) for i in range(k)])
        tightnesses = np.vstack([np.sum(K[clusters[i],:][:,clusters[i]])/(Ws[i]**2) if Ws[i]>0 else 0  for i in range(k)])           
            
        # Build point-to-centroid distance matrix and compute new cluster indicators
        similarities = affinities-tightnesses
        new_ci = np.argmax(similarities,0)

        # Check if there were no changes to the indicators
        if (new_ci == cluster_indicators).all():
            converged = True

        cluster_indicators = np.copy(new_ci)
        iterations += 1
        
        if iterations >= max_iterations:
            converged = True
    return cluster_indicators

##################################################
# The element-wise matrix exponentiation algorithm

def square_search_fast(k, Ku, indicators, s, precision=1e-3, max_depth=10):

    intial_s = np.copy(s)
    final_sigma = intial_s
    numerator=1
    denominator=1

    K = np.copy(Ku)
    
    done = False
    changes=False

    # If there are never any changes we need to return 1/max_depth
    any_changes = False
      
    while not done:
        numerator *= 2
        denominator *= 2
        Ku **= .5
        
        Z = Ku > 0
        if not changes:
            numerator -= 1
            K = np.multiply(1/Ku, K, where=Z, out=K)
        else:
            numerator += 1
            K = np.multiply(Ku, K)
            any_changes = True
            
        new_indicators = kernel_kmeans(k, K, cluster_indicators=indicators, max_iterations=1)
        changes = (new_indicators != indicators).any()
        if changes:
            final_sigma = intial_s*denominator/numerator

        exponent = numerator/denominator
        done = np.log2(denominator) >= max_depth

    # If no changes were ever made, we just return the max sigma we can get with current depth
    if not any_changes:
        final_sigma = intial_s*denominator/numerator

    # We keep the last K that induced changes
    elif not changes:
        K = np.multiply(1/Ku, K, where=Z, out=K)

    return final_sigma, K


def binary_search(X, k, K, indicators, s, precision=1e-3):
    
    upper = 1e6
    lower = s
    previous = lower
    current = (upper+lower)/2

    K = compute_kernel_matrix(X, sigma=current)

    new_indicators = kernel_kmeans(k, K, cluster_indicators=indicators, max_iterations=1)
    changes = (new_indicators != indicators).any()

    done = False
    
    while not done:        
        if not changes:
            lower = current
        else:
            upper = current
        current = (upper+lower)/2            

        K = compute_kernel_matrix(X, sigma=current)
        previous = current
        
        new_indicators = kernel_kmeans(k, K, cluster_indicators=indicators, max_iterations=1)
        changes = (new_indicators != indicators).any()
        
        done = upper-lower < precision
        
    return upper


