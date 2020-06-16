import numpy as np
import time
import kernel_kmeans as kkm
import collections
from sklearn import metrics
import sys


def search(K, y, D, k, argsort_D, initial_sigma, upper_bound, indicators, max_depth=4):

    sigmas = []
    nmis = []
    nncs = []
    times = []
    
    converged = False
    sigma=initial_sigma

    while not converged:
        preK = np.copy(K)
        pre_sigma = np.copy(sigma)
        pre_indicators = np.copy(indicators)

        start = time.time()
        sigma,K = kkm.square_search_fast(k, preK, pre_indicators, sigma, precision=1e-2, max_depth=max_depth)

        indicators = kkm.kernel_kmeans(k, K, cluster_indicators=pre_indicators)
        end = time.time()

        nmi = metrics.normalized_mutual_info_score(y, indicators)
        nnc = kkm.compute_nearest_neighbour_cost(D, k, indicators, argsort_D)

        nmis.append(float(nmi))
        nncs.append(float(nnc))
        sigmas.append(float(sigma))
        times.append(end-start)

        converged = ((pre_indicators==indicators).all()) and sigma >= upper_bound
    
    return nmis, nncs, sigmas, times
        

def run(X, y, D, k, argsort_D, pre_indicators=None, second_pass=True):

    N=X.shape[0]

    # The lower bound for sigma is quite pessimistic, so we can afford starting slightly above 
    s = kkm.get_minimum_pairwise_distance(X, D=D)
    sigmas = [s]

    global_start = time.time()

    start = time.time()
    K = kkm.compute_kernel_matrix(X, sigma=s)

    if pre_indicators is None:
        pre_indicators = np.random.choice(np.arange(k), N)
    indicators = kkm.kernel_kmeans(k, K, cluster_indicators=pre_indicators)
    end = time.time()
    
    nmis = [metrics.normalized_mutual_info_score(y, indicators)]
    nncs = [kkm.compute_nearest_neighbour_cost(D, k, indicators, argsort_D)]

    times = [end-start]

    depths = [1,2]
    
    lower = s
    upper = 1e6
    
    for depth in depths:

        pnmis, pnncs, psigmas, ptimes = search(K, y, D, k, argsort_D, lower, upper, indicators, max_depth=depth)
        best = np.argmin(pnncs)
        best_last = len(pnncs) - np.argmin(pnncs[::-1]) - 1

        lower_i = np.max([0, best])
        lower = psigmas[lower_i]
        upper_i = np.min([len(psigmas)-1, best_last+1])
        upper = psigmas[upper_i]

        nmis.extend(pnmis)
        nncs.extend(pnncs)
        times.extend(ptimes)
        sigmas.extend(psigmas)

        start = time.time()
        K = kkm.compute_kernel_matrix(X, sigma=lower)
        if pre_indicators is None:
            pre_indicators = np.random.choice(np.arange(k), N)
        indicators = kkm.kernel_kmeans(k, K, cluster_indicators=pre_indicators)
        end = time.time()
        times.append(end-start)

    global_end = time.time()
    total_time = global_end-global_start
    
    results = {}
    results['sigmas'] = sigmas
    results['nmis'] = nmis
    results['nncs'] = nncs
    results['times'] = times
    results['total_time'] = total_time
    
    return results
