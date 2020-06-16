import numpy as np
import time
import kernel_kmeans as kkm
import collections
from sklearn import metrics
import sys
        

def run(X, y, D, k, argsort_D, pre_indicators=None, second_pass=True, tries=10):
    
    N=X.shape[0]

    s=kkm.get_percentile_pairwise_distance(X, q=1)
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

    sigma = np.copy(s)      

    for _ in range(tries):

        start = time.time()
        sigma,K = kkm.square_search_fast(k, K, pre_indicators, sigma, precision=1e-2, max_depth=2)
        indicators = kkm.kernel_kmeans(k, K, cluster_indicators=indicators)
        end = time.time()

        nmi = metrics.normalized_mutual_info_score(y, indicators)
        nnc = kkm.compute_nearest_neighbour_cost(D, k, indicators, argsort_D)

        nmis.append(nmi)
        nncs.append(nnc)
        times.append(end-start)
        sigmas.append(sigma)
    
    global_end = time.time()
    total_time = global_end-global_start
    
    results = {}
    results['sigmas'] = sigmas
    results['nmis'] = nmis
    results['nncs'] = nncs
    results['times'] = times
    results['total_time'] = total_time
    
    return results
