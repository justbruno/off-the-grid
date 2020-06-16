import numpy as np
import time
import kernel_kmeans as kkm
import collections
from sklearn import metrics
import sys
import baselines

def run(X, y, D, k, argsort_D, pre_indicators=None, tries=None):

    N=X.shape[0]

    global_start = time.time()

    sigmas = []
    nmis = []
    nncs = []
    times = []

    if tries == None:
        tries = int(2*(np.log(N)+1))
    k_values = np.arange(1, np.min([tries+1, N]))
    
    for k_value in k_values:
        start = time.time()
        s=baselines.mean_sdist_to_knn(X, sq_distance_matrix=D, params={'k':k_value})        
        sigmas.append(s)

        K = kkm.compute_kernel_matrix(X, sigma=s)

        if pre_indicators is None:
            pre_indicators = np.random.choice(np.arange(k), N)
        indicators = kkm.kernel_kmeans(k, K, cluster_indicators=pre_indicators)
        end = time.time()
        
        nmi = metrics.normalized_mutual_info_score(y, indicators)
        nnc = kkm.compute_nearest_neighbour_cost(D, k, indicators, argsort_D)

        nmis.append(float(nmi))
        nncs.append(float(nnc))
        sigmas.append(float(s))
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
