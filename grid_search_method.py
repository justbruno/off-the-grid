import numpy as np
import time
import kernel_kmeans as kkm
import collections
from sklearn import metrics
import sys
import baselines

def run(X, y, D, k, argsort_D, pre_indicators=None, tries=10):

    N=X.shape[0]
    sigmas = []

    global_start = time.time()

    nmis = []
    nncs = []
    times = []

    # Will actually do tries+3 tries
    step = 10/tries    
    for s in 10.**np.arange(-6, 6.00000001, step):
        start = time.time()
        
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
