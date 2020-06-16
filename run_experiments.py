import numpy as np
import kernel_kmeans as kkm
import sys
import ours
import vl_method
import grid_search_method

REPS = 1

input_path = sys.argv[1]
output_path = sys.argv[2]

# Read data
X = np.loadtxt(input_path+'.csv', delimiter=',')
y = np.loadtxt(input_path + '_labels.csv')

k=len(np.unique(y))
N=X.shape[0]
print(input_path)
print(X.shape, y.shape, k)

# In the experiments, we scale real data sets, as it is beneficial for kernel k-means
#X = X/np.std(X,axis=0)

D = kkm.compute_square_distance_matrix(X)

# User for the cNNC cost
argsort_D = np.argsort(D, axis=1)

# Run the algorithms
for i in range(REPS):
    # Use the same initialization for all
    pre_indicators = np.random.choice(np.arange(k), N)

    print('\n Ours')
    results = ours.run(X, y, D, k, argsort_D, pre_indicators=pre_indicators, second_pass=True)
    with open(output_path, 'a') as out:
        out.write('{};{};{};{};{};{};{}\n'.format(input_path, 'ours', results['sigmas'], results['nmis'], results['nncs'], results['times'], results['total_time']))

    print('Max NMI: {}'.format(np.max(results['nmis'])))
    print('Min cNCC: {}'.format(np.min(results['nncs'])))
        
    print('Time sums: {}'.format(np.sum(results['times'])))
    print('Avg iteration time: {}'.format(np.mean(results['times'])))
    print('Total time: {}'.format(results['total_time']))

    
    print('\n MKNN')
    # Set number of tries to trade off performance and running time
    results = vl_method.run(X, y, D, k, argsort_D, pre_indicators=pre_indicators)
    with open(output_path, 'a') as out:
        out.write('{};{};{};{};{};{};{}\n'.format(input_path, 'mknn', results['sigmas'], results['nmis'], results['nncs'], results['times'], results['total_time']))

    print('Max NMI: {}'.format(np.max(results['nmis'])))
    print('Min cNCC: {}'.format(np.min(results['nncs'])))

    print('Time sums: {}'.format(np.sum(results['times'])))
    print('Avg iteration time: {}'.format(np.mean(results['times'])))
    print('Total time: {}'.format(results['total_time']))
        
    print('\n Grid search')
    # Set number of tries to trade off performance and running time (will actually do tries+3 tries)
    results = grid_search_method.run(X, y, D, k, argsort_D, pre_indicators=pre_indicators, tries=10)
    with open(output_path, 'a') as out:
        out.write('{};{};{};{};{};{};{}\n'.format(input_path, 'grid_search', results['sigmas'], results['nmis'], results['nncs'], results['times'], results['total_time']))
    
    print('Max NMI: {}'.format(np.max(results['nmis'])))
    print('Min cNCC: {}'.format(np.min(results['nncs'])))

    print('Time sums: {}'.format(np.sum(results['times'])))
    print('Avg iteration time: {}'.format(np.mean(results['times'])))
    print('Total time: {}'.format(results['total_time']))
