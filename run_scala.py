import numpy as np
import kernel_kmeans as kkm
import sys
import ours_limit
import vl_method
import grid_search_method

REPS = 2

input_path = sys.argv[1]
output_path = sys.argv[2]

# Read data
X_raw = np.loadtxt(input_path+'.csv', delimiter=',')
y_raw = np.loadtxt(input_path + '_labels.csv')

k=len(np.unique(y_raw))
N=X_raw.shape[0]
print(input_path)
print(X_raw.shape, y_raw.shape, k)

stds = np.std(X_raw,axis=0)
stds[stds==0]=1
X_raw = X_raw/stds

sizes = [1000,2000,4000,8000]

# As explained in the paper, we choose the number of tries to match grid search
TRIES = 13

# Run the algorithms
for i in range(REPS):

    for size in sizes:

        sample = np.random.choice(np.arange(X_raw.shape[0]), size, replace=False)
        X = X_raw[sample,:]
        y = y_raw[sample]
                
        D = kkm.compute_square_distance_matrix(X)

        # Used for the cNNC cost
        argsort_D = np.argsort(D, axis=1)

        # Use the same initialization for all
        pre_indicators = np.random.choice(np.arange(k), X.shape[0])

        print('\n Ours')
        results = ours_limit.run(X, y, D, k, argsort_D, pre_indicators=pre_indicators, second_pass=True, tries=TRIES)
        with open(output_path, 'a') as out:
            out.write('{};{};{};{};{};{};{};{}\n'.format(input_path, 'ours', results['sigmas'], results['nmis'], results['nncs'], results['times'], results['total_time'], size))

        print('Time sums: {}'.format(np.sum(results['times'])))
        print('Total time: {}'.format(results['total_time']))

        print('\n MKNN')
        results = vl_method.run(X, y, D, k, argsort_D, pre_indicators=pre_indicators, tries=TRIES)
        with open(output_path, 'a') as out:
            out.write('{};{};{};{};{};{};{};{}\n'.format(input_path, 'mknn', results['sigmas'], results['nmis'], results['nncs'], results['times'], results['total_time'], size))

        print('Time sums: {}'.format(np.sum(results['times'])))
        print('Total time: {}'.format(results['total_time']))

        print('\n Grid search')
        results = grid_search_method.run(X, y, D, k, argsort_D, pre_indicators=pre_indicators, tries=10)
        with open(output_path, 'a') as out:
            out.write('{};{};{};{};{};{};{};{}\n'.format(input_path, 'grid_search', results['sigmas'], results['nmis'], results['nncs'], results['times'], results['total_time'], size))

        print('Time sums: {}'.format(np.sum(results['times'])))
        print('Total time: {}'.format(results['total_time']))
