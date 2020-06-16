import numpy as np
import time
import kernel_kmeans as kkm
import sys

REPS = 2

input_path = sys.argv[1]
output_path = sys.argv[2]

# Read data
X = np.loadtxt(input_path+'.csv', delimiter=',')
y = np.loadtxt(input_path + '_labels.csv')

#X = X/np.std(X,axis=0)

k=len(np.unique(y))
N=X.shape[0]
print(input_path)
print(X.shape, y.shape, k)

s=kkm.get_percentile_pairwise_distance(X, q=1)
sigmas = [s]
print('First sigma: {}'.format(s))

K = kkm.compute_kernel_matrix(X, sigma=s)

indicators = kkm.kernel_kmeans(k, K)

PREC=10

square_times = []
square_errors = []
square_sigmas = []

binary_times = []
binary_errors = []
binary_sigmas = []

for _ in range(REPS):

    s=kkm.get_percentile_pairwise_distance(X, q=1)
    K = kkm.compute_kernel_matrix(X, sigma=s)    
    indicators = kkm.kernel_kmeans(k, K)

    for i in range(10):

        Ku = np.copy(K)
        sigma_true = kkm.binary_search(X, k, Ku, indicators, s, precision=1e-9)
        print('True: {}'.format(sigma_true))

        start = time.time()
        sigma_square,_ = kkm.square_search_fast(k, Ku, indicators, s, max_depth=PREC)
        end = time.time()
        square_times.append(end-start)
        square_errors.append((np.abs(sigma_true-sigma_square)))
        square_sigmas.append(sigma_square)
        print('By square: {}'.format(sigma_square))
        print('Time: {}'.format(end-start))
        print('Error: {}'.format((np.abs(sigma_true-sigma_square))))

        Ku = np.copy(K) # Unnecessary copy if everything's correct under the hood, but better safe than sorry
        start = time.time()
        sigma_binary = kkm.binary_search(X, k, Ku, indicators, s, precision=1e-2)
        end = time.time()
        print('By binary search: {}'.format(sigma_binary))
        print('Time: {}'.format(end-start))
        binary_times.append(end-start)
        binary_errors.append((np.abs(sigma_true-sigma_binary)))
        binary_sigmas.append(sigma_binary)
        print('Error: {}'.format((np.abs(sigma_true-sigma_binary))))

        print()
        s = sigma_square
        K = kkm.compute_kernel_matrix(X, sigma=s)
        indicators = kkm.kernel_kmeans(k, K)    

        Ku = np.copy(K)

    with open(output_path, 'a') as out:
        out.write('{};{};{};{};{}\n'.format(input_path, 'ours', square_times, square_errors, square_sigmas))

    with open(output_path, 'a') as out:
        out.write('{};{};{};{};{}\n'.format(input_path, 'binary', binary_times, binary_errors, binary_sigmas))


        
        
