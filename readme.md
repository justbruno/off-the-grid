Replication code for Bruno Ordozgoiti and Lluís A. Belanche Muñoz, Off-the-grid: fast and effective hyperparameter search for kernel clustering, ECML-PKDD 2020.

Example runs in test_run.sh.

You can check the source to figure out the output format for each script.

## Background
The paper cited above gives an algorithm for efficient hyperparameter search for RBF kernel $k$-means. The RBF kernel is often defined as $\kappa(x,y) =  \exp\left(-\frac{\|x-y|\_2^2}{2\sigma^2}\right)$, where $\sigma$ is the hyperparameter.

The key observation is that if we change $\sigma$, the new kernel matrix can be computed using element-wise exponentiation. Relying on the theory of dyadic rationals, we can approximate the new matrix to **arbitrary precision** using **very fast exponentiation operations**.

The paper also analyzes the behaviour of kernel $k$-means for certain regimes of $\sigma$.
