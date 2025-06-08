import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def PCA(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform PCA on data with features in rows and samples in columns.
    Returns data projected onto principal components.
    """
    X_mean = np.mean(X, axis=1)  # Mean per feature
    X_bar = X - X_mean[:, None]  # Center data

    # Optional: standardize features (uncomment if needed)
    # X_std = np.std(X, axis=1)
    # X_bar = X_bar / X_std[:, None]

    U, s, VT = np.linalg.svd(X_bar, full_matrices=False)
    Phi = U.T @ X_bar  # Project onto PCs
    return s, Phi


def plot_2_principal_components(Phi, y) -> None:
    plt.scatter(Phi[0, y == 0], Phi[1, y == 0], color="r", label="")
    plt.scatter(Phi[0, y == 1], Phi[1, y == 1], color="g", label="")
    plt.legend()


def singular_values_metrics(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute cumulative fraction and explained variance from singular values.
    """
    cumulative_fraction = np.cumsum(s) / np.sum(s)  # Fraction of singular values
    explained_variance = np.cumsum(s**2) / np.sum(s**2)  # Variance explained by PCs
    return cumulative_fraction, explained_variance


def randomized_SVD(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a low-rank approximation of matrix X using randomized SVD.

    Parameters:
        X (np.ndarray): Input matrix of shape (n, m)
        k (int): Target rank for approximation

    Returns:
        U (np.ndarray): Approximate left singular vectors (n, k)
        s (np.ndarray): Approximate singular values (k,)
        VT (np.ndarray): Approximate right singular vectors (k, m)
    """
    __build_class__, m = X.shape
    G = np.random.randn(m, k)  # Step 1: Random projection matrix (m x k)
    Y = X @ G  # Step 2: Project X onto lower-dimensional subspace (n x k)
    Q, _ = np.linalg.qr(Y)  # Step 3: Orthonormalize the projection
    B = Q.T @ X  # Step 4: Project X into the subspace - Shape: (k, m)
    U_Y, s, VT = np.linalg.svd(
        B, full_matrices=False
    )  # Step 5: Compute SVD on the small matrix
    U = Q @ U_Y  # Step 6: Map back to original space

    return U, s, VT


def randomized_SVD_oversampling(
    X: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes a low-rank approximation of matrix X using randomized SVD with oversampling.

    Parameters:
        X (np.ndarray): Input matrix of shape (n, m)
        k (int): Target rank for approximation

    Returns:
        U (np.ndarray): Approximate left singular vectors (n, k_approx)
        s (np.ndarray): Approximate singular values (k_approx,)
        VT (np.ndarray): Approximate right singular vectors (k_approx, m)
    """
    _, m = X.shape
    oversampled_l = round(k * 1.5)  # Oversampling (e.g., k + 5 or k + 10 also common)
    # Step 1: Generate random Gaussian test matrix
    G = np.random.randn(m, oversampled_l)
    # Step 2: Project X onto a lower-dimensional subspace
    Y = X @ G
    # Step 3: Orthonormalize Y using QR decomposition
    Q, _ = np.linalg.qr(Y)
    # Step 4: Project X into the subspace spanned by Q
    B = Q.T @ X
    # Step 5: Compute SVD on the small matrix B
    U_Y, s, VT = np.linalg.svd(B, full_matrices=False)
    # Step 6: Lift the left singular vectors back to the original space
    U = Q @ U_Y
    return U, s, VT


def SVT_approximation(
    X_noise: np.ndarray, tau: float, max_iter: int, tol: float
) -> tuple[np.ndarray, int]:
    """
    Perform Singular Value Thresholding (SVT) to approximate a low-rank matrix.
    It is used for denoising.

    Parameters:
        X_noise (np.ndarray): The noisy input matrix.
        tau (float): Threshold value for singular value shrinkage.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence (relative Frobenius norm).

    Returns:
        X_hat (np.ndarray): Denoised (low-rank) matrix.
        rank_estimate (int): Estimated rank of the output matrix.
    """
    X_prev = X_noise.copy()

    for _ in range(max_iter):
        # Step 1: SVD decomposition
        U, s, VT = np.linalg.svd(X_prev, full_matrices=False)

        # Step 2: Soft-threshold the singular values
        S_thresholded = np.maximum(s - tau, 0)

        # Step 3: Reconstruct the matrix
        X_hat = U @ np.diag(S_thresholded) @ VT

        # Step 4: Check for convergence
        if np.linalg.norm(X_hat - X_prev, "fro") / np.linalg.norm(X_hat, "fro") < tol:
            break

        X_prev = X_hat.copy()

    # Estimate the rank (number of non-zero singular values)
    rank_estimate = np.sum(S_thresholded > 0)

    return X_hat, rank_estimate


def SVT_matrix_completion(
    X_full,
    rows_train,
    cols_train,
    vals_train,
    rows_test,
    cols_test,
    vals_test,
    n_max_iter=100,
    threshold=1e-2,
    increment_tol=1e-5,
    print_progress=True,
):
    """
    Performs matrix completion using Singular Value Thresholding (SVT).
    Used for reccomender systems, with partial data.

    Args:
        X_full (np.ndarray): Initial matrix with missing entries filled (e.g., zeros).
        rows_train (np.ndarray): Row indices of known training entries.
        cols_train (np.ndarray): Column indices of known training entries.
        vals_train (np.ndarray): Known values at training indices.
        rows_test (np.ndarray): Row indices of test entries.
        cols_test (np.ndarray): Column indices of test entries.
        vals_test (np.ndarray): Ground-truth values at test indices.
        n_max_iter (int): Maximum number of iterations.
        threshold (float): Threshold below which singular values are set to zero.
        increment_tol (float): Stop if matrix change is below this norm.
        print_progress (bool): Whether to print metrics each iteration.

    Returns:
        A tuple with:
            - Completed matrix (np.ndarray)
            - List of RMSE values per iteration
            - List of Pearson correlation values per iteration
    """
    A = X_full.copy()
    RMSE_list = []
    rho_list = []

    for i in range(n_max_iter):
        A_old = A.copy()

        # Singular Value Thresholding
        U, s, VT = np.linalg.svd(A, full_matrices=False)
        s[s < threshold] = 0
        A = U @ np.diag(s) @ VT

        # Restore known values
        A[rows_train, cols_train] = vals_train

        # Convergence check
        increment = np.linalg.norm(A - A_old)

        # Predict and evaluate
        vals_predicted = A[rows_test, cols_test]
        errors = vals_test - vals_predicted

        RMSE = np.sqrt(np.mean(errors**2))
        rho = pearsonr(vals_test, vals_predicted)[0]

        RMSE_list.append(RMSE)
        rho_list.append(rho)

        if increment < increment_tol:
            break

    return A, RMSE_list, rho_list


def reconstruct_rank_k_matrix(X: np.ndarray, k: int) -> np.ndarray:
    """
    Recuntruct a matrix into a rank k approximation

    Args:
        X (np.ndarray): matrix to be reconstructed
        k (int): rank

    Returns:
        np.ndarray: reconstructed matrix
    """

    U, s, VT = np.linalg.svd(X, full_matrices=False)
    # U, s, VT = randomized_SVD(X, k)
    # U, s, VT = randomized_SVD_oversampling(X, k)
    X_k = U[:, :k] @ np.diag(s[:k]) @ VT[:k, :]
    return X_k


def compute_frobenius_error(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Compute frobenius error between 2 matrices (typically original and reconstructed one)

    Args:
        X1 (np.ndarray): first matrix
        X2 (np.ndarray): second matrix

    Returns:
        float: error
    """
    error = np.linalg.norm(X1 - X2, ord="fro") / np.linalg.norm(X1, ord="fro")
    return error


def kth_rank1_matrix(X: np.ndarray, k: int) -> np.ndarray:
    """
    Find the k-th rank-1 matrix given the matrix X

    Args:
        X (np.ndarray): original matrix
        k (int): position of the desired rank-1 matrix

    Returns:
        np.ndarray: k-th rank-1 matrix
    """
    U, _, VT = np.linalg.svd(X, full_matrices=False)
    ukvk = np.outer(U[:, k - 1], VT[k - 1, :])
    return ukvk


def pseudo_inverse(A: np.ndarray) -> np.ndarray:
    """
    Computes the Moore-Penrose pseudo-inverse of a matrix A using Singular Value Decomposition (SVD).

    This is useful in solving least squares problems, where you want to find the vector `x` that minimizes
    the Euclidean norm ||Ax - b||^2. The solution is given by:
        x = A^+ b
    where A^+ is the pseudo-inverse of A.

    Args:
        A (np.ndarray): Input matrix of shape (m, n)

    Returns:
        np.ndarray: The pseudo-inverse of A, of shape (n, m)
    """
    U, s, VT = np.linalg.svd(A, full_matrices=False)
    s[s > 0] = 1 / s[s > 0]
    return VT.T @ np.diag(s) @ U.T


def linear_regression_pseudo_inverse(
    X: np.ndarray, Y: np.ndarray, N: int
) -> tuple[float, float]:
    """
    Performs simple linear regression using the Moore-Penrose pseudo-inverse.

    The model assumes:
        Y ≈ m * X + q

    It constructs a design matrix Phi with a bias term and solves the least squares
    problem to estimate the slope (m_hat) and intercept (q_hat).

    Args:
        X (np.ndarray): Input feature vector of shape (N,)
        Y (np.ndarray): Target output vector of shape (N,)
        N (int): Number of data points (should match len(X) and len(Y))

    Returns:
        tuple[float, float]: Estimated slope (m_hat) and intercept (q_hat)
    """
    Phi = np.block([X[:, np.newaxis], np.ones((N, 1))])  # Shape: (N, 2)
    w = pseudo_inverse(Phi) @ Y  # Shape: (2,)

    m_hat = w[0]
    q_hat = w[1]
    return m_hat, q_hat


def linear_regression_solve(X: np.ndarray, Y: np.ndarray, N: int):
    """
    Linear regression by solving the matrix equation
    """
    Phi = np.block([X[:, np.newaxis], np.ones((N, 1))])  # Shape: (N, 2)
    w = np.linalg.solve(Phi.transpose() @ Phi, Phi.transpose() @ Y)
    m_hat = w[0]
    q_hat = w[1]
    return m_hat, q_hat


def ridge_regression_pseudo_inverse(
    X: np.ndarray, Y: np.ndarray, N: int, lam: float
) -> tuple[float, float]:
    """
    Performs ridge regression (L2-regularized least squares) using the pseudo-inverse approach.

    This method solves:
        w = argmin ||Y - Φw||^2 + λ||w||^2

    Args:
        X (np.ndarray): Input feature vector of shape (N,)
        Y (np.ndarray): Target output vector of shape (N,)
        N (int): Number of data points
        lam (float): Regularization parameter (λ > 0)

    Returns:
        tuple[float, float]: Estimated slope (m_hat) and intercept (q_hat)
    """
    Phi = np.block([X[:, np.newaxis], np.ones((N, 1))])  # Design matrix (N × 2)
    PhiPhiT = Phi @ Phi.T  # (N × N)
    alpha = np.linalg.solve(PhiPhiT + lam * np.eye(N), Y)  # Solve dual system
    w = Phi.T @ alpha  # Recover primal weights
    m_hat = w[0]
    q_hat = w[1]
    return m_hat, q_hat


def scalar_product_kernel_q(q: float):
    """
    Constructs a polynomial kernel function of the form:
        K(xi, xj) = (xi * xj + 1)^q

    Args:
        q (float): Degree of the polynomial kernel

    Returns:
        Callable: A kernel function that takes two scalars xi and xj
    """

    def scalar_product_kernel(xi, xj):
        return (xi * xj + 1) ** q

    return scalar_product_kernel


def gaussian_kernel_sigma(sigma: float):
    """
    Constructs a Gaussian (RBF) kernel function:
        K(xi, xj) = exp(-|xi - xj|^2 / (2 * sigma^2))

    Args:
        sigma (float): Bandwidth parameter (controls spread of the kernel)

    Returns:
        Callable: A kernel function that takes two scalars xi and xj
    """

    def gaussian_kernel(xi, xj):
        return np.exp(-(np.abs(xi - xj) ** 2) / (2 * sigma**2))

    return gaussian_kernel


def kernel_regression(
    X: np.ndarray,
    Y: np.ndarray,
    N: int,
    kernel,
    lam: float,
    X_test: np.ndarray,
    N_test: int,
) -> np.ndarray:
    """
    Performs kernel ridge regression using a provided kernel function.

    Solves the dual problem:
        alpha = (K + λI)^(-1) Y
    and makes predictions:
        Y_test = K_test @ alpha

    Args:
        X (np.ndarray): Training input data of shape (N,)
        Y (np.ndarray): Training target values of shape (N,)
        N (int): Number of training samples
        kernel (Callable): Kernel function of the form kernel(xi, xj)
        lam (float): Regularization parameter λ
        X_test (np.ndarray): Test input data of shape (N_test,)
        N_test (int): Number of test samples

    Returns:
        np.ndarray: Predicted values for the test inputs, shape (N_test,)
    """
    K = np.array([[kernel(X[i], X[j]) for j in range(N)] for i in range(N)])
    alpha = np.linalg.solve(K + lam * np.eye(N), Y)

    K_test = np.array(
        [[kernel(X_test[i], X[j]) for j in range(N)] for i in range(N_test)]
    )
    Y_test_KR = K_test @ alpha
    return Y_test_KR
