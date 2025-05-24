import numpy as np
import jax.numpy as jnp
from jax import grad, hessian, jit


def PCA(X: np.ndarray) -> np.ndarray:
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
    return Phi


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
    n, m = X.shape

    # Step 1: Random projection matrix (m x k)
    G = np.random.randn(m, k)

    # Step 2: Project X onto lower-dimensional subspace (n x k)
    Y = X @ G

    # Step 3: Orthonormalize the projection
    Q, _ = np.linalg.qr(Y)

    # Step 4: Project X into the subspace
    B = Q.T @ X  # Shape: (k, m)

    # Step 5: Compute SVD on the small matrix
    U_Y, s, VT = np.linalg.svd(B, full_matrices=False)

    # Step 6: Map back to original space
    U = Q @ U_Y

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


def SVT(
    X_noise: np.ndarray, tau: float, max_iter: int, tol: float
) -> tuple[np.ndarray, int]:
    """
    Perform Singular Value Thresholding (SVT) to approximate a low-rank matrix.

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
        U, S, V = np.linalg.svd(X_prev, full_matrices=False)

        # Step 2: Soft-threshold the singular values
        S_thresholded = np.maximum(S - tau, 0)

        # Step 3: Reconstruct the matrix
        X_hat = U @ np.diag(S_thresholded) @ V

        # Step 4: Check for convergence
        if np.linalg.norm(X_hat - X_prev, "fro") / np.linalg.norm(X_hat, "fro") < tol:
            break

        X_prev = X_hat.copy()

    # Estimate the rank (number of non-zero singular values)
    rank_estimate = np.sum(S_thresholded > 0)

    return X_hat, rank_estimate


def compute_recon_errors(ks: list[int], X_bar: np.ndarray) -> list[float]:
    """
    Compute reconstruction errors for different ranks k using randomized SVD.

    Parameters:
        ks (list[int]): List of target ranks for low-rank approximation.
        X_bar (np.ndarray): Centered data matrix (features × samples).

    Returns:
        recon_error (list[float]): Relative reconstruction errors for each k.
    """
    # (Optional) Override ks if you want fixed ranks
    # ks = [5, 10, 25, 50, 100, 200, 400]

    recon_error = []

    for k in ks:
        # Compute randomized SVD of rank k
        randU, rands, randVT = randomized_SVD(X_bar, k)

        # Reconstruct low-rank approximation of X_bar
        Xk = randU[:, :k] @ np.diag(rands[:k]) @ randVT[:k, :]

        # Compute relative Frobenius norm reconstruction error
        error = np.linalg.norm(X_bar - Xk, ord="fro") / np.linalg.norm(X_bar, ord="fro")
        recon_error.append(error)

    return recon_error


def gradient_descent(
    J: callable, w_init: jnp.ndarray, alpha: float, n_iter: int
) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """
    Performs gradient descent with automatic differentiation using JAX.

    Parameters:
        J (callable): Cost function J(w) -> float, where w is a 1D jax.numpy array.
        w_init (jnp.ndarray): Initial weights (1D array).
        alpha (float): Learning rate.
        n_iter (int): Number of iterations.

    Returns:
        tuple:
            - jnp.ndarray: Final optimized parameter vector.
            - list[jnp.ndarray]: History of weight vectors (including initial and final).
    """
    w = w_init
    J = jit(J)
    gradJ = jit(grad(J))
    history = [w]

    for _ in range(n_iter):
        w = w - alpha * gradJ(w)
        history.append(w)

    return w, history


def gradient_descent_with_data(
    loss_fn: callable,
    w_init: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    learning_rate: float,
    n_iter: int,
) -> tuple[jnp.ndarray, list[float]]:
    """
    Perform gradient descent optimization on a loss function with data.

    Parameters:
        loss_fn (callable): Loss function taking parameters w, data X, and targets y.
        w_init (jnp.ndarray): Initial parameters.
        X (jnp.ndarray): Input data.
        y (jnp.ndarray): Target labels or values.
        learning_rate (float): Step size for gradient updates.
        n_iter (int): Number of iterations to run.

    Returns:
        w (jnp.ndarray): Optimized parameters after gradient descent.
        history (list[float]): Loss values recorded at each iteration.
    """
    w = w_init

    # JIT-compile the loss function and its gradient for speed
    loss_fn = jit(loss_fn)
    grad_fn = jit(grad(loss_fn, argnums=0))  # Gradient w.r.t. parameters w

    history = []
    for _ in range(n_iter):
        # Compute gradient of loss w.r.t. parameters
        g = grad_fn(w, X, y)

        # Update parameters by taking a step opposite to gradient
        w = w - learning_rate * g

        # Evaluate and record current loss
        current_loss = loss_fn(w, X, y)
        history.append(float(current_loss))

    return w, history


def stochastic_gradient_descent(
    loss_fn: callable,
    w_init: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    learning_rate: float,
    epochs: int,
    batch_size: int,
) -> tuple[list[float], jnp.ndarray]:
    """
    Performs stochastic gradient descent using JAX for automatic differentiation.

    Parameters:
        loss_fn (callable): Vectorized loss function: loss_fn(w, X, y) -> float.
        w_init (jnp.ndarray): Initial weights.
        X (jnp.ndarray): Input features (n_samples, n_features).
        y (jnp.ndarray): Target values (n_samples,).
        learning_rate (float): Step size.
        epochs (int): Number of epochs.
        batch_size (int): Size of minibatch.

    Returns:
        tuple:
            - List of loss values (one per epoch).
            - Final optimized weights.
    """
    grad_fn = jit(grad(loss_fn))
    loss_fn = jit(loss_fn)

    n_samples = X.shape[0]
    w = w_init
    loss_history = []

    for _ in range(epochs):
        idx = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[idx]
        y_batch = y[idx]
        g = grad_fn(w, X_batch, y_batch)
        w = w - learning_rate * g
        loss_history.append(float(loss_fn(w, X, y)))  # full-batch loss (for logging)

    return loss_history, w


def gradient_descent_momentum(
    f: callable,
    x_init: jnp.ndarray,
    learning_rate: float,
    momentum_coeff: float,
    tol: float,
    max_iter: int,
) -> tuple[jnp.ndarray, list[float]]:
    """
    Gradient descent with momentum using JAX.

    Parameters:
        f (callable): Scalar-valued objective function f(x).
        x_init (jnp.ndarray): Initial point.
        learning_rate (float): Learning rate.
        momentum_coeff (float): Momentum factor (e.g., 0.9).
        tol (float): Convergence tolerance (based on function value change).
        max_iter (int): Maximum number of iterations.

    Returns:
        - Final optimized x
        - History of function values or norms (for plotting or analysis)
    """
    f = jit(f)
    grad_f = jit(grad(f))

    x = x_init
    x_old = x_init
    momentum = jnp.zeros_like(x)
    history = []

    for _ in range(max_iter):
        g = grad_f(x)
        momentum = momentum_coeff * momentum + g
        x = x - learning_rate * momentum

        # Error metric based on change in function value
        E = jnp.abs(f(x) - f(x_old))
        history.append(float(E))

        if E < tol:
            break
        x_old = x

    return x, history


def gradient_descent_momentum_data(
    f: callable,  # loss function f(w, X, y)
    w_init: jnp.ndarray,
    X: jnp.ndarray,
    y: jnp.ndarray,
    learning_rate: float,
    momentum_coeff: float,
    tol: float,
    max_iter: int,
) -> tuple[jnp.ndarray, list[float]]:
    """
    Gradient descent with momentum on data-driven loss function.

    Parameters:
        f (callable): Loss function f(w, X, y).
        w_init (jnp.ndarray): Initial parameter vector.
        X (jnp.ndarray): Input data.
        y (jnp.ndarray): Target data.
        learning_rate (float): Learning rate.
        momentum_coeff (float): Momentum factor.
        tol (float): Convergence tolerance.
        max_iter (int): Max iterations.

    Returns:
        - Optimized parameters w
        - History of loss values
    """
    f = jit(f)
    grad_f = jit(grad(f, argnums=0))  # grad wrt w

    w = w_init
    w_old = w_init
    momentum = jnp.zeros_like(w)
    history = []

    for _ in range(max_iter):
        g = grad_f(w, X, y)
        momentum = momentum_coeff * momentum + g
        w = w - learning_rate * momentum

        loss_diff = jnp.abs(f(w, X, y) - f(w_old, X, y))
        history.append(float(loss_diff))

        if loss_diff < tol:
            break
        w_old = w

    return w, history


def newton_method(
    J: callable,
    x_init: jnp.ndarray,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> tuple[list[jnp.ndarray], list[float]]:
    """
    Newton's method with automatic differentiation using JAX.

    Args:
        J: Scalar objective function J(x).
        x_init: Initial guess vector.
        tolerance: Convergence tolerance.
        max_iterations: Maximum iterations.

    Returns:
        x_history: List of iterates.
        J_history: List of function values.
    """

    J = jit(J)
    grad_J = jit(grad(J))
    hess_J = jit(hessian(J))

    x_current = x_init
    x_history = [x_current]
    J_history = [float(J(x_current))]

    for _ in range(max_iterations):
        g = grad_J(x_current)
        H = hess_J(x_current)

        if jnp.linalg.norm(g) < tolerance:
            break

        # Solve for Newton step: H p = -g
        p = jnp.linalg.solve(H, -g)
        x_current = x_current + p

        x_history.append(x_current)
        J_history.append(float(J(x_current)))

        if jnp.abs(J_history[-1] - J_history[-2]) < tolerance:
            break

    return x_history, J_history
