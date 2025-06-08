import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.optimize
from jax import grad, hessian, jit


def plot_surface(f: callable, min_val: int, max_val: int, n_points: int) -> None:
    x = np.linspace(min_val, max_val, n_points)
    y = np.linspace(min_val, max_val, n_points)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9, edgecolor="none")


def plot_contour(f: callable, x_history) -> None:
    # Plot the contour plot of the function
    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(label="J(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Contour plot of J(x)")

    # Extract trajectory
    x_values = [x[0] for x in x_history]
    y_values = [x[1] for x in x_history]
    plt.plot(x_values, y_values, marker="o")

    # Get final point
    final_x, final_y = x_history[-1]
    final_val = f(final_x, final_y)

    # Annotate convergence point
    plt.plot(final_x, final_y, "rx")  # red X marker
    plt.text(
        final_x + 0.1,
        final_y + 0.1,
        f"({final_x:.2f}, {final_y:.2f})\nJ={final_val:.2f}",
        fontsize=8,
        color="red",
    )


def gradient_descent(
    J: callable, w_init: jnp.ndarray, alpha: float, max_iter: int, tol: float
) -> tuple[jnp.ndarray, list[jnp.ndarray], list[float]]:
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
    history_weight = [w]
    history_loss = [J(w)]

    for _ in range(max_iter):
        grad_val = gradJ(w)
        w = w - alpha * grad_val
        history_weight.append(w)
        history_loss.append(w)
        if jnp.linalg.norm(grad_val) < tol:
            break

    return w, history_weight, history_loss


def gradient_descent_backtracking(
    func, grad_func, x0, alpha=0.3, beta=0.8, tol=1e-6, max_iter=100
):
    """
    Perform gradient descent with backtracking line search.

    Args:
        func (callable): The objective function to minimize.
        grad_func (callable): Gradient of the objective function.
        x0 (jnp.ndarray): Initial point.
        alpha (float): Parameter for sufficient decrease condition (default 0.3).
        beta (float): Backtracking step size reduction factor (default 0.8).
        tol (float): Convergence tolerance on gradient norm.
        max_iter (int): Maximum number of iterations.

    Returns:
        x (jnp.ndarray): Estimated minimizer.
        path (list): List of iterates for visualization purposes.
    """
    x = x0
    path = [x]
    for _ in range(max_iter):
        grad_val = grad_func(x)
        t = 1.0
        # Backtracking line search
        while func(x - t * grad_val) > func(x) - alpha * t * jnp.dot(
            grad_val, grad_val
        ):
            t *= beta
        x = x - t * grad_val
        path.append(x)
        if jnp.linalg.norm(grad_val) < tol:
            break
    return x, path


def exact_line_search_quadratic(A, b, x0, tol=1e-6, max_iter=100):
    """
    Perform gradient descent with exact line search for a quadratic function.

    Minimizes: f(x) = 0.5 * xᵀ A x + bᵀ x

    Args:
        A (jnp.ndarray): Positive definite matrix defining the quadratic form.
        b (jnp.ndarray): Linear term.
        x0 (jnp.ndarray): Initial point.
        tol (float): Convergence tolerance on gradient norm.
        max_iter (int): Maximum number of iterations.

    Returns:
        x (jnp.ndarray): Estimated minimizer.
        path (list): List of iterates for visualization purposes.
    """
    x = x0
    path = [x]
    for _ in range(max_iter):
        grad_val = jnp.dot(A, x) + b
        t = jnp.dot(grad_val, grad_val) / jnp.dot(grad_val, jnp.dot(A, grad_val))
        x -= t * grad_val
        path.append(x)
        if jnp.linalg.norm(grad_val) < tol:
            break
    return x, path


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


def bfgs_inverse_update(
    loss_fn,
    grad_fn,
    x_init,
    max_epochs=1000,
    tol=1e-8,
    line_search_fn=None,
    verbose=False,
):
    """
    BFGS optimization using inverse Hessian approximation (manual update).

    Parameters:
        loss_fn (callable): The scalar loss function to minimize.
        grad_fn (callable): Gradient of the loss function.
        x_init (np.ndarray): Initial guess.
        max_epochs (int): Maximum number of iterations.
        tol (float): Gradient norm tolerance for stopping.
        line_search_fn (callable): Optional custom line search (default: SciPy).
        verbose (bool): If True, print debug info.

    Returns:
        x (np.ndarray): Optimized parameters.
        history (list): List of loss values at each iteration.
    """
    x = x_init.copy()
    n = x.size
    Identity = np.eye(n)
    Binv = Identity
    grad = grad_fn(x)
    history = [loss_fn(x)]

    for epoch in range(1, max_epochs + 1):
        if np.linalg.norm(grad) < tol:
            break

        # Step 1: Compute search direction
        p = -Binv @ grad

        # Step 2: Line search
        if line_search_fn is not None:
            alpha = line_search_fn(loss_fn, grad_fn, x, p)
        else:
            result = scipy.optimize.line_search(loss_fn, grad_fn, x, p)
            alpha = result[0]
        alpha = 1e-8 if alpha is None else alpha

        # Step 3: Update parameters
        x_new = x + alpha * p
        s = x_new - x
        x = x_new

        # Step 4: Compute new gradient and y
        grad_new = grad_fn(x)
        y = grad_new - grad
        grad = grad_new

        # Step 5: Update inverse Hessian using Sherman-Morrison
        rho = 1.0 / (np.dot(y, s))
        E = Identity - rho * np.outer(y, s)
        Binv = E.T @ Binv @ E + rho * np.outer(s, s)

        # Step 6: Record loss
        history.append(loss_fn(x))

        if verbose:
            print(
                f"Epoch {epoch}: loss = {history[-1]:.3e}, ||grad|| = {np.linalg.norm(grad):.2e}"
            )

    return x, history
