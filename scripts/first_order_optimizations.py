import numpy as np
import jax.numpy as jnp
import jax
from copy import deepcopy

# In this implementation argnums=2 because loss_func are implemented as
# loss_func(model, x, y, params)

# If it is implemented without taking the model as parameter just change
# argnums = 2


def gradient_descent(
    loss_jit: callable,
    grad_jit: callable,
    xx: np.ndarray,
    yy: np.ndarray,
    init_params: list[float],
    num_epochs: int = 2000,
    learning_rate: float = 1e-1,
) -> tuple[list[float], list[list[float]], list[float]]:
    params = deepcopy(init_params)
    history_params = []
    history_params.append(params)

    history_loss = []
    history_loss.append(loss_jit(xx, yy, params))

    for epoch in range(num_epochs):
        grads = grad_jit(xx, yy, params)
        for i in range(len(params)):
            params[i] -= learning_rate * grads[i]
        history_params.append(params)
        history_loss.append(loss_jit(xx, yy, params))

    return params, history_params, history_loss


def stochastic_gradient_descent(
    loss_jit: callable,
    grad_jit: callable,
    xx: np.ndarray,
    yy: np.ndarray,
    n_training_points: int,
    init_params: list[float],
    num_epochs: int = 20000,
    learning_rate: float = 1e-1,
    batch_size: int = 10,
):
    params = deepcopy(init_params)
    history_params = []
    history_params.append(params)

    history_loss = []
    history_loss.append(loss_jit(xx, yy, params))

    for epoch in range(num_epochs):
        idxs = np.random.choice(n_training_points, batch_size)
        grads = grad_jit(xx[idxs, :], yy[idxs, :], params)
        for i in range(len(params)):
            params[i] -= learning_rate * grads[i]

        history_loss.append(loss_jit(xx, yy, params))

    return params, history_params, history_loss


def stochastic_gradient_descent_decay(
    loss_jit: callable,
    grad_jit: callable,
    xx: np.ndarray,
    yy: np.ndarray,
    n_training_points: int,
    init_params: list[float],
    num_epochs: int = 20000,
    learning_rate_max: float = 1e-1,
    learning_rate_min: float = 2e-2,
    learning_rate_decay: int = 10000,
    batch_size: int = 10,
):
    params = deepcopy(init_params)

    history_params = []
    history_params.append(params)

    history_loss = []
    history_loss.append(loss_jit(xx, yy, params))

    for epoch in range(num_epochs):
        learning_rate = max(
            learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay)
        )
        idxs = np.random.choice(n_training_points, batch_size)
        grads = grad_jit(xx[idxs, :], yy[idxs, :], params)
        for i in range(len(params)):
            params[i] -= learning_rate * grads[i]

        history_loss.append(loss_jit(xx, yy, params))

    return params, history_params, history_loss


def stochastic_gradient_descent_momentum(
    loss_jit: callable,
    grad_jit: callable,
    xx: np.ndarray,
    yy: np.ndarray,
    n_training_points: int,
    init_params: list[float],
    num_epochs: int = 20000,
    learning_rate_max: float = 1e-1,
    learning_rate_min: float = 2e-2,
    learning_rate_decay: int = 10000,
    batch_size: int = 10,
    alpha: float = 0.9,
):
    params = deepcopy(init_params)

    history_params = []
    history_params.append(params)

    history_loss = []
    history_loss.append(loss_jit(xx, yy, params))

    velocity = [0.0 for _ in range(len(params))]
    for epoch in range(num_epochs):
        learning_rate = max(
            learning_rate_min, learning_rate_max * (1 - epoch / learning_rate_decay)
        )
        idxs = np.random.choice(n_training_points, batch_size)
        grads = grad_jit(xx[idxs, :], yy[idxs, :], params)

        for i in range(len(params)):
            velocity[i] = alpha * velocity[i] - learning_rate * grads[i]
            params[i] += velocity[i]

        history_params.append(params)
        history_loss.append(loss_jit(xx, yy, params))

    return params, history_params, history_loss


def adagrad(
    loss_jit: callable,
    grad_jit: callable,
    xx: np.ndarray,
    yy: np.ndarray,
    n_training_points: int,
    init_params: list[float],
    num_epochs: int = 20000,
    batch_size: int = 10,
    learning_rate: float = 1e-1,
    delta: float = 1e-7,
):
    params = deepcopy(init_params)

    history_params = []
    history_params.append(params)

    history_loss = []
    history_loss.append(loss_jit(xx, yy, params))

    cumulated_square_grad = [0.0 for i in range(len(params))]
    for epoch in range(num_epochs):
        idxs = np.random.choice(n_training_points, batch_size)
        grads = grad_jit(xx[idxs, :], yy[idxs, :], params)

        for i in range(len(params)):
            cumulated_square_grad[i] += grads[i] * grads[i]
            params[i] -= (
                learning_rate / (delta + jnp.sqrt(cumulated_square_grad[i])) * grads[i]
            )
        history_params.append(params)
        history_loss.append(loss_jit(xx, yy, params))

    return params, history_params, history_loss


def rmsprop(
    loss_jit: callable,
    grad_jit: callable,
    xx: np.ndarray,
    yy: np.ndarray,
    n_training_points: int,
    init_params: list[float],
    num_epochs: int = 20000,
    batch_size: int = 50,
    learning_rate: float = 1e-3,
    decay_rate: float = 0.9,
    delta: float = 1e-7,
):
    params = deepcopy(init_params)
    history_params = []
    history_params.append(params)

    history_loss = []
    history_loss.append(loss_jit(xx, yy, params))

    cumulated_square_grad = [0.0 for i in range(len(params))]
    for epoch in range(num_epochs):
        idxs = np.random.choice(n_training_points, batch_size)
        grads = grad_jit(xx[idxs, :], yy[idxs, :], params)

        for i in range(len(params)):
            cumulated_square_grad[i] = (
                decay_rate * cumulated_square_grad[i]
                + (1 - decay_rate) * grads[i] * grads[i]
            )
            params[i] -= (
                learning_rate / (delta + jnp.sqrt(cumulated_square_grad[i])) * grads[i]
            )
        history_params.append(params)
        history_loss.append(loss_jit(xx, yy, params))

    return params, history_params, history_loss
