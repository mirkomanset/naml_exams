import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, hessian, jit
import jax.scipy.optimize
from jaxopt import BoxOSQP


class SupportVectorRegressor:
    def __init__(self, epsilon=0.1, lmbda=1.0):
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.w = None

    def loss(self, params, X, y):
        predictions = jnp.dot(X, params[:-1]) + params[-1]

        # Compute epsilon-insensitive loss
        epsilon_loss = jnp.maximum(0, jnp.abs(predictions - y) - self.epsilon)

        # Regularization term (L2 norm of w)
        reg_term = self.lmbda * jnp.sum(params**2)

        # Total loss
        return reg_term + jnp.mean(epsilon_loss)

    def train(self, X, y):
        _, n_features = X.shape

        # Initialize weights and bias
        self.w = jnp.zeros(n_features + 1)

        # Solve optimization problem
        opt_res = jax.scipy.optimize.minimize(
            self.loss, self.w, method="BFGS", args=(X, y)
        )
        self.w = opt_res.x

    def predict(self, X):
        return jnp.dot(X, self.w[:-1]) + self.w[-1]


class SupportVectorMachine:
    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda
        self.w = None

    def loss(self, params, X, y):
        # Compute the decision function
        decision = jnp.dot(X, params[:-1]) + params[-1]
        # Compute the hinge loss
        loss_val = jnp.maximum(0, 1 - y * decision)
        # Regularization term (L2 norm of w)
        reg_term = self.lmbda * jnp.sum(params**2)
        # Total loss
        return reg_term + jnp.mean(loss_val)

    def train(self, X, y):
        _, n_features = X.shape

        # Initialize weights and bias
        self.w = jnp.zeros(n_features + 1)

        # Solve optimization problem
        opt_res = jax.scipy.optimize.minimize(
            self.loss, self.w, method="BFGS", args=(X, y)
        )
        self.w = opt_res.x

    def predict(self, X):
        # Decision function
        decision = jnp.dot(X, self.w[:-1]) + self.w[-1]
        return jnp.sign(decision)


class KernelSVM:
    def __init__(self, gamma: float, C: float):
        """
        Kernel SVM using RBF kernel and OSQP solver.

        Args:
            gamma (float): Parameter for the RBF kernel.
            C (float): Regularization parameter.
        """
        self.gamma = gamma
        self.C = C
        self.X_train = None
        self.y_train = None
        self.beta = None
        self.b = None

    def rbf_kernel(self, X1, X2):
        """Compute the RBF kernel between two matrices X1 and X2."""
        sq_dists = (
            -2 * jnp.dot(X1, X2.T)
            + jnp.sum(X1**2, axis=1)[:, None]
            + jnp.sum(X2**2, axis=1)
        )
        return jnp.exp(-self.gamma * sq_dists)

    def fit(self, X, y):
        """
        Fit the Kernel SVM model using the OSQP solver.

        Args:
            X (jnp.ndarray): Training feature matrix.
            y (jnp.ndarray): Training labels (+1 or -1).
        """
        self.X_train = X
        self.y_train = y
        K = self.rbf_kernel(X, X)

        def matvec_Q(Q, beta):
            return jnp.dot(Q, beta)

        def matvec_A(_, beta):
            return beta, jnp.sum(beta)

        l = -jax.nn.relu(-y * self.C), 0.0
        u = jax.nn.relu(y * self.C), 0.0

        # Assuming BoxOSQP is imported and accessible here
        osqp = BoxOSQP(matvec_Q=matvec_Q, matvec_A=matvec_A, tol=1e-6)
        params, _ = osqp.run(
            init_params=None, params_obj=(K, -y), params_eq=None, params_ineq=(l, u)
        )
        self.beta = params.primal[0]
        self.b = self.compute_bias()

    def compute_bias(self):
        """Compute the bias term b using a support vector."""
        support_indices = jnp.where(jnp.abs(self.beta) > 1e-4)[0]
        if len(support_indices) > 0:
            i = support_indices[0]
            return self.y_train[i] - jnp.sum(
                self.beta[support_indices]
                * self.rbf_kernel(
                    self.X_train[support_indices], self.X_train[i : i + 1]
                ).reshape((-1,))
            )
        else:
            return 0.0

    def decision_function(self, X_test):
        """
        Compute the decision function for test data.

        Args:
            X_test (jnp.ndarray): Test feature matrix.

        Returns:
            jnp.ndarray: Predicted decision function values.
        """
        K_test = self.rbf_kernel(X_test, self.X_train)
        return K_test @ self.beta + self.b


def relu(x):
    return jnp.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return jnp.where(x > 0, x, alpha * x)


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def initialize_params(layers_size):
    np.random.seed(0)  # for reproducibility
    params = list()
    for i in range(len(layers_size) - 1):
        W = np.random.randn(layers_size[i + 1], layers_size[i])
        # Xavier initialization
        # W = np.random.randn(layers_size[i + 1], layers_size[i]) * np.sqrt(
        #     2 / (layers_size[i + 1] + layers_size[i])
        # )
        b = np.zeros((layers_size[i + 1], 1))
        params.append(W)
        params.append(b)
    return params


def ANN(x, params, a=0, b=10):
    # layer = x.T
    # Scale in [-1, 1] -> good with tanh where a is minimum, b is maximum
    layer = (2 * x.T - (a + b)) / (b - a)
    num_layers = int(len(params) / 2 + 1)
    weights = params[0::2]
    biases = params[1::2]
    for i in range(num_layers - 1):
        layer = weights[i] @ layer - biases[i]
        if i < num_layers - 2:
            layer = jnp.tanh(layer)
            # layer = sigmoid(layer)
            # layer = relu(layer)

    return layer.T


def loss_quadratic(x, y, params):
    error = ANN(x, params) - y
    return jnp.sum(error * error)


def loss_crossentropy(x, y, params):
    y_app = ANN(x, params)
    return -jnp.sum(y * jnp.log(y_app) + (1 - y) * jnp.log(1 - y_app))


def MSE(x, y, params):
    error = ANN(x, params) - y
    return jnp.mean(error * error)


def MSW(params):
    weights = params[::2]
    partial_sum = 0.0
    n_weights = 0
    for W in weights:
        partial_sum += jnp.sum(W * W)
        n_weights += W.shape[0] * W.shape[1]
    return partial_sum / n_weights


def regularization_loss(x, y, params, penalization):
    return MSE(x, y, params) + penalization * MSW(params)
