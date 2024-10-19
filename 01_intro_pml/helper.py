import numpy as np

from typing import Tuple, Dict, Union
from abc import ABC, abstractmethod

def generate_housing_data(price_per_square_meter: float = 1_000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate housing data for a simple linear regression problem.
    - X represents the input data, e.g. square meters of a house
    - y represents the output data, e.g. price of the house

    Parameters
    ----------
    price_per_square_meter : float, optional
        Price per square meter of the house, by default 1_000

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the input data and the output data
    """

    X = np.array([130, 140, 150, 160, 170])
    price_per_square_meter = 1000
    y = price_per_square_meter * X + np.random.normal(0, price_per_square_meter*5, size=X.shape)

    return X, y


class LinearModel(ABC):
    """
    Abstract class for linear models.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the data.

        Parameters
        ----------
        X : np.ndarray
            Training data
        y : np.ndarray
            Target values
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict the target values for the given data.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Predicted target values
        """
        pass

    @abstractmethod
    def get_params(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Get the parameters of the model.

        Returns
        -------
        Union[np.ndarray, Dict[str, np.ndarray]]
            Model parameters
        """
        pass

class LinearDeterministicModel(LinearModel):
    """Linear model that uses the normal equation to find the optimal weights."""

    def __init__(self) -> None:
        """Constructor"""
        super().__init__()
        self.W = None # Weights: [w, b]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model using the normal equation.

        np.linalg.inv is not numerically stable for ill-conditioned matrices
        as an alternative, we can use np.linalg.pinv to compute the pseudo-inverse
        or numpy.linalg.solve to solve the linear system of equations

        Derivation of the normal equation:
        -------------------------------
        Given the linear model y = XW, where:
        - y is the target values
        - X is the input data
        - W are the weights

        We can define the loss function as the mean squared error (MSE):
        L = (1/N) * ||XW - y||^2

        To find the optimal weights, we can minimize the loss function:
        dL/dW = 0
        => 2 * X^T(XW - y) = 0
        => X^TXW = X^Ty
        => W = (X^TX)^(-1)X^Ty

        Parameters
        ----------
        X : np.ndarray
            Training data
        y : np.ndarray
            Target values
        """
        # We add a column of ones to X to account for the bias term
        X = np.column_stack([X, np.ones(X.shape)])

        # We use the normal equation to find the optimal weights
        self.W = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target values for the given data.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Predicted target values
        """

        X = np.column_stack([X, np.ones(X.shape)])
        return X @ self.W
    
    def get_params(self) -> np.ndarray:
        """Get the parameters of the model.

        Returns
        -------
        np.ndarray
            Model parameters, [w, b]
        """
        return self.W
    
class LinearProbabilisticModel(LinearModel):
    """Linear model that uses a probabilistic approach to find the optimal weights."""
    
    def __init__(self, noise_variance: float = 1.0) -> None:
        """Constructor

        Parameters
        ----------
        noise_variance : float, optional
            Variance of the noise, by default 1.0
        """

        super().__init__()
        self.noise_variance = noise_variance
        self.posterior_mean = None
        self.posterior_cov = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model using a probabilistic approach.

        Derivation of the posterior distribution:
        -----------------------------------------
        Given the linear model y = XW + epsilon, where:
        - y is the target values
        - X is the input data
        - W are the weights
        - epsilon is the noise

        We can define the likelihood as a Gaussian distribution:
        p(y|X, W) = N(y|XW, noise_variance)
        
        We can define a Gaussian prior for the weights:
        p(W) = N(W|0, prior_variance)

        We can compute the posterior distribution using Bayes' theorem:
        p(W|X, y) = N(W|posterior_mean, posterior_cov)
        
        where:
        posterior_cov = (prior_cov^(-1) + noise_cov^(-1)X^TX)^(-1)
        posterior_mean = posterior_cov * (prior_cov^(-1) * prior_mean + noise_cov^(-1)X^Ty)

        Parameters
        ----------
        X : np.ndarray
            Training data
        y : np.ndarray
            Target values
        """

        # Add a column of ones to X to account for the bias term
        X = np.column_stack([X, np.ones(X.shape)])

        prior_variance = 1e6  # Weak prior, large variance
        
        # Posterior covariance and mean
        inv_prior_cov = np.eye(2) / prior_variance
        noise_cov_inv = np.eye(len(X)) / self.noise_variance
        posterior_cov = np.linalg.inv(inv_prior_cov + X.T @ noise_cov_inv @ X)
        posterior_mean = posterior_cov @ X.T @ noise_cov_inv @ y

        self.posterior_mean = posterior_mean
        self.posterior_cov = posterior_cov

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the target values for the given data.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Predicted target values and standard deviation
        """
        X = np.column_stack([X, np.ones(X.shape)])

        # Predictive mean
        pred_mean = X @ self.posterior_mean
        
        # Predictive variance (diagonal elements of X * posterior_cov * X.T + noise variance)
        pred_variance = np.sum(X @ self.posterior_cov * X, axis=1) + self.noise_variance
        pred_std = np.sqrt(pred_variance)
        
        return pred_mean, pred_std
    
    def get_params(self, cov: bool = False) -> Dict[str, np.ndarray]:
        """Get the parameters of the model.

        Parameters
        ----------
        cov : bool, optional
            Return the posterior covariance, by default False

        Returns
        -------
        Dict[str, np.ndarray]
            Model parameters
        """

        if cov:
            return {
                'posterior_mean': self.posterior_mean,
                'posterior_cov': self.posterior_cov
            }
        else:
            return {
                'posterior_mean': self.posterior_mean,
                'posterior_std': np.sqrt(np.diag(self.posterior_cov))  # Return only diagonal elements (standard deviation of weights)
            }