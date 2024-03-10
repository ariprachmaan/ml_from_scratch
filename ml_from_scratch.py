# Import Module
import numpy as np

class LogisticRegression:
    """
    Initialize logistic regression model.

    Parameters:
    -----------
    learning_rate : float, default=0.001
        The learning rate for gradient descent.

    num_iterations : int, default=10000
        The number of iterations for the gradient descent.

    tol : float, default=1e-4
        The tolerance for the optimization. If the update is smaller than `tol`, the optimization will stop.

    fit_intercept : bool, default=True
        Whether to fit the intercept term.

    Attributes:
    -----------
    learning_rate : float
        The learning rate for gradient descent.

    num_iterations : int
        The number of iterations for the gradient descent.

    tol : float
        The tolerance for the optimization.

    fit_intercept : bool
        Whether to fit the intercept term.

    weights : array-like, shape (n_features,)
        The coefficients of the logistic regression model.
    """
    
    def __init__(self, learning_rate=0.001, num_iterations=10000, tol=1e-4, fit_intercept=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.weights = None
    
    def fit(self, X, y):
        """
        Fit logistic regression model to the training data.

        Parameters:
        -----------
        X : array-like
            Training data, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target labels.

        Returns:
        --------
        self : object
            Returns self.
        """
        if self.fit_intercept:
            X = self.__add_intercept(X)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)

        for _ in range(self.num_iterations):
            y_pred = self.__sigmoid(np.dot(X, self.weights))
            grad_weights = np.dot(X.T, (y_pred - y)) / y.size
            self.weights -= self.learning_rate * grad_weights

            if np.all(np.abs(grad_weights) < self.tol):
                break

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        """
        Compute logistic sigmoid function.

        Parameters:
        -----------
        z : array-like
            Input to the sigmoid function.

        Returns:
        --------
        array-like
            The output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X):
        """
        Predict class probabilities for input data.

        Parameters:
        -----------
        X : array-like
            Data to predict class probabilities for.

        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted probabilities of the positive class.
        """
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.weights))
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels for input data.

        Parameters:
        -----------
        X : array-like
            Data to predict class labels for.

        threshold : float, default=0.5
            Threshold value for classification. If the predicted probability is greater than or equal to
            the threshold, the class label is 1, otherwise it is 0.

        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted class labels.
        """
        return (self.predict_proba(X) >= threshold).astype(int)
