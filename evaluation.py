# Import Module
import numpy as np

def confusion_matrix(y, y_pred):
    """
    Function to compute the confusion matrix and evaluation metrics for the logistic regression model.

    Parameters
    ----------
    y : array-like
        True labels of the samples.

    y_pred : array-like, shape (n_samples,)
        Predicted labels of the samples.

    Returns
    -------
    cm : array-like, shape (2, 2)
        Confusion matrix.

    accuracy : float
        Accuracy of the model.

    sens : float
        Sensitivity (recall) of the model.

    prec : float
        Precision of the model.

    f_score : float
        F1-score of the model.
    """
    # Calculate True Positive, True Negative, False Positive, and False Negative
    tp = np.sum((y == 1) & (y_pred == 1))
    tn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))

    # Construct the confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])

    # Calculate evaluation metrics
    epsilon = 1e-9
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    sens = tp / (tp + fn + epsilon)
    prec = tp / (tp + fp + epsilon)
    f_score = (2 * prec * sens) / (prec + sens + epsilon)

    return cm, accuracy, sens, prec, f_score