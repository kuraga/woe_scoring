from typing import List, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import l1_min_c


class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """

    def __init__(self):
        self.model_ = None

    def fit(self, data, target):
        """
        Fits a logistic regression model to the given data and target.

        Args:
            data: pandas.DataFrame or numpy.ndarray
                The input data matrix of shape (n_samples, n_features).
            target: pandas.Series or numpy.ndarray
                The target vector of shape (n_samples,).

        Returns:
            self: LogisticRegression
                The fitted model.
        """
        # Add constant term (intercept) and fit the model
        X_with_const = sm.add_constant(data)
        self.model_ = sm.Logit(target, X_with_const).fit(disp=0)  # Suppress convergence messages
        return self

    def predict(self, data):
        """
        Predict the binary outcome of a dataset using the fitted logistic regression model.

        Args:
            data (array-like): The dataset to predict, with shape (n_samples, n_features).

        Returns:
            numpy.ndarray: The predicted binary outcomes, with shape (n_samples,).
        """
        # Add constant term and predict
        X_with_const = sm.add_constant(data)
        # Convert probabilities to binary predictions using 0.5 threshold
        decision = self.model_.predict(X_with_const) > 0.5
        # Ensure integer type for classification results
        return np.array(decision, dtype=np.int64)

    def predict_proba(self, data):
        """
        Predict class probabilities for input data.

        Args:
            data (array-like): Input data to predict probabilities for.

        Returns:
            array: An array of shape (n_samples, 2) containing the predicted
            probabilities for each class, where n_samples is the number of
            samples in the input data. The first column contains the probability
            of the negative class and the second column contains the probability
            of the positive class.
        """
        # Add constant term (intercept)
        X_with_const = sm.add_constant(data)

        # Get positive class probabilities
        pos_proba = self.model_.predict(X_with_const)

        # Stack negative and positive probabilities
        decision_2d = np.column_stack((1 - pos_proba, pos_proba))

        return decision_2d


class Model:
    """
    Initialize and manage a predictive model with standardized interface.

    This class provides a uniform API for different model types (sklearn or statsmodels),
    handling model creation, training, and evaluation.

    Args:
        model_type (str): Model implementation to use ('sklearn' or 'statsmodels')
        l1_exp_scale (int): Exponent scale for L1 regularization grid
        l1_grid_size (int): Number of grid points for L1 regularization search
        cv (int): Number of cross-validation folds
        class_weight (str): Class weighting strategy for imbalanced datasets
        random_state (int): Random seed for reproducibility
        n_jobs (int): Number of CPU cores for parallelization
        scoring (str): Metric for model evaluation

    Attributes:
        coef_: Model coefficients
        intercept_: Model intercept
        feature_names_: Features used in the model
        model_score_: Cross-validation performance score
        pvalues_: Statistical significance of each feature
    """

    def __init__(
            self, model_type: str, l1_exp_scale: int, l1_grid_size: int, cv: int = None, class_weight: str = None,
            random_state: int = None, n_jobs: int = None, scoring: str = None
    ) -> None:
        self.model_type = model_type
        self.cv = cv
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.l1_exp_scale = l1_exp_scale
        self.l1_grid_size = l1_grid_size

        self.model = self._get_model(model_type)
        self.coef_ = []
        self.intercept_ = 0.0
        self.feature_names_ = []
        self.model_score_ = 0.0
        self.pvalues_ = []

    def get_model(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> callable:
        """
        Returns a callable object that can be used to make predictions based on the provided data and target.

        :param data: The input data to use for making predictions.
        :type data: pd.DataFrame
        :param target: The target values to use for making predictions.
        :type target: Union[pd.Series, np.ndarray]
        :return: A callable object that can be used to make predictions.
        :rtype: callable
        """

        return self.model(data, target)

    def _get_model(self, model_type: str) -> callable:
        if model_type == 'sklearn':
            return self._get_sklearn_model
        elif model_type == 'statsmodels':
            return self._get_statsmodels_model
        else:
            raise ValueError(f'Unknown model type: {model_type}. Should be either "sklearn" or "statsmodels"')

    def _get_sklearn_model(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> callable:
        """
        Trains a scikit-learn logistic regression model with regularization.

        Args:
            data (pd.DataFrame): The input data to train the model on.
            target (Union[pd.Series, np.ndarray]): The target values to train the model on.

        Returns:
            callable: The trained logistic regression model.
        """
        # Ensure target is in the right format
        target_array = target.values if hasattr(target, 'values') else np.array(target)

        # Calculate optimal regularization strengths
        base_c = l1_min_c(data, target_array, loss="log", fit_intercept=True)
        Cs = base_c * np.logspace(0, self.l1_exp_scale, self.l1_grid_size)

        # Create and fit model
        model = LogisticRegressionCV(
            Cs=Cs,
            cv=self.cv,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tol=1e-5,
            max_iter=5000,
            scoring=self.scoring,
            penalty='l1',
            solver='liblinear'  # Best for L1 penalty
        ).fit(data, target_array)

        # Extract model parameters
        self.coef_ = list(model.coef_[0])
        self.intercept_ = model.intercept_[0]
        self.feature_names_ = data.columns.tolist()

        # Calculate cross-validation score
        self.model_score_ = cross_val_score(
            model, data, target_array, cv=self.cv, n_jobs=self.n_jobs, scoring=self.scoring
        ).mean()

        # Calculate statistical significance
        self.pvalues_ = list(self._calc_pvalues(model, data))

        return model

    def _get_statsmodels_model(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> callable:
        """ Fits a statsmodels model to the given data and target, and returns the trained model.

        Args:
            data (pd.DataFrame): The input data to be used for training the model.
            target (Union[pd.Series, np.ndarray]): The target variable to be used for training the model.

        Returns:
            callable: The trained statsmodels model.
        """

        model = SMWrapper().fit(data, target)
        self.coef_ = list(model.model_.params[1:])
        self.intercept_ = model.model_.params[0]
        self.feature_names_ = data.columns.to_list()
        self.model_score_ = cross_val_score(
            model, data, target, cv=self.cv, n_jobs=self.n_jobs, scoring=self.scoring
        ).mean()
        self.pvalues_ = list(model.model_.pvalues)[1:]
        return model

    def _calc_pvalues(self, model: callable, data: pd.DataFrame) -> np.ndarray:
        """
        Calculates p-values for a logistic regression model using the Wald test.

        This implements the statistical test for coefficient significance based on
        the asymptotic normality of maximum likelihood estimates.

        Args:
            model: A logistic regression model fit using scikit-learn.
            data: A Pandas DataFrame of features.

        Returns:
            A NumPy array of p-values for each feature.
        """
        # Get predicted probabilities
        p = model.predict_proba(data)[:, 1]

        # Combine intercept and coefficients
        coefs = np.concatenate([model.intercept_, model.coef_[0]])

        # Add column of ones for intercept
        x_full = np.insert(np.array(data), 0, 1, axis=1)

        # Calculate variance-covariance matrix using Fisher Information Matrix
        # This is more numerically stable for large datasets
        weights = p * (1 - p)
        weighted_x = x_full * np.sqrt(weights[:, np.newaxis])
        xTx = weighted_x.T @ weighted_x

        # Get inverse of Fisher Information Matrix
        try:
            # Try more stable SVD-based pseudo-inverse first
            vcov = np.linalg.pinv(xTx)
        except np.linalg.LinAlgError:
            # Fall back to standard inverse
            vcov = np.linalg.inv(xTx)

        # Calculate standard errors
        se = np.sqrt(np.diag(vcov))

        # Calculate t-statistics
        t = coefs / se

        # Calculate two-tailed p-values
        p_values = (1 - norm.cdf(abs(t))) * 2

        # Return p-values for features (excluding intercept)
        return p_values[1:]
