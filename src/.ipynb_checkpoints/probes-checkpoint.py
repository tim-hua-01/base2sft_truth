import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from abc import ABC, abstractmethod
import pickle
import os
import warnings


class BaseProbe(ABC):
    """Abstract base class for different types of probes."""
    
    def __init__(self):
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X, y=None):
        """Fit the probe to the data."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions using the fitted probe."""
        pass
    
    @abstractmethod
    def decision_function(self, X):
        """Calculate a score for X and return the value on the 1 probe dimension."""
        pass
    
    @abstractmethod
    def get_direction(self):
        """Get the direction of the probe."""
        pass


class LogisticRegressionProbe(BaseProbe):
    """Logistic Regression probe with standard scaling."""
    
    def __init__(self, reg):
        super().__init__()
        
        self.model = SGDClassifier(
            loss='log_loss',  # For logistic regression
            penalty='l2',     # L2 regularization similar to original code
            alpha=reg,        # Note: alpha in SGD is equivalent to 1/C in LogisticRegression
            max_iter=50000,
            tol=1e-5, # default 1e-4
            n_jobs=-1,
            verbose=0,
            random_state=42   # For reproducibility
        )
    
    def fit(self, X, y):
        """Fit the logistic regression probe."""
        self.model.fit(X, y)
        self.is_fitted = True

        # normalize the weights
        norm = np.linalg.norm(self.model.coef_)
        self.model.coef_ /= norm
        self.model.intercept_ /= norm
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted before making predictions.")
        return self.model.predict(X)
    
    def decision_function(self, X):
        """Calculate score - return the value on the 1 probe dimension."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted before scoring.")

        return self.model.decision_function(X)

    def get_direction(self):
        """Get the logistic regression coefficients."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted to get coefficients.")
        return self.model.coef_


class DifferenceOfMeansProbe(BaseProbe):
    """Difference of means probe for binary classification."""
    
    def __init__(self):
        super().__init__()
        self.mean_class_0 = None
        self.mean_class_1 = None
        self.threshold = None
        self.coef_ = None
    
    def fit(self, X, y):
        """Fit by calculating means for each class."""
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("DifferenceOfMeansProbe only supports binary classification.")
        
        # Calculate mean for each class
        mask_0 = (y == 0)
        mask_1 = (y == 1)
        
        self.mean_class_0 = np.mean(X[mask_0], axis=0)
        self.mean_class_1 = np.mean(X[mask_1], axis=0)
        
        # Calculate difference vector and threshold
        self.coef_ = self.mean_class_1 - self.mean_class_0
        
        # Threshold is halfway between projected means
        proj_mean_0 = np.dot(self.mean_class_0, self.coef_)
        proj_mean_1 = np.dot(self.mean_class_1, self.coef_)
        self.threshold = (proj_mean_0 + proj_mean_1) / 2
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions based on projection onto difference vector."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted before making predictions.")
        
        # Project data onto difference vector
        projections = np.dot(X, self.coef_)
        
        # Classify based on threshold
        predictions = np.where(projections > self.threshold, 1, 0)
        return predictions
    
    def decision_function(self, X):
        """Calculate score - return the value on the 1 probe dimension."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted before scoring.")
        
        # Project data onto difference vector (this is the 1D probe output)
        projections = np.dot(X, self.coef_)
        return projections
    
    def get_direction(self):
        """Get the difference vector between class means."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted to get difference vector.")
        return self.coef_


class PCAProbe(BaseProbe):
    """PCA probe for dimensionality reduction and analysis."""
    
    def __init__(self):
        super().__init__()        
        self.n_components = 1
        self.pca = PCA(n_components=1, random_state=42)
        self.threshold = None  # Default threshold
    
    def fit(self, X, y=None):
        """Fit PCA to the data."""        
        # Fit PCA
        self.pca.fit(X)
        self.is_fitted = True
        self.coef_ = self.pca.components_.squeeze()
        
        # Find optimal threshold if labels are provided
        best_accuracy, best_threshold = self._find_optimal_threshold(X, y)
        self.coef_ = -self.coef_
        best_accuracy_negative, best_threshold_negative = self._find_optimal_threshold(X, y)
        if best_accuracy > best_accuracy_negative:
            self.coef_ = -self.coef_
        
        return self
    
    def _find_optimal_threshold(self, X, y):
        """Find optimal threshold using grid search on decision function scores."""
        # Get decision function scores
        scores = self.decision_function(X)
        
        # Try different thresholds
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        best_accuracy = 0
        best_threshold = 0
        
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            accuracy = accuracy_score(y, predictions)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        self.threshold = best_threshold
        return best_accuracy, best_threshold
    
    def predict(self, X):
        """Make binary predictions using the optimal threshold."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted before making predictions.")
        
        # Get decision function scores
        scores = self.decision_function(X)
        
        # Apply threshold to make binary predictions
        predictions = (scores > self.threshold).astype(int)
        
        return predictions
    
    def decision_function(self, X):
        """Calculate score - return the value on the 1 probe dimension."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted before scoring.")
        
        return np.dot(X, self.coef_)
    
    def get_direction(self):
        """Get the principal components."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted to get components.")
        return self.coef_

class LDAProbe(BaseProbe):
    """LDA probe for linear discriminant analysis and classification."""
    
    def __init__(self):
        super().__init__()        
        self.n_components = 1
        self.lda = LinearDiscriminantAnalysis(n_components=self.n_components)
        self.threshold = None  # Default threshold
        # self.use_builtin_predict = False  # Option to use LDA's built-in predict
    
    def fit(self, X, y=None):
        """Fit LDA to the data."""
        if y is None:
            raise ValueError("LDA requires labels (y) for supervised learning.")
        
        # Fit LDA
        self.lda.fit(X, y)
        self.is_fitted = True
        self.coef_ = self.lda.coef_.squeeze()
        
        # Find optimal threshold on decision function scores
        self._find_optimal_threshold(X, y)
        
        return self
    
    def _find_optimal_threshold(self, X, y):
        """Find optimal threshold using grid search on decision function scores."""
        # Get decision function scores
        scores = self.decision_function(X)
        
        # Try different thresholds
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        best_accuracy = 0
        best_threshold = 0
        
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            accuracy = accuracy_score(y, predictions)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        self.threshold = best_threshold
        
        # # Compare with LDA's built-in predict
        # builtin_predictions = self.lda.predict(X)
        # builtin_accuracy = accuracy_score(y, builtin_predictions)
        # threshold_accuracy = best_accuracy
        
        # # Use whichever method gives better accuracy
        # self.use_builtin_predict = builtin_accuracy >= threshold_accuracy
        
        return best_threshold
    
    def predict(self, X):
        """Make binary predictions using optimal method."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted before making predictions.")
        
        # if self.use_builtin_predict:
        #     # Use LDA's built-in predict method
        #     return self.lda.predict(X)
        # else:
        # Use threshold on decision function scores
        scores = self.decision_function(X)
        predictions = (scores > self.threshold).astype(int)
        return predictions
    
    def decision_function(self, X):
        """Calculate discriminant scores."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted before scoring.")
        
        # For binary classification, LDA decision function returns signed distances
        # to the separating hyperplane
        return np.dot(X, self.coef_)
    
    def get_direction(self):
        """Get the LDA coefficients (discriminant direction)."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted to get coefficients.")
        return self.coef_
    
    def get_means(self):
        """Get class means in the original space."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted to get class means.")
        return self.lda.means_
    
    def get_explained_variance_ratio(self):
        """Get the percentage of variance explained by each component."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted to get explained variance ratio.")
        return self.lda.explained_variance_ratio_

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class MLPProbe(BaseProbe):
    """MLP probe with 3 hidden layers (256, 128, 64) and ReLU activations."""
    
    def __init__(self, learning_rate=0.001, batch_size=32, device=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_dim = None
        
    def _build_model(self, input_dim):
        """Build the MLP architecture."""
        class MLP(nn.Module):
            def __init__(self, reg):
                super().__init__()
                self.feature_layers = nn.Sequential(
                    nn.Linear(8192, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x, return_logits=False):
                logits = self.feature_layers(x)
                if return_logits:
                    return logits
                return self.sigmoid(logits)
        
        return MLP(input_dim)
    
    def fit(self, X, y):
        """Fit the MLP probe for 5 epochs using Adam optimizer."""
        # Convert to numpy if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        self.input_dim = X.shape[1]
        
        # Build model
        self.model = self._build_model(self.input_dim).to(self.device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Train for 5 epochs
        self.model.train()
        for epoch in range(5):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make binary predictions."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted before making predictions.")
        
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = (outputs >= 0.5).float().cpu().numpy().reshape(-1)
        
        return predictions.astype(int)
    
    def decision_function(self, X):
        """Calculate score - return the logit values (pre-sigmoid)."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted before scoring.")
        
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor, return_logits=True)
            decision_scores = logits.cpu().numpy().reshape(-1)
        
        return decision_scores
    
    def get_direction(self):
        """Get the weights of the first layer as the probe direction."""
        if not self.is_fitted:
            raise ValueError("Probe must be fitted to get direction.")
        
        return None