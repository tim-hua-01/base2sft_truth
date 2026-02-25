"""
Shared-First INLP: A principled approach to finding general vs domain-specific
truth directions.

Phase 1: Pool all domains, run INLP → finds GENERAL directions
Phase 2: Project out general directions, run INLP per domain → finds DOMAIN-SPECIFIC directions

This works because:
- General directions must classify well across ALL domains
- Pooling forces INLP to find what's shared (domain-specific signal gets diluted)
- After projecting out shared structure, whatever remains is domain-specific
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import h5py
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings


# ============================================================================
# Data Structures
# ============================================================================

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class DomainData:
    """Data for a single domain."""
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    
    # Original data (before PCA, if applied)
    X_train_original: Optional[np.ndarray] = None
    X_test_original: Optional[np.ndarray] = None
    
    @property
    def n_train(self) -> int:
        return len(self.y_train)
    
    @property
    def n_test(self) -> int:
        return len(self.y_test)
    
    @property
    def dim(self) -> int:
        return self.X_train.shape[1]
    
    @property
    def original_dim(self) -> int:
        if self.X_train_original is not None:
            return self.X_train_original.shape[1]
        return self.X_train.shape[1]


@dataclass
class Direction:
    """A single extracted direction with metadata."""
    vector: np.ndarray  # Normalized direction vector
    bias: float
    train_accuracy: float
    test_accuracy: float
    source: str  # 'general' or domain name
    iteration: int
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate accuracy on given data."""
        scores = X @ self.vector + self.bias
        preds = (scores > 0).astype(int)
        return accuracy_score(y, preds)
        # return roc_auc_score(y, scores)


@dataclass 
class SharedFirstResult:
    """Results from Shared-First INLP."""
    general_directions: List[Direction]
    domain_specific_directions: Dict[str, List[Direction]]  # domain_name -> directions
    
    # Evaluation matrices
    general_cross_accuracy: np.ndarray  # (n_general, n_domains)
    specific_cross_accuracy: Dict[str, np.ndarray]  # domain -> (n_specific, n_domains)
    
    domain_names: List[str]

# ============================================================================
# Core Algorithm
# ============================================================================

class SharedFirstINLP:
    """
    Shared-First INLP Algorithm.
    
    Phase 1: Find general directions by pooling all domains
    Phase 2: Find domain-specific directions in residual space
    """
    
    def __init__(
        self,
        n_general: int = 5,
        n_specific: int = 5,
        classifier_alpha: float = 1e-4,
        classifier_max_iter: int = 10000,
        classifier_tol: float = 1e-6,
        classifier_loss: str = 'log_loss',  # 'hinge' for SVM, 'log_loss' for logistic
        normalize: bool = False,  # Whether to use StandardScaler before fitting
        center: bool = False,  # Whether to center data (subtract mean)
        center_mode: str = 'shared',  # 'shared' = global mean, 'individual' = per-domain mean
        pca_dim: Optional[int] = None,  # None = no PCA, int = target dimension
        pca_mode: str = 'shared',  # 'shared' = fit on all data, 'individual' = fit per domain
        leave_one_out: bool = False,  # Whether to use leave-one-out for general directions
        balance_domains: bool = False,  # Whether to balance domains in Phase 1 pooling
        random_state: int = 42,
    ):
        """
        Args:
            n_general: Number of general directions to extract
            n_specific: Number of domain-specific directions per domain
            classifier_alpha: Regularization strength for SGDClassifier
            classifier_max_iter: Max iterations for SGDClassifier
            classifier_tol: Tolerance for stopping criterion
            classifier_loss: 'hinge' for SVM, 'log_loss' for logistic regression
            normalize: Whether to use StandardScaler before fitting classifier
            center: Whether to center data by subtracting mean
            center_mode: 'shared' (global mean) or 'individual' (per-domain mean)
            pca_dim: Target dimensionality for PCA (None = no PCA)
            pca_mode: 'shared' (fit PCA on all data) or 'individual' (fit per domain)
            leave_one_out: If True, for each domain i, compute general directions from
                          all domains EXCEPT i. This prevents general directions from
                          capturing domain-specific signal.
            balance_domains: If True, downsample each domain to the minimum domain size
                            when pooling for Phase 1 (general directions).
            random_state: Random seed
        """
        self.n_general = n_general
        self.n_specific = n_specific
        self.classifier_alpha = classifier_alpha
        self.classifier_max_iter = classifier_max_iter
        self.classifier_tol = classifier_tol
        self.classifier_loss = classifier_loss
        self.normalize = normalize
        self.center = center
        self.center_mode = center_mode
        self.pca_dim = pca_dim
        self.pca_mode = pca_mode
        self.leave_one_out = leave_one_out
        self.balance_domains = balance_domains
        self.random_state = random_state
        
        self.domains: Dict[str, DomainData] = {}
        self.result: Optional[SharedFirstResult] = None
        
        # PCA objects (stored for potential inverse transform)
        self.pca_shared: Optional[PCA] = None
        self.pca_individual: Dict[str, PCA] = {}
        
        # Centering means (stored for potential use)
        self.center_mean_shared: Optional[np.ndarray] = None
        self.center_mean_individual: Dict[str, np.ndarray] = {}
        
        # Leave-one-out general directions (one set per domain)
        self.general_directions_loo: Dict[str, List[Direction]] = {}
    
    def load_data(
        self,
        h5_path: str,
        task_list: List[str],
        layer_name: str = "layer_33",
        train_split: float = 0.8,
        enforce_balance: bool = True,
    ) -> None:
        """Load data from HDF5 file.
        
        Args:
            h5_path: Path to HDF5 file
            layer_name: Name of layer group in HDF5
            train_split: Fraction of data to use for training
            enforce_balance: If True, subsample to ensure 50/50 class balance
        """
        print(f"Loading data from {h5_path}...")
        
        with h5py.File(h5_path, 'r') as f:
            layer_group = f[layer_name]
            
            # for domain_name in layer_group.keys():
            for domain_name in task_list:
                domain_group = layer_group[domain_name]
                
                X = np.array(domain_group['X'], dtype=np.float32)
                y = np.array(domain_group['y'], dtype=np.int64)
                
                # if domain_name.startswith("ethics"):
                #     # subsample 1200 samples
                #     X = X[:1200]
                #     y = y[:1200]
                # elif domain_name.startswith("repe_honesty__IF"):
                #     # subsample 400 samples
                #     X = X[:400]
                #     y = y[:400]
                # elif domain_name.startswith("sycophancy"):
                #     # subsample
                #     indices = np.random.choice(len(X), size=3000, replace=False)
                #     X = X[:600]
                #     y = y[:600]
                # elif 'logical' in domain_name:
                #     X = X[:600]
                #     y = y[:600]
                
                # Check original balance
                n_pos = np.sum(y == 1)
                n_neg = np.sum(y == 0)
                balance_ratio = min(n_pos, n_neg) / max(n_pos, n_neg)
                
                # Enforce balance if requested
                if enforce_balance and balance_ratio < 0.99:
                    min_count = min(n_pos, n_neg)
                    
                    # Subsample majority class
                    idx_pos = np.where(y == 1)[0]
                    idx_neg = np.where(y == 0)[0]
                    
                    np.random.seed(self.random_state)
                    idx_pos_sub = np.random.choice(idx_pos, min_count, replace=False)
                    idx_neg_sub = np.random.choice(idx_neg, min_count, replace=False)
                    
                    idx_balanced = np.concatenate([idx_pos_sub, idx_neg_sub])
                    np.random.shuffle(idx_balanced)
                    
                    X = X[idx_balanced]
                    y = y[idx_balanced]
                    
                    print(f"  {domain_name}: {n_pos}+ / {n_neg}- -> balanced to {min_count}/{min_count}")
                else:
                    print(f"  {domain_name}: {n_pos}+ / {n_neg}- (ratio={balance_ratio:.3f})")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    train_size=train_split,
                    random_state=self.random_state,
                    stratify=y,
                )
                
                # Verify train/test balance
                train_pos = np.sum(y_train == 1)
                train_neg = np.sum(y_train == 0)
                test_pos = np.sum(y_test == 1)
                test_neg = np.sum(y_test == 0)
                
                self.domains[domain_name] = DomainData(
                    name=domain_name,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                )
                
                print(f"    -> train: {train_pos}+/{train_neg}-, test: {test_pos}+/{test_neg}-")
        
        print(f"\nLoaded {len(self.domains)} domains.\n")
        
        # Apply centering if requested (before PCA)
        if self.center:
            self._apply_centering()
        
        # Apply PCA if requested
        if self.pca_dim is not None:
            self._apply_pca()
    
    def _apply_centering(self) -> None:
        """Center data by subtracting mean."""
        print(f"Applying centering (mode: {self.center_mode})")
        
        if self.center_mode == 'shared':
            # Compute global mean from all training data
            X_all_train = np.vstack([d.X_train for d in self.domains.values()])
            self.center_mean_shared = np.mean(X_all_train, axis=0)
            
            print(f"  Global mean norm: {np.linalg.norm(self.center_mean_shared):.4f}")
            
            # Subtract global mean from all domains
            for domain_name, domain in self.domains.items():
                domain.X_train = domain.X_train - self.center_mean_shared
                domain.X_test = domain.X_test - self.center_mean_shared
        
        elif self.center_mode == 'individual':
            # Compute and subtract per-domain mean
            for domain_name, domain in self.domains.items():
                domain_mean = np.mean(domain.X_train, axis=0)
                self.center_mean_individual[domain_name] = domain_mean
                
                domain.X_train = domain.X_train - domain_mean
                domain.X_test = domain.X_test - domain_mean
                
                print(f"  {domain_name}: mean norm = {np.linalg.norm(domain_mean):.4f}")
        
        else:
            raise ValueError(f"Unknown center_mode: {self.center_mode}")
        
        print()
    
    def _apply_pca(self) -> None:
        """Apply PCA dimensionality reduction to all domains."""
        original_dim = list(self.domains.values())[0].dim
        target_dim = min(self.pca_dim, original_dim)
        
        print(f"Applying PCA: {original_dim} -> {target_dim} dimensions (mode: {self.pca_mode})")
        
        if self.pca_mode == 'shared':
            # Fit PCA on all training data together
            X_all_train = np.vstack([d.X_train for d in self.domains.values()])
            
            self.pca_shared = PCA(n_components=target_dim, random_state=self.random_state)
            self.pca_shared.fit(X_all_train)
            
            explained_var = np.sum(self.pca_shared.explained_variance_ratio_) * 100
            print(f"  Shared PCA explains {explained_var:.1f}% of variance")
            
            # Transform all domains with the same PCA
            for domain_name, domain in self.domains.items():
                # Store original data
                domain.X_train_original = domain.X_train.copy()
                domain.X_test_original = domain.X_test.copy()
                
                # Apply PCA
                domain.X_train = self.pca_shared.transform(domain.X_train).astype(np.float32)
                domain.X_test = self.pca_shared.transform(domain.X_test).astype(np.float32)
        
        elif self.pca_mode == 'individual':
            # Fit separate PCA for each domain
            for domain_name, domain in self.domains.items():
                pca = PCA(n_components=target_dim, random_state=self.random_state)
                pca.fit(domain.X_train)
                
                explained_var = np.sum(pca.explained_variance_ratio_) * 100
                print(f"  {domain_name}: PCA explains {explained_var:.1f}% of variance")
                
                # Store original data
                domain.X_train_original = domain.X_train.copy()
                domain.X_test_original = domain.X_test.copy()
                
                # Apply PCA
                domain.X_train = pca.transform(domain.X_train).astype(np.float32)
                domain.X_test = pca.transform(domain.X_test).astype(np.float32)
                
                self.pca_individual[domain_name] = pca
        
        else:
            raise ValueError(f"Unknown pca_mode: {self.pca_mode}")
        
        new_dim = list(self.domains.values())[0].dim
        print(f"  New dimensionality: {new_dim}\n")
    
    def direction_to_original_space(self, direction: np.ndarray, domain_name: Optional[str] = None) -> np.ndarray:
        """
        Transform a direction from PCA space back to original space.
        
        Args:
            direction: Direction vector in PCA space (pca_dim,)
            domain_name: For 'individual' PCA mode, which domain's PCA to use.
                        For 'shared' mode, this is ignored.
        
        Returns:
            Direction vector in original space (original_dim,)
        """
        if self.pca_dim is None:
            # No PCA was applied, direction is already in original space
            return direction
        
        if self.pca_mode == 'shared':
            if self.pca_shared is None:
                raise ValueError("Shared PCA not fitted yet")
            # w_orig = components.T @ w_pca
            w_orig = self.pca_shared.components_.T @ direction
        
        elif self.pca_mode == 'individual':
            if domain_name is None:
                raise ValueError("domain_name required for individual PCA mode")
            if domain_name not in self.pca_individual:
                raise ValueError(f"No PCA fitted for domain: {domain_name}")
            pca = self.pca_individual[domain_name]
            w_orig = pca.components_.T @ direction
        
        else:
            raise ValueError(f"Unknown pca_mode: {self.pca_mode}")
        
        # Normalize
        w_orig = w_orig / np.linalg.norm(w_orig)
        return w_orig
    
    def get_all_directions_original_space(self) -> Dict[str, np.ndarray]:
        """
        Get all directions transformed back to original space.
        
        Returns:
            Dict with keys:
                'general': (n_general, original_dim)
                'specific_{domain}': (n_specific, original_dim) for each domain
        """
        if self.result is None:
            raise ValueError("Run the algorithm first")
        
        directions = {}
        
        # General directions (use shared PCA or no transform)
        general_orig = []
        for d in self.result.general_directions:
            w_orig = self.direction_to_original_space(d.vector)
            general_orig.append(w_orig)
        directions['general'] = np.vstack(general_orig)
        
        # Domain-specific directions
        for domain_name, dirs in self.result.domain_specific_directions.items():
            specific_orig = []
            for d in dirs:
                # For individual PCA, use the domain's own PCA
                # For shared PCA, domain_name is ignored
                w_orig = self.direction_to_original_space(d.vector, domain_name)
                specific_orig.append(w_orig)
            directions[f'specific_{domain_name}'] = np.vstack(specific_orig)
        
        return directions
    
    def _fit_classifier(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit linear classifier using SGDClassifier and return normalized direction + bias."""
        
        if self.normalize:
            # SGDClassifier works better with scaled features
            X_scaled = self._inlp_scaler.transform(X) # MY CODE
            scaler = self._inlp_scaler
        else:
            X_scaled = X
        
        clf = SGDClassifier(
            loss=self.classifier_loss,  # 'hinge' for SVM, 'log_loss' for logistic
            alpha=self.classifier_alpha,
            max_iter=self.classifier_max_iter,
            tol=self.classifier_tol,
            random_state=self.random_state,
            learning_rate='optimal',
            early_stopping=False,
            shuffle=True,
            verbose=0,
            fit_intercept=False
        )
        # clf = SGDClassifier(
        #     loss='log_loss',  # For logistic regression
        #     penalty='l2',     # L2 regularization similar to original code
        #     alpha=1,        # Note: alpha in SGD is equivalent to 1/C in LogisticRegression
        #     max_iter=50000,
        #     tol=1e-5, # default 1e-4
        #     n_jobs=-1,
        #     verbose=0,
        #     random_state=42   # For reproducibility
        # )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_scaled, y)
        
        if self.normalize:
            # Transform coefficients back to original space
            # For StandardScaler: X_scaled = (X - mean) / std
            # So in original space: w_orig = w_scaled / std, b_orig = b_scaled - w_orig @ mean
            direction = clf.coef_[0] / scaler.scale_
            bias = clf.intercept_[0] - np.dot(direction, scaler.mean_)
        else:
            direction = clf.coef_[0]
            bias = clf.intercept_[0]
        
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            # Classifier failed to find a direction, return random
            print("    WARNING: Classifier returned zero weights, using random direction")
            direction = np.random.randn(X.shape[1])
            direction = direction / np.linalg.norm(direction)
            bias = 0.0
        else:
            direction = direction / norm
            bias = bias / norm
        
        return direction, bias
    
    def _project_to_nullspace(self, X: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Project X to null space of direction."""
        d = direction / np.linalg.norm(direction)
        return X - np.outer(X @ d, d)
    
    def _run_inlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_directions: int,
        source_name: str,
    ) -> List[Direction]:
        """Run standard INLP and return list of directions."""
        directions = []
        X_train_work = X_train.copy()
        X_test_work = X_test.copy()

        # Fit scaler ONCE on original data
        if self.normalize:
            self._inlp_scaler = StandardScaler()
            self._inlp_scaler.fit(X_train)  # Fit only, don't transform yet
        
        for k in range(n_directions):
            # Fit classifier
            direction, bias = self._fit_classifier(X_train_work, y_train)
            
            # Evaluate
            train_acc = accuracy_score(
                y_train, 
                (X_train_work @ direction + bias > 0).astype(int)
            )
            test_acc = accuracy_score(
                y_test,
                (X_test_work @ direction + bias > 0).astype(int)
            )
            # train_acc = roc_auc_score(
            #     y_train, 
            #     X_train_work @ direction + bias
            # )
            # test_acc = roc_auc_score(
            #     y_test,
            #     X_test_work @ direction + bias
            # )
            
            directions.append(Direction(
                vector=direction,
                bias=bias,
                train_accuracy=train_acc,
                test_accuracy=test_acc,
                source=source_name,
                iteration=k,
            ))
            
            # Project to null space
            X_train_work = self._project_to_nullspace(X_train_work, direction)
            X_test_work = self._project_to_nullspace(X_test_work, direction)
        
        return directions
    
    def run(self, verbose: bool = True) -> SharedFirstResult:
        """
        Run the Shared-First INLP algorithm.
        
        If leave_one_out=False (default):
            Phase 1: Pool all data, run INLP → general directions
            Phase 2: Project out general, run INLP per domain → specific directions
        
        If leave_one_out=True:
            For each domain i:
                Phase 1: Pool all data EXCEPT i, run INLP → general directions for i
                Phase 2: Project domain i using these directions, run INLP → specific for i
        """
        domain_names = list(self.domains.keys())
        
        if self.leave_one_out:
            return self._run_leave_one_out(domain_names, verbose)
        else:
            return self._run_standard(domain_names, verbose)
    
    def _pool_domains_balanced(
        self, 
        domains_to_pool: List[DomainData],
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pool domains with optional balancing (downsample to min domain size).
        
        Returns:
            X_train_pooled, y_train_pooled, X_test_pooled, y_test_pooled
        """
        if not self.balance_domains:
            # No balancing - just concatenate
            X_train = np.vstack([d.X_train for d in domains_to_pool])
            y_train = np.concatenate([d.y_train for d in domains_to_pool])
            X_test = np.vstack([d.X_test for d in domains_to_pool])
            y_test = np.concatenate([d.y_test for d in domains_to_pool])
            return X_train, y_train, X_test, y_test
        
        # Find minimum domain size (training)
        min_train_size = min(d.n_train for d in domains_to_pool)
        min_test_size = min(d.n_test for d in domains_to_pool)
        
        if verbose:
            print(f"  Balancing domains: downsampling to {min_train_size} train, {min_test_size} test per domain")
        
        X_train_parts = []
        y_train_parts = []
        X_test_parts = []
        y_test_parts = []
        
        np.random.seed(self.random_state)
        
        for d in domains_to_pool:
            # Stratified subsampling for train
            train_idx = self._stratified_subsample(d.y_train, min_train_size)
            X_train_parts.append(d.X_train[train_idx])
            y_train_parts.append(d.y_train[train_idx])
            
            # Stratified subsampling for test
            test_idx = self._stratified_subsample(d.y_test, min_test_size)
            X_test_parts.append(d.X_test[test_idx])
            y_test_parts.append(d.y_test[test_idx])
        
        return (
            np.vstack(X_train_parts),
            np.concatenate(y_train_parts),
            np.vstack(X_test_parts),
            np.concatenate(y_test_parts),
        )
    
    def _stratified_subsample(self, y: np.ndarray, n: int) -> np.ndarray:
        """Stratified subsampling to n samples, preserving class balance."""
        if len(y) <= n:
            return np.arange(len(y))
        
        # Get indices per class
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        
        # Compute how many to take from each class
        n_0 = int(n * len(idx_0) / len(y))
        n_1 = n - n_0
        
        # Subsample
        idx_0_sub = np.random.choice(idx_0, min(n_0, len(idx_0)), replace=False)
        idx_1_sub = np.random.choice(idx_1, min(n_1, len(idx_1)), replace=False)
        
        idx = np.concatenate([idx_0_sub, idx_1_sub])
        np.random.shuffle(idx)
        return idx
    
    def _run_standard(self, domain_names: List[str], verbose: bool) -> SharedFirstResult:
        """Standard run: general directions from all pooled data."""
        
        # ==================================================
        # PHASE 1: Find general directions from pooled data
        # ==================================================
        if verbose:
            print("=" * 60)
            print("PHASE 1: Finding GENERAL directions (pooled data)")
            if self.balance_domains:
                print("(Domain balancing enabled)")
            print("=" * 60)
        
        # Pool all training data (with optional balancing)
        domains_list = list(self.domains.values())
        X_train_pooled, y_train_pooled, X_test_pooled, y_test_pooled = \
            self._pool_domains_balanced(domains_list, verbose=verbose)
        
        if verbose:
            print(f"Pooled data: {len(y_train_pooled)} train, {len(y_test_pooled)} test\n")
        
        # Run INLP on pooled data
        general_directions = self._run_inlp(
            X_train_pooled, y_train_pooled,
            X_test_pooled, y_test_pooled,
            self.n_general,
            source_name="general",
        )
        
        if verbose:
            print("General directions found:")
            for d in general_directions:
                print(f"  k={d.iteration}: train={d.train_accuracy:.3f}, "
                      f"test={d.test_accuracy:.3f}")
        
        # Evaluate general directions on each domain
        general_cross_accuracy = np.zeros((self.n_general, len(domain_names)))
        
        for i, direction in enumerate(general_directions):
            for j, domain_name in enumerate(domain_names):
                domain = self.domains[domain_name]
                acc = direction.evaluate(domain.X_test, domain.y_test)
                general_cross_accuracy[i, j] = acc
        
        if verbose:
            print("\nGeneral directions cross-domain accuracy:")
            header = "     " + " ".join([f"{n[:8]:>8}" for n in domain_names])
            print(header)
            for i in range(self.n_general):
                row = f"k={i}: " + " ".join([f"{general_cross_accuracy[i,j]:.3f}" 
                                              for j in range(len(domain_names))])
                print(row)
        
        # ==================================================
        # PHASE 2: Find domain-specific directions
        # ==================================================
        if verbose:
            print("\n" + "=" * 60)
            print("PHASE 2: Finding DOMAIN-SPECIFIC directions")
            print("=" * 60)
        
        # First, project ALL data to null space of general directions
        projected_domains = {}
        
        for domain_name, domain in self.domains.items():
            X_train_proj = domain.X_train.copy()
            X_test_proj = domain.X_test.copy()
            
            for direction in general_directions:
                X_train_proj = self._project_to_nullspace(X_train_proj, direction.vector)
                X_test_proj = self._project_to_nullspace(X_test_proj, direction.vector)
            
            projected_domains[domain_name] = (X_train_proj, X_test_proj)
        
        # Run INLP on each domain's projected data
        domain_specific_directions = {}
        specific_cross_accuracy = {}
        
        for domain_name in domain_names:
            if verbose:
                print(f"\n--- {domain_name} ---")
            
            domain = self.domains[domain_name]
            X_train_proj, X_test_proj = projected_domains[domain_name]
            
            # Run INLP on projected data
            specific_dirs = self._run_inlp(
                X_train_proj, domain.y_train,
                X_test_proj, domain.y_test,
                self.n_specific,
                source_name=domain_name,
            )
            
            domain_specific_directions[domain_name] = specific_dirs
            
            if verbose:
                for d in specific_dirs:
                    print(f"  k={d.iteration}: train={d.train_accuracy:.3f}, "
                          f"test={d.test_accuracy:.3f}")
            
            # Evaluate on all domains (using projected data)
            cross_acc = np.zeros((self.n_specific, len(domain_names)))
            
            for i, direction in enumerate(specific_dirs):
                for j, other_name in enumerate(domain_names):
                    _, other_X_test_proj = projected_domains[other_name]
                    other_y_test = self.domains[other_name].y_test
                    
                    acc = direction.evaluate(other_X_test_proj, other_y_test)
                    cross_acc[i, j] = acc
            
            specific_cross_accuracy[domain_name] = cross_acc
        
        # Store result
        self.result = SharedFirstResult(
            general_directions=general_directions,
            domain_specific_directions=domain_specific_directions,
            general_cross_accuracy=general_cross_accuracy,
            specific_cross_accuracy=specific_cross_accuracy,
            domain_names=domain_names,
        )
        
        return self.result
    
    def _run_leave_one_out(self, domain_names: List[str], verbose: bool) -> SharedFirstResult:
        """Leave-one-out run: for each domain i, compute general from all except i."""
        
        if verbose:
            print("=" * 60)
            print("LEAVE-ONE-OUT MODE")
            print("For each domain i: general directions from all domains EXCEPT i")
            if self.balance_domains:
                print("(Domain balancing enabled)")
            print("=" * 60)
        
        domain_specific_directions = {}
        specific_cross_accuracy = {}
        
        # We'll store LOO general directions for each domain
        # For the result, we'll report cross-accuracy where each domain is evaluated
        # using the general directions computed without it
        general_cross_accuracy = np.zeros((self.n_general, len(domain_names)))
        
        # Also store a "representative" set of general directions (from first LOO)
        representative_general_directions = None
        
        for target_idx, target_domain_name in enumerate(domain_names):
            if verbose:
                print(f"\n{'='*60}")
                print(f"TARGET DOMAIN: {target_domain_name}")
                print(f"{'='*60}")
            
            # ==================================================
            # PHASE 1: General directions from all EXCEPT target
            # ==================================================
            if verbose:
                print(f"\nPhase 1: General directions from all except {target_domain_name}")
            
            # Pool all domains except target (with optional balancing)
            other_domains = [d for name, d in self.domains.items() if name != target_domain_name]
            
            X_train_pooled, y_train_pooled, X_test_pooled, y_test_pooled = \
                self._pool_domains_balanced(other_domains, verbose=verbose)
            
            if verbose:
                print(f"  Pooled (excl. target): {len(y_train_pooled)} train, {len(y_test_pooled)} test")
            
            # Run INLP on pooled data (without target)
            general_directions_for_target = self._run_inlp(
                X_train_pooled, y_train_pooled,
                X_test_pooled, y_test_pooled,
                self.n_general,
                source_name=f"general_excl_{target_domain_name}",
            )
            
            # Store LOO general directions
            self.general_directions_loo[target_domain_name] = general_directions_for_target
            
            # Use first set as representative
            if representative_general_directions is None:
                representative_general_directions = general_directions_for_target
            
            if verbose:
                print("  General directions (excl. target):")
                for d in general_directions_for_target:
                    print(f"    k={d.iteration}: train={d.train_accuracy:.3f}, "
                          f"test={d.test_accuracy:.3f}")
            
            # Evaluate these general directions on TARGET domain (held out)
            for i, direction in enumerate(general_directions_for_target):
                target_domain = self.domains[target_domain_name]
                acc = direction.evaluate(target_domain.X_test, target_domain.y_test)
                general_cross_accuracy[i, target_idx] = acc
            
            if verbose:
                print(f"  General directions accuracy on held-out {target_domain_name}:")
                for i in range(self.n_general):
                    print(f"    k={i}: {general_cross_accuracy[i, target_idx]:.3f}")
            
            # ==================================================
            # PHASE 2: Domain-specific for target
            # ==================================================
            if verbose:
                print(f"\nPhase 2: Specific directions for {target_domain_name}")
            
            # Project target domain data to null space of its LOO general directions
            target_domain = self.domains[target_domain_name]
            X_train_proj = target_domain.X_train.copy()
            X_test_proj = target_domain.X_test.copy()
            
            for direction in general_directions_for_target:
                X_train_proj = self._project_to_nullspace(X_train_proj, direction.vector)
                X_test_proj = self._project_to_nullspace(X_test_proj, direction.vector)
            
            # Run INLP on projected target data
            specific_dirs = self._run_inlp(
                X_train_proj, target_domain.y_train,
                X_test_proj, target_domain.y_test,
                self.n_specific,
                source_name=target_domain_name,
            )
            
            domain_specific_directions[target_domain_name] = specific_dirs
            
            if verbose:
                print(f"  Specific directions for {target_domain_name}:")
                for d in specific_dirs:
                    print(f"    k={d.iteration}: train={d.train_accuracy:.3f}, "
                          f"test={d.test_accuracy:.3f}")
            
            # Evaluate specific directions on all domains
            # For this, we need to project each domain using target's LOO general directions
            cross_acc = np.zeros((self.n_specific, len(domain_names)))
            
            for i, direction in enumerate(specific_dirs):
                for j, other_name in enumerate(domain_names):
                    other_domain = self.domains[other_name]
                    
                    # Project other domain using target's LOO general directions
                    other_X_test_proj = other_domain.X_test.copy()
                    for gen_dir in general_directions_for_target:
                        other_X_test_proj = self._project_to_nullspace(other_X_test_proj, gen_dir.vector)
                    
                    acc = direction.evaluate(other_X_test_proj, other_domain.y_test)
                    cross_acc[i, j] = acc
            
            specific_cross_accuracy[target_domain_name] = cross_acc
            
            if verbose:
                target_idx_in_list = domain_names.index(target_domain_name)
                print(f"  Specific directions cross-accuracy (self vs others):")
                for i in range(self.n_specific):
                    self_acc = cross_acc[i, target_idx_in_list]
                    other_accs = [cross_acc[i, j] for j in range(len(domain_names)) 
                                  if j != target_idx_in_list]
                    mean_other = np.mean(other_accs)
                    print(f"    k={i}: self={self_acc:.3f}, other_mean={mean_other:.3f}")
        
        # Store result
        self.result = SharedFirstResult(
            general_directions=representative_general_directions,  # Representative set
            domain_specific_directions=domain_specific_directions,
            general_cross_accuracy=general_cross_accuracy,
            specific_cross_accuracy=specific_cross_accuracy,
            domain_names=domain_names,
        )
        
        return self.result
    
    def print_summary(self) -> None:
        """Print a summary of results."""
        if self.result is None:
            print("No results yet. Run the algorithm first.")
            return
        
        r = self.result
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        # Settings info
        settings = f"loss={self.classifier_loss}, normalize={self.normalize}, center={self.center}"
        if self.leave_one_out:
            settings += ", leave_one_out=True"
        if self.balance_domains:
            settings += ", balance_domains=True"
        print(f"\nSettings: {settings}")
        if self.center:
            print(f"Centering mode: {self.center_mode}")
        
        # PCA info
        if self.pca_dim is not None:
            original_dim = list(self.domains.values())[0].original_dim
            current_dim = list(self.domains.values())[0].dim
            print(f"PCA: {original_dim} -> {current_dim} dimensions (mode: {self.pca_mode})")
        
        # General directions
        print("\n## General Directions")
        if self.leave_one_out:
            print("(Leave-One-Out mode: each column shows accuracy on held-out domain)")
            print("General directions for domain i were trained on all domains EXCEPT i.\n")
        else:
            print("These directions were found from pooled data and should work across all domains.\n")
        
        print(f"{'k':<4} {'Train':>8} {'Test':>8} | Per-domain test accuracy")
        print("-" * 70)
        for i, d in enumerate(r.general_directions):
            domain_accs = " ".join([f"{r.general_cross_accuracy[i,j]:.2f}" 
                                    for j in range(len(r.domain_names))])
            print(f"{i:<4} {d.train_accuracy:>8.3f} {d.test_accuracy:>8.3f} | {domain_accs}")
        
        # Domain-specific directions
        print("\n## Domain-Specific Directions")
        if self.leave_one_out:
            print("These directions were found AFTER projecting out domain-specific general directions.")
            print("(Each domain used general directions computed from all OTHER domains.)")
        else:
            print("These directions were found AFTER projecting out general directions.")
        print("They should work on their source domain but NOT on others.\n")
        
        for domain_name in r.domain_names:
            specific_dirs = r.domain_specific_directions[domain_name]
            cross_acc = r.specific_cross_accuracy[domain_name]
            
            # Find column index for this domain
            domain_idx = r.domain_names.index(domain_name)
            
            print(f"\n### {domain_name}")
            print(f"{'k':<4} {'Self':>8} | {'Mean Other':>10} | Δ (Self - Other)")
            print("-" * 50)
            
            for i, d in enumerate(specific_dirs):
                self_acc = cross_acc[i, domain_idx]
                other_accs = [cross_acc[i, j] for j in range(len(r.domain_names)) 
                             if j != domain_idx]
                mean_other = np.mean(other_accs)
                delta = self_acc - mean_other
                
                print(f"{i:<4} {self_acc:>8.3f} | {mean_other:>10.3f} | {delta:>+.3f}")

# ============================================================================
# Utility Functions
# ============================================================================

def load_directions(results_dir: str, original_space: bool = True) -> Dict[str, np.ndarray]:
    """
    Load saved directions from results directory.
    
    Args:
        results_dir: Path to results directory
        original_space: If True and PCA was used, load original-space directions.
                       If False, load PCA-space directions.
    
    Returns:
        Dict with 'general' and 'specific_{domain}' keys
    """
    results_path = Path(results_dir)
    
    if original_space and (results_path / 'directions_original_space.npz').exists():
        data = np.load(results_path / 'directions_original_space.npz')
    else:
        data = np.load(results_path / 'directions.npz')
    
    return dict(data)


def apply_direction(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Apply a direction to data to get scores.
    
    Args:
        X: Data matrix (n_samples, dim)
        direction: Direction vector (dim,)
    
    Returns:
        scores: (n_samples,)
    """
    return X @ direction

# ============================================================================
# Main
# ============================================================================

import argparse
parser = argparse.ArgumentParser(description="Shared-First INLP")
parser.add_argument("--data", "-d", type=str, required=False,
                    default="../activations_all_layers_llama-8b.h5",
                    help="Path to HDF5 file")
parser.add_argument("--layer", type=str, default="layer_15",
                    help="Layer name in HDF5")
parser.add_argument("--n-general", type=int, default=5,
                    help="Number of general directions")
parser.add_argument("--n-specific", type=int, default=5,
                    help="Number of domain-specific directions per domain")
parser.add_argument("--alpha", type=float, default=1e-6,
                    help="Regularization strength for SVM")
parser.add_argument("--max-iter", type=int, default=50000,
                    help="Max iterations for SGDClassifier")
parser.add_argument("--tol", type=float, default=1e-4,
                    help="Tolerance for convergence")
parser.add_argument("--loss", type=str, default="log_loss",
                    choices=["hinge", "log_loss"],
                    help="Loss function: 'hinge' for SVM, 'log_loss' for logistic regression")
parser.add_argument("--normalize", action="store_true", default=False,
                    help="Use StandardScaler before fitting classifier")
parser.add_argument("--center", action="store_true", default=False,
                    help="Center data by subtracting mean before processing")
parser.add_argument("--center-mode", type=str, default="shared",
                    choices=["shared", "individual"],
                    help="Centering mode: 'shared' (global mean) or 'individual' (per-domain mean)")
parser.add_argument("--pca-dim", type=int, default=None,
                    help="Target dimensionality for PCA (default: None = no PCA)")
parser.add_argument("--pca-mode", type=str, default="shared",
                    choices=["shared", "individual"],
                    help="PCA mode: 'shared' (fit on all data) or 'individual' (fit per domain)")
parser.add_argument("--leave-one-out", "--loo", action="store_true", default=False,
                    help="Use leave-one-out mode: for each domain i, compute general directions "
                         "from all domains EXCEPT i to prevent leakage of domain-specific signal")
parser.add_argument("--balance-domains", action="store_true", default=False,
                    help="Balance domains in Phase 1 pooling: downsample each domain to the "
                         "size of the smallest domain to prevent larger domains from dominating")
parser.add_argument("--output", "-o", type=str, default="results",
                    help="Output directory")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--no-balance", action="store_true", default=False,
                    help="Disable enforcing 50/50 class balance (default: enforce balance)")

args = parser.parse_args("")

args.data = "./results/activations_layer_33_llama-70b-3.3.h5"
args.layer = 'layer_33'
args.output = 'results/stratified_inlp'
# args.data = "./results/activations_layer_15_llama-8b.h5"
# args.layer = 'layer_15'
# args.output = 'results/stratified_inlp_llama8b'

# args.n_general=3
# args.n_specific=3
args.n_general=10
args.n_specific=5

task_list = [
    'claims__definitional_gemini_600_full', 'claims__evidential_gemini_600_full',
    'claims__fictional_gemini_600_full', 
    'claims__logical_gemini_600_full',
    'ethics__commonsense',
    'sycophancy__mmlu_stem_same_conf_all',
    'repe_honesty__IF_dishonest',
    'roleplaying__plain',
    'insider_trading__upscale', 
    'sandbagging_v2__wmdp_mmlu',
    # 'internal_state__animals', 'internal_state__cities', 'internal_state__companies',
    # 'internal_state__elements', 'internal_state__facts', 'internal_state__inventions',
    # 'got__best',
    # 'repe_honesty__plain'
]

# Run algorithm
inlp = SharedFirstINLP(
    n_general=args.n_general,
    n_specific=args.n_specific,
    classifier_alpha=args.alpha,
    classifier_max_iter=args.max_iter,
    classifier_tol=args.tol,
    classifier_loss=args.loss,
    normalize=args.normalize,
    center=args.center,
    center_mode=args.center_mode,
    pca_dim=args.pca_dim,
    pca_mode=args.pca_mode,
    leave_one_out=args.leave_one_out,
    balance_domains=args.balance_domains,
    random_state=args.seed,
)
    
inlp.load_data(args.data, task_list, args.layer, enforce_balance=not args.no_balance)

result = inlp.run(verbose=True)
inlp.print_summary()