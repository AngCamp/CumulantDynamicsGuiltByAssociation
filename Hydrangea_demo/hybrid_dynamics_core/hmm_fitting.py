import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class HMMFittingStep:
    """Fit Gaussian HMM models."""

    def _hmm_param_count(self, n_states, n_features):
        return n_states * (n_states - 1) + (n_states - 1) + n_states * n_features + n_states * n_features

    def build_folds(self, n_folds=5, strategy="temporal_block_randomized", seed=None):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")

        n_samples = self.spike_matrix.shape[0]
        if n_samples < n_folds:
            raise ValueError("Need more samples than folds.")

        rng = np.random.default_rng(seed if seed is not None else self.random_state)

        if strategy == "contiguous":
            edges = np.linspace(0, n_samples, n_folds + 1, dtype=int)
            self.fold_idx = [np.arange(edges[i], edges[i + 1]) for i in range(n_folds)]

        elif strategy == "temporal_block_randomized":
            edges = np.linspace(0, n_samples, n_folds + 1, dtype=int)
            block_ranges = list(zip(edges[:-1], edges[1:]))
            block_splits = [np.array_split(np.arange(a, b), n_folds) for a, b in block_ranges]

            fold_idx = []
            for fold in range(n_folds):
                selected = []
                for b in range(n_folds):
                    perm = np.argsort(rng.random(len(block_splits[b])))
                    selected.append(block_splits[b][perm[fold]])
                fold_idx.append(np.concatenate(selected))
            self.fold_idx = fold_idx

        else:
            raise ValueError(f"Unknown fold strategy: {strategy}")

        return self.fold_idx

    def fit_hmm(self, k_values=None, n_iter=100, covariance_type="diag", verbose=False, use_cv=True, report="full"):
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")

        if k_values is None:
            k_values = list(range(10, 1, -1))

        self.hmm_models = {}
        rows = []

        n_obs = self.spike_matrix.shape[0]

        for n_states in k_values:
            hmm = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=self.random_state,
                verbose=verbose,
            )
            hmm.fit(self.spike_matrix)

            loglik = float(hmm.score(self.spike_matrix))
            n_params = self._hmm_param_count(n_states, self.spike_matrix.shape[1])

            if self.fold_idx is not None and use_cv:
                cv_scores = []
                for fold in self.fold_idx:
                    train_idx = np.setdiff1d(np.arange(n_obs), fold, assume_unique=True)
                    cv_hmm = GaussianHMM(
                        n_components=n_states,
                        covariance_type=covariance_type,
                        n_iter=n_iter,
                        random_state=self.random_state,
                        verbose=False,
                    )
                    cv_hmm.fit(self.spike_matrix[train_idx])
                    cv_scores.append(float(cv_hmm.score(self.spike_matrix[fold])))
                median_cv = float(np.median(cv_scores))
                cv_arr = np.asarray(cv_scores, dtype=float)
            else:
                median_cv = loglik
                cv_arr = np.array([loglik], dtype=float)

            rows.append({
                "n_states": n_states,
                "loglik": loglik,
                "AIC": -2 * loglik + 2 * n_params,
                "BIC": -2 * loglik + n_params * np.log(n_obs),
                "n_params": n_params,
                "median_cv_loglik": median_cv,
                "cv_fold_loglik": cv_arr.tolist(),
            })

            self.hmm_models[n_states] = hmm

        self.hmm_scores = pd.DataFrame(rows).set_index("n_states").sort_index()
        self.best_k = int(self.hmm_scores["median_cv_loglik"].idxmax())
        self.best_hmm = self.hmm_models[self.best_k]

        if report in ["full", "selected"]:
            print(self.hmm_scores[["AIC", "BIC", "loglik", "median_cv_loglik"]])

        return self.hmm_scores.copy()
