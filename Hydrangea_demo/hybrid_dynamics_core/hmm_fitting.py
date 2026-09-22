import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class HMMFittingStep:
    """Fit Gaussian HMM models across a state-count range.

    Default behaviour is to sweep a range of K values, while cross-validation is
    governed by the number of folds and the fold strategy. The default fold
    strategy is to split the recording into contiguous temporal segments, with an
    optional randomization inside each temporal block.
    """

    def _hmm_param_count(self, n_states, n_features):
        return n_states * (n_states - 1) + (n_states - 1) + n_states * n_features + n_states * n_features

    def build_folds(self, k_fold=5, fold_strategy="temporal_segments", shuffle_within_segments=True, seed=None, event_intervals=None):
        """Build validation folds for HMM cross-validation.

        Parameters
        ----------
        k_fold : int
            Number of folds used for cross-validation.
        fold_strategy : {'temporal_segments', 'contiguous', 'event_intervals'}
            How the time series is partitioned into validation sets.
            - temporal_segments: split the recording into contiguous temporal blocks
            - contiguous: naive contiguous folds
            - event_intervals: accept a pynapple IntervalSet of event blocks and
              shuffle these blocks before assigning them to folds
        shuffle_within_segments : bool
            If True, randomize within each temporal segment before assigning the
            held-out portion. This preserves long-term temporal structure while
            preventing a single local region dominating the validation set.
        seed : int or None
            Random seed for reproducible fold assignment.
        event_intervals : pynapple.IntervalSet or None
            Explicit event-based interval set used when fold_strategy='event_intervals'.
        """
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")

        n_samples = self.spike_matrix.shape[0]
        if n_samples < k_fold:
            raise ValueError("Need more samples than folds.")

        rng = np.random.default_rng(seed if seed is not None else self.random_state)
        self.fold_idx = None

        if fold_strategy == "contiguous":
            edges = np.linspace(0, n_samples, k_fold + 1, dtype=int)
            self.fold_idx = [np.arange(edges[i], edges[i + 1]) for i in range(k_fold)]

        elif fold_strategy == "temporal_segments":
            edges = np.linspace(0, n_samples, k_fold + 1, dtype=int)
            block_ranges = list(zip(edges[:-1], edges[1:]))
            block_splits = [np.array_split(np.arange(a, b), k_fold) for a, b in block_ranges]

            fold_idx = []
            for fold in range(k_fold):
                selected = []
                for b in range(k_fold):
                    if shuffle_within_segments:
                        perm = np.argsort(rng.random(len(block_splits[b])))
                        chosen = block_splits[b][perm[fold]]
                    else:
                        chosen = block_splits[b][fold]
                    selected.append(chosen)
                fold_idx.append(np.concatenate(selected))
            self.fold_idx = fold_idx

        elif fold_strategy == "event_intervals":
            if event_intervals is None:
                raise ValueError("event_intervals requires a pynapple IntervalSet object.")

            if not hasattr(event_intervals, "start") or not hasattr(event_intervals, "end"):
                raise TypeError("event_intervals must be a pynapple IntervalSet-like object with start/end attributes.")

            interval_starts = np.asarray(event_intervals.start)
            interval_ends = np.asarray(event_intervals.end)
            time_points = self.bin_times_s

            candidates = []
            for s, e in zip(interval_starts, interval_ends):
                mask = (time_points >= s) & (time_points < e)
                if np.any(mask):
                    candidates.append(np.nonzero(mask)[0])

            if len(candidates) == 0:
                raise ValueError("No time points fell inside the supplied event intervals.")

            order = np.arange(len(candidates))
            if len(candidates) > 1:
                order = rng.permutation(order)

            fold_idx = [[] for _ in range(k_fold)]
            for idx, block_idx in enumerate(order):
                fold_idx[idx % k_fold].append(candidates[block_idx])

            if any(len(x) == 0 for x in fold_idx):
                raise ValueError(
                    "event_intervals produced empty folds. Reduce k_fold or provide more interval blocks."
                )
            self.fold_idx = [np.concatenate(x) for x in fold_idx]

        else:
            raise ValueError(f"Unknown fold strategy: {fold_strategy}")

        if any(len(fold) == 0 for fold in self.fold_idx):
            raise ValueError(
                "One or more validation folds are empty. Check fold strategy and k_fold for this dataset."
            )

        return self.fold_idx

    def fit_hmm(self, n_states_min=2, n_states_max=10, k_fold=5, fold_strategy="temporal_segments", shuffle_within_segments=True, n_iter=100, covariance_type="diag", verbose=False, use_cv=True, report="full", event_intervals=None):
        """Fit a sweep of Gaussian HMMs across a simple integer state-count range.

        Parameters
        ----------
        n_states_min : int
            Minimum number of latent states to test.
        n_states_max : int
            Maximum number of latent states to test.
        k_fold : int
            Number of cross-validation folds used when use_cv is True.
        fold_strategy : {'temporal_segments', 'contiguous', 'event_intervals'}
            How the time series is partitioned into validation sets.
        shuffle_within_segments : bool
            Whether to randomize subsegments inside each temporal block.
        n_iter : int
            Maximum EM iterations for each HMM fit.
        covariance_type : {'diag', 'full'}
            Gaussian HMM covariance form.
        verbose : bool
            Verbosity for hmmlearn.
        use_cv : bool
            Whether to perform cross-validation using the configured folds.
        report : {'full', 'selected', 'none'}
            Reporting mode for model comparison summary.
        event_intervals : pynapple.IntervalSet or None
            Interval-based validation blocks when fold_strategy='event_intervals'.
        """
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")

        if n_states_min > n_states_max:
            raise ValueError("n_states_min must be <= n_states_max.")

        k_values = list(range(n_states_max, n_states_min - 1, -1))
        if self.fold_idx is None:
            self.build_folds(
                k_fold=k_fold,
                fold_strategy=fold_strategy,
                shuffle_within_segments=shuffle_within_segments,
                seed=self.random_state,
                event_intervals=event_intervals,
            )

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
                    if len(fold) == 0:
                        raise ValueError("Encountered an empty validation fold; cannot run CV scoring.")
                    train_idx = np.setdiff1d(np.arange(n_obs), fold, assume_unique=True)
                    if len(train_idx) == 0:
                        raise ValueError("Encountered an empty training fold; cannot run CV scoring.")
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
