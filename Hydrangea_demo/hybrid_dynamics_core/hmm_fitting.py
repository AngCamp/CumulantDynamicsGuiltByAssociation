import pickle
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

try:
    from sklearn.exceptions import ConvergenceWarning
except ImportError:  # pragma: no cover - sklearn always ships this
    ConvergenceWarning = UserWarning


class HMMFittingStep:
    """Fit Gaussian HMM models across a state-count range.

    Default behaviour is to sweep a range of K values, while cross-validation is
    governed by the number of folds and the fold strategy. The default fold
    strategy is to split the recording into contiguous temporal segments, with an
    optional randomization inside each temporal block.
    """

    def _hmm_param_count(self, n_states, n_features):
        return n_states * (n_states - 1) + (n_states - 1) + n_states * n_features + n_states * n_features

    def fit_states(self, *args, **kwargs):
        """Fit latent states using the object's locked state discovery method."""
        if not hasattr(self, "state_discovery_method"):
            raise ValueError("Object missing state_discovery_method lock.")
        if self.state_discovery_method == "gaussian_hmm":
            return self.fit_hmm(*args, **kwargs)
        raise NotImplementedError(
            f"State discovery method '{self.state_discovery_method}' is not implemented yet."
        )

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

    def _fit_one_hmm(self, matrix, n_states, covariance_type, n_iter, verbose):
        """Fit a single Gaussian HMM and report how the EM run went."""
        hmm = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=self.random_state,
            verbose=verbose,
        )
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            hmm.fit(matrix)
        elapsed = time.perf_counter() - started

        monitor = getattr(hmm, "monitor_", None)
        converged = bool(getattr(monitor, "converged", False))
        iterations = int(getattr(monitor, "iter", 0) or len(getattr(monitor, "history", []) or []))
        return hmm, elapsed, converged, iterations

    def fit_hmm(self, n_states_min=2, n_states_max=10, k_fold=5, fold_strategy="temporal_segments", shuffle_within_segments=True, n_iter=100, covariance_type="diag", verbose=False, use_cv=True, report="full", event_intervals=None, show_progress=True):
        """Fit a sweep of Gaussian HMMs across a simple integer state-count range.

        Each K is fit once on the whole matrix (for AIC/BIC and the state
        sequence) and then k_fold more times on held-out splits (for the CV
        score that selects K). That is K * (1 + k_fold) EM runs, so progress
        reporting matters: `show_progress` prints where the sweep is, how long
        each fit took, and whether EM converged.

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
            hmmlearn's own EM monitor: one line per EM iteration with the
            running log-likelihood and its delta. Useful when diagnosing a
            single fit that will not converge, but it says nothing about which
            K or fold is running. Leave it off and use `show_progress` instead.
        use_cv : bool
            Whether to perform cross-validation using the configured folds.
        report : {'full', 'selected', 'none'}
            Reporting mode for the model comparison table.
        event_intervals : pynapple.IntervalSet or None
            Interval-based validation blocks when fold_strategy='event_intervals'.
        show_progress : bool
            Print one line per fit: K, fold, elapsed seconds, EM iterations, and
            whether EM converged, plus a per-K summary.

        Returns
        -------
        pandas.DataFrame
            One row per K. Log-likelihoods, information criteria, the spread of
            the CV folds (mean/median/std/min/max), EM convergence, and timings.
        """
        if self.spike_matrix is None:
            raise ValueError("Run normalize() first.")
        if hasattr(self, "state_discovery_method") and self.state_discovery_method != "gaussian_hmm":
            raise NotImplementedError(
                f"This object is locked to state_discovery_method='{self.state_discovery_method}', "
                "but fit_hmm implements 'gaussian_hmm'."
            )

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

        n_obs, n_features = self.spike_matrix.shape
        run_cv = use_cv and self.fold_idx is not None
        n_folds = len(self.fold_idx) if run_cv else 0
        n_fits = len(k_values) * (1 + n_folds)
        sweep_started = time.perf_counter()

        if show_progress:
            print(
                f"Fitting {len(k_values)} models (K={min(k_values)}..{max(k_values)}) on "
                f"{n_obs} bins x {n_features} units | {covariance_type} covariance | "
                f"{n_folds} CV folds -> {n_fits} EM runs"
            )

        for model_i, n_states in enumerate(k_values, start=1):
            if show_progress:
                print(f"  K={n_states:<3} [{model_i}/{len(k_values)}]", flush=True)

            hmm, fit_seconds, converged, iterations = self._fit_one_hmm(
                self.spike_matrix, n_states, covariance_type, n_iter, verbose
            )
            loglik = float(hmm.score(self.spike_matrix))
            n_params = self._hmm_param_count(n_states, n_features)

            if show_progress:
                flag = "converged" if converged else f"HIT n_iter={n_iter}"
                print(
                    f"      full fit   {fit_seconds:6.1f}s  {iterations:>3} iter  {flag:<16}"
                    f"loglik={loglik:,.0f}",
                    flush=True,
                )

            cv_seconds = 0.0
            if run_cv:
                cv_scores = []
                for fold_i, fold in enumerate(self.fold_idx, start=1):
                    if len(fold) == 0:
                        raise ValueError("Encountered an empty validation fold; cannot run CV scoring.")
                    train_idx = np.setdiff1d(np.arange(n_obs), fold, assume_unique=True)
                    if len(train_idx) == 0:
                        raise ValueError("Encountered an empty training fold; cannot run CV scoring.")

                    cv_hmm, fold_seconds, fold_converged, fold_iterations = self._fit_one_hmm(
                        self.spike_matrix[train_idx], n_states, covariance_type, n_iter, False
                    )
                    score = float(cv_hmm.score(self.spike_matrix[fold]))
                    cv_scores.append(score)
                    cv_seconds += fold_seconds

                    if show_progress:
                        flag = "converged" if fold_converged else f"HIT n_iter={n_iter}"
                        print(
                            f"      fold {fold_i}/{n_folds}  {fold_seconds:6.1f}s  "
                            f"{fold_iterations:>3} iter  {flag:<16}"
                            f"held-out loglik={score:,.0f}",
                            flush=True,
                        )

                cv_arr = np.asarray(cv_scores, dtype=float)
            else:
                cv_arr = np.array([loglik], dtype=float)

            rows.append({
                "n_states": n_states,
                "n_params": n_params,
                "loglik": loglik,
                "loglik_per_bin": loglik / n_obs,
                "AIC": -2 * loglik + 2 * n_params,
                "BIC": -2 * loglik + n_params * np.log(n_obs),
                "mean_cv_loglik": float(np.mean(cv_arr)),
                "median_cv_loglik": float(np.median(cv_arr)),
                "std_cv_loglik": float(np.std(cv_arr)),
                "min_cv_loglik": float(np.min(cv_arr)),
                "max_cv_loglik": float(np.max(cv_arr)),
                "converged": converged,
                "n_iter_run": iterations,
                "fit_seconds": fit_seconds,
                "cv_seconds": cv_seconds,
                "total_seconds": fit_seconds + cv_seconds,
                "cv_fold_loglik": cv_arr.tolist(),
            })

            self.hmm_models[n_states] = hmm

            if show_progress:
                elapsed = time.perf_counter() - sweep_started
                done = model_i * (1 + n_folds)
                remaining = (elapsed / done) * (n_fits - done) if done else 0.0
                print(
                    f"      K={n_states} done in {fit_seconds + cv_seconds:.1f}s | "
                    f"median CV loglik={np.median(cv_arr):,.0f} | "
                    f"elapsed {elapsed / 60:.1f} min, ~{remaining / 60:.1f} min left",
                    flush=True,
                )

        self.hmm_scores = pd.DataFrame(rows).set_index("n_states").sort_index()
        self.best_k = int(self.hmm_scores["median_cv_loglik"].idxmax())
        self.best_hmm = self.hmm_models[self.best_k]
        self.state_labels_ = self.best_hmm.predict(self.spike_matrix)
        self.hmm_fit_seconds = time.perf_counter() - sweep_started
        if hasattr(self, "_record_timing"):
            self._record_timing("fit_hmm", self.hmm_fit_seconds)
        if hasattr(self, "_mark_checkpoint"):
            self._mark_checkpoint("states_discovered")

        if show_progress:
            n_unconverged = int((~self.hmm_scores["converged"]).sum())
            print(
                f"\nSweep finished in {self.hmm_fit_seconds / 60:.1f} min. "
                f"Best K={self.best_k} by median CV log-likelihood."
                + (f" {n_unconverged} model(s) hit the n_iter ceiling." if n_unconverged else "")
            )

        if report in ["full", "selected"]:
            print("\nModel comparison:")
            print(self.hmm_score_table())

        return self.hmm_scores.copy()

    def hmm_score_table(self, sort_by="n_states"):
        """Readable model-comparison table: scores, CV spread, and timings."""
        if self.hmm_scores is None:
            raise ValueError("Run fit_hmm() first.")
        columns = [
            "n_params",
            "loglik_per_bin",
            "AIC",
            "BIC",
            "mean_cv_loglik",
            "median_cv_loglik",
            "std_cv_loglik",
            "min_cv_loglik",
            "max_cv_loglik",
            "converged",
            "n_iter_run",
            "total_seconds",
        ]
        columns = [c for c in columns if c in self.hmm_scores.columns]
        table = self.hmm_scores[columns].copy()
        if sort_by in ("median_cv_loglik", "mean_cv_loglik"):
            table = table.sort_values(sort_by, ascending=False)
        return table.round(3)

    def _summarize_sequence(self, seq, n_states, dt):
        seq = np.asarray(seq, dtype=int)
        total = len(seq)
        occ_pct = np.array([(seq == s).mean() * 100 for s in range(n_states)])

        visit_counts = np.zeros(n_states, dtype=int)
        dwell_runs = {s: [] for s in range(n_states)}

        start = 0
        for i in range(1, total + 1):
            if i == total or seq[i] != seq[start]:
                s = seq[start]
                run_len = i - start
                dwell_runs[s].append(run_len * dt)
                visit_counts[s] += 1
                start = i

        mean_dwell = np.array([
            np.mean(dwell_runs[s]) if len(dwell_runs[s]) else 0.0
            for s in range(n_states)
        ])

        median_dwell = np.array([
            np.median(dwell_runs[s]) if len(dwell_runs[s]) else 0.0
            for s in range(n_states)
        ])

        switch_rate = np.sum(np.diff(seq) != 0) / ((total - 1) * dt) if total > 1 else 0.0
        return {
            "occupancy_pct": occ_pct,
            "visit_counts": visit_counts,
            "mean_dwell_s": mean_dwell,
            "median_dwell_s": median_dwell,
            "switch_rate_per_s": switch_rate,
        }

    def plot_selected_cv_curve(self, ax=None, show=True):
        if self.hmm_scores is None:
            raise ValueError("Run fit_hmm() first.")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        else:
            fig = ax.figure
        ax.plot(self.hmm_scores.index, self.hmm_scores["median_cv_loglik"], marker="o", color="tab:green")
        ax.set_title("Median temporal CV log-likelihood by K")
        ax.set_xlabel("n states")
        ax.set_ylabel("CV log-likelihood")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_state_sequence(self, seq, n_states, ax=None, show=True, title=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        else:
            fig = ax.figure
        ax.scatter(self.bin_times_s, seq, c=seq, cmap="tab10", s=2, marker="s")
        ax.set(
            title=title or f"State sequence (K={n_states})",
            xlabel="time (s)",
            ylabel="state",
            yticks=range(n_states),
        )
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_state_occupancy(self, occ, n_states, ax=None, show=True, title="Occupancy (%)"):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        else:
            fig = ax.figure
        bars = ax.bar(np.arange(n_states), occ, color="tab:blue")
        ax.set(title=title, xlabel="state", ylabel="% of time", xticks=range(n_states))
        for b, v in zip(bars, occ):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_state_dwell(self, dwell, n_states, ax=None, show=True, title="Mean dwell time"):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        else:
            fig = ax.figure
        bars = ax.bar(np.arange(n_states), dwell, color="tab:green")
        ax.set(title=title, xlabel="state", ylabel="seconds", xticks=range(n_states))
        for b, v in zip(bars, dwell):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_transition_matrix(self, T, n_states, ax=None, show=True, title="Transition matrix"):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        else:
            fig = ax.figure
        im = ax.imshow(T, cmap="viridis", vmin=0, vmax=1)
        ax.set(title=title, xlabel="to state", ylabel="from state", xticks=range(n_states), yticks=range(n_states))
        for i in range(n_states):
            for j in range(n_states):
                ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", color="w" if T[i, j] < 0.6 else "k", fontsize=7)
        fig.colorbar(im, ax=ax, label="P(to|from)")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def hmm_report(self, report=None):
        """Display a selected or full HMM diagnostic report.

        Display-only: returns None. Model comparison scores stay on
        ``hmm_scores``; the full report also stores ``hmm_diagnostics``.
        """
        if self.hmm_scores is None:
            raise ValueError("Run fit_hmm() first.")

        report = self.report if report is None else report

        if report == "none":
            return None

        if report == "selected":
            print("Models ranked by median CV log-likelihood:")
            print(self.hmm_score_table(sort_by="median_cv_loglik"))
            self.plot_selected_cv_curve(show=True)
            return None

        if report == "full":
            print("Models ranked by median CV log-likelihood:")
            print(self.hmm_score_table(sort_by="median_cv_loglik"))

            dt = float(np.median(np.diff(self.bin_times_s))) if len(self.bin_times_s) > 1 else 1.0
            diagnostic_rows = []

            for n_states in sorted(self.hmm_models):
                hmm = self.hmm_models[n_states]
                seq = hmm.predict(self.spike_matrix)
                T = hmm.transmat_
                diag = self._summarize_sequence(seq, n_states, dt)
                occ = diag["occupancy_pct"]
                mean_dwell = diag["mean_dwell_s"]

                diagnostic_rows.append({
                    "n_states": n_states,
                    "switch_rate_per_s": diag["switch_rate_per_s"],
                    "min_occupancy_pct": occ.min(),
                    "median_occupancy_pct": np.median(occ),
                    "n_states_lt_1pct": int((occ < 1.0).sum()),
                    "n_states_lt_5pct": int((occ < 5.0).sum()),
                    "max_mean_dwell_s": mean_dwell.max(),
                    "median_cv_loglik": self.hmm_scores.loc[n_states, "median_cv_loglik"],
                })

                fig, axes = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={"height_ratios": [2, 1]})
                self.plot_state_sequence(seq, n_states, ax=axes[0, 0], show=False)
                self.plot_state_occupancy(occ, n_states, ax=axes[0, 1], show=False)
                self.plot_state_dwell(mean_dwell, n_states, ax=axes[1, 0], show=False)
                self.plot_transition_matrix(T, n_states, ax=axes[1, 1], show=False)

                fig.suptitle(
                    f"K={n_states} | switch_rate={diag['switch_rate_per_s']:.3f}/s | "
                    f"<1%={int((occ < 1).sum())} | <5%={int((occ < 5).sum())} | "
                    f"median CV loglik={self.hmm_scores.loc[n_states, 'median_cv_loglik']:.1f}",
                    y=1.02,
                )
                fig.tight_layout()
                plt.show()

            summary = pd.DataFrame(diagnostic_rows).sort_values("n_states", ascending=False)
            self.hmm_diagnostics = summary
            print(summary)
            return None

        raise ValueError("report must be one of: 'full', 'selected', or 'none'")

    def save_hmm_model(self, path, n_states=None):
        """Serialize a fitted HMM and its selection metadata to disk."""
        if n_states is None:
            if self.best_k is None:
                raise ValueError("Run fit_hmm() first or provide n_states explicitly.")
            n_states = self.best_k

        if n_states not in self.hmm_models:
            raise KeyError(f"No fitted HMM available for n_states={n_states}.")

        artifact = {
            "n_states": int(n_states),
            "best_k": int(self.best_k) if self.best_k is not None else int(n_states),
            "hmm_model": self.hmm_models[n_states],
            "hmm_scores": self.hmm_scores,
            "bin_size_s": self.bin_size_s,
            "random_state": self.random_state,
            "report": self.report,
            "embedding_method": getattr(self, "embedding_method", None),
            "state_discovery_method": getattr(self, "state_discovery_method", None),
            "analysis_stage": getattr(self, "analysis_stage", None),
        }

        with open(path, "wb") as handle:
            pickle.dump(artifact, handle)

        return path

    def load_hmm_model(self, path):
        """Restore a previously serialized HMM artifact from disk."""
        with open(path, "rb") as handle:
            artifact = pickle.load(handle)

        n_states = int(artifact["n_states"])
        hmm_model = artifact["hmm_model"]

        saved_embedding_method = artifact.get("embedding_method", None)
        saved_state_method = artifact.get("state_discovery_method", None)
        if saved_embedding_method is not None and hasattr(self, "embedding_method"):
            if str(saved_embedding_method).lower() != str(self.embedding_method).lower():
                raise ValueError(
                    "Saved model embedding method does not match object lock: "
                    f"saved='{saved_embedding_method}', current='{self.embedding_method}'."
                )
        if saved_state_method is not None and hasattr(self, "state_discovery_method"):
            if str(saved_state_method).lower() != str(self.state_discovery_method).lower():
                raise ValueError(
                    "Saved model state discovery method does not match object lock: "
                    f"saved='{saved_state_method}', current='{self.state_discovery_method}'."
                )

        self.hmm_models = {n_states: hmm_model}
        self.best_k = int(artifact.get("best_k", n_states))
        self.best_hmm = hmm_model
        self.hmm_scores = artifact.get("hmm_scores", self.hmm_scores)
        if self.spike_matrix is not None:
            self.state_labels_ = self.best_hmm.predict(self.spike_matrix)
        if hasattr(self, "_mark_checkpoint"):
            self._mark_checkpoint("states_discovered")

        return artifact
