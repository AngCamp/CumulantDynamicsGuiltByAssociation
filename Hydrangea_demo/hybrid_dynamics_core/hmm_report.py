import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HMMReportStep:
    """HMM summaries and plots."""

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

    def hmm_report(self, report=None):
        if self.hmm_scores is None:
            raise ValueError("Run fit_hmm() first.")

        report = self.report if report is None else report

        if report == "none":
            return None

        if report == "selected":
            ranked = self.hmm_scores.sort_values("median_cv_loglik", ascending=False)
            print("Ranked CV table:")
            print(ranked[["median_cv_loglik", "AIC", "BIC", "loglik"]])

            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(self.hmm_scores.index, self.hmm_scores["median_cv_loglik"], marker="o", color="tab:green")
            ax.set_title("Median temporal CV log-likelihood by K")
            ax.set_xlabel("n states")
            ax.set_ylabel("CV log-likelihood")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
            return fig

        if report == "full":
            ranked = self.hmm_scores.sort_values("median_cv_loglik", ascending=False)
            print("Full model comparison:")
            print(ranked[["median_cv_loglik", "AIC", "BIC", "loglik"]])

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
                axes[0, 0].scatter(self.bin_times_s, seq, c=seq, cmap="tab10", s=2, marker="s")
                axes[0, 0].set(title=f"State sequence (K={n_states})", xlabel="time (s)", ylabel="state", yticks=range(n_states))

                bars = axes[0, 1].bar(np.arange(n_states), occ, color="tab:blue")
                axes[0, 1].set(title="Occupancy (%)", xlabel="state", ylabel="% of time", xticks=range(n_states))
                axes[0, 1].grid(alpha=0.3, axis="y")
                for b, v in zip(bars, occ):
                    axes[0, 1].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

                bars = axes[1, 0].bar(np.arange(n_states), mean_dwell, color="tab:green")
                axes[1, 0].set(title="Mean dwell time", xlabel="state", ylabel="seconds", xticks=range(n_states))
                axes[1, 0].grid(alpha=0.3, axis="y")
                for b, v in zip(bars, mean_dwell):
                    axes[1, 0].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=8)

                im = axes[1, 1].imshow(T, cmap="viridis", vmin=0, vmax=1)
                axes[1, 1].set(title="Transition matrix", xlabel="to state", ylabel="from state", xticks=range(n_states), yticks=range(n_states))
                for i in range(n_states):
                    for j in range(n_states):
                        axes[1, 1].text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", color="w" if T[i, j] < 0.6 else "k", fontsize=7)
                fig.colorbar(im, ax=axes[1, 1], label="P(to|from)")

                fig.suptitle(
                    f"K={n_states} | switch_rate={diag['switch_rate_per_s']:.3f}/s | "
                    f"<1%={int((occ < 1).sum())} | <5%={int((occ < 5).sum())} | "
                    f"median CV loglik={self.hmm_scores.loc[n_states, 'median_cv_loglik']:.1f}",
                    y=1.02,
                )
                fig.tight_layout()
                plt.show()

            summary = pd.DataFrame(diagnostic_rows).sort_values("n_states", ascending=False)
            print(summary)
            return summary

        raise ValueError("report must be one of: 'full', 'selected', or 'none'")
