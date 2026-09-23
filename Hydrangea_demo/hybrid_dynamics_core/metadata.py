"""Session metadata tables: units, trials, conditions, and regions.

Every dataset carries different metadata. Units may or may not have a region
label, and most datasets have no cell-type annotation at all. Trials,
conditions, and regions arrive as tables whose columns are dataset-specific.
This module normalizes all of that into pandas DataFrames held on the analysis
object, so later steps (state occupancy by region, state-by-condition
contrasts, unit-level covariates of state membership) have one place to look.

Nothing here assumes a particular column name. Region and cell-type columns are
resolved from a candidate list, and every accessor returns None rather than
raising when the annotation is absent.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_REGION_KEYS = (
    "cell_area",
    "region",
    "brain_region",
    "location",
    "structure_acronym",
    "area",
)

DEFAULT_CELL_TYPE_KEYS = (
    "cell_type",
    "celltype",
    "cell_class",
    "unit_type",
    "putative_cell_type",
)


def resolve_metadata_key(table, requested=None, candidates=()):
    """Return a usable column name from a metadata table, or None.

    Parameters
    ----------
    table : pandas.DataFrame or None
        Metadata table to search.
    requested : str or None
        Explicitly requested column. Returned only if it exists.
    candidates : sequence of str
        Fallback column names, tried in order, when nothing is requested.
    """
    if table is None or not isinstance(table, pd.DataFrame) or table.shape[1] == 0:
        return None
    if requested is not None:
        return requested if requested in table.columns else None
    for key in candidates:
        if key in table.columns:
            return key
    return None


def as_metadata_table(data, index_name=None):
    """Coerce assorted metadata containers into a DataFrame.

    Accepts a DataFrame, a Series, a dict of arrays, a dict of scalars, a list
    of row dicts, a pynapple IntervalSet, or None.
    """
    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.DataFrame):
        table = data.copy()
    elif isinstance(data, pd.Series):
        table = data.to_frame()
    elif hasattr(data, "start") and hasattr(data, "end"):
        # pynapple IntervalSet-like
        table = pd.DataFrame(
            {"start": np.asarray(data.start, dtype=float), "end": np.asarray(data.end, dtype=float)}
        )
        for key in list(getattr(data, "metadata_columns", []) or []):
            try:
                values = np.asarray(data.get_info(key))
            except Exception:
                continue
            if values.ndim == 1 and len(values) == len(table):
                table[key] = values
    elif isinstance(data, dict):
        if len(data) == 0:
            table = pd.DataFrame()
        elif all(np.ndim(value) == 0 for value in data.values()):
            table = pd.DataFrame([data])
        else:
            table = pd.DataFrame(dict(data))
    elif isinstance(data, (list, tuple)):
        table = pd.DataFrame(list(data))
    else:
        raise TypeError(f"Cannot interpret metadata of type {type(data)!r} as a table.")

    if index_name is not None and index_name in table.columns:
        table = table.set_index(index_name)
    if index_name is not None and len(table.columns) > 0:
        table.index.name = index_name
    return table


def extract_unit_metadata(spike_group, index_name="unit_id"):
    """Pull per-unit metadata out of a pynapple TsGroup into a DataFrame.

    Returns an empty DataFrame when the group carries no annotation, which is
    the common case for datasets without region or cell-type labels.
    """
    if spike_group is None:
        return pd.DataFrame()

    index = pd.Index(np.asarray(getattr(spike_group, "index", [])), name=index_name)
    if len(index) == 0:
        return pd.DataFrame()

    # Newer pynapple exposes the metadata DataFrame directly.
    stored = getattr(spike_group, "metadata", None)
    if isinstance(stored, pd.DataFrame) and len(stored) == len(index):
        table = stored.copy()
        table.index = index
        return table

    data = {}
    for key in list(getattr(spike_group, "metadata_columns", []) or []):
        try:
            values = np.asarray(spike_group.get_info(key))
        except Exception:
            continue
        if values.ndim == 1 and len(values) == len(index):
            data[key] = values
    return pd.DataFrame(data, index=index)


def unit_inventory(unit_table, region_key=None, cell_type_key=None):
    """Unit counts as a region-by-cell-type crosstab with a leading Total.

    Datasets without region annotation collapse to a single 'all units' row;
    datasets without cell-type annotation return a Total-only column. Both are
    valid tables, so downstream plotting never needs a special case.
    """
    if unit_table is None or len(unit_table) == 0:
        return pd.DataFrame()

    region_key = resolve_metadata_key(unit_table, region_key, DEFAULT_REGION_KEYS)
    cell_type_key = resolve_metadata_key(unit_table, cell_type_key, DEFAULT_CELL_TYPE_KEYS)

    if region_key is None:
        regions = pd.Series(["all units"] * len(unit_table), index=unit_table.index, name="region")
    else:
        regions = unit_table[region_key].astype(str).rename("region")

    if cell_type_key is None:
        tab = pd.DataFrame(index=pd.Index(sorted(regions.unique()), name="region"))
    else:
        types = unit_table[cell_type_key].astype(str).rename("cell_type")
        tab = pd.crosstab(regions, types)

    totals = regions.value_counts().reindex(tab.index).fillna(0).astype(int)
    tab.insert(0, "Total", totals.values)
    return tab.sort_values("Total", ascending=False)


def summarize_table(table, name):
    """One-row description of a metadata table for the object summary."""
    if table is None or len(table) == 0:
        return {"table": name, "n_rows": 0, "n_columns": 0, "columns": []}
    return {
        "table": name,
        "n_rows": int(len(table)),
        "n_columns": int(table.shape[1]),
        "columns": list(table.columns),
    }


class MetadataStep:
    """Metadata tables attached to a single session.

    Four tables are tracked:

    - ``unit_metadata``     one row per recorded unit (region, cell type, depth,
                            quality metrics, gene panels later on)
    - ``trial_table``       one row per trial
    - ``condition_table``   one row per experimental condition
    - ``region_table``      one row per brain region

    All four are optional. The unit table is the only one that is filtered
    alongside the spike matrix: :meth:`normalize` drops silent units, and
    ``unit_metadata_filtered`` tracks the survivors so that per-unit covariates
    stay aligned with the columns of the observation matrix.
    """

    def _init_metadata(
        self,
        spike_group=None,
        unit_metadata=None,
        trial_table=None,
        condition_table=None,
        region_table=None,
    ):
        self.unit_metadata = extract_unit_metadata(spike_group)
        if unit_metadata is not None:
            self.set_unit_metadata(unit_metadata, merge=True)
        self.unit_metadata_filtered = self.unit_metadata.copy()

        self.trial_table = as_metadata_table(trial_table, index_name="trial_id")
        self.condition_table = as_metadata_table(condition_table, index_name="condition_id")
        self.region_table = as_metadata_table(region_table, index_name="region_id")

    def set_unit_metadata(self, table, merge=True):
        """Attach or merge a per-unit metadata table.

        Rows are matched on the index (unit id). With ``merge=True`` the new
        columns are joined onto whatever was extracted from the spike group.
        """
        incoming = as_metadata_table(table, index_name="unit_id")
        current = getattr(self, "unit_metadata", pd.DataFrame())

        if merge and current is not None and len(current) > 0 and len(incoming) > 0:
            if len(incoming) == len(current) and not incoming.index.equals(current.index):
                incoming = incoming.set_axis(current.index)
            new_columns = [c for c in incoming.columns if c not in current.columns]
            self.unit_metadata = current.join(incoming[new_columns], how="left")
        else:
            self.unit_metadata = incoming

        if getattr(self, "spike_matrix", None) is None:
            self.unit_metadata_filtered = self.unit_metadata.copy()
        return self.unit_metadata

    def set_trial_table(self, table):
        """Attach a per-trial metadata table."""
        self.trial_table = as_metadata_table(table, index_name="trial_id")
        return self.trial_table

    def set_condition_table(self, table):
        """Attach a per-condition metadata table."""
        self.condition_table = as_metadata_table(table, index_name="condition_id")
        return self.condition_table

    def set_region_table(self, table):
        """Attach a per-region metadata table."""
        self.region_table = as_metadata_table(table, index_name="region_id")
        return self.region_table

    def get_unit_metadata(self, filtered=True):
        """Return the per-unit table, optionally restricted to analyzed units."""
        if filtered:
            table = getattr(self, "unit_metadata_filtered", None)
            if table is not None and len(table) > 0:
                return table
        table = getattr(self, "unit_metadata", None)
        return table if table is not None else pd.DataFrame()

    def resolved_region_key(self, requested=None, filtered=True):
        """Region column actually present in the unit table, or None."""
        requested = requested if requested is not None else getattr(self, "region_key", None)
        return resolve_metadata_key(self.get_unit_metadata(filtered), requested, DEFAULT_REGION_KEYS)

    def resolved_cell_type_key(self, requested=None, filtered=True):
        """Cell-type column actually present in the unit table, or None."""
        requested = requested if requested is not None else getattr(self, "cell_type_key", None)
        return resolve_metadata_key(
            self.get_unit_metadata(filtered), requested, DEFAULT_CELL_TYPE_KEYS
        )

    def _unit_meta(self, key, filtered=True):
        """Per-unit values for one column as an array, or None if absent."""
        if key is None:
            return None
        table = self.get_unit_metadata(filtered)
        if table is None or key not in table.columns:
            return None
        return np.asarray(table[key])

    def unit_inventory(self, region_key=None, cell_type_key=None, filtered=True, print_table=False):
        """Unit counts by region and cell type; see :func:`unit_inventory`."""
        tab = unit_inventory(
            self.get_unit_metadata(filtered),
            region_key=self.resolved_region_key(region_key, filtered),
            cell_type_key=self.resolved_cell_type_key(cell_type_key, filtered),
        )
        if print_table and len(tab) > 0:
            print("\nUnit counts (region x cell_type, with Total):")
            print(tab)
        return tab

    def plot_unit_inventory(
        self,
        region_key=None,
        cell_type_key=None,
        filtered=True,
        ax=None,
        show=True,
        print_table=False,
    ):
        """Grouped bar chart of unit counts per region and cell type."""
        tab = self.unit_inventory(
            region_key=region_key,
            cell_type_key=cell_type_key,
            filtered=filtered,
            print_table=print_table,
        )

        if ax is None:
            width_in = 1.8 * max(len(tab), 1) + 3
            fig, ax = plt.subplots(1, 1, figsize=(min(width_in, 18), 5))
        else:
            fig = ax.figure

        if len(tab) == 0:
            ax.text(0.5, 0.5, "No unit metadata available", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return fig, ax

        areas = [str(a) for a in tab.index.tolist()]
        cols = tab.columns.tolist()
        n_bars = len(cols)
        x = np.arange(len(areas), dtype=float)
        width = 0.8 / n_bars
        cmap = plt.get_cmap("tab10")

        for i, col in enumerate(cols):
            offsets = x + (i - (n_bars - 1) / 2) * width
            bars = ax.bar(offsets, tab[col].values, width, label=str(col), color=cmap(i % 10))
            ax.bar_label(bars, padding=2, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(areas, rotation=30 if len(areas) > 4 else 0, ha="right" if len(areas) > 4 else "center")
        ax.set(xlabel="brain region", ylabel="unit count", title="Unit counts by region")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def metadata_summary(self):
        """Row-count and column listing for each attached metadata table."""
        rows = [
            summarize_table(self.get_unit_metadata(filtered=False), "unit_metadata"),
            summarize_table(getattr(self, "trial_table", None), "trial_table"),
            summarize_table(getattr(self, "condition_table", None), "condition_table"),
            summarize_table(getattr(self, "region_table", None), "region_table"),
        ]
        return pd.DataFrame(rows).set_index("table")

    def describe_metadata(self, as_text=True):
        """Print which metadata tables and annotation columns are available."""
        summary = self.metadata_summary()
        region_key = self.resolved_region_key(filtered=False)
        cell_type_key = self.resolved_cell_type_key(filtered=False)

        if as_text:
            print("Metadata tables")
            print("-" * 36)
            for name, row in summary.iterrows():
                cols = ", ".join(map(str, row["columns"])) if row["columns"] else "-"
                print(f"{name:<16} rows={row['n_rows']:<6} cols={row['n_columns']:<3} {cols}")
            print(f"region column:    {region_key or 'none'}")
            print(f"cell-type column: {cell_type_key or 'none'}")
        return summary
