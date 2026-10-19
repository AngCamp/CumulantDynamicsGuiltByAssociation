# CumulantDynamicsGuiltByAssociation
A repo to analyze the relation between cumulants in spiking activity, their associated dynamic systems relations, the LFP, behaviour, and regional gene expression.

## NWB/DANDI workflow

This repository now includes a small Python package for:

- pulling only NWB assets from DANDI
- downloading spike and LFP files in parallel with a configurable worker count
- defining reusable time-interval sets for later cumulant searches

### Example

```python
from cumulant_dynamics import DandiNWBDownloader, TimeInterval, TimeIntervalSet

downloader = DandiNWBDownloader("000001", max_workers=4)
files = downloader.download_grouped_assets(
    {"spiking": ["spike"], "lpf": ["lfp"]},
    "./data",
)

intervals = TimeIntervalSet([TimeInterval(0.0, 2.5, "task")])
```
