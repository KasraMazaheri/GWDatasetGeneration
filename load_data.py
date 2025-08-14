from gwpy.timeseries import TimeSeries, TimeSeriesDict
from pathlib import Path
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Point this to whatever directory you want to house
# all of the data products this notebook creates
data_dir = Path("./data")

# And this to the directory where you want to download the data
background_dir = data_dir / "background_data"
background_dir.mkdir(parents=True, exist_ok=True)

# These are the GPS time of the start and end of the segments.
# There's no particular reason for these times, other than that they
# contain analysis-ready data
segments = [
    (1240579783, 1240587612), 
    (1240594562, 1240606748), 
    (1240624412, 1240644412),
    (1240644412, 1240654372),
    (1240658942, 1240668052),
]

for (start, end) in segments:
    # Download the data from GWOSC. This will take a few minutes.
    duration = end - start
    fname = background_dir / f"background-{start}-{duration}.hdf5"
    if fname.exists():
        continue

    ts_dict = TimeSeriesDict()
    for ifo in config['general']['ifos']:
        ts_dict[ifo] = TimeSeries.fetch_open_data(ifo, start, end, cache=True)
    ts_dict = ts_dict.resample(config['general']['sample_rate'])
    ts_dict.write(fname, format="hdf5")
