from gwpy.timeseries import TimeSeries, TimeSeriesDict
import yaml
from utils import load_config
from pathlib import Path

def load_data(config, data_dir: str):

    background_dir = data_dir / "background_data"
    background_dir.mkdir(parents=True, exist_ok=True)

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
        for ifo in config.general.ifos:
            ts_dict[ifo] = TimeSeries.fetch_open_data(ifo, start, end, cache=True)
        ts_dict = ts_dict.resample(config.general.sample_rate)
        ts_dict.write(fname, format="hdf5")

if __name__ == "__main__":

    config = load_config(config_path='config.yaml')
    data_dir = Path("./data")
    load_data(config, data_dir)
