# Dataset generation for gravitational-wave physics

This repo is meant to generate data for ML tasks in gravitational wave physics. It is based on `ml4gw` and primarily uses its functionality.

To generate data, run
```python
python main.py --config config.yaml --data path/to/data/folder --out path/to/output/directory
```
with
- `config`: is the config file containing all general and signal-specific setups. Currently, `config.yaml` will only generate BBH signals.
- `data`: path to directory containing open (background) data. The data can be downloaded using the `load_data.py` script.
- `out`: output directory where to store the dataset.

The generated output are `sig.h5` and `bkg.h5` files, which contain the time series for the two detectors (`data`) to be used for classification. The `sig.h5` files additionally contain all parameters defining the signal waveform to be used in regression tasks or for more detailed studies.

