from utils import load_config
from injections import injection
import torch
import argparse
from pathlib import Path
import h5py
import os

def main(config_path: str, data_dir: str, output_dir: str):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)

    data_dir = Path(data_dir)
    out_dir = Path(output_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # generate signals
    signals, params = injection(config, data_dir=data_dir, device=device, inject=True)
    sig_data = signals.cpu().numpy()
    with h5py.File(out_dir / 'sig.h5', 'w') as h5f:
        h5f.create_dataset('data', data=sig_data)
        for k in params.keys():
            h5f.create_dataset(k, data=params[k].cpu().numpy())
    del signals, sig_data, params

    # generate backgrounds
    backgrounds, _ = injection(config, data_dir=data_dir, device=device, inject=False)
    bkg_data = backgrounds.cpu().numpy()
    with h5py.File(out_dir / 'bkg.h5', 'w') as h5f:
        h5f.create_dataset('data', data=bkg_data)
    del backgrounds, bkg_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Data generation script for GW signals"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="path to config file"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to folder containing public data"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Path to output folder for .hdf5 files"
    )
    args = parser.parse_args()

    main(config_path=args.config, data_dir=args.data, output_dir=args.out)
