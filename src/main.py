import argparse
import gc
import os
from pathlib import Path

import h5py
import torch
from tqdm import tqdm
from set_seed import set_seed 

from injections import injection
from utils import load_config


def main(config_path: str, data_dir: str, output_dir: str, batch_size_cl: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(config_path)

    data_dir = Path(data_dir)
    out_dir = Path(output_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    set_seed(config.general.seed)

    total = 0
    with tqdm(
        total=config.general.num_waveforms, desc="Processing", unit="step"
    ) as pbar:
        while total < config.general.num_waveforms:
            # generate signals
            signals, params = injection(
                config, data_dir=data_dir, device=device, inject=True, batch_size_cl=batch_size_cl
            )
            sig_data = signals.cpu().numpy()
            with h5py.File(out_dir / "sig_{0}.h5".format(total), "w") as h5f:
                h5f.create_dataset("data", data=sig_data)
                for k in params.keys():
                    h5f.create_dataset(k, data=params[k].cpu().numpy())
            del signals, sig_data, params
            gc.collect()
            torch.cuda.empty_cache()

            # Only generate backgrounds if batch_size_cl is 1.
            # Otherwise, contrastive learning doesn't need separate backgrounds.
            if batch_size_cl == 1:
                # generate backgrounds
                backgrounds, _ = injection(
                    config, data_dir=data_dir, device=device, inject=False
                )
                bkg_data = backgrounds.cpu().numpy()
                with h5py.File(out_dir / "bkg_{0}.h5".format(total), "w") as h5f:
                    h5f.create_dataset("data", data=bkg_data)
                del backgrounds, bkg_data
                gc.collect()
                torch.cuda.empty_cache()

            total += config.general.batch_size
            pbar.update(config.general.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data generation script for GW signals"
    )

    parser.add_argument(
        "--config", type=str, default="config.yaml", help="path to config file"
    )
    parser.add_argument(
        "--data", type=str, help="Path to folder containing public data"
    )
    parser.add_argument("--out", type=str, help="Path to output folder for .hdf5 files")
    parser.add_argument("--batch_size_cl", type=int, default=1, help="Contrastive learning batch size")
    args = parser.parse_args()

    main(config_path=args.config, data_dir=args.data, output_dir=args.out, batch_size_cl=args.batch_size_cl)
