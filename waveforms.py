import torch
import matplotlib.pyplot as plt
from ml4gw.distributions import PowerLaw, Sine, Cosine, DeltaFunction
from torch.distributions import Uniform
import yaml
import importlib
from ml4gw.waveforms import IMRPhenomD
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components
from ml4gw.gw import get_ifo_geometry, compute_observed_strain
import numpy as np
from utils import load_config

def generate_signals(config, device: str, save: bool):
    
    waveform_duration = config.general.waveform_duration
    num_waveforms = config.general.num_waveforms
    sample_rate = config.general.sample_rate
    ifos = config.general.ifos
    f_min = config.general.f_min
    f_max = config.general.f_max
    f_ref = config.general.f_ref
    waveform_dict = config.waveform
    right_pad = config.general.right_pad

    nyquist = sample_rate / 2
    num_samples = int(waveform_duration * sample_rate)
    num_freqs = num_samples // 2 + 1

    frequencies = torch.linspace(0, nyquist, num_freqs).to(device)
    freq_mask = (frequencies >= f_min) * (frequencies < f_max).to(device)

    param_dict = {}
    attrs = [x for x in dir(waveform_dict) if '__' not in x]
    for k in attrs:
        attrs_config = getattr(waveform_dict, k)
        func_path = getattr(attrs_config, 'func')

        module_name, func_name = func_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)

        if 'args' in dir(attrs_config):
            args = getattr(attrs_config, 'args')
            param_dict[k] = func(*args)
        else:
            param_dict[k] = func()

    # And then sample from each of those distributions
    params = {
            k: v.sample((num_waveforms,)).to(device) for k, v in param_dict.items()
    }

    approximant = IMRPhenomD().to(device)

    waveform_generator = TimeDomainCBCWaveformGenerator(
        approximant=approximant,
        sample_rate=sample_rate,
        f_min=f_min,
        duration=waveform_duration,
        right_pad=right_pad,
        f_ref=f_ref,
    ).to(device)

    #TODO: generalize this for BNS signals
    params["mass_1"], params["mass_2"] = chirp_mass_and_mass_ratio_to_components(
        params["chirp_mass"], params["mass_ratio"]
    )
    params["s1z"], params["s2z"] = params["chi1"], params["chi2"]

    hc, hp = waveform_generator(**params)

    # Waveform projection
    #TODO: check if isotropic
    dec = Cosine()
    psi = Uniform(0, torch.pi)
    phi = Uniform(-torch.pi, torch.pi)

    tensors, vertices = get_ifo_geometry(*ifos)

    waveforms = compute_observed_strain(
        dec=dec.sample((num_waveforms,)).to(device),
        psi=psi.sample((num_waveforms,)).to(device),
        phi=phi.sample((num_waveforms,)).to(device),
        detector_tensors=tensors.to(device),
        detector_vertices=vertices.to(device),
        sample_rate=sample_rate,
        cross=hc,
        plus=hp,
    )
    if save:
        return True
    else:
        return waveforms

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(config_path='config.yaml')

    generate_signals(config, device=device, save=True)
