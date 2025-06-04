__all__ = [
    "norm_data",
    "normalize_fluorescence_xas",
    "align_spec",
    "load_h5_data",
    "apply_shifts",
    "fluorescence_XAS_generate",
    "plot_spectra_energy_range"
]

import numpy as np
import pandas as pd
import h5py

from datetime import datetime
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from scipy.signal import correlate
from numpy.polynomial import Polynomial
import plotly.graph_objects as go
import matplotlib.cm as cm



def norm_data(data):
    return (data - data.min()) / (data.max() - data.min())

def normalize_fluorescence_xas(energy, mu_raw, e0,
                                pre_range=(-200, -30),
                                post_range=(100, 400),
                                plot=False):
    """
    Normalize and flatten fluorescence μ(E) using linear pre- and post-edge fits.

    Parameters:
        energy (np.ndarray): Energy array (in eV)
        mu_raw (np.ndarray): Raw μ (e.g. fluorescence / I0)
        e0 (float): Edge energy in eV
        pre_range (tuple): Offset from e0 for pre-edge region
        post_range (tuple): Offset from e0 for post-edge region
        plot (bool): Whether to show an interactive plot (using Plotly)

    Returns:
        mu_flat (np.ndarray): Flattened and normalized μ(E)
    """
    pre_mask = (energy > e0 + pre_range[0]) & (energy < e0 + pre_range[1])
    post_mask = (energy > e0 + post_range[0]) & (energy < e0 + post_range[1])

    if pre_mask.sum() < 3 or post_mask.sum() < 3:
        raise ValueError("Not enough points in pre- or post-edge region for fitting.")

    pre_fit = Polynomial.fit(energy[pre_mask], mu_raw[pre_mask], 1)
    post_fit = Polynomial.fit(energy[post_mask], mu_raw[post_mask], 1)

    pre_bkg = pre_fit(energy)
    post_line = post_fit(energy)
    edge_step = post_line - pre_bkg
    mu_flat = (mu_raw - pre_bkg) / edge_step

    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=energy, y=mu_raw, mode='lines', name='Raw μ'))
        fig.add_trace(go.Scatter(x=energy, y=pre_bkg, mode='lines', name='Pre-edge fit', line=dict(dash='dash', color='green')))
        fig.add_trace(go.Scatter(x=energy, y=post_line, mode='lines', name='Post-edge fit', line=dict(dash='dash', color='orange')))
        fig.add_trace(go.Scatter(x=energy, y=mu_flat, mode='lines', name='Flattened μ', line=dict(color='crimson')))
        fig.add_vline(x=e0, line=dict(dash='dot', color='gray'), annotation_text='E₀', annotation_position='top right')

        fig.update_layout(title="Fluorescence XAS Normalization",
                          xaxis_title="Energy (eV)",
                          yaxis_title="μ(E)",
                          template="plotly_white")
        fig.show()

    return mu_flat

def align_spec(specs, align_to_idx=0, plot=False):
    """
    Align spectra to a reference using cross-correlation with sub-pixel shifts.

    Parameters
    ----------
    specs : list of 1D np.ndarray
        Spectra to align; each must have the same length.
    align_to_idx : int, default=0
        Index of the spectrum in `specs` to use as reference.
    plot : bool, default=False
        If True, plot the original and aligned spectra.

    Returns
    -------
    aligned_specs : list of 1D np.ndarray
        The input spectra shifted by the computed sub-pixel shifts.
    shifts : list of float
        Sub-pixel index shifts applied (positive means shift right).
    """
    N = len(specs[0])
    # reference
    ref = specs[align_to_idx]
    ref0 = ref - np.mean(ref)
    
    # cross-correlation lags
    lags = np.arange(-N + 1, N)
    
    shifts = []
    aligned_specs = []
    
    for spec in specs:
        spec0 = spec - np.mean(spec)
        corr = correlate(spec0, ref0, mode='full')
        i_max = np.argmax(corr)
        lag = lags[i_max]
        
        # Sub-pixel refinement: quadratic interpolation around the peak
        if 0 < i_max < len(corr) - 1:
            y_minus, y0, y_plus = corr[i_max-1], corr[i_max], corr[i_max+1]
            denom = (y_minus - 2*y0 + y_plus)
            if denom != 0:
                delta = (y_minus - y_plus) / (2 * denom)
            else:
                delta = 0.0
        else:
            delta = 0.0
        
        shift = lag + delta
        shifts.append(shift)
        
        # Apply sub-pixel shift via interpolation
        x = np.arange(N)
        x_shifted = x - shift
        aligned = np.interp(x, x_shifted, spec, left=np.nan, right=np.nan)
        aligned_specs.append(aligned)
    
    if plot:
        # plot original
        plt.figure()
        for i, spec in enumerate(specs):
            plt.plot(spec, label=str(i))
        plt.title("Original Spectra")
        plt.xlabel("Index"); plt.ylabel("Intensity")
        plt.legend(title="Spec #")
        plt.show()
        
        # plot aligned
        plt.figure()
        for i, spec in enumerate(aligned_specs):
            plt.plot(spec, label=str(i))
        plt.title("Sub-pixel Aligned Spectra")
        plt.xlabel("Index"); plt.ylabel("Intensity")
        plt.legend(title="Spec #")
        plt.show()

    return shifts

# example usage
# align_spec(test['data']['ge_8element'][-1,:,:], align_to_idx=2, plot=True)


def load_h5_data(filepath, 
                 detector: str = 'ge_8element', 
                 num_elements: int = 8):
    """
    Load Bluesky HDF5 scan file into a dict, including:
      - top-level metadata (duration, entry_identifier, plan_name, sample_name,
        start_time, stop_time)
      - all arrays under /…/data in data['data']
      - sample_wheel in data['sample_wheel']
      - deadtime_factor in data['deadtime_factor'], shape (N_points, num_elements)
      - ion chamber net counts in data['ion_chamber'] (dict of arrays)
    """
    data = {}
    with h5py.File(filepath, 'r') as f:
        # assume exactly one top-level group (the UUID)
        root = f[list(f.keys())[0]]
        
        # 1) top-level metadata
        for key in ('duration', 'entry_identifier', 'plan_name', 
                    'sample_name', 'start_time', 'stop_time'):
            val = root[key][()]
            if isinstance(val, bytes):
                val = val.decode()
            data[key] = val

        # 2) scan arrays
        data['data'] = {
            name: root['data'][name][()]
            for name in root['data'].keys()
        }

        # 3) sample wheel position
        data['sample_wheel'] = root['instrument'] \
            ['bluesky']['streams']['baseline']['sam_wheel']['value'][()]

        # 4) deadtime factors for Ge elements
        primary = root['instrument']['bluesky']['streams']['primary']
        dt_list = []
        for i in range(num_elements):
            key = f'{detector}-element{i}-deadtime_factor'
            if key in primary:
                dt_list.append(primary[key]['value'][()])
        data['deadtime_factor'] = np.column_stack(dt_list) if dt_list else np.empty((0, 0))

        # 5) ion chamber net counts as individual arrays
        ion_keys = [
            'Ipreslit-mcs-scaler-channels-1-net_count',
            'IpreKB-mcs-scaler-channels-2-net_count',
            'I0-mcs-scaler-channels-3-net_count',
            'It-mcs-scaler-channels-4-net_count',
            'Iref-mcs-scaler-channels-5-net_count',
            'IpreKB_ds_v1-mcs-scaler-channels-6-net_count',
            'I0_ds_v2-mcs-scaler-channels-7-net_count',
            'It_ds_h1-mcs-scaler-channels-8-net_count'
        ]
        data['ion_chamber'] = {}
        for key in ion_keys:
            if key in primary:
                data['ion_chamber'][key] = primary[key]['value'][()]

    return data


def apply_shifts(
    data, 
    shifts=1, 
    spec_axis=-1, 
    axis_index=None, 
    fill_value=np.nan, 
    plot_data=None
):
    """
    Apply per-index float shifts to an N-D array by interpolation along spec_axis,
    with optional plotting of one selected slice before/after shifting.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (...).
    shifts : array-like of float
        Shift values (positive moves data to the right), shape matching data.shape[axis_index].
    spec_axis : int
        Axis along which to interpolate/shift (the spectral dimension).
    axis_index : int, optional
        Axis whose length equals len(shifts). If None, it is inferred.
    fill_value : scalar, default=nan
        Value used for out-of-bounds regions after shifting.
    plot_data : int or None, default=None
        If int, index along the *other* axis (not spec_axis or axis_index) whose
        slice to plot. Use negative indices as usual.

    Returns
    -------
    out : np.ndarray
        Same shape as `data`, with each 1D slice along `spec_axis`
        shifted by the corresponding (float) amount from `shifts`.
    """
    data = np.asarray(data)
    shifts = np.asarray(shifts, dtype=float)
    ndim = data.ndim

    # normalize negative axes
    spec_axis = spec_axis % ndim
    if axis_index is not None:
        axis_index = axis_index % ndim

    # infer axis_index if not given
    if axis_index is None:
        candidates = [ax for ax in range(ndim)
                      if ax != spec_axis and data.shape[ax] == shifts.shape[0]]
        if len(candidates) == 1:
            axis_index = candidates[0]
        else:
            raise ValueError(f"Cannot infer axis_index, matches: {candidates}")

    # validate shapes
    if data.shape[axis_index] != shifts.shape[0]:
        raise ValueError(
            f"shifts length ({shifts.shape[0]}) "
            f"must equal data.shape[{axis_index}] ({data.shape[axis_index]})"
        )

    # prepare output
    out = np.empty_like(data)
    N = data.shape[spec_axis]
    x = np.arange(N)

    # apply shifts slice-by-slice
    it = np.nditer(shifts, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index[0] if shifts.ndim == 1 else it.multi_index
        sh = shifts[it.multi_index]

        slicer = [slice(None)] * ndim
        slicer[axis_index] = idx
        sl = tuple(slicer)

        sub = data[sl]
        # compute new sample grid
        new_x = x - sh
        # identify sub_spec_axis index in sub
        other_axes = [ax for ax in range(ndim) if ax != axis_index]
        new_spec_axis = other_axes.index(spec_axis)
        # interpolate along spectral axis
        aligned = np.apply_along_axis(
            lambda row: np.interp(x, new_x, row, left=fill_value, right=fill_value),
            new_spec_axis,
            sub
        )
        out[sl] = aligned

    # optional plotting of one slice
    if plot_data is not None:
        # determine the axis to plot over (other than spec_axis/axis_index)
        other_axes = [ax for ax in range(ndim) if ax not in (spec_axis, axis_index)]
        if len(other_axes) != 1:
            raise ValueError(
                "plot_data works only when there's exactly one other axis "
                "besides spec_axis and axis_index"
            )
        plot_axis = other_axes[0]
        idx = plot_data

        # extract that slice before and after
        orig_slice = np.take(data, indices=idx, axis=plot_axis)
        shifted_slice = np.take(out, indices=idx, axis=plot_axis)

        # move axes so axis_index->0 and spec_axis->1 for plotting
        sub_axes = [ax for ax in range(ndim) if ax != plot_axis]
        new_idx0 = sub_axes.index(axis_index)
        new_idx1 = sub_axes.index(spec_axis)
        orig2 = np.moveaxis(orig_slice, [new_idx0, new_idx1], [0, 1])
        shifted2 = np.moveaxis(shifted_slice, [new_idx0, new_idx1], [0, 1])

        # Plot original
        plt.figure()
        for i in range(orig2.shape[0]):
            plt.plot(x, orig2[i], label=f"{i}")
        plt.xlabel("Index along spectral axis")
        plt.ylabel("Intensity")
        plt.title(f"Original Slice at axis {plot_axis} index {idx}")
        plt.legend(title="Spec #")
        plt.show()

        # Plot shifted
        plt.figure()
        for i in range(shifted2.shape[0]):
            plt.plot(x, shifted2[i], label=f"{i}")
        plt.xlabel("Index along spectral axis")
        plt.ylabel("Intensity")
        plt.title(f"Shifted Slice at axis {plot_axis} index {idx}")
        plt.legend(title="Spec #")
        plt.show()

    return out

def fluorescence_XAS_generate(
    fluorescence: Optional[np.ndarray] = None,
    hdf: Optional[dict] = None,
    detector: str = 'ge_8element',
    align_shifts: Optional[Sequence[float]] = None,
    deadtime_factor: Optional[np.ndarray] = None,
    energy: Optional[np.ndarray] = None,
    I0: Optional[np.ndarray] = None,
    roi: Optional[Union[Sequence[int], Sequence[Sequence[int]]]] = None,
    plot_fluo: bool = False,
    plot_xas: bool = True,
    plot_shifts_index: Optional[int] = None,
    export: bool = False,
    export_prefix: str = "",
    norm: bool = True,
    newfig: bool = True
) -> np.ndarray:
    """
    Generate fluorescence XAS with ROI support, deadtime correction, alignment,
    automatic HDF input, plotting, and CSV export with energy as first column.
    """
    # Auto-load from HDF if needed
    if fluorescence is None and hdf:
        fluorescence = hdf['data'][detector]
    if deadtime_factor is None and hdf:
        deadtime_factor = hdf['deadtime_factor']
    if energy is None and hdf:
        try:
            energy = hdf['data']['energy']
        except KeyError:
            energy = hdf['data']['monochromator-energy']


    # 1) Deadtime correction
    if deadtime_factor is not None:
        dt = np.asarray(deadtime_factor, dtype=float)
        if dt.ndim == 2:
            dt = dt[:, :, np.newaxis]
        fluorescence = fluorescence * dt

    # 2) Alignment
    if align_shifts is not None:
        fluorescence = apply_shifts(
            fluorescence, align_shifts,
            spec_axis=2, axis_index=1,
            plot_data=plot_shifts_index
        )

    # 3) Build ROI list
    n_chan = fluorescence.shape[2]
    if roi is None:
        roi_list = [(0, n_chan)]
    elif isinstance(roi, (list, tuple)) and len(roi) == 2 and all(isinstance(x, int) for x in roi):
        roi_list = [tuple(roi)]
    else:
        roi_list = [tuple(r) for r in roi]
    mask = np.zeros(n_chan, dtype=bool)
    for start, end in roi_list:
        mask[start:end] = True
    roi_str = ", ".join(f"[{s}:{e}]" for s, e in roi_list)

    # 4) Sum over masked channels → processed_fluo
    processed_fluo = fluorescence[:, :, mask].sum(axis=(1, 2))

    # 5) Compute XAS if I0 provided (not saved)
    xas = processed_fluo / I0 if I0 is not None else processed_fluo

    # 6) Plot XAS
    if plot_xas:
        if newfig: plt.figure()
        x = energy if energy is not None else np.arange(processed_fluo.size)
        y = (xas - xas.min())/(xas.max() - xas.min()) if norm else xas
        plt.plot(x, y, label=f"ROI {roi_str}")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Fluorescence XAS")
        plt.legend()

    # 7) Plot example fluorescence
    if plot_fluo:
        if newfig: plt.figure()
        plt.plot(fluorescence[0,0,:], label="first scan, det 0")
        plt.plot(fluorescence[-1,0,:], label="last scan, det 0")
        plt.xlabel("Channel")
        plt.ylabel("Fluorescence intensity")
        plt.legend()
        plt.show()

    # 8) Export CSV with energy as first column
    if export:
        data_dict = hdf['data'] if hdf else {}
        ion_dict  = hdf.get('ion_chamber', {}) if hdf else {}

        if hdf:
            cols = []
            # 1) energy first
            cols.append(("energy", energy))
            # 2) other scan data (excluding raw detector array)
            for key, arr in data_dict.items():
                if key != detector:
                    cols.append((key, arr))
            # 3) ion‐chamber channels
            for key, arr in ion_dict.items():
                cols.append((key, arr))
            # 4) processed fluorescence last
            cols.append(("processed_fluo", processed_fluo))
        else:
            # minimal export: energy first
            cols = [
                ("energy", energy),
                ("I0", I0),
                ("processed_fluo", processed_fluo)
            ]

        # Build filename
        sn = ""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        uid = ""
        if hdf:
            sn = hdf.get('sample_name', '')
            if isinstance(sn, bytes):
                sn = sn.decode()
            st = hdf.get('start_time', '')
            try:
                dt0 = datetime.fromisoformat(st)
                ts = dt0.strftime("%Y%m%d_%H%M")
            except Exception:
                pass
            uid = hdf.get('entry_identifier', '')
        fname = (f"{export_prefix}{sn}_{ts}_{uid}.csv"
                 if sn else f"{export_prefix}fluorescence_xas_{ts}.csv")

        # Write CSV
        df = pd.DataFrame({name: arr for name, arr in cols})
        with open(fname, 'w') as f:
            # column comments
            for i, (name, _) in enumerate(cols, start=1):
                unit = ' A' if 'net_current' in name else ''
                f.write(f"# Column.{i}: {name}{unit}\n")
            # ROI info
            f.write(f"# ROIs used: {roi_str}\n")
            # scan metadata
            if align_shifts is not None:
                f.write(f"# Shifts: {list(align_shifts)}\n")
                f.write("# These are the shifts to align all MCA spectra for each element\n")
            if hdf:
                f.write(f"# Scan duration: {hdf.get('duration')}\n")
                f.write(f"# Scan.start_time: {hdf.get('start_time')}\n")
                f.write(f"# uid: {hdf.get('entry_identifier')}\n")
            # commented header row
            f.write("# " + ",".join(name for name, _ in cols) + "\n")
            # data rows
            df.to_csv(f, index=False, header=False)
        print(f"Exported results to {fname}")

    return processed_fluo

def plot_spectra_energy_range(
    energies: np.ndarray,
    fluorescence: np.ndarray,
    I0: Optional[np.ndarray] = None,
    energy_min: float = 17000,
    energy_max: float = 18000,
    energy_steps: int = 10,
    channel: int = 0,
    cmap_name: str = 'rainbow_r'
):
    """
    Plot spectra for scans whose energies lie within [energy_min, energy_max],
    subsampling every `energy_steps`-th scan, with optional I0 normalization.

    Parameters
    ----------
    energies : 1D array, shape (n_scans,)
        Energy values for each scan.
    fluorescence : 3D array, shape (n_scans, n_detectors, n_channels)
        Fluorescence data.
    I0 : 1D array, shape (n_scans,), optional
        If provided, divide each spectrum by I0[i].
    energy_min : float
        Lower bound of energy range (inclusive).
    energy_max : float
        Upper bound of energy range (inclusive).
    energy_steps : int
        Step size for subsampling the filtered scans.
    channel : int
        Detector channel index to plot.
    cmap_name : str
        Name of the matplotlib colormap to use.

    Returns
    -------
    selected_indices : 1D array of ints
        Indices of scans actually plotted.
    """
    # Mask and select indices
    mask = (energies >= energy_min) & (energies <= energy_max)
    indices = np.where(mask)[0][::energy_steps]
    if len(indices) == 0:
        raise ValueError("No scans found in the specified energy range.")

    # Prepare colormap
    cmap = cm.get_cmap(cmap_name, len(indices))

    # Plot
    plt.figure()
    for j, idx in enumerate(indices):
        color = cmap(j)
        e = energies[idx]
        spec = fluorescence[idx, channel, :]
        if I0 is not None:
            spec = spec / I0[idx]
        plt.plot(spec, color=color, lw=1, alpha=0.7, label=f'{e:.1f} eV')

    plt.xlabel('Channel')
    plt.ylabel('Intensity')
    plt.title(f'Spectra from {energy_min} to {energy_max} eV (step {energy_steps})')
    plt.legend(title='Energy', fontsize=6, ncol=2, frameon=False)
    plt.tight_layout()
    plt.show()

    return indices