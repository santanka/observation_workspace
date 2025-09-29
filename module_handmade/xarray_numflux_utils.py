
import xarray as xr
import numpy as np
from typing import Optional

def eflux_to_numflux(
    da_eflux: xr.DataArray,
    energy_coord: Optional[str] = None,
    safe_min_e: float = 1e-30,
    copy_attrs: bool = True,
) -> xr.DataArray:
    """
    Convert differential energy flux [eV/(cm^2 sr s eV)] -> differential number flux [/cm^2 sr s eV]
    by dividing by energy E, using an energy coordinate aligned with (time, energy-bin).

    Parameters
    ----------
    da_eflux : xr.DataArray
        Input array with dims like ('time', 'v_dim') or ('time','spec_bins'), values in eV/(cm^2 sr s eV).
        Must have an energy coordinate that matches these dims. Common names: 'spec_bins', 'v'.
    energy_coord : str, optional
        Explicit name of the energy coordinate to use. If None, try in order:
        ['spec_bins', 'v', 'energy', 'E'].
    safe_min_e : float
        Minimum energy used to avoid division by zero/negative (values <=0 treated as NaN).
    copy_attrs : bool
        If True, copy attrs from input and update units/long_name.

    Returns
    -------
    xr.DataArray
        Same shape/dims/coords as input, with units changed to 1/(cm^2 sr s eV).
    """
    if not isinstance(da_eflux, xr.DataArray):
        raise TypeError("da_eflux must be an xarray.DataArray")

    # Resolve energy coordinate
    cand = [energy_coord] if energy_coord else ["spec_bins", "v", "energy", "E"]
    e_name = None
    for name in cand:
        if name in da_eflux.coords:
            e_name = name
            break
    if e_name is None:
        # also accept a coord that shares dims with the spectral dim (second dim)
        non_time = [d for d in da_eflux.dims if d != "time"]
        if len(non_time) == 1 and non_time[0] in da_eflux.coords:
            e_name = non_time[0]
        else:
            raise ValueError("Energy coordinate not found. Provide energy_coord explicitly.")

    E = da_eflux.coords[e_name]

    # Broadcast to data shape
    E_b = xr.broadcast(da_eflux, E)[1]

    # Sanitize energies: <=0 -> NaN to avoid invalid division
    E_b_safe = xr.where(E_b > 0, E_b, np.nan)
    E_b_safe = xr.where(np.isfinite(E_b_safe), E_b_safe, np.nan)

    # Perform conversion
    numflux = da_eflux / xr.where(E_b_safe > safe_min_e, E_b_safe, np.nan)

    # attrs
    if copy_attrs:
        attrs = dict(da_eflux.attrs) if da_eflux.attrs else {}
        old_units = attrs.get("units", None)
        attrs["units"] = "1/(cm^2 sr s eV)"
        ln = attrs.get("long_name", "differential energy flux")
        attrs["long_name"] = ln.replace("energy", "number") if "energy" in ln else "differential number flux"
        numflux = numflux.copy()
        numflux.attrs = attrs

    return numflux
