"""Synthetic, local-only validation of the installed public LEP-i moment pipeline."""
from pathlib import Path
from copy import deepcopy
from contextlib import redirect_stdout, redirect_stderr
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
import io
import json
import logging
import fnmatch
import warnings
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.constants import proton_mass, elementary_charge
import pyspedas as psp

BASE = Path(__file__).resolve().parent
INPUT_NAME = 'erg_lepi_l2_3dflux_FPDU_synthetic_validation'
MAG_NAME = 'lepi_synthetic_validation_mag'
TIMES = np.array([1_600_000_000., 1_600_000_008.])
GEOMETRY = json.loads((BASE/'lepi_l2_geometry.json').read_text())
PUBLIC_MODULES = {
    'pyspedas': importlib.import_module('pyspedas.projects.erg.satellite.erg.particle.erg_lep_part_products'),
    'ergpyspedas': importlib.import_module('ergpyspedas.erg.satellite.erg.particle.erg_lep_part_products'),
}
ENERGY_RANGE = [30., 30_000.]
DENSITY_FLOOR = 1e-10


def source_manifest():
    result = {'geometry': GEOMETRY, 'packages': {}, 'sources': {}}
    for name in ['numpy', 'scipy', 'pyspedas', 'ergpyspedas', 'cdflib']:
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = 'distribution metadata unavailable'
    for package in PUBLIC_MODULES:
        for leaf in ['erg_lep_part_products', 'erg_lepi_get_dist', 'erg_pgs_clean_data',
                     'erg_pgs_limit_range', 'erg_convert_flux_units', 'get_lepi_flux_angle_in_sga']:
            root = ('pyspedas.projects.erg' if package == 'pyspedas' else 'ergpyspedas.erg')
            mod = importlib.import_module(root+'.satellite.erg.particle.'+leaf)
            path = Path(mod.__file__)
            result['sources'][mod.__name__] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ['pyspedas.particles.moments.moments_3d',
                 'pyspedas.particles.moments.moments_3d_omega_weights']:
        mod = importlib.import_module(name)
        path = Path(mod.__file__)
        result['sources'][name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def store_synthetic(flux):
    """flux shape [30 energy, 8 anode, 16 sector], units per keV cm² s sr."""
    if np.shape(flux) != (30, 8, 16):
        raise ValueError('Expected [30, 8, 16] flux array')
    ok = psp.store_data(INPUT_NAME, data={'x': TIMES, 'y': np.repeat(flux[None], 2, axis=0),
                         'v1': np.array(GEOMETRY['energy_kev']),
                         'v2': np.arange(8), 'v3': np.arange(16)})
    if not ok:
        raise RuntimeError('Failed to store synthetic tplot variable')
    psp.store_data(MAG_NAME, data={'x': TIMES, 'y': np.tile([17., 23., 31.], (2, 1))})


@lru_cache(maxsize=2)
def native_distribution(package='pyspedas'):
    store_synthetic(np.zeros((30, 8, 16)))
    return PUBLIC_MODULES[package].erg_lepi_get_dist(INPUT_NAME, [0], species='proton')


def native_bin_table(package='pyspedas'):
    d = native_distribution(package)
    e = d['energy'][:, 0, 0, 0]
    de = d['denergy'][:, 0, 0, 0]
    use = (e >= ENERGY_RANGE[0]) & (e <= ENERGY_RANGE[1]) & (np.arange(30) != 0)
    return {'channel': np.arange(30), 'energy_ev': e, 'width_ev': de,
            'getter_lower_edge_ev': _getter_edges(e)[0],
            'getter_upper_edge_ev': _getter_edges(e)[1], 'selected': use}


def _getter_edges(e):
    """Recover actual asymmetrical widths for documentation (not reference moments)."""
    boundaries = np.sqrt(e[:-1]*e[1:])
    high = np.insert(boundaries, 0, boundaries[0])
    high[[0, 1]] = e[1]+(e[1]-boundaries[1])
    low = np.append(boundaries, boundaries[-1])
    low[0] = low[1]
    low[29] = e[29]-(boundaries[28]-e[29])
    return low, high


def unit_vectors(theta_deg, phi_deg):
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.stack([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), np.sin(th)], axis=-1)


def maxwell_flux(energy_ev, directions, theta_ev, velocity_kms, density_cm3, mass):
    """Independent SI f -> differential number flux, per keV cm² s sr.

    j_eV = f_SI v² (J/eV)/m_SI * 1e-4. No library flux converter used.
    mass is the getter's eV/(km/s)^2 value; using it isolates discretization.
    """
    mass_si = mass*elementary_charge/1e6
    vth = np.sqrt(2*theta_ev*elementary_charge/mass_si)
    v = np.sqrt(2*np.asarray(energy_ev)*elementary_charge/mass_si)
    u = np.asarray(velocity_kms)*1000
    vec = v[..., None]*directions
    exponent = -np.sum((vec-u)**2, axis=-1)/vth**2
    f_si = density_cm3*1e6/(np.pi**1.5*vth**3)*np.exp(exponent)
    return f_si*v*v*elementary_charge/mass_si*1e-4*1000


def make_flux(theta_ev, velocity_kms, density_cm3=1., package='pyspedas', sampling='centre', quadrature_order=8):
    d = native_distribution(package)
    e = d['energy'][..., 0]
    dirs = unit_vectors(d['theta'][..., 0], d['phi'][..., 0])
    if sampling == 'centre':
        raw = maxwell_flux(e, dirs, theta_ev, velocity_kms, density_cm3, d['mass'])
    elif sampling == 'bin_average':
        # Uniform top-hat model, not a calibrated instrument-response model.
        low, high = _getter_edges(e[:, 0, 0])
        eroot, ew = leggauss(quadrature_order)
        aroot, aw = leggauss(quadrature_order)
        th0 = np.deg2rad(d['theta'][..., 0])
        ph0 = np.deg2rad(d['phi'][..., 0])
        dth = np.deg2rad(d['dtheta'][..., 0])
        dph = np.deg2rad(d['dphi'][..., 0])
        raw = np.zeros_like(e)
        for i in range(e.shape[0]):
            th = th0[i, ..., None, None]+dth[i, ..., None, None]/2*aroot[None, None, :, None]
            ph = ph0[i, ..., None, None]+dph[i, ..., None, None]/2*aroot[None, None, None, :]
            vx, vy, vz = np.broadcast_arrays(np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), np.sin(th))
            dirs_quad = np.stack([vx, vy, vz], axis=-1)
            angular_measure = np.cos(th)*aw[:, None]*aw[None, :]
            angular_norm = angular_measure.sum(axis=(-2, -1))
            values = np.zeros((16, 8))
            for node, weight in zip(eroot, ew):
                en = (high[i]+low[i])/2+(high[i]-low[i])/2*node
                values += weight*np.sum(maxwell_flux(en, dirs_quad, theta_ev,
                                      velocity_kms, density_cm3, d['mass'])*angular_measure,
                                      axis=(-2, -1))/angular_norm/2
            raw[i] = values
    else:
        raise ValueError('sampling must be centre or bin_average')
    return raw.transpose(0, 2, 1)  # native [E, sector, anode] -> L2 [E, anode, sector]


@lru_cache(maxsize=2)
def independent_angular_weights(package='pyspedas'):
    """Independent Gauss quadrature of solid angle, vector, and dyad.

    Does NOT call moments_3d_omega_weights. Latitude variable is mu=sin(theta).
    """
    d = native_distribution(package)
    th = np.deg2rad(d['theta'][..., 0].reshape(30, -1))
    ph = np.deg2rad(d['phi'][..., 0].reshape(30, -1))
    dth = np.deg2rad(d['dtheta'][..., 0].reshape(30, -1))
    dph = np.deg2rad(d['dphi'][..., 0].reshape(30, -1))
    # Integrate in latitude theta rather than mu to avoid endpoint square-root
    # singularities of vector components at the poles.
    roots, weights = leggauss(16)
    tq = th[..., None, None]+dth[..., None, None]/2*roots[None, None, :, None]
    pq = ph[..., None, None]+dph[..., None, None]/2*roots[None, None, None, :]
    vx, vy, vz = np.broadcast_arrays(np.cos(tq)*np.cos(pq), np.cos(tq)*np.sin(pq), np.sin(tq))
    dirs = np.stack([vx, vy, vz], axis=-1)
    measure = np.cos(tq)*dth[..., None, None]/2*dph[..., None, None]/2*weights[:, None]*weights[None, :]
    omega = np.sum(measure, axis=(-2, -1))
    vector = np.sum(measure[..., None]*dirs, axis=(-3, -2))
    tensor = np.sum(measure[..., None, None]*dirs[..., :, None]*dirs[..., None, :], axis=(-4, -3))
    np.testing.assert_allclose(omega.sum(axis=1), 4*np.pi, rtol=1e-12)
    np.testing.assert_allclose(vector.sum(axis=1), 0., atol=1e-12)
    np.testing.assert_allclose(tensor.sum(axis=1), np.tile(np.eye(3)*4*np.pi/3, (30, 1, 1)), atol=1e-12)
    return omega, vector, tensor


def independent_discrete_moments(flux, package='pyspedas', energy_range=ENERGY_RANGE):
    """Physics-based independent midpoint-energy reference on selected bins.

    j is per eV cm² s sr; n=sum(j dE/v_cm * dOmega). Angular integration
    uses independent quadrature; public cleaning/unit/moment routines are not used.
    """
    d = native_distribution(package)
    e = d['energy'][..., 0].reshape(30, -1)
    de = d['denergy'][..., 0].reshape(30, -1)
    j = np.asarray(flux).transpose(0, 2, 1).reshape(30, -1)/1000
    valid = np.isfinite(j) & (e >= energy_range[0]) & (e <= energy_range[1])
    valid[0] = False
    j = np.where(valid, np.maximum(j, 0.), 0.)
    v_kms = np.sqrt(2*e/d['mass'])
    # Same finite-bin potential weight, inferred from the installed definition;
    # it is exactly 1 for all selected native bins in the present runs.
    weight = np.clip(e/de+0.5, 0., 1.)
    omega, vec, ten = independent_angular_weights(package)
    particle_measure = j*de*weight/(v_kms*1e5)
    density = np.sum(particle_measure*omega)
    flux_vector = np.sum((j*de*weight)[..., None]*vec, axis=(0, 1))
    velocity = flux_vector/(density*1e5)
    raw = np.sum((particle_measure*d['mass']*v_kms*v_kms)[..., None, None]*ten, axis=(0, 1))
    pressure_tensor = raw-density*d['mass']*np.outer(velocity, velocity)
    temperature_tensor = pressure_tensor/density
    return dict(density=density, velocity=velocity, flux=flux_vector,
                ptens=pressure_tensor, ttens=temperature_tensor,
                avgtemp=np.trace(temperature_tensor)/3)


def run_public(flux, package='pyspedas', energy_range=ENERGY_RANGE, suffix='_public'):
    """Calls the actual public function with actual tplot data; no monkeypatches."""
    store_synthetic(flux)
    log = io.StringIO()
    previous_log_disable = logging.root.manager.disable
    previous_handlers = logging.root.handlers[:]
    diagnostic_records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            diagnostic_records.append(record)

    try:
        logging.root.handlers = [CaptureHandler()]
        logging.disable(logging.INFO)
        with redirect_stdout(log), redirect_stderr(log), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            stale = [name for name in psp.tplot_names(quiet=True)
                     if fnmatch.fnmatchcase(name, INPUT_NAME+'_*'+suffix)]
            if stale:
                psp.del_data(stale)
            names = PUBLIC_MODULES[package].erg_lep_part_products(
                INPUT_NAME, species='proton', outputs=['moments'], energy=list(energy_range),
                mag_name=MAG_NAME, suffix=suffix)
    finally:
        logging.root.handlers = previous_handlers
        logging.disable(previous_log_disable)
    required = ['density', 'velocity', 'flux', 'ptens', 'ttens', 'avgtemp']
    out = {'stdout': log.getvalue(), 'warnings': [str(w.message) for w in caught], 'names': names,
           'logging_warnings': [r.getMessage() for r in diagnostic_records if r.levelno >= logging.WARNING]}
    for key in required:
        if not isinstance(names, list) or INPUT_NAME+'_'+key+suffix not in names:
            raise RuntimeError('Public return value does not include '+key)
        data = psp.get_data(INPUT_NAME+'_'+key+suffix)
        if data is None or len(data[0]) != 2:
            raise RuntimeError('Public function did not produce '+key)
        np.testing.assert_allclose(data[1][0], data[1][1], rtol=1e-12, atol=1e-12)
        value = data[1][0]
        if key == 'ptens':
            xx, yy, zz, xy, xz, yz = value
            value = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        out[key] = value
    return out


def clear_synthetic():
    # Only variables owned by this validation, never del_data('*').
    psp.del_data(INPUT_NAME+'*')
    psp.del_data(MAG_NAME+'*')


@lru_cache(maxsize=1)
def _continuous_namespace():
    """Reuse only parameter/definition cells of the existing continuous notebook."""
    path = BASE/'LEPi_drifting_Maxwellian_partial_moments.ipynb'
    notebook = json.loads(path.read_text())
    ns = dict(np=np, quad=quad, proton_mass=proton_mass, elementary_charge=elementary_charge)
    for marker in ['# θ = k_B T [eV]', 'def angular_moments(', 'REFERENCE_DENSITY_CM3 =']:
        matches = [''.join(c['source']) for c in notebook['cells']
                   if c['cell_type'] == 'code' and ''.join(c['source']).startswith(marker)]
        if len(matches) != 1:
            raise RuntimeError('Continuous reference definition cell changed: '+marker)
        exec(compile(matches[0], str(path), 'exec'), ns)
    return ns


def continuous_reference(theta_ev, speed_kms, energy_range=ENERGY_RANGE, density_cm3=1., package='pyspedas'):
    # Baseline uses SI proton_mass; rescale its velocity to match the mass
    # convention in the installed getter. This is an exact dimensionless mapping.
    mass_si = native_distribution(package)['mass']*elementary_charge/1e6
    factor = np.sqrt(mass_si/proton_mass)
    result = _continuous_namespace()['partial_thermal_moment'](
        theta_ev, speed_kms*factor, *energy_range, density_cm3=density_cm3)
    result['partial_speed_kms'] /= factor
    return result


def result_record(theta_ev, velocity_kms, density_cm3=1., package='pyspedas', sampling='centre'):
    u = np.asarray(velocity_kms, dtype=float)
    speed = np.linalg.norm(u)
    flux = make_flux(theta_ev, u, density_cm3, package, sampling)
    public = run_public(flux, package)
    discrete = independent_discrete_moments(flux, package)
    # Independent agreement is separate from bias versus the full distribution.
    discrete_error = check_public_vs_discrete(public, discrete, package)
    continuous = continuous_reference(theta_ev, speed, density_cm3=density_cm3, package=package)
    table = native_bin_table(package)
    use = table['selected']
    support = [table['getter_lower_edge_ev'][use].min(), table['getter_upper_edge_ev'][use].max()]
    shell = continuous_reference(theta_ev, speed, support, density_cm3, package)
    nhat = u/speed if speed else np.array([1., 0., 0.])
    projected = float(public['velocity']@nhat)
    off_axis = float(np.linalg.norm(public['velocity']-projected*nhat))
    pressure = np.trace(public['ptens'])/3
    pressure_z = float(nhat@public['ptens']@nhat)
    pressure_x = (np.trace(public['ptens'])-pressure_z)/2
    nratio = float(public['density']/density_cm3)
    reliable = nratio > DENSITY_FLOOR
    return dict(package=package, sampling=sampling, theta_ev=theta_ev, true_speed_kms=speed,
                ux_kms=u[0], uy_kms=u[1], uz_kms=u[2], density_cm3=float(public['density']),
                density_ratio=nratio, projected_speed_kms=projected,
                off_axis_speed_kms=off_axis, temperature_ev=float(public['avgtemp']),
                temperature_ratio=float(public['avgtemp']/theta_ev) if reliable else np.nan,
                pressure_ratio=float(pressure/(density_cm3*theta_ev)) if reliable else np.nan,
                pressure_pa=float(pressure*1e6*elementary_charge),
                velocity_ratio=projected/speed if speed and reliable else np.nan,
                temperature_anisotropy=pressure_z/pressure_x if speed and pressure_x > 0 and reliable else np.nan,
                continuous_density_ratio=continuous['density_fraction'],
                continuous_speed_kms=continuous['partial_speed_kms'],
                continuous_temperature_ratio=continuous['temperature_ratio'],
                continuous_pressure_ratio=continuous['pressure_ratio'],
                shell_density_ratio=shell['density_fraction'], shell_speed_kms=shell['partial_speed_kms'],
                shell_temperature_ratio=shell['temperature_ratio'], shell_pressure_ratio=shell['pressure_ratio'],
                velocity_error_vs_continuous_kms=projected-continuous['partial_speed_kms'],
                temperature_error_vs_continuous_pct=100*(public['avgtemp']/theta_ev-continuous['temperature_ratio']),
                max_discrete_scaled_error=discrete_error,
                max_discrete_temperature_difference_ev=float(np.max(np.abs(public['ttens']-discrete['ttens']))),
                public_warning_count=len(public['warnings']),
                public_logging_warning_count=len(public['logging_warnings']),
                public_logging_messages=' | '.join(public['logging_warnings']))


def refined_core_moments(theta_ev, velocity_kms, ne=48, nphi=32, ntheta=16,
                         density_cm3=1., package='pyspedas'):
    """Convergence diagnostic of actual downstream core on a refined ideal grid.

    This does NOT call the public LEP-i getter (hardcoded native 30 channels).
    Public native-grid comparisons are performed separately by run_public.
    """
    edges = np.geomspace(*ENERGY_RANGE, ne+1)
    centres = np.sqrt(edges[:-1]*edges[1:])
    phi0 = (np.arange(nphi)+0.5)*360/nphi
    theta0 = -90+(np.arange(ntheta)+0.5)*180/ntheta
    energy, phi, theta = np.broadcast_arrays(centres[:, None, None],
                        phi0[None, :, None], theta0[None, None, :])
    mass = native_distribution(package)['mass']
    dist = dict(time=np.array([TIMES[0]]), end_time=np.array([TIMES[0]+8]),
                charge=1., mass=mass, species='proton', units_name='flux',
                energy=energy.copy(), denergy=np.broadcast_to(np.diff(edges)[:, None, None], energy.shape).copy(),
                theta=theta.copy(), phi=phi.copy(), dtheta=np.full_like(energy, 180/ntheta),
                dphi=np.full_like(energy, 360/nphi), bins=np.ones_like(energy, dtype=np.int8))
    dist['data'] = maxwell_flux(energy, unit_vectors(theta, phi), theta_ev,
                               velocity_kms, density_cm3, mass)/1000
    module = PUBLIC_MODULES[package]
    cleaned = module.erg_pgs_clean_data(dist, units='flux', for_moments=True,
                                       magf=np.array([17.,23.,31.]))
    cleaned = module.erg_pgs_limit_range(cleaned, energy=ENERGY_RANGE)
    eflux = module.erg_convert_flux_units(cleaned, units='eflux')
    return module.spd_pgs_moments(eflux)


def check_public_vs_discrete(public, reference, package='pyspedas'):
    """Scale-aware tolerances retain meaningful checks of zero vector components."""
    worst = 0.
    for key in ['density', 'velocity', 'flux', 'ptens', 'ttens', 'avgtemp']:
        scale = max(1., float(np.max(np.abs(reference[key]))))
        if key == 'flux':
            scale = max(scale, reference['density']*np.sqrt(2*reference['avgtemp']/native_distribution(package)['mass'])*1e5)
        delta = float(np.max(np.abs(np.asarray(public[key])-reference[key])))
        worst = max(worst, delta/scale)
        np.testing.assert_allclose(public[key], reference[key], rtol=2e-10, atol=2e-10*scale)
    return worst
