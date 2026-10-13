"""Strict, vector-only DSI -> J2000 -> GSM rotation of spacecraft-relative velocities."""
import numpy as np

FLAGS = {'attitude_unavailable': 1, 'orbit_unavailable': 2,
         'degenerate_dsi_basis': 4, 'invalid_rotation': 8}


def series(times, values):
    times, values = np.asarray(times, float), np.asarray(values)
    if times.ndim != 1 or values.shape[0] != len(times) or not len(times):
        raise ValueError('Empty or incompatible support series')
    if not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError('Support timestamps are nonfinite or reversed')
    duplicates = np.flatnonzero(np.diff(times) == 0)
    for i in duplicates:
        if not np.array_equal(values[i], values[i+1], equal_nan=values.dtype.kind in 'biufc'):
            raise ValueError('Conflicting duplicate support records')
    keep = np.r_[np.diff(times) != 0, True]
    return times[keep], values[keep]


def interpolate(times, values, targets, max_gap):
    times, values = series(times, values)
    values = np.asarray(values, float)
    targets = np.asarray(targets, float)
    right = np.searchsorted(times, targets, side='left')
    exact = (right < len(times)) & (times[np.minimum(right, len(times)-1)] == targets)
    left = np.where(exact, right, right-1)
    covered = (left >= 0) & (right < len(times)) & np.isfinite(targets)
    left = np.clip(left, 0, len(times)-1)
    right = np.clip(right, 0, len(times)-1)
    gaps = times[right]-times[left]
    finite = np.isfinite(values.reshape(len(values), -1)).all(axis=1)
    valid = covered & (gaps <= max_gap) & finite[left] & finite[right]
    fraction = np.divide(targets-times[left], gaps, out=np.zeros(len(targets)), where=gaps > 0)
    shape = (len(targets),) + (1,)*(values.ndim-1)
    result = values[left] + fraction.reshape(shape)*(values[right]-values[left])
    result[~valid] = np.nan
    gaps = np.where(covered, gaps, np.nan)
    return result, valid, {'left': left, 'right': right, 'gap_seconds': gaps,
                           'left_time_unix': np.where(covered, times[left], np.nan),
                           'right_time_unix': np.where(covered, times[right], np.nan)}


def normalize(vectors):
    length = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(vectors).all(axis=1) & (length > 1.e-12)
    out = np.full(vectors.shape, np.nan)
    out[valid] = vectors[valid]/length[valid, None]
    return out, valid


def rotation(targets, attitude_times, ra_dec, orbit_times, pos_gse_km,
             max_attitude_gap=16., max_orbit_gap=120., transform=None):
    if transform is None:
        from pyspedas.cotrans_tools.cotrans import cotrans
        transform = cotrans
    at, angles = series(attitude_times, ra_dec)
    ot, pos = series(orbit_times, pos_gse_km)
    if angles.shape != (len(at), 2) or pos.shape != (len(ot), 3):
        raise ValueError('Attitude must be RA/Dec; orbit position must have three km components')
    angles = np.asarray(angles, float).copy()
    angles[~(np.isfinite(angles).all(axis=1) & (np.abs(angles[:, 0]) <= 360.)
             & (np.abs(angles[:, 1]) <= 90.))] = np.nan
    ra, dec = np.deg2rad(angles).T
    z_samples = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))
    z, att_valid, ab = interpolate(at, z_samples, targets, max_attitude_gap)
    # L2 definitive orbit pos_gse is km. Reject finite fill/sentinel and implausible positions.
    pos = np.asarray(pos, float).copy()
    pos[~(np.isfinite(pos).all(axis=1) & (np.linalg.norm(pos, axis=1) < 1.e6) & (np.linalg.norm(pos, axis=1) > 1.))] = np.nan
    position, orb_valid, ob = interpolate(ot, pos, targets, max_orbit_gap)
    flags = ((~att_valid).astype(np.uint8)*FLAGS['attitude_unavailable']
             | (~orb_valid).astype(np.uint8)*FLAGS['orbit_unavailable'])
    matrices = np.full((len(targets), 3, 3), np.nan)
    indices = np.flatnonzero(flags == 0)
    if len(indices):
        t = np.asarray(targets)[indices]
        # Matches public dsi2j2000: satellite-to-Sun direction, nominal Sun distance in km.
        sun_gse = np.array([1.496e8, 0., 0.])-position[indices]
        sun_j2000 = np.asarray(transform(time_in=t, data_in=sun_gse, coord_in='gse',
                                        coord_out='j2000', position=False, quiet=True))
        ez, vz = normalize(z[indices])
        ey, vy = normalize(np.cross(ez, sun_j2000))
        ex, vx = normalize(np.cross(ey, ez))
        good = vz & vy & vx
        flags[indices[~good]] |= FLAGS['degenerate_dsi_basis']
        indices = indices[good]
        if len(indices):
            t = np.asarray(targets)[indices]
            basis = np.stack((ex[good], ey[good], ez[good]), axis=2)
            for j in range(3):
                matrices[indices, :, j] = transform(time_in=t, data_in=basis[:, :, j],
                                                   coord_in='j2000', coord_out='gsm',
                                                   position=False, quiet=True)
            gram = np.einsum('nji,njk->nik', matrices[indices], matrices[indices])
            orthogonal = (np.isfinite(matrices[indices]).all(axis=(1, 2))
                          & (np.max(np.abs(gram-np.eye(3)), axis=(1, 2)) < 1.e-10)
                          & (np.abs(np.linalg.det(matrices[indices])-1.) < 1.e-10))
            flags[indices[~orthogonal]] |= FLAGS['invalid_rotation']
    matrices[flags != 0] = np.nan
    diagnostics = {'rotation_quality': flags, 'rotation_valid': flags == 0,
                   'attitude_bracket_gap_seconds': ab['gap_seconds'],
                   'orbit_bracket_gap_seconds': ob['gap_seconds']}
    for label, brackets in [('attitude', ab), ('orbit', ob)]:
        for name in ('left_time_unix', 'right_time_unix'):
            diagnostics[label+'_'+name] = brackets[name]
    return matrices, diagnostics, ab


def apply(matrices, velocity):
    velocity = np.asarray(velocity, float)
    if velocity.shape != matrices.shape[:2]:
        raise ValueError('Velocity timestamps or component count do not match rotation')
    valid = np.isfinite(matrices).all(axis=(1, 2)) & np.isfinite(velocity).all(axis=1)
    out = np.full(velocity.shape, np.nan)
    out[valid] = np.einsum('nij,nj->ni', matrices[valid], velocity[valid])
    np.testing.assert_allclose(np.linalg.norm(out[valid], axis=1),
                               np.linalg.norm(velocity[valid], axis=1), rtol=1.e-11, atol=1.e-10)
    return out, valid
