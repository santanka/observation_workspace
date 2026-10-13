"""Small known-truth checks for epoch merging and magnetic interpolation guards."""
import tempfile
from pathlib import Path
import worker

np = worker.np
from cdflib import cdfwrite, cdfepoch


def write_field(path, seconds, field, cadence):
    label = '64hz' if cadence == '64hz' else '8sec'
    epoch_name = 'epoch_' + label
    with cdfwrite.CDF(str(path), delete=True) as c:
        epoch = cdfepoch.compute_tt2000([[2022, 9, 29, 5, 5, int(s), 0, 0, 0] for s in seconds])
        c.write_var({'Variable': epoch_name, 'Data_Type': 33, 'Num_Elements': 1,
                     'Rec_Vary': True, 'Dim_Sizes': []}, var_data=epoch)
        c.write_var({'Variable': 'mag_' + label + '_dsi', 'Data_Type': 21,
                     'Num_Elements': 1, 'Rec_Vary': True, 'Dim_Sizes': [3]},
                    var_attrs={'FILLVAL': [-1e31, 'CDF_REAL4']},
                    var_data=np.asarray(field, dtype=np.float32))
        c.write_var({'Variable': 'quality_' + label, 'Data_Type': 41,
                     'Num_Elements': 1, 'Rec_Vary': True, 'Dim_Sizes': []},
                    var_data=np.zeros(len(seconds), dtype=np.int8))
    return {'expanded_path': str(path), 'source': {'path': str(path)}}


def main():
    epoch = np.array([4, 8, 12, 8, 16], dtype=np.int64)
    np.testing.assert_array_equal(worker.unique_order(epoch), [0, 3, 2, 4])
    lo = np.datetime64('2022-09-29T05:05:00', 'ns')
    hi = lo + np.timedelta64(16, 's')
    t0 = lo.astype('int64') / 1e9
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # Two granules with one duplicated timestamp; second has priority at the overlap.
        a = write_field(root/'a.cdf', [0, 8], [[10, 0, 20], [20, 0, 20]], '8sec')
        b = write_field(root/'b.cdf', [8, 16], [[30, 0, 20], [50, 0, 20]], '8sec')
        arrays, _ = worker.magnetic([a, b], lo, hi, np.array([t0+4, t0+12]), '8sec', 16.)
        np.testing.assert_array_equal(arrays['mgf_b_dsi'][:, 0], [10, 30, 50])
        np.testing.assert_array_equal(arrays['mgf_bracket_gap_seconds'], [8, 8])
        actual = worker.psp.get_data('batch_mgf_dsi')
        np.testing.assert_array_equal(np.interp([t0+4, t0+12], actual.times, actual.y[:, 0]), [20, 40])
        try:
            worker.magnetic([a, b], lo, hi, np.array([t0+4]), '8sec', 1.)
        except worker.MissingMagneticData:
            pass
        else:
            raise AssertionError('Interpolation across an excessive gap was allowed')
        bad = write_field(root/'bad.cdf', [0, 8], [[10, 0, 20], [np.nan, 0, 20]], '64hz')
        try:
            worker.magnetic([bad], lo, hi, np.array([t0+4]), '64hz', 16.)
        except worker.MissingMagneticData:
            pass
        else:
            raise AssertionError('Interpolation using nonfinite field was allowed')
        try:
            worker.magnetic([a], lo, hi, np.array([t0+12]), '8sec', 16.)
        except worker.MissingMagneticData:
            pass
        else:
            raise AssertionError('Magnetic extrapolation was allowed')
    with tempfile.TemporaryDirectory() as directory:
        from test_acquisition import fixture
        from batch import fingerprint
        from cdflib import cdfepoch
        path = Path(directory)/'duplicates.cdf'
        fixture(path, product='lepi', seconds=(4, 4, 12))
        staged = [{'expanded_path': str(path), 'source': fingerprint(path)}]
        records = worker.particle_records(staged, lo, hi)
        merged = np.concatenate([r['epoch'] for r in records])
        keep = worker.unique_order(merged)
        np.testing.assert_array_equal(keep, [1, 2])
        targets = np.concatenate([r['unix'] for r in records])[keep]
        name, arrays, _ = worker.species_data(records, keep, 'FPDU', targets)
        assert worker.psp.get_data(name).y.shape[0] == 2
        np.testing.assert_array_equal(arrays['FPDU__counts_raw'], np.ones_like(arrays['FPDU__counts_raw']))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'input.cdf'
        path.write_bytes(b'original verified input')
        request = {'lepi': [{'source': worker.fingerprint(path)}], 'mgf': []}
        worker.check_input_files(request)
        path.write_bytes(b'changed during calculation')
        try:
            worker.check_input_files(request)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Changed verified input was accepted')
    print('Worker edge checks passed: duplicate epochs, 8sec merge, gap, NaN, extrapolation, input-change guard')


if __name__ == '__main__':
    main()
