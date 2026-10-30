"""Known-truth regression for MGF quarantine and TT2000 interpolation; local CDF only."""
import json
from pathlib import Path
import tempfile
import worker
from batch import fingerprint
from test_acquisition import fixture
from cdflib import cdfwrite, cdfepoch
from validate_input import validate

np = worker.np


def write(path, epoch, b, q=None, omit_quality=False):
    with cdfwrite.CDF(str(path), delete=True) as cdf:
        values = [('epoch_64hz', 33, [], epoch), ('mag_64hz_dsi', 21, [3], b)]
        if not omit_quality:
            values.append(('quality_64hz', 41, [], np.zeros(len(epoch), dtype=np.int8) if q is None else q))
        for name, typ, dims, data in values:
            cdf.write_var({'Variable': name, 'Data_Type': typ, 'Num_Elements': 1,
                           'Rec_Vary': True, 'Dim_Sizes': dims}, var_data=np.asarray(data))
    return {'source': fingerprint(path), 'expanded_path': str(path), 'nominal_utc': '2022-09-29T05:00:00'}


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        lepi = root/'lepi.cdf'
        fixture(lepi, product='lepi', seconds=(0, 4, 8, 12, 16), energy=np.arange(32, 0, -1))
        epoch = cdfepoch.compute_tt2000([[2022, 9, 29, 5, 5, s, 0, 0, 0] for s in (0, 4, 8, 12, 16)])
        b = np.tile([10., 20., 30.], (5, 1))
        req = {'input_verification': 'verified', 'profile_hash': 'synthetic', 'input_signature': 'test',
               'acquisition_evidence': [], 'start_utc': '2022-09-29T05:05:00+00:00',
               'end_utc_exclusive': '2022-09-29T05:10:00+00:00',
               'lepi': [{'source': fingerprint(lepi), 'expanded_path': str(lepi)}],
               'config': {'mgf_cadence': '64hz', 'max_mgf_gap_seconds': 4.,
                          'species': {'FPDU': 'proton', 'FHEDU': 'heplus', 'FODU': 'oplus'},
                          'energy_range_ev_per_q': [30., 30000.]}}
        def execute(label, epochs, field, expected, reason=None, q=None):
            src = write(root/(label+'.cdf'), epochs, field, q)
            before = worker.sha_file(src['source']['path'])
            report = validate(src['expanded_path'], '64hz', '2022-09-29T05:00:00')
            assert report['full_required_payload_read']
            req.update(mgf=[src], output=str(root/(label+'.npz')))
            result = worker.execute(req)
            assert result['status'] == expected, result
            assert worker.sha_file(src['source']['path']) == before
            with np.load(req['output'], allow_pickle=False) as data:
                saved = {key: data[key] for key in data.files}
            if reason is not None:
                np.testing.assert_array_equal(saved['moment_invalid_reason'], reason)
            for var in ('FPDU', 'FHEDU', 'FODU'):
                np.testing.assert_array_equal(saved[var+'__b_dsi_at_moment'], saved['mgf_b_dsi_at_particle'])
            return saved, report
        baseline, _ = execute('clean', epoch, b, 'complete')
        def compare_valid(data):
            valid = data['moment_valid']
            for var in ('FPDU', 'FHEDU', 'FODU'):
                for key in worker.pilot.MOMENTS:
                    name = var+'__'+key
                    np.testing.assert_array_equal(data[name][valid], baseline[name][valid], err_msg=name)
                    assert np.isnan(data[name][~valid]).all()
                for key in ('counts_raw', 'quality_raw'):
                    np.testing.assert_array_equal(data[var+'__'+key], baseline[var+'__'+key])
            np.testing.assert_array_equal(data['epoch_tt2000'], baseline['epoch_tt2000'])
        # For a regular varying field, independently run the public function on
        # its original MGF grid, then compare all 48 series to the new path.
        regular_b = np.array([[10.,20.,30.], [30.,20.,30.], [50.,20.,30.]])
        req['config']['max_mgf_gap_seconds'] = 8.
        regular, _ = execute('regular', epoch[[0,2,4]], regular_b, 'complete')
        lo = np.datetime64('2022-09-29T05:05:00', 'ns'); hi = lo+np.timedelta64(20, 's')
        records = worker.particle_records(req['lepi'], lo, hi)
        order = worker.unique_order(np.concatenate([r['epoch'] for r in records]))
        targets = baseline['time_unix']
        original_times = np.asarray(cdfepoch.to_datetime(epoch[[0,2,4]])).astype('datetime64[ns]').astype('int64')/1e9
        worker.psp.store_data('reference_mgf', data={'x':original_times, 'y':regular_b})
        for var, species in req['config']['species'].items():
            name, _, _ = worker.species_data(records, order, var, targets)
            worker.pilot.erg_lep_part_products(name, species=species, outputs=['moments'],
                energy=req['config']['energy_range_ev_per_q'], mag_name='reference_mgf')
            for key in worker.pilot.MOMENTS:
                np.testing.assert_allclose(regular[var+'__'+key], worker.psp.get_data(name+'_'+key).y,
                                           rtol=1e-13, atol=1e-13, equal_nan=True, err_msg=var+'__'+key)
        req['config']['max_mgf_gap_seconds'] = 4.
        # Distinct native epochs collapse in public Unix seconds, but neither is discarded.
        close_epochs = np.insert(epoch, 2, epoch[1]+128)
        close_b = np.insert(b, 2, [40., 20., 30.], axis=0)
        data, _ = execute('precision', close_epochs, close_b, 'complete', [0]*5)
        assert data['mgf_time_unix'][1] == data['mgf_time_unix'][2]
        assert len(data['mgf_epoch_tt2000']) == 6
        compare_valid(data)
        src = req['mgf'][0]
        lo = np.datetime64('2022-09-29T05:05:00', 'ns'); hi = lo+np.timedelta64(20, 's')
        # An independent interior interpolation at the 64 ns midpoint must use both points.
        target = np.array([epoch[1]+64])
        unix = np.asarray(cdfepoch.to_datetime(target)).astype('datetime64[ns]').astype('int64')/1e9
        arrays, _ = worker.magnetic([src], lo, hi, unix, '64hz', 4., target_epochs=target)
        np.testing.assert_array_equal(arrays['mgf_b_dsi_at_particle'], [[25., 20., 30.]])
        # A conflicting duplicate invalidates its exact target; other 48-series rows are unchanged.
        dup_epoch = np.insert(epoch, 3, epoch[2])
        dup_b = np.insert(b, 3, [99., 20., 30.], axis=0)
        data, report = execute('conflict', dup_epoch, dup_b, 'partial', [0, 0, 16, 0, 0])
        assert report['magnetic_record_quarantine']['conflicting_epoch_tt2000'] == [int(epoch[2])]
        compare_valid(data)
        # Both adjacent brackets are blocked; removal must not bridge the conflict.
        src = req['mgf'][0]
        target = epoch[0] + np.array([0, 6, 8, 10, 16], dtype=np.int64)*10**9
        unix = np.asarray(cdfepoch.to_datetime(target)).astype('datetime64[ns]').astype('int64')/1e9
        arrays, _ = worker.magnetic([src], lo, hi, unix, '64hz', 20., target_epochs=target)
        np.testing.assert_array_equal(arrays['moment_invalid_reason'], [0, 16, 16, 16, 0])
        # A conflict in quality also blocks the group, even with identical B.
        q = np.zeros(6, dtype=np.int8);q[3] = 1
        data, _ = execute('quality_conflict', dup_epoch, np.insert(b, 3, b[2], axis=0), 'partial', [0, 0, 16, 0, 0], q=q)
        compare_valid(data)
        old = np.asarray(cdfepoch.compute_tt2000([[2016, 12, 22, 0, 0, 0, 0, 0, 0]])).item()
        data, report = execute('old_prefix', np.insert(epoch, 0, old), np.insert(b, 0, [99., 99., 99.], axis=0), 'complete')
        assert report['magnetic_record_quarantine']['outside_record_indices'] == [0]
        np.testing.assert_array_equal(data['mgf_source_record_index'], [1, 2, 3, 4, 5])
        compare_valid(data)
        # An interior isolated record is a barrier, even when remaining gap is short.
        src = write(root/'interior.cdf', np.insert(epoch, 2, old), np.insert(b, 2, b[1], axis=0))
        target = epoch[0]+np.array([0, 6, 16], dtype=np.int64)*10**9
        unix = np.asarray(cdfepoch.to_datetime(target)).astype('datetime64[ns]').astype('int64')/1e9
        arrays, _ = worker.magnetic([src], lo, hi, unix, '64hz', 20., target_epochs=target)
        np.testing.assert_array_equal(arrays['moment_invalid_reason'], [0, 32, 0])
        # All conflicts / all outside / all nonfinite / all zero -> terminal exclusion, no product.
        for label, epochs, field in (
            ('all_conflicts', np.repeat(epoch, 2), np.tile([[10., 20., 30.], [99., 20., 30.]], (5, 1))),
            ('all_outside', np.array([old]), b[:1]),
            ('all_nonfinite', epoch, np.full((5, 3), np.nan)),
            ('all_zero', epoch, np.zeros((5, 3)))):
            src = write(root/(label+'.cdf'), epochs, field)
            validate(src['expanded_path'], '64hz', '2022-09-29T05:00:00')
            req.update(mgf=[src], output=str(root/(label+'.npz')))
            try:
                worker.execute(req)
            except worker.MissingMagneticData as exc:
                result = worker.magnetic_failure_result(req, exc)
                assert result['status'] == 'excluded' and not result['retryable'], result
                assert result['computed_records_per_species'] == 0
                assert result['invalid_records_per_species'] == 5
            else:
                raise AssertionError(label+' should be excluded')
            assert not Path(req['output']).exists()
        # Quarantine never excuses schema damage or unexplained retained-time reversal.
        for label, src in (
            ('schema', write(root/'schema.cdf', epoch, b, omit_quality=True)),
            ('reversal', write(root/'reverse.cdf', epoch[::-1], b))):
            try:
                validate(src['expanded_path'], '64hz', '2022-09-29T05:00:00')
            except (ValueError, KeyError):
                pass
            else:
                raise AssertionError(label+' was accepted')
        # Cross-granule conflict also blocks, instead of keeping whichever file is last.
        a = write(root/'a.cdf', epoch[:3], b[:3])
        bsrc = write(root/'b.cdf', epoch[2:], np.array([[99.,20.,30.], b[3], b[4]]))
        target = epoch[[0, 2, 4]]
        unix = np.asarray(cdfepoch.to_datetime(target)).astype('datetime64[ns]').astype('int64')/1e9
        arrays, _ = worker.magnetic([a, bsrc], lo, hi, unix, '64hz', 4., target_epochs=target)
        np.testing.assert_array_equal(arrays['moment_invalid_reason'], [0, 16, 0])
        print('Recovery regression passed: 128 ns and midpoint truth; all 48 moments; conflicts and barriers; quarantine provenance; all-invalid exclusions; schema/reversal guards')


if __name__ == '__main__':
    main()
