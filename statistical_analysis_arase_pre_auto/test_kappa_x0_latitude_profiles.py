import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import plot_kappa_x0_latitude_profiles as profiles


def make_frame():
    rows = []
    for i, (latitude, mlt, value) in enumerate([(7., 21., 1.), (8., 21., 2.),
                                               (7., 21., 3.), (-7., 21., 100.), (8., 12., 999.)]):
        row = {"range_id": i + 1, "phase_mode": "all", "status": "ok",
               "n_occupied_blocks": 10, "block_duration": "10s",
               "range_L_90deg_mean": 4.2, "range_MLAT_deg_mean": latitude,
               "range_MLT_circular_mean_hour": mlt}
        for component in "EBS":
            row[f"kappa_{component}"] = value
            row[f"log10_X0_{component}"] = -value
        rows.append(row)
    return pd.DataFrame(rows)


class LatitudeProfileTests(unittest.TestCase):
    def test_correlations_are_per_l_bin_and_use_signed_or_folded_range_latitudes(self):
        frame = make_frame()
        # A separate L bin has a different slope and must not enter the first correlation.
        other = frame.iloc[:3].copy()
        other['range_L_90deg_mean'] = 5.2
        other['kappa_E'] = [200., 100., 200.]
        frame = pd.concat([frame, other], ignore_index=True)
        for coordinate in profiles.COORDINATES:
            points, table, _ = profiles.profile_data(frame, 'all', coordinate)
            for l_bin, cell in points.groupby('_l_bin'):
                rows = table.loc[table.l_bin.eq(l_bin) & table.family.eq('kappa') & table.component.eq('E')]
                expected = profiles.correlations.pair_correlations(cell._latitude, cell.kappa_E)
                self.assertTrue(rows.correlation_n.eq(len(cell)).all())
                np.testing.assert_allclose(rows.pearson_r, expected['pearson_r'])
                np.testing.assert_allclose(rows.spearman_rho, expected['spearman_rho'])
                self.assertEqual(rows.pearson_x_coordinate.iloc[0],
                                 'abs_MLAT_deg' if coordinate == 'l_abs_mlat' else 'MLAT_deg')
            outside = table.loc[table.l_bin.eq(0)]
            self.assertTrue(outside.correlation_n.eq(0).all())
            self.assertTrue(outside.pearson_r.isna().all())

    def test_finite_pairs_and_plot_labels_ignore_bin_display_threshold(self):
        frame = make_frame()
        frame.loc[0, 'kappa_E'] = np.nan
        points, table, _ = profiles.profile_data(frame, 'all', 'l_abs_mlat')
        index = int(points._l_bin.iloc[0])
        table = table.loc[table.l_bin.eq(index)]
        row = table.loc[table.family.eq('kappa') & table.component.eq('E')].iloc[0]
        self.assertEqual(row.correlation_n, 3)
        self.assertEqual(int(table.loc[table.family.eq('kappa') & table.component.eq('B'), 'correlation_n'].iloc[0]), 4)
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(profiles.PLOT_CONFIG, {'dpi': 30}):
            with mock.patch.object(profiles.plt, 'close'):
                profiles.plot_profile(points, table, 'all', 'l_abs_mlat', 4., 5., 'test',
                                      Path(temp) / 'profile', minimum_bin_count=100)
                figure = profiles.plt.gcf()
                self.assertTrue(all(r'r_{\mathrm{P}}' in ax.get_title() for ax in figure.axes))
                self.assertTrue(all(r'\rho_{\mathrm{S}}' in ax.get_title() for ax in figure.axes))
            profiles.plt.close(figure)

    def test_folded_profiles_pool_raw_ranges_and_preserve_distribution(self):
        points, table, _ = profiles.profile_data(make_frame(), "all", "l_abs_mlat")
        self.assertEqual(len(points), 4)
        self.assertTrue(points._latitude.ge(0).all())
        row = table.loc[table.family.eq("kappa") & table.component.eq("E") & table.n.gt(0)].iloc[0]
        self.assertEqual(row["n"], 4)
        self.assertEqual(row["median"], 2.5)
        self.assertEqual(row["std"], np.std([1., 2., 3., 100.], ddof=1))
        np.testing.assert_allclose([row.q16, row.q84], np.percentile([1., 2., 3., 100.], [16, 84]))

    def test_signed_profiles_keep_hemispheres_separate(self):
        _, table, _ = profiles.profile_data(make_frame(), "all", "l_mlat")
        rows = table.loc[table.family.eq("kappa") & table.component.eq("E") & table.n.gt(0)]
        self.assertEqual(rows.n.tolist(), [1, 3])
        self.assertEqual(rows["median"].tolist(), [100., 2.])

    def test_cli_outputs_both_coordinates_with_selection_and_preserves_master(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "inputs").mkdir()
            frame = make_frame()
            sparse = frame.iloc[:1].copy()
            sparse["n_occupied_blocks"] = 9
            sparse["kappa_E"] = 9999.
            frame = pd.concat([frame, sparse], ignore_index=True)
            master = root / "inputs/arase_phase_master.csv"
            frame.to_csv(master, index=False)
            before = master.read_bytes()
            (root / "inputs/provenance.json").write_text(json.dumps({
                "year": 2022, "analysis_version": profiles.batch.ANALYSIS_VERSION,
                "annual_master_sha256": profiles.batch.file_sha256(master),
            }))
            with mock.patch.dict(profiles.PLOT_CONFIG, {"dpi": 40}):
                self.assertEqual(profiles.main([
                    "--year", "2022", "--statistics-dir", str(root),
                    "--phase", "all", "--l-bin", "4",
                ]), 0)
            output = root / "min_blocks_10/kappa_x0_latitude_profiles"
            self.assertEqual(len(list(output.rglob("*.png"))), 2)
            self.assertEqual(len(list(output.rglob("*.pdf"))), 2)
            self.assertEqual(len(list(output.rglob("*.csv"))), 2)
            table = pd.read_csv(next((output / "l_abs_mlat/L_4_5").glob("*.csv")))
            row = table.loc[table.family.eq("kappa") & table.component.eq("E") & table.n.gt(0)].iloc[0]
            self.assertEqual(row["n"], 4)
            self.assertEqual(row["median"], 2.5)
            self.assertEqual(master.read_bytes(), before)
            metadata = json.loads((output / "provenance.json").read_text())
            self.assertEqual(metadata["selection"]["min_occupied_blocks"], 10)


if __name__ == "__main__":
    unittest.main()
