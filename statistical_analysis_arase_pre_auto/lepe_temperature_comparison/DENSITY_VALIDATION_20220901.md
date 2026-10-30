# density_hfa_v1検証（2026-10-22）

2022-09-01の既存候補9イベントで密度比較を実行した。月全体の計算は行っていない。
ローカルLEP-e L2 v05、PIの全raw/低1min/高1min、HFA L3 v06_11を使用した。

## テスト

`test_compare.py`、`test_multiband.py`、`test_density.py` の計31件が通過した。
密度の追加テストは12件。単位変換、両温度/両フラックス条件、磁場/温度用maskとの独立性、
HFAフラグ、分中心への1対1対応、許容差境界、不良/欠測を補間しないこと、分境界、
CSV再読込時の密度比精度、共有分の重複除外、破損出力の再利用拒否、dry-runを確認した。

## 実データ

- 9イベントすべてで密度/比の2行×3列の図、CSV、診断NPZ、来歴JSONを作成した。
- native rawのIntegral/PI時刻対応はイベント内562点、未対応0点。
- HFAはイベントごとに延べ111分すべてで分中心から1秒以内に対応し、品質条件も通過した。
  隣接イベントが同じ分を含むため、ユニークな参照分数は109分。
- Integralの全/低/高の元密度は、以前の温度比較に保存した密度と相対許容差1e-12で一致した。
- 全域密度と低/高の和を計616レコードで照合した。これらはイベントと重なる完全な分区間の
  レコードで、隣接イベント間の重複を含む。最大差は約2.34e-15 cm^-3。
- PIの元ne（m^-3）と換算密度（cm^-3）の関係を照合した。
- HFAは元float32を保持し、比較値をfloat64で保存した。CSVから再計算した比が相対許容差1e-14で一致した。
- HFAの元時刻、対応ずれ、品質フラグと比の欠測条件を照合した。
- 最初のイベントで全/低/高を補助磁場方向(0,0,1)から(0,1,0)へ変えて再計算し、密度差0 cm^-3を確認した。
- 全入力、コード、出力のSHA256を照合した。
- 同じコマンドの再実行で9件すべてが `Skip verified` となり、密度を再計算しなかった。

検証コマンド:

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/lepe_temperature_comparison/validate_density.py \
  --day-root /tmp/lepe_density_hfa_v1_validation/Arase_valid_time_ranges_20220901_000000_to_20221001_000000_b25627655ffe/2022-09-01 \
  --check-auxiliary-field --reference-temperature-v3 --report /tmp/lepe_density_hfa_v1_real_validation.json
```

数値処理と対応付けの検証であり、Integral・PIモデル密度・HFAが同じ推定量であることや、
PIフィットの科学的妥当性を証明するものではない。同帯域PIモデルの前向き積分は未実装。
