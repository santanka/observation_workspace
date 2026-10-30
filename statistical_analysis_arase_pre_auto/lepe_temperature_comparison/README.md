# LEP-e PI multiband温度比較 v3

数密度のIntegral・PI fit・HFA比較は独立した `density.py` と `DENSITY.md` を参照。

既存KAW候補イベントについて、`erg_lep_part_products` の中心モーメント温度と
PI提供のbi-Maxwellianフィット温度を比較する。既存batch、ノートブック、KAW選別条件は変更しない。

図は3行（effective・parallel・perpendicular）×3列（全・低・高エネルギー）。

| 列 | Integralのチャネル中心範囲 | PI fit | 時間処理 |
|---|---|---|---|
| 全 | E >= 67 eV、上限追加なし | raw 50–20000 eV | 双方のネイティブ値 |
| 低 | 67 <= E <= 1000 eV | 1min 50–1000 eV | Integral温度の1分平均と、1分平均分布のフィット |
| 高 | 1000 <= E <= 20000 eV | 1min 1000–20000 eV | 同上 |

中心でチャネルを選び、端のビン幅を切り詰めない。1000 eVは両帯域に含む。
現在の実チャネルは約963 eVと1179 eVの間に境界があり、重複しない。
実使用範囲とPI側のフィットチャネル中心を来歴へ記録する。
全Integralは上限追加なしなので、将来20 keV超の実チャネルがあればPIとの範囲差が残る。

## 実行

ワークスペースのルートから実行する。日付はUTC。`compare.py` は既定で `multiband.py` に渡す。

```bash
# 対象とローカルファイルの存在確認のみ。計算・取得・出力書き込みなし。
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/lepe_temperature_comparison/compare.py \
  --month 202209 --dry-run

# 1日だけ計算・保存
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/lepe_temperature_comparison/compare.py \
  --date 2022-09-01

# 月単位。不足するLEP-e/MGFだけ直列取得する場合。
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/lepe_temperature_comparison/compare.py \
  --month 202209 --download-missing

# 一時出力での検証。表示上限100 keVを超える点もマーカーと最大値を表示し、生値は保持。
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/lepe_temperature_comparison/compare.py \
  --date 2022-09-01 --output-root /tmp/lepe_multiband_v3_validation --plot-temperature-max 100000

.venv_pyspedas/bin/python -m unittest discover \
  -s statistical_analysis_arase_pre_auto/lepe_temperature_comparison -p 'test_*.py'

# 保存済み結果の整合確認。--day-rootにはdataset直下の日フォルダを指定。
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/lepe_temperature_comparison/validate_saved.py \
  --day-root /tmp/lepe_multiband_v3_validation/Arase_valid_time_ranges_20220901_000000_to_20221001_000000_b25627655ffe/2022-09-01 \
  --reference-dual-energy
```

月指定は202209 / 2022-09に対応。複数datasetに同じ日/月がある場合は `--dataset <auto/dataset key>` を明示する。
対象は既存候補フォルダすべて。秒以下のイベント境界は既存 `run_state/.../ranges/range_*.json` から読む。
日をまたぐイベントは未対応として当日を失敗扱いにする。

## 入力・時間処理

- L2 LEP-e v05: `/mnt/j/observation_data/ergsc/satellite/erg/lepe/l2/3dflux/`。
- MGF L2 8sec DSI: 同じraw rootの `mgf/l2/8sec/`。
- PI 1min: `/mnt/j/observation_data/lepe-fitmom-data-v2/fitmom-1min/1min/YYYY/`。
  `--fit-1min-root` は `1min/` を指定する。
- PI raw: `/mnt/j/observation_data/lepe-fitmom-data-v2/fitmom-raw/YYYY/`。
  `--fit-raw-root` は `fitmom-raw/` を指定する。

PIのversion、energy_min/max、resampling、時刻の単調増加/重複を確認する。
変数単位属性がないため温度eV・密度m^-3はPI資料/生成コードに基づく。
新raw生成コード自体の実装は独立に検証していない。新入力を旧1min値で自動補完しない。

rawは8秒等間隔ではない。実際の8秒/16秒間隔と欠測を保持し、点で表示する。時間幅を一律±4秒としない。
raw同士の比は `--raw-match-tolerance-ms`（既定1 ms）以内の最近傍に限定し、1対1にならない競合は双方とも未対応。
符号付き時刻差・成否・元時刻を保存し、未対応を補間で埋めない。native比のイベント集計は [start,end) に限定する。

低/高はイベントと重なる完全なUTC分区間を使用する。CSVは左端、図の点は中心。
PI横棒は±30秒の平均区間であり温度誤差ではない。イベント外は灰色背景。
Integralは温度の1分平均、PIは分布の1分平均後のフィットで、処理順序は異なる。
スピン露出を分境界で分割しない。各分/各成分の有効数と割合を保存し、最低カバレッジ条件は設定しない。

## 温度と品質

公開 `erg_lep_part_products` のDSI中心温度テンソルから
`Tpara=bhat^T T_DSI bhat`, `Tperp=(trace(T_DSI)-Tpara)/2`,
`Teff=trace(T_DSI)/3=(Tpara+2*Tperp)/3` を計算する。すべてeV、衛星電位0 V。
MGF 8sec DSIをスピン開始時刻へ線形補間し、16秒超のギャップと外挿を拒否する。
MGF品質フラグは保持し追加選別は行わない。既存KAWの64 Hz背景磁場とは異なる。

IntegralはPI属性のpacket_info bitmask、正の密度、有効磁場を共通条件とし、成分別に有限正の温度を採用する。
一成分の欠損で他成分を除外しない。成分ごとの平均有効点が異なる場合、平均後のTeffと成分の恒等式は成立しないことがある。
PIは種別/成分ごとに有限正温度と `num_valid_data > 3` を要求する。Teffは両成分が有効な場合だけ採用する。
この点数条件だけでフィット品質は保証されない。

| オプション | 既定 |
|---|---|
| `--low-fit-flux-min` | 対応成分eflux_mean > 1e4 |
| `--high-fit-flux-min` | 閾値なし |
| `--raw-fit-flux-min` | 閾値なし |
| `--fit-temperature-max` | 品質上限なし |
| `--plot-temperature-max` | 表示上限なし |

残差による除外は追加しない。高/rawの非正・非有限フラックスは警告列として保持し、閾値が無効ならその警告だけでは除外しない。
成分別の除外ビットは1=温度非有限/非正、2=点数不足、4=フラックス条件、8=温度上限。
残差、点数、フラックス、生温度を保存する。有限正の除外値も灰色×で表示する。
表示上限指定時は範囲外を三角と最大値の注記で示し、CSV/NPZと統計対象は変更しない。

PIモデルフィット、部分積分、元データの較正は異なる。全域を単一bi-Maxwellianで近似できるとは限らない。
低/高のフィット温度・密度を加算/平均して全域フィットとみなさない。
PI資料は本データを非公式quick look用としており、比較図だけで科学的妥当性の判断は完結しない。

## 出力・再開

```text
<event>/lepe_temperature_comparison/pi_multiband_v3/
    temperature_comparison.png
    temperature_comparison.pdf
    native_comparison.csv
    minute_comparison.csv
    diagnostics.npz
    provenance.json
```

native CSVはIntegralとPI rawの実時刻の和集合。`record_has_integral` / `record_has_pi_raw` でレコード種別を区別する。
`raw_match__*` はIntegral時刻でのPI対応値・時刻差・比。元のPI rawは `pi_raw__*`、Integralは `all__*` / `low__*` / `high__*`。
minute CSVは低/高の接頭辞で値・品質・有効数・比を分ける。
NPZは元DSIテンソル、補間磁場、速度、PI生値/診断値、LEP-e/MGF品質フラグを含む。
JSONは全入力SHA256、両コードSHA256、パッケージ版、設定、実チャネル、PI属性を記録する。

日集計は `<date>/lepe_temperature_comparison/pi_multiband_v3/event_summary.csv`。
月集計/日ログは `<dataset>/lepe_temperature_comparison/pi_multiband_v3/` 以下。
月処理は1日1プロセスの直列実行。イベント/日の失敗後も次へ進み、1件でも失敗すると終了コード1。
入力読み込み失敗は当日全体の失敗となる。中断時は処理中の子プロセスを停止する。
再実行は入力・コード・版・設定・境界・出力SHA256が一致するイベントだけ再利用する。`--force` で全再計算。

既定はローカルのみ。不足LEP-e/MGFの取得は `--download-missing` を明示する。PIは自動取得しない。
取得失敗を入力不存在とはみなさず失敗として残す。
dry-runはファイル存在と区間対応のみで、全日の時間被覆/内容の検証ではない。入力不足時は終了コード1。

旧2条件は `compare.py --legacy-dual-energy ...` で再現可能。`README_dual_energy.md` を参照。旧出力は変更しない。

検証内容と実施範囲は `VALIDATION_20220901.md` に記載する。
