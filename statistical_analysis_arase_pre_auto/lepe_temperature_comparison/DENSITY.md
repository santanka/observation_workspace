# LEP-e密度とPI fit、HFAの比較

`density.py` は既存KAW batchと温度比較から独立した密度比較。
温度処理のコード/既存出力を変更せず、数密度専用の品質マスクを使用する。

| 列 | Integralのチャネル中心範囲 | PI fit |
|---|---|---|
| 全 | E >= 67 eVの全観測チャネル | raw 50–20000 eV |
| 低 | 67 <= E <= 1000 eV | 1min 50–1000 eV |
| 高 | 1000 <= E <= 20000 eV | 1min 1000–20000 eV |

Integralは観測範囲内の部分密度、PIのneは全速度空間に外挿されたモデル規格化密度。
HFA L3の `ne_mgf` はUHR周波数と磁場から得る独立な参照密度。HFAを厳密な真値と固定しない。
低/高のPI neは足さない。Integral/HFAが1未満という事実だけでは積分の誤りとは判断しない。
初版は提供PI neとの比較であり、同じ観測帯域に制限したPIモデルの前向き積分は含まない。

## 実行

ワークスペースのルートで科学環境のPythonを使用する。時刻はUTC。

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/lepe_temperature_comparison/density.py \
  --date 2022-09-01 --dry-run

.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/lepe_temperature_comparison/density.py \
  --date 2022-09-01

# 一時保存先で検証
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/lepe_temperature_comparison/density.py \
  --date 2022-09-01 --output-root /tmp/lepe_density_hfa_v1_validation

# 月単位も指定可能（今回の実データ検証は9月1日のみ）
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/lepe_temperature_comparison/density.py \
  --month 202209

.venv_pyspedas/bin/python -m unittest discover \
  -s statistical_analysis_arase_pre_auto/lepe_temperature_comparison -p 'test_*.py'
```

`--month` は202209 / 2022-09に対応。複数datasetが該当すれば `--dataset <auto/dataset key>` を指定する。
最新のローカルLEP-e L2 v05とHFA L3、PIの低/高1minと全rawが必要。
HFAは `<raw-root>/pwe/hfa/l3/YYYY/MM/` から最新ローカル版を選ぶ。
PIパスの既定値は温度比較と同じ。`--raw-root` / `--fit-1min-root` / `--fit-raw-root` で変更できる。
ネットワーク取得は行わない。不足入力は失敗とし、データ不存在を科学的に確定する扱いにはしない。

## 単位・品質

IntegralとHFAの元単位はcm^-3（HFA属性は/cc）。PI neはm^-3なので1e-6倍する。
CSV/NPZは元のPI `ne` もm^-3のまま残し、換算後の列は `density_raw_cm3` / `density_cm3` と明記する。
HFAの元float32密度は診断データに保持し、比較用密度/比はfloat64で保存する。
原密度と判定理由を保持し、既定で密度上限による除外やクリッピングを行わない。

- Integral: 有限正密度、PI属性のpacket_info bitmask、帯域内に有限フラックスが1セル以上あること。
  温度・磁場方向・温度用valid列には依存させない。負フラックスは公開ルーチンの扱いを維持する。
  原フラックスの有限/負値数を保存し、不完全なエネルギー被覆を記録できるようにする。
- PI: 有限正ne、両温度が有限正、num_valid_data > 3。
  `--low-fit-flux-min` は既定1e4で両方向の平均フラックスが厳密にこれを超えることを要求する。
  `--high-fit-flux-min` / `--raw-fit-flux-min` は既定で無効。非正フラックスは警告として残す。
  残差や温度上限による除外は追加しない。
- HFA: 有限正ne_mgfかつquality_flagが厳密に0。ビットフラグを連続値やVALIDMIN/MAX範囲として扱わない。
  元の整数フラグを保持する。

Integral除外ビット: 1=密度、2=packet、4=有限フラックスなし。
PI除外ビット: 1=密度、2=温度、4=点数、8=フラックス。
HFA除外ビット: 1=密度、2=quality_flag。

公開 `erg_lep_part_products` は全モーメントを一括出力するため、捨てるテンソル診断の計算用に
非ゼロの補助磁場ベクトルを与える。スカラー密度の数式にこの方向は入らず、実MGFを必要としない。
補助ベクトルを変えても密度が変化しないことを実データで検証する。

## 時間・図

図は上段が密度、下段がHFAとの比、全/低/高を列とする2×3。
全域はIntegralとPI rawのnative点を薄く、その密度の算術1分平均を太く表示する。
低/高はIntegral密度の1分平均と提供PI fitを表示する。全域raw neの平均は、平均PSDへの再フィットではない。
どの列にも同じHFA参照値を元Epochの点で表示し、HFAの横棒や補間線は描かない。
PIの1min横棒だけは平均区間を表す。有限正のPI/HFA除外値は灰色×で表示する。

HFAはCDF属性上beginning of intervalだが、202209の実Epochは各分の約30秒。
元Epochを変更せず、PIの公称分中心から `--hfa-match-tolerance-s`（既定1秒）以内の
最近傍を一対一対応させる。元時刻・符号付きずれ・成否を残す。
raw Integral/PIは `--raw-match-tolerance-ms`（既定1 ms）以内の同様の対応。
一対一にならない競合は未対応。HFA不良/欠測を補間・持ち越ししない。
HFAが使えない分もIntegral/PI密度は残し、HFAに対する比だけを欠測にする。

完全なUTC分区間で平均/比を計算する。CSV時刻は分先頭、表示点は分中心。
イベント外の分区間部分は灰色背景とする。raw比較のイベント集計だけは厳密に [start,end) 内。
各分の総数・有効数・割合を保存する。露出時間重みや最低カバレッジ閾値は導入しない。
`--plot-density-max <cm^-3>` は表示上限だけを変更し、範囲外を三角と注記で示す。生値/統計対象は変更しない。

## 出力と再開

```text
<event>/lepe_temperature_comparison/density_hfa_v1/
    density_comparison.png
    density_comparison.pdf
    native_comparison.csv
    minute_comparison.csv
    diagnostics.npz
    provenance.json
```

native CSVはIntegral、PI raw、HFA元Epochの和集合。レコード種別とイベント内外の列を持つ。
全域rawの対応値・ずれ・比はIntegral時刻の `raw_match__*` に保存する。
minute CSVはHFA対応情報と各帯域の密度、品質、有効数/割合、HFA比を含む。
NPZは各入力系列を元時刻/元dtypeで保持する。JSONは入力・コード・出力のSHA256と全条件を記録する。

日集計は `<date>/lepe_temperature_comparison/density_hfa_v1/event_summary.csv`。
イベントごとに有効分数とHFA対応分数、密度比の中央値と四分位点を保存する。
月処理は1日1プロセスで直列に実行し、日失敗後も続行する。失敗があれば終了コード1。
日別成否、成功した日のイベント集計、1分比の月統計をdataset直下の比較用フォルダへ保存する。
月統計は共有分を重複カウントせず、同じ分の重複比が不整合ならエラーとする。
入力/コード/設定/境界/出力SHA256が一致するイベントだけ再利用し、`--force` で再計算する。
既存のKAW処理結果・温度比較結果を変更しない。

保存済み結果は `validate_density.py --day-root <dataset内の日フォルダ>` で再確認できる。
`--reference-temperature-v3` で既存の温度比較に保存された密度と比較し、
`--check-auxiliary-field` で最初のイベントのLEP-e CDFを再計算して補助方向の不変性を検査する。
9月1日の実施記録は `DENSITY_VALIDATION_20220901.md` を参照。
