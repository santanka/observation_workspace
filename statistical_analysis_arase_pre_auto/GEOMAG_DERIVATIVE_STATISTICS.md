# Dst・AE・SYM-H・ALの変化率とβₑの一括追加集計

既存の2020–2022年統合masterを使い、追加の環境統計と依存性図だけを作る。
元Notebook、batch runner、完了marker、master、κ・X₀のfitは変更しない。
追加CDF処理を含む全期間実行は利用者側で行う。

## 一括実行

βₑのrange並列数を指定する入口は
[並列版の説明](PARALLEL_ENVIRONMENT_STATISTICS.md)を参照。
旧実行を止めてから`run_environment_statistics_parallel.py --workers 2`等へ切り替える。

最初は1 rangeの計画だけ確認する。CDFの配列は読み込まず、元kappa CSVのhash確認等を行う。

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/run_environment_statistics.py \
  --start-month 202001 --end-month 202212 --max-ranges 1 --plan-only
```

最初の1 rangeの追加集計（作図なし）:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/run_environment_statistics.py \
  --start-month 202001 --end-month 202212 --max-ranges 1 --prepare-only
```

確認後、まとめて全対象を追加集計・作図:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/run_environment_statistics.py \
  --start-month 202001 --end-month 202212 \
  --min-occupied-blocks 10 --min-bin-count 5 \
  --difference-method centered --dst-lag-hours 1 --ae-lag-minutes 5 \
  --symh-lag-minutes 5 --al-lag-minutes 5
```

既定の作図対象はTᵢ/Tₑ・βₑ・dDst/dt・dAE/dt・SYM-H・AL・dSYM-H/dt・dAL/dtの8変数×4方向=32図。
各図でPNG・PDF・bin統計CSVを出力する。既存nₑ・Dst・βᵢ図はこの既定コマンドでは再作図しない。
処理順は地磁気変化率の集計→βₑの集計→依存性図。
各準備段階で失敗すると後続を実行せず停止する。結果の欠測は有効率として記録するが、有効率では選別しない。
`environment_inputs/pipeline/latest_run.json`に実行コマンドと段階別statusを残す。

同じコマンドで再開可能。各追加集計は入力・設定・コードのsignatureが合うrange結果を再利用する。
`--prepare-only`で追加集計のみ、`--plot-only`で作図のみ、`--parameter d_ae_dt`等で対象限定も可能。
`--parameter`、`--phase`は反復できる。`--max-ranges`は部分結果を全期間図として扱わないため
`--prepare-only`または`--plan-only`と併用する。
`--phase`は作図の限定であり、追加集計は全phaseで必要なrangeの和集合を対象とする。

既定はoffline。地磁気指数はローカルOMNI CDFを直接読むだけで、ネットワーク処理自体を含まない。
βₑも既定はネットワークを禁止する。βₑの不足入力をloaderが取得することを許可する場合だけ
`--allow-network`を追加する（地磁気変化率のCDFはこの指定でも取得しない）。
`--spedas-dir`でCDF cacheのroot、`--statistics-dir`で期間の統合入力rootを変更できる。

## 差分の定義

ユーザー指定により既定は中心差分。両方の単位をnT/hにする。

| 指標 | 元のcadence | 既定lag h | 計算 |
|---|---|---|---|
| dDst/dt | 1 h | 1 h | `[Dst(t+1h)−Dst(t−1h)] / 2h` |
| dAE/dt | 1 min | 5 min | `[AE(t+5min)−AE(t−5min)] / (10/60 h)` |
| dSYM-H/dt | 1 min | 5 min | `[SYM-H(t+5min)−SYM-H(t−5min)] / (10/60 h)` |
| dAL/dt | 1 min | 5 min | `[AL(t+5min)−AL(t−5min)] / (10/60 h)` |

中心差分の両端間隔はDstが2時間、AE・SYM-H・ALが10分。未来側の値も含むため、観測時点までの予測指標ではない。
64 Hzに補間した指数や、イベントごとの中央値の間を微分しない。追加の平滑化は行わない。
後退差分に変える場合は`--difference-method backward`とし、`[X(t)−X(t−h)]/h`を使う。
`--dst-lag-hours`は正の整数時間、`--ae-lag-minutes`・`--symh-lag-minutes`・`--al-lag-minutes`は正の整数分に限る。

SYM-H・AL自体の図は既存masterの`range_geomag_SYM_H_nT_median`・`range_geomag_AL_nT_median`を使用し、
対象指数のrange中央値が有限であることを要求し、有効率による選別は行わない。値を追加CDFから取り直した中央値に置き換えない。
ALの負値・正値・0を含め、元の符号を保つ。`|AL|`、`−AL`、微分の絶対値には変換しない。

native時系列を正規cadenceにreindexし、欠測時刻・FILLVAL・VALIDMIN/MAX外・非有限値をNaNとする。
**差分区間内の全native標本が有効である場合だけ**計算する。両端だけが有効でも、間に欠測があれば無効。
端点の不足を片側差分で補わず、ファイル境界では前後の月/半年CDFも含めて読む。

計算後のnative変化率を、保存済みkappa時系列CSVのrange内時刻へnearestで対応させる。
許容差はnative cadenceの半分（Dst:30分、AE:30秒）。等距離なら後側のnative時刻を使う。
NaNのnative時刻を除去せず、遠い有効値で欠測を埋めない。
この時刻列でrange中央値・q16・q84・平均・標準偏差・有効率を集計する。
gridの標本数が既存masterと一致することも確認する。

range中央値は符号付きの代表的な増減率であり、range内の正負の変動が相殺され得る。
絶対値の中央値・変動強度・回帰傾きとは異なる量。

## 保存先と出典

```text
auto/statistics/202001-202212/
  environment_inputs/
    beta_e/                         # βₑの独立集計
    geomag_derivatives/
      centered_dst_1h_ae_5min_symh_5min_al_5min/
        plan.json
        ranges/                     # unique rangeごとのJSON
        failures.json
        range_summary.csv
        range_summary.provenance.json
    pipeline/latest_run.json
  min_blocks_10/
    ti_te_statistics/
    beta_e_statistics/
    d_dst_dt_statistics/
    d_ae_dt_statistics/
    sym_h_statistics/
    al_statistics/
    d_sym_h_dt_statistics/
    d_al_dt_statistics/
```

差分設定ごとに追加集計の保存先を分ける。CDFは月/半年ごとに最新のcached versionを選び、pathとSHA256を記録する。
以前のDst・AEのみの追加集計ディレクトリは変更せず、新しい4微分の集計を別の設定名で保存する。
元の解析時点のCDF checksumはmasterにないため、現在のcacheと過去入力のbyte単位の一致は保証できない。
必要なCDF unitがない場合はplan/provenanceに記録し、該当差分を欠測にする。
対象indexのCDFが全くない、schemaが違う、元時刻CSVがない等の処理失敗はfailuresに記録し、集約CSV更新前に停止する。

## 作図条件と調整

上段log10 X₀・下段κ、列E/B/S、方向別、all MLT。
occupied block≥10、bin中央値・分布幅はn≥5を既定とする。range内有効率≥80%の条件は廃止した。
100秒移動平均の端点NaNで短いrangeを除かないよう、有効率を採否判定には使わない。
全欠測で中央値が得られないrangeは対象変数の図から除く。
nは有限なfit値を持つrange数。有効率の分母は元kappa時刻列の標本数であり、独立なnative標本数ではない。
散布点は採用rangeをすべて表示し、bin統計CSVにはn<5のbinも残す。
q16–q84はrange間分布幅であり、中央値の信頼区間ではない。

変化率・SYM-H・ALは符号付き線形軸。bin幅の既定値と変更オプション:

| 指標 | bin幅 | オプション |
|---|---|---|
| dDst/dt | 10 nT/h | `--ddst-bin-width-nt-per-hour` |
| dAE/dt | 600 nT/h | `--dae-bin-width-nt-per-hour` |
| SYM-H | 20 nT | `--symh-bin-width-nt` |
| AL | 200 nT | `--al-bin-width-nt` |
| dSYM-H/dt | 60 nT/h | `--dsymh-bin-width-nt-per-hour` |
| dAL/dt | 600 nT/h | `--dal-bin-width-nt-per-hour` |

`--min-bin-count`を調整可能。`--min-valid-fraction`は明示した場合のみ有効率選別を再有効化する互換オプション。
今回の一括実行では指定しない。準備scriptが出力する「valid fraction >= 0.8」の件数は従来の診断表示であり、選別条件ではない。
差分方式・lag・平滑化なしを図とprovenanceに明記する。

SYM-H・ALとその微分だけに限定する例（βₑの時系列処理は行わない）:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/run_environment_statistics.py \
  --start-month 202001 --end-month 202212 \
  --parameter sym_h --parameter al --parameter d_sym_h_dt --parameter d_al_dt
```

微分の追加集計scriptは4種類の微分を一緒に集計する。`--parameter`は一括runnerの
準備段階の要否と作図対象を選ぶもので、準備内部の微分を個別に省く指定ではない。

作図先は設定を変えても同じ内容ディレクトリで上書きされる。条件比較の図を残す場合は
一括runnerの`--plot-output-dir /path/to/alternate_parent`を指定する。
単独の作図scriptでは`--derivative-summary /path/to/range_summary.csv`で使用する追加集計を選び、
`--output-dir`で図の親ディレクトリを変更する。

βₑ・Tᵢ/Tₑの定義と追加集計の制約は[環境依存性の説明](ENVIRONMENT_PARAMETER_STATISTICS.md)を参照。

出典: [NASA OMNI high-resolution説明](https://omniweb.gsfc.nasa.gov/html/omni_min_data.html)、
[WDC京都のDst/AE時間値](https://wdcvmweb.kugi.kyoto-u.ac.jp/dstae/index.html)。
