# Arase statistical-analysis batch runner

`statistical_analysis_arase/Arase_BE_dispersion_relation_second.ipynb` を、
`valid_time_ranges` JSON の各 range に対して再開可能な独立 process として実行する。
元Notebookは変更せず、batch workerがメモリ上のコピーへ時刻とbatch専用設定を適用する。

## 固定した解析条件

- component: toroidal
- phase: north traveling / south traveling / standing / all
- 最低データ量: 共通E/B fit-valid sampleが64以上
- 時間方向の最低分布: occupied 10秒blockが20以上
- HFA数密度: quality flag `< 1` かつ有限値のみ
- HFA線形補間: 有効native sample間隔が5分未満の場合のみ。5分以上は補間しない
- 数密度依存量: HFA数密度から計算した値のみを統計出力に使用
- plot: 元Notebookの設定を維持。ただしmedian PSDだけ無効
- `PLOT_EB_EACHTIME_FIGURES=False` は元Notebookどおり維持
- random spectrum: 元Notebookどおり最大50枚/range

## 追加出力

各rangeの既存phase summaryへ、range全体とphase別fit時刻の背景量・位置統計を追加する。

- numeric summary: mean, std, median, q16, q84, n_valid, valid_fraction
- position: R, MLAT, circular MLT, GSM XYZ
- McIlwain L: pitch angle 90°, 60°, 30°の3成分
- GSM position covariance
- native HFA coverage/quality counts
- bootstrap raw samples (`*.npz`)
- k-rho bin statistics (`*.npz`)

range JSONのファイル名と内容hashからdataset keyを作り、実行状態と解析出力を分離する。

```text
<dataset key> = <range JSONのstem>_<JSON SHA-256の先頭12文字>
```

実行状態と全range集約は
`/mnt/j/statistical_analysis_arase/preanalysis/KAW_observation/run_state/<dataset key>/`
に保存する。図とrange別解析結果は
`/mnt/j/statistical_analysis_arase/preanalysis/KAW_observation/auto/<dataset key>/`
に保存する。
master tableには、このrunnerが `complete` と記録したrangeだけを含める。出力先に残る
旧Notebook実行のsummaryは自動では混入しない。部分実行後もmasterは全rangeについて
再構築される。

```text
KAW_observation/run_state/<dataset key>/
  downloads/range_<id>/<product>.json
  ranges/range_<id>.json
  logs/downloads/...
  logs/ranges/...
  aggregate/range_status.csv
  aggregate/arase_phase_master.csv
  aggregate/arase_phase_master.parquet  # parquet engineがある場合
```

## 実行

使用するPythonはNotebookと同じ仮想環境にする。

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py run
```

別の同形式JSONを処理する場合:

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py run \
  --ranges /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/別のranges.json
```

JSONの内容が変わるとdataset keyも変わるため、同名ファイルを更新した場合も以前の状態や
出力とは混在しない。`--state-dir`または`--output-root`を明示すれば既定値を上書きできる。

現在の環境にはParquet engineが入っていないためCSVは常に作成し、Parquetは
`pyarrow`または`fastparquet`が利用可能な場合だけ作成する。

最初は代表的な3 rangeで確認する。

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py run \
  --range-id 1 --range-id 11 --range-id 15
```

download済みデータだけを使う場合:

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py run \
  --skip-prefetch
```

事前downloadのみ:

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py prefetch
```

集約のみ:

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py aggregate
```

## GSM軌道coverage図

既定では、valid rangesに含まれる全candidate時間の図と、phase別に解析成功rangeだけを
抽出した4図を両方生成する。

```bash
.venv_pyspedas/bin/python \
  statistical_analysis_arase_pre_auto/plot_orbit_coverage.py
```

全candidateの図は`KAW_observation/auto/<dataset key>/orbit_coverage/`、phase別4図はその下の
`detected_phase/`へ保存する。各図のPNG、PDFと、再描画用の軌道NPZ cacheを保存する。
図の期間、colormap、点サイズ、透明度、軸範囲・反転、colorbar
tickは`plot_orbit_coverage.py`冒頭の`PLOT_CONFIG`で変更できる。軌道データを読み直す場合は
`--refresh-cache`を指定する。別のranges JSONにはbatch runnerと同じ`--ranges`を使う。
表示期間、出力ファイル名、colorbar tickはranges JSONから自動決定されるため、2022年に
限定されない。期間を固定したい場合は`plot_orbit_coverage.py`冒頭の`PLOT_CONFIG`で
`time_start`と`time_end`を指定する。colorbarの位置と太さは`colorbar_axes`で調整できる。
地球はGSMの昼夜を表し、XY・XZでは`X>0`を白、`X<0`を黒にする。Xが面外となる
YZでは参考図に合わせて全体を黒で表示する。

phase別に解析成功rangeだけの軌道へ限定する場合:

```bash
.venv_pyspedas/bin/python \
  statistical_analysis_arase_pre_auto/plot_orbit_coverage.py \
  --detected-only
```

`status == "ok"`でκ_E・κ_Bが有限なrange IDをphaseごとに抽出し、all、north、south、
standingを別々のPNG・PDFとして`orbit_coverage/detected_phase/`へ保存する。4図の時間色と
GSM軸範囲は共通にする。

全candidateの図だけに限定する場合は`--all-ranges-only`を指定する。

## κの初期時系列図

`status == "ok"`で、κとbootstrap q16/q84が有限な結果だけをplotする。range中央時刻を
横軸、κを縦軸とし、κ_Eはgreen、κ_Bはpurple、errorbarはbootstrap q16--q84とする。
north/south/standing/allは別々のPNG・PDFに保存する。
各phaseでplotされた点のκ_E・κ_Bを等重みで算術平均し、対応色の横破線とtitle内の
数値として表示する。

```bash
.venv_pyspedas/bin/python \
  statistical_analysis_arase_pre_auto/plot_kappa_time_series.py
```

出力先は`KAW_observation/auto/<dataset key>/kappa_time_series/`。色、marker、errorbar、
図サイズ、共通y軸範囲、表示期間はscript冒頭の`PLOT_CONFIG`で調整できる。別のranges
JSONには`--ranges`を使い、masterや出力先を直接指定する場合は`--master`、
`--output-dir`を使う。

`--force` を付けると完了markerを無視して再実行する。wavelet cache自体の上書き設定は
元Notebookの `WAVELET_OVERWRITE=False` を維持する。

## 障害分離

- downloadはrange別・product別のprocessで実行する
- EFD/MGF/orbit/LEP/HFAは各rangeの開始・終了時刻をそのまま指定する
- OMNI 1minはgeomagnetic-index処理に必要なrange前後3時間を取得する
- OMNI hourlyはその拡張区間を含む日付範囲を取得する
- 既定timeoutは15分、最大3回retryする
- range解析は1 rangeずつ独立kernel/processで実行する
- 既定timeoutは4時間
- timeout時はprocess groupを終了する
- 解析workerでは全loaderへ `no_update=True` を適用し、ネットワークへ接続しない
- 1 rangeの失敗状態は保存される。再実行時は未完了rangeだけが対象になる

指定する`trange`は必要最小限だが、実際の保存量は配布元CDFの粒度に依存する。例えば
MGF 64 Hzは1時間単位なので不要な時間ファイルを避けられる一方、EFD 64 HzとHFAは
日単位ファイルのため、短いrangeでも該当日全体のCDFが保存される。

## 注意

既存wavelet cacheは元Notebookのファイル存在判定に従って再利用される。古いcacheの
科学的互換性を保証するmanifestはまだ存在しないため、最初の3-range検証では既存cacheを
使った結果と新規計算結果を比較する必要がある。
