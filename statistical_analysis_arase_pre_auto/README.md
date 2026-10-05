# Arase statistical-analysis batch runner

`statistical_analysis_arase/Arase_BE_dispersion_relation_second.ipynb` を、
`valid_time_ranges` JSON の各 range に対して再開可能な独立 process として実行する。
元Notebookは変更せず、batch workerがメモリ上のコピーへ時刻とbatch専用設定を適用する。

## 固定した解析条件

- component: toroidal
- phase: north traveling / south traveling / standing / all
- 最低データ量: 共通E/B fit-valid sampleが64以上
- 時間方向の最低分布: occupied 10秒blockが6以上
- HFA数密度: quality flag `< 1` かつ有限値のみ
- HFA線形補間: 有効native sample間隔が5分未満の場合のみ。5分以上は補間しない
- 数密度依存量: HFA数密度から計算した値のみを統計出力に使用
- 電場スピン帯域判定: 0.8--6倍スピン周波数、coherence有効、かつKAW解析波数範囲内の点だけで`logrmse_E_spinband_kaw`を計算
- 電場スピン帯域閾値: 有効点10以上では`logrmse_E_spinband_kaw <= 0.75 dex`。10点未満は適用対象外として通過し、点数自体の欠損は除外
- 局所E/B残差: 縦オフセット補正後の`EB_residual_peak <= 0.70 dex`
- `logrmse_EB_shape`: 計算・保存のみ。KAW選定には使用しない
- plot: 元Notebookの設定を維持。ただしmedian PSDだけ無効
- `PLOT_EB_EACHTIME_FIGURES=False` は元Notebookどおり維持
- random spectrum: 元Notebookどおり最大50枚/range

batch workerは上記の列名・閾値・KAW波数mask・3か所の選定maskがNotebookに存在することを実行前に確認する。条件が欠けた古いNotebookを`--notebook`で指定した場合は、range解析を開始せずエラーにする。

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
  downloads/<product>/<file-unit>.json
  downloads/_blocks/<product>/<block>_attempt_<n>.json
  ranges/range_<id>.json
  logs/downloads/<product>/<block>.log
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
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py prefetch \
  --download-workers 2
```

既定では全rangeの一周後に、一時的なdownload失敗とATT通信失敗を最大2回再試行する。
kernel deathは低並列設定のまま1回だけ自動再試行する。待機時間は60秒、300秒である。
決定論的なdata gap、parse、重複時刻、plot入力エラーは同じ条件で反復しない。
postpass retryを無効化する場合は
`--postpass-retries 0`を指定する。

runner hashが変わった後でも既存complete rangeを再実行せず、missing/running/failed/timeout
markerだけを処理する場合:

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py run \
  --noncomplete-only --skip-prefetch
```

空入力、E64/B64非重複、有効FAC segmentなし、ATT dataなしは`excluded`をterminal status
として保存し、`--noncomplete-only`でも再実行しない。

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

## event別GSM・SM軌道図

1 analysis rangeを1 eventとして、GSMとSMを別図へ出力する。各図はXY、XZ、YZの
3 panelで、ORB native cadenceの軌道を実線、range startを緑の大丸、startから2分ごとの
位置を小点、range endを赤い星で示す。start/endと途中点はORB位置を指定時刻へ線形補間
する。

既処理期間へのbackfillはKAW解析を再実行せず、次で行う。

```bash
.venv_pyspedas/bin/python \
  statistical_analysis_arase_pre_auto/batch_arase.py event-orbits \
  --ranges /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20200101_000000_to_20200201_000000.json
```

通常のbatch終了後に続けて作図する場合は`run`へ`--plot-event-orbits`を付ける。既定では
manifest中の全rangeを対象とする。`--completed-only`、反復可能な`--range-id`、
`--png-only`、`--refresh-cache`をbackfill subcommandで使用できる。既存図はskipするため、
中断後に同じcommandで再開でき、上書きには`--force`を使う。

出力は`KAW_observation/auto/<dataset key>/event_orbits/range_NNN/`に保存する。GSM・SM
それぞれのPNG/PDFと、manifest hashで検証する共通ORB cacheを生成する。途中点間隔、色、
marker、線幅、軸範囲、画像形式は`plot_event_orbits.py`冒頭の`PLOT_CONFIG`で変更できる。
短いrangeでは地球スケール上でstart/endが重なるため、上段を地球を含む全体図、下段を
対応するXY・XZ・YZのevent-localな詳細図とする。詳細図は元図へ重ねず、衛星位置、地球、
目盛ラベルとの重なりを避ける。`show_event_zoom`、`event_zoom_min_span_re`、
`event_zoom_padding_fraction`で調整できる。

## κの初期時系列図

`status == "ok"`で、κとbootstrap q16/q84が有限な結果だけをplotする。range中央時刻を
横軸、κを縦軸とし、κ_Eはgreen、κ_Bはpurple、κ_Sはorange、errorbarはbootstrap
q16--q84とする。
north/south/standing/allは別々のPNG・PDFに保存する。
各phaseでplotされた点のκ_E・κ_B・κ_Sを等重みで算術平均し、対応色の横破線とtitle内の
数値として表示する。

```bash
.venv_pyspedas/bin/python \
  statistical_analysis_arase_pre_auto/plot_kappa_time_series.py
```

出力先は`KAW_observation/auto/<dataset key>/kappa_time_series/`。色、marker、errorbar、
図サイズ、共通y軸範囲、表示期間はscript冒頭の`PLOT_CONFIG`で調整できる。別のranges
JSONには`--ranges`を使い、masterや出力先を直接指定する場合は`--master`、
`--output-dir`を使う。

`--force` を付けると完了markerを無視して再実行する。batch実行中のwavelet NetCDFは
`/tmp/arase_wavelet_spectra/` 以下のrange固有directoryに一時保存し、rangeのterminal
statusを書いた後に、成功・失敗・timeoutによらず削除する。保存先の親directoryは
`--wavelet-scratch-root` で変更できる。元Notebookを直接実行した場合の永続cache設定は
変更しない。

## KAW occurrence probability図

rangeの代表位置をbin分けし、L--MLAT面とL--MLT面について、対象期間全体のArase
dwelling time、KAW observation time、occurrence probabilityの3 panel図を作成する。
L--MLATは双極子磁力線形状へ写像し、両座標系とも中央に昼夜を示す地球を描く。
L--MLATの分母・分子はともに18--06 MLTのnightsideだけを使用する。

```bash
.venv_pyspedas/bin/python \
  statistical_analysis_arase_pre_auto/plot_occurrence_probability.py
```

既定ではall、north traveling、south traveling、standingの4 phaseを個別に出力する。
allだけを作成する場合:

```bash
.venv_pyspedas/bin/python \
  statistical_analysis_arase_pre_auto/plot_occurrence_probability.py \
  --phase all
```

対象期間のORB L2日次fileを確認し、local cacheにない日だけ最大3回downloadする。
接続不良時は待機して再試行し、全日が揃わない場合は部分的なdwelling timeで図を作らず
停止する。local dataだけを使う場合は`--no-download`、ORB cacheを再構築する場合は
`--refresh-dwelling-cache`を指定する。

出力先は`KAW_observation/auto/<dataset key>/occurrence_probability/`で、各図のPNG、
PDF、各binの時間・確率・range数を含むCSV、再利用用ORB dwelling cacheを保存する。
確率の分母・分子、代表位置、
bin境界、coverage maskの定義は
[`OCCURRENCE_PROBABILITY.md`](OCCURRENCE_PROBABILITY.md)に記載する。bin幅、表示範囲、
log/linear scale、colormap、minimum dwelling durationは`plot_occurrence_probability.py`冒頭の
`PLOT_CONFIG`で変更できる。
## κ・X0空間統計図

`status == "ok"`のrangeを代表位置でbin分けし、κと`log10(X0)`のmedianを
L--MLAT面およびL--MLT面へ描く。各図はE、B、Sの3 panelで、L--MLATは18--06 MLTに
限定する。これらに加え、上段を`log10(X0)`、下段をκとした(a)--(f)の2行3列統合図も
出力する。統合図の各panelには、表示binに属するrange全体の
`median [q16, q84], N`を示す。

```bash
.venv_pyspedas/bin/python \
  statistical_analysis_arase_pre_auto/plot_kappa_x0_statistics.py
```

既定ではn > 0の全binを表示する。表示に必要なrange数は
`plot_kappa_x0_statistics.py`冒頭の`PLOT_CONFIG["min_bin_count"]`で変更できる。
mean、std、median、q16、q84、min、max、nを含むCSVも保存する。詳細な定義は
[`KAPPA_X0_STATISTICS.md`](KAPPA_X0_STATISTICS.md)に記載する。カラーバーは6パラメータ
別に表示bin中央値のq5--q95を使い、`PLOT_CONFIG["color_percentiles"]`で変更できる。

## AE指数に対するκ・X0統計図

`status == "ok"`の各rangeについて、range中のAE中央値に対するκと`log10(X0)`を描く。
各phaseを2行3列の図として出力し、raw range点に加えてAE bin内のmedian、q16--q84、
range数を示す。

```bash
.venv_pyspedas/bin/python \
  statistical_analysis_arase_pre_auto/plot_ae_parameter_statistics.py
```

既定のAE bin幅は200 nT、表示条件はn > 0、対象は全MLTである。bin幅、最小range数、
夜側限定などは`plot_ae_parameter_statistics.py`冒頭の`PLOT_CONFIG`で変更できる。
定義と注意点は[`AE_PARAMETER_STATISTICS.md`](AE_PARAMETER_STATISTICS.md)に記載する。



## Download plan・障害分離

prefetchはrangeごとには実行しない。選択した全rangeを、配布CDFの物理的な
ファイル時間単位へ変換してから重複を除去し、連続unitをblock化する。

| product | file unit | 1 blockの上限 |
|---|---:|---:|
| MGF L2 64 Hz | 1時間 | 24時間 |
| PWE-EFD L2 64 Hz | 1日 | 7日 |
| ORB L2 def | 1日 | 7日 |
| ATT L2 txt | 1日 | 7日 |
| LEPe/LEPi L2 3dflux | 1日 | 7日 |
| PWE-HFA L2/L3 | 1日 | 7日 |
| OMNI 1 min | 1か月 | 1 file |
| OMNI hourly | 半年 | 1 file |

ATTには座標変換時の補間余裕としてrange前後60秒、OMNIには従来どおりrange前後3時間の
padを加えてからfile unitへ変換する。
月・半年fileは代表1日だけをloaderへ渡し、同じremote file名の反復列挙を避ける。
blockのtrange終端は次unit境界の1秒前とし、不要な次fileを取得しない。

- download markerは`range/product`ではなく`product/file-unit`単位で保存する
- markerにはloaderが返したlocal file一覧を保存し、全fileが存在する場合だけ再利用する
- ATTはmarkerがない初回でもlocal日次fileの全coverageを先に確認し、揃っていればremote indexへ接続しない
- MGFもlocal 1時間fileの全coverageを先に確認する。不足時だけremoteへ接続し、取得後に要求hourを再検証する
- 完了unitを除外した後で残りを再block化するため、再実行時は欠損unitだけを取得する
- block内の一部CDFだけが欠損しても、取得済みunitは完了として保存する
- 欠損・失敗unitを必要とするrangeだけを`failed_download`にする
- `missing_remote`は再利用し、通常の再実行では再取得しない。再確認には`--force`を使う
- 既定では8 blockを並列downloadする。`--download-workers`で変更できる
- 各downloadは独立processなのでPySPEDAS/tplotのglobal stateを共有しない
- 既定timeoutはblockごとに15分、最大3回retryする
- range解析は1 rangeずつ独立kernel/processで実行する
- ATTはdownload後に必要数値列をparse検証する。公式textでGZ-Deltaと負の補助値の間の
  空白が欠ける既知形式は、raw fileを変更せずtolerant parserで分離する
- ATT検証モジュールはscript直接実行と`python -m`実行の両方で読み込む。
  workspaceの`PYTHONPATH`設定には依存しない
- ATTの数値列は既存PySPEDASと同じ`astype(float)`で変換する。公式の`NaN`は欠損値のまま
  保持し、不正な文字列は拒否する。日次file内のrange外欠損をparse失敗とは扱わない
- 解析workerではATTをlocal cacheから明示的に読み込み、local ORBから太陽方向を事前生成した上で
  `erg_cotrans(..., noload=True)`を使い、内部の暗黙的なATT/ORB downloadを抑止する
- 解析workerでは全loaderへ`no_update=True`を適用し、ネットワークへ接続しない
- range失敗markerには`failure_class`、`retryable`、`attempt`、`auto_retry_count`を保存する
- ATT remote-indexのtimeout・接続失敗とkernel deathだけを解析postpass retry対象とする。
  kernel deathは1回、ATT通信失敗は2回までとし、EFD/LEPi等のデータ欠損や有効segment不足は
  再試行しない
- batch変換ではwavelet、ProcessPool、bootstrapの内部worker数を1に制限し、range内の
  多重並列によるOOMを避ける
- 補間前にtime座標をsort・deduplicateし、100秒rolling windowは正のmedian cadenceから
  最低1 sampleとして計算する
- batch plottingは非対話backendを使い、正の有限値がないlog軸はlinear軸へ退避して注記する
- prefetchの`failed`/`timeout` unitも一周後に再取得し、成功したrangeだけ解析へ戻す。
  `missing_remote`はterminalとして再試行しない
- timeout時はprocess groupを終了する

旧`downloads/range_<id>/<product>.json` markerはunit markerへ移行しないため、
変更後の初回prefetchでは計画を作り直す。ただし、既存CDFはPySPEDASのlocal cacheとして
再利用される。

並列数を過度に増やすと配布サーバーへの負荷と一時的な失敗が増えるため、通常は2を推奨する。
実際の保存量は指定した解析時間幅ではなく配布CDFのfile unitで決まる。

## 注意

batch runnerは既存の永続wavelet cacheを再利用せず、各rangeでCWT/XWTを再計算する。
NetCDFは後続セルのbacking storeとしてrange実行中だけ保持される。range終了時には
Notebook kernel/processを先に終了してfile handleを閉じ、それから親runnerが一時directoryを
削除する。batch runner自体がSIGKILLされた場合だけ清掃処理を実行できないため、再開前に
`/tmp/arase_wavelet_spectra/` の残存directoryを確認する。
