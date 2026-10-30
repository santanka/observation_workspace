# 年統合のrange数と発生確率

複数年も統合できる。先にκ・X0/AE用の統合入力を同じ期間で作成してから実行する。

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/plot_yearly_occurrence.py \
  --start-month 202001 --end-month 202212 --min-occupied-blocks 10 --download-missing-orbit
```

保存先は`auto/statistics/202001-202212/min_blocks_10/occurrence_probability/`。
`--min-occupied-blocks`の既定値は10。6以上を比較用に出す場合は6を指定する。
分子時間とORB滞在時間を全期間でそれぞれ合計してから確率を計算し、年別確率を平均しない。
明示的なno_data月は全期間から除き、その一覧を図とprovenanceへ記録する。
`--year`との同時指定は不可。開始・終了月はともに含む。

`plot_yearly_occurrence.py`は年統合済みの`inputs/arase_phase_master.csv`と
`inputs/provenance.json`を読み、κ・X0と同じ位置・方向分類で作図する。
通常実行で作るのは、参照先の旧図と同じ「Arase滞在時間・KAW観測時間・発生確率」の
3 panel図。イベント数の2 panel図は`--counts-only`を明示した場合だけ作る。
Notebookの再解析は行わない。`plot_yearly_statistics.py --year 2022 --prepare-only`で
統合入力を準備できる。2022年分はすでに準備済み。

## 実行

workspaceのルートで実行する。イベント数だけならORBの読込み・追加取得は不要。

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/plot_yearly_occurrence.py --year 2022 --counts-only
```

発生確率も含め、既存のローカルORBだけを使う場合:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/plot_yearly_occurrence.py --year 2022
```

分母の対象日にORBが不足すれば停止し、部分的な分母から
発生確率を作らない。不足日だけを公式ERGサイトから取得して続ける場合:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/plot_yearly_occurrence.py --year 2022 --download-missing-orbit
```

取得は逐次実行。年ごとのdirectory listingを1回取得し、不足日のCDFだけをGETする。
既存CDFの更新確認や再取得はしない。GETのconnect/read timeoutは15/90秒。
一時ファイルへ保存し、CDFの必須変数と対象日のrecordを検証してから確定する。
中断・通信失敗後は同じコマンドで残りだけを取得できる。

既定の保存先:

```text
auto/statistics/202201-202212/
  inputs/                          # 全rangeの統合入力（読取りのみ）
  cache/
    orbit_dwelling_bins.npz         # 閾値によらないORB分母cache
  min_blocks_10/
    event_counts/                  # --counts-onlyで生成
    occurrence_probability/        # 3 panel図、bin統計、provenance.json
```

各内容について4 phase × 3座標（L–MLAT、L–|MLAT|、L–MLT）の12図をPNG/PDFで保存し、
12個のbin統計CSVも保存する。同じ出力先への再実行では対象図・CSVを上書きする。
`--statistics-dir`で期間ディレクトリ、`--spedas-dir`でローカル観測データのrootを変更できる。

## 数え方と分母

イベント数の単位は**解析range**。長いparent eventの分割rangeを含むため、独立した
物理イベント数とは区別する。2 panelは次の件数を示す。

- `n_batch_ranges`: completeとして集約済みのrange数（okとinsufficient_dataを含む）
- `n_kaw_ranges`: phaseの`status == ok`かつ`n_occupied_blocks >= N`を満たす採用KAWイベントのrange数

イベント採用条件は旧図の`status == ok`に、指定した最低occupied block数Nを追加する。これはKAW選別後に
κ fitのサンプル数・分布条件を満たすrangeである。`insufficient_data`は、個別の
検出サンプルがあってもイベント採用条件を満たさないため分子へ加えない。
全図のcount color scaleを共通にする。

発生確率は参照先の旧図を踏襲し、KAWイベントとして採用したrangeの全区間長を使う。

```text
KAW observation time = sum(full event duration where status == ok and n_occupied_blocks >= N)
denominator = ORB dwelling time during the months with analysis results
occurrence probability = KAW observation time / denominator
```

block数は保存済みの10秒blockで数え、phaseごとに判定する。閾値変更では分子に含むイベントを
絞り、採用イベントの区間長は全時間のまま使う。分母のORB滞在時間と対象月は変更しない。
したがって6と10で同じ分母cacheを再利用できる。図の期間表示とprovenanceに閾値を記録する。

2022年はユーザー指定により、解析結果のある1–5月・8–12月（304日）を分母の対象とする。
必須データ欠測の6–7月は分母にも加えない。個々の日のEFD/MGF/HFA/LEP-i等の利用可能性を
分母へmaskする処理は行わない。有効観測時間に限定した分母とは異なるため、その条件を
provenanceへ記録する。分子はKAWイベントの**全時間**であり、イベント内で個別に
KAWを判定した時刻の時間合計（`detected_duration_sec`）は使わない。
`excluded`のrangeをゼロ検出として分子へ加えることはなく、その日自体のORB時間は
対象月に含まれる限り分母へ残る。

ORBはCDFを直接読むので、`--download-missing-orbit`なしではネットワークに接続しない。
native cadenceを使い、時間差が30秒を超える区間は滞在時間へ加えない。時刻はUTC。
`pos_rmlatmlt`のMLAT・MLTと、`pos_Lm`の第0成分（90 deg pitch angle）を使用する。
CDF fill値をNaNへ変換する。実際の滞在時間と暦上の対象時間はprovenanceに記録する。

L–MLATとL–|MLAT|は18–06 MLTの夜側、L–MLTは全MLT。bin境界と代表位置はκ・X0と同じ。
方向は符号付きMLATで先にtoward ionosphere / toward magnetic equatorへ分類し、standingと
allを維持する。L–|MLAT|は南北の**分子時間と分母時間をそれぞれ合計してから割る**。
南北の発生確率を平均しない。分母が0のbinはNaNにする。

rangeの代表位置への時間集約で分子が分母を超えた場合は、binningの不整合として停止する。
100%へのclippingは行わない。図にはArase滞在時間、KAW観測時間、発生確率の3 panelを表示する。
time/percentそれぞれ全phase・全座標で共通のlog color scaleを使う。

取得した全ORBのパス・SHA256、対象月、入力masterのSHA256、コードと設定をprovenanceへ
保存する。分母cacheはORBのSHA256、対象月、設定、読込み・bin集計コードが一致した場合
だけ再利用する。元の月別結果とκ・X0の図は書き換えない。

## 検証

```bash
cd statistical_analysis_arase_pre_auto
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  ../.venv_pyspedas/bin/python -m unittest \
  test_yearly_occurrence test_plot_occurrence_probability test_statistics_phases \
  test_plot_yearly_statistics test_plot_kappa_x0_statistics test_plot_ae_parameter_statistics
```
