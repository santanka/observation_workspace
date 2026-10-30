# 年単位のκ・X0空間統計とAE依存性

任意の月範囲にも対応する。2020–2022年を一つの母集団として統合する場合:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/plot_yearly_statistics.py \
  --start-month 202001 --end-month 202212 --min-occupied-blocks 10
```

開始月・終了月をともに含む。図の保存先は`auto/statistics/202001-202212/min_blocks_10/`。
年別統計の平均ではなく、全期間の元rangeを統合してから中央値・分位点を計算する。
統合入力を新規作成・更新するときは、全月のstatusがcompleteまたはexcludedであることを要求し、failed等が残れば停止する。
`--year 2022`は`--start-month 202201 --end-month 202212`と同じ意味で維持する。
統合入力のファイル名`annual_ranges.json`は複数年でも互換性のため維持する。

`plot_yearly_statistics.py` は月別の完了済みmasterを検証・統合し、既存の2つのplot
scriptで年単位の図を作る。Notebookの再解析、CDF読込み、downloadは行わない。

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/plot_yearly_statistics.py --year 2022
```

既定の出力先は
`/mnt/j/statistical_analysis_arase/preanalysis/KAW_observation/lepe_ge67eV_vion_v2/auto/statistics/202201-202212/min_blocks_10/`。
`statistics/<期間>/<最低block数>/<内容>/` の順に分ける。既定の最低block数は10。

```text
auto/statistics/
  202201-202212/
    inputs/                       # 閾値によらない全rangeの統合入力
    cache/                        # 閾値によらないORB滞在時間cache
    min_blocks_6/
      kappa_x0_statistics/
      ae_statistics/
      occurrence_probability/
    min_blocks_10/
      provenance.json             # 選別条件・phase別の選別前後件数・描画設定
      kappa_x0_statistics/         # L–MLAT、L–|MLAT|、L–MLT
      ae_statistics/
      occurrence_probability/
```

別の場所には `--output-dir /path/to/output` を使う。既存の月別結果には書き込まない。
`--output-dir`は期間ディレクトリを指定し、その下に`min_blocks_N`と内容別ディレクトリを作る。
同じ閾値での再実行ではその閾値の図・集計CSVを上書きする。
6以上の結果を作るには`--min-occupied-blocks 6`、10以上なら`--min-occupied-blocks 10`。
6未満は保存済みfitの元条件を緩めるため、この後処理では指定できない。

既存の`inputs/`が揃っていれば、期間・解析version・masterのSHA256を確認して再利用する。
閾値変更だけでは月別結果を読み直さず、統合入力も書き換えない。月別結果そのものを更新した後に
統合し直す場合だけ`--refresh-inputs`を付ける。どちらもNotebookの再解析は行わない。

- `inputs/annual_ranges.json`: 月別range IDを年内一意に再採番したmanifest。元dataset・IDも保持
- `inputs/arase_phase_master.csv`: 完了rangeの4 phase。`source_dataset`, `source_range_id`,
  `range_uid`, `source_phase_mode`で元の結果を追跡できる。
  `phase_mode`は下記の半球に対する方向分類を保存する
- `inputs/provenance.json`: 対象月の入力パス・SHA256、欠測理由、complete/excluded件数、phase別ok件数
- `min_blocks_N/provenance.json`: 閾値と10秒blockの単位、phase別の選別前後件数、描画設定・コードのSHA256
- `min_blocks_N/kappa_x0_statistics/`: 4 phase × 3座標 × 3種類（κ、log10 X0、両者の統合）の
  PNG/PDF計72ファイルと、bin統計CSV計24ファイル
- `min_blocks_N/ae_statistics/`: 4 phaseのPNG/PDF計8ファイルとbin統計CSV計4ファイル

準備・検証だけなら `--prepare-only`。phaseを限定する場合は `--phase all` などを使い、
複数phaseにはオプションを反復する。入力元の変更には `--ranges-dir` と `--state-root`。

## 採用条件

作図時にphaseごとの`status == ok`かつ`n_occupied_blocks >= N`を共通に適用する。
`n_occupied_blocks`は有効sampleが1点以上ある10秒blockの数であり、N=10でも
検出サンプル時間100秒以上を意味しない。保存済みのκ・X0・bootstrap結果は変更しない。
これは空間binの`min_bin_count`とは別の条件である。

κ・X0カラーマップは既定で各パラメータの有限なrange数が`n >= 5`のbinだけ着色する。
`plot_yearly_statistics.py`と`plot_kappa_x0_statistics.py`の`--min-bin-count 10`等で変更できる。
L–MLAT、L–|MLAT|、L–MLTの単独図・統合図に共通の表示条件であり、統計的有意性の判定ではない。
図のタイトルと年次作図provenanceに閾値を記録する。集計CSVには閾値未満のbinも保持する。
AE依存性、発生確率、L別緯度profile図の表示条件はこのオプションでは変更しない。
同じ出力先で再実行するとカラーマップを上書きするため、異なる閾値の図を並存させる場合は
別の`--output-dir`を指定する。

統合入力の作成時には対象月のmanifestを必須とする。明示的な `analysis_status=no_data` と空のrangesを
持つ月は欠測理由を記録して統合対象から除く。単にファイルがない月は停止する。
通常月は全rangeのstatusがcompleteまたはexcludedでなければ停止する。masterには
complete rangeごとに4 phaseを要求し、解析version、重複、manifestとの時刻一致も確認する。
入力不足で解析前にexcludedとなったmarkerではversion欠落を許容し、その件数を記録する。
completeのversion欠落や、明示的な別versionは許容しない。
元の4 phaseのstatusがすべてokである必要はなく、insufficient_dataを保持してplot側で除く。

## 半球に対する伝播方向

range単位の位置統計と揃えるため、`range_MLAT_deg_mean`（符号付き磁気緯度、deg）を使う。
fit時刻だけのMLATではなく、range全体の代表MLATの符号で分類する。

| 元phase | MLAT > 0 | MLAT < 0 |
|---|---|---|
| north_traveling | toward_ionosphere | toward_magnetic_equator |
| south_traveling | toward_magnetic_equator | toward_ionosphere |
| standing | standing | standing |
| all | all | all |

MLAT = 0、欠損、非有限値ではtraveling行を`unclassified`として統合CSVに保持し、
方向別の図から除く。理由を`direction_classification_reason`とprovenanceへ保存する。
allとstandingはこれによって除かない。`source_phase_mode`には元のnorth/southを保持する。
κやX0の値、statusを入れ替えたり再fitしたりせず、各行の統計上の分類だけを変える。
赤道をまたぐrangeも代表MLATで1分類とするため、時刻ごとの方向分類ではない。

月別のκ・X0およびAE plot scriptにも同じ分類を適用する。`--phase`には`all`、
`toward_ionosphere`、`toward_magnetic_equator`、`standing`を指定できる。
元のNotebook・月別masterのnorth/south分類は変更しない。range数・出現率も同じ方向分類へ
対応し、別コマンドで出力する。実行方法は[`YEARLY_OCCURRENCE.md`](YEARLY_OCCURRENCE.md)を参照。

初回のnorth/south分類で生成した2022年の図と入力は、再作図の検証後に
`202201-202212/legacy/north_south/`へ保存する。

2022年の既存manifestでは6–7月が必須データ欠測。従って図の期間表示は2022年全体だが、
実際の統計対象は残る10か月の完了rangeである。欠測月をゼロ検出として補わない。

月別の中央値を平均せず、全rangeを結合してからbin統計を再計算する。解析条件は
[`KAPPA_X0_STATISTICS.md`](KAPPA_X0_STATISTICS.md) と
[`AE_PARAMETER_STATISTICS.md`](AE_PARAMETER_STATISTICS.md) を維持する。

- 1 rangeを1 sampleとして等重み。all/toward ionosphere/toward magnetic equator/standingは別集計
- κ E/B/Sと保存済みlog10 X0 E/B/Sを表示
- L–MLTは全MLT、L–MLATは18–06 MLTの夜側
- L–|MLAT|も18–06 MLTの夜側。符号付きMLATで方向分類後、南北のrangeをまとめて再集計
- 空間bin代表値は中央値、描画下限は1 range。色域は年統合の全phase・3座標で共通
- AEは有限・非負のrange中央値、200 nT bin。有効率による選別は無効。縦軸範囲は全phaseで共通
- q16–q84はrange間の分布幅であり、中央値の信頼区間ではない

AEと位置の共変動、隣接rangeの相関、欠測による観測条件の偏りは、この図だけでは
補正しない。保存済みAE集約値を使うため、native OMNIの再検証も行わない。
range数・出現率図はこのcommandの対象外で、`plot_yearly_occurrence.py`を使う。

## 検証

```bash
cd statistical_analysis_arase_pre_auto
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  ../.venv_pyspedas/bin/python -m unittest \
  test_statistics_phases test_plot_yearly_statistics \
  test_plot_kappa_x0_statistics test_plot_ae_parameter_statistics
```
