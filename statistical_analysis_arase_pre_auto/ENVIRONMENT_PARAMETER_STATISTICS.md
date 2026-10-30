# nₑ・Dst・βᵢ・Tᵢ/Tₑに対するκ・X₀依存性

保存済みの期間統合CSVから、各変数・各phaseに独立した2行3列の図を作る。
上段log10 X₀、下段κ、列はE/B/S。データ取得・fit再解析・月別状態の再検証は行わない。

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/plot_environment_parameter_statistics.py \
  --start-month 202001 --end-month 202212 \
  --min-occupied-blocks 10 --min-bin-count 5
```

既定は4変数×4 phaseの16図。PNG・PDF・bin統計CSVと変数別provenanceを出力する。
`--year 2022`も使用可能。既存の`inputs/`を期間・解析version・SHA256で確認して読む。
`--statistics-dir`で入力の期間ディレクトリを変更できる。

```text
auto/statistics/202001-202212/min_blocks_10/
  ne_statistics/
  dst_statistics/
  beta_i_statistics/
  ti_te_statistics/
```

各ディレクトリにはphase別のPNG/PDF/CSVと`provenance.json`が入る。
同じ変数・phaseの再実行は上書きする。provenanceはその変数の直近実行内容。
別条件を並存させる場合は`--output-dir /path/to/parent`を指定し、その下に変数ごとのディレクトリを作る。

## 入力と条件

| 横軸 | range代表値の列 | 軸・既定bin幅 |
|---|---|---|
| nₑ [cm⁻³] | `range_number_density_HFA_cc_median` | 対数、0.5 dex |
| Dst [nT] | `range_geomag_Dst_nT_median` | 線形、20 nT |
| βᵢ [無次元] | `range_ion_plasma_beta_HFA_median` | 対数、0.5 dex |
| Tᵢ/Tₑ [無次元] | `range_i_e_temp_ratio_median` | 対数、0.5 dex |

- AEと同じrange全体の中央値を使用する。fit対象時刻のみの中央値、native HFA標本の中央値には切り替えない。
- `status=ok`、occupied 10 s block数≥10（既定）。range内有効率による選別は行わない。
- nₑ・βᵢ・Tᵢ/Tₑは有限な正のrange中央値、Dstは負値・0・正値を含む有限なrange中央値を要求する。
- AEと同じall MLT。方向は符号付きMLATからtoward ionosphere/equatorへ分類する。
- βᵢは保存済み解析値を用い、range中央値のnₑ・Tᵢ・Bから再計算しない。
- Tᵢ/Tₑも保存済みの時系列比のrange中央値を用い、温度の中央値同士の比には置き換えない。
- βₑは統合CSVに未保存。下記の独立した追加集計を行い、βᵢ中央値をTᵢ/Tₑ中央値で割って代用しない。
- 各変数の中央値の有限性・正値条件により、変数間で採用range集合が異なり得る。
- 100秒移動平均の端点NaNで短いrangeが除かれることを避けるため、有効率≥80%の条件は廃止した。
  有効率は診断値として保存し続けるが、既定の採否判定には使わない。
  図のタイトル・注記にも有効率を表示しない。選別設定を明示した場合の条件はprovenanceに残す。

## 表示と調整

灰色は各range、色付き点はbin中央値、棒はq16–q84のrange間分布幅で、信頼区間ではない。
各rangeは等重み。n≥5のbinに中央値・分布幅を表示し、全binの統計はCSVに残す。
binのnはパラメータ別の有限なrange数。標本数不足・欠測binをまたいで線を結ばない。
対数binの位置は幾何中心、Dstは算術中心。binは下端を含み上端を含まない。
bin境界は幅の整数倍（対数ではlog10空間）で、選択した全phaseを含む範囲へ自動拡張する。
同一変数内ではphase間でbin・縦軸を共通にする。異なる変数の図では採用range集合が異なり縦軸も異なり得る。

- `--parameter ne` / `dst` / `beta_i` / `ti_te` / `beta_e`：変数を限定、反復可能
- `--phase all`等：方向を限定、反復可能
- `--min-bin-count 10`：bin中央値の表示最低数
- `--min-valid-fraction 0.8`：明示した場合のみ有効率選別を再有効化する互換オプション。既定は無効
- `--log-bin-width-dex 0.25`：nₑ・βᵢ・Tᵢ/Tₑのbin幅
- `--dst-bin-width-nt 10`：Dstのbin幅
- `--min-occupied-blocks 6`：採用rangeのblock条件（6未満不可）

panelのNとSpearman ρは、bin表示閾値にかかわらず採用された全有限ペアから計算する。
ρは記述的指標で、隣接rangeの依存やL/MLATによる交絡を補正していない。p値や因果関係は示さない。
これはKAW採用イベント内のfit値の依存性であり、発生確率ではない。

## βₑの独立追加集計

`prepare_electron_beta_statistics.py`で保存済みNotebookのプラズマパラメータ作成部分までを
メモリ上で変換して実行する。元Notebook、batch runner、完了marker、既存master・fitは変更しない。
CWT・E/B fit・bootstrapは実行しないが、CDFの読込み、座標処理、LEP momentsの再構成等は必要で重い。
実データの全期間処理は利用者側で実行する。

各時刻で`βₑ = 2 μ₀ nₑ e Tₑ / B²`（nₑ:m⁻³、Tₑ:eV、B:T）を計算し、
既存βᵢと同じ100秒のcentered rolling meanを適用する。
原解析のkappa時系列CSVの時刻に補間し、range全体の中央値・q16・q84・有効率を保存する。
平滑化後のβᵢを平滑化後のTᵢ/Tₑで割る方法にも置き換えない。
再構成したβᵢ・Tᵢ/Tₑ・nₑ・Tₑのn_total、n_valid、中央値、q16、q84が保存値と
一致することを要求する（rtol=1e-6、atol=1e-10）。不一致はfailedとなりログに残す。
これは再構成の整合性確認であり、元解析の再実行・全期間の科学的再検証ではない。

最初の1 rangeの計画だけ確認（CDF処理なし、元CSVは読む）:

```bash
.venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/prepare_electron_beta_statistics.py \
  --start-month 202001 --end-month 202212 --max-ranges 1 --plan-only
```

最初の1 rangeを実際に処理する場合は`--plan-only`を外す。動作確認後、全対象を処理:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/prepare_electron_beta_statistics.py \
  --start-month 202001 --end-month 202212 --min-occupied-blocks 10
```

既定はofflineで、暗黙のlisting/updateを含むsocketアクセスを禁止する。欠けたCDFは自動取得せずfailedとする。
βₑに不要な衛星速度の計算・座標変換・診断図は実行対象から外している。
以前の実装では`transform_v_sc_to_dsi`内の`dsi2j2000`がORB再読込みを要求し、
local cacheがあってもoffline guardで失敗した。この経路は追加集計コード内で除去し、元Notebookは変更しない。
offline接続要求や処理計画後のソース変更を検出した場合は、最初の失敗を`failures.json`に保存して停止する。
ソース修正時は、実行中の親コマンドを止めてから再開する。異なるコード版を混在させた成功結果は作らない。
ローカルcacheが不足し、loaderによる取得を許可する場合だけ`--allow-network`を追加する。
rangeは並列化せず1件ずつ独立したprocessで実行する。既定timeoutは4時間/rangeで`--timeout`で変更可能。
`--range-uid`で対象を限定できる（複数指定可）。

出力は期間直下の`environment_inputs/beta_e/`:

- `plan.json`: 対象range、時間範囲、元kappa CSVとSHA256、再構成設定
- `ranges/`: range別の結果JSON、request、log。入力・コードのsignatureが一致するcomplete結果は再利用
- `failures.json`: 失敗range。失敗があれば集約CSVの更新前に停止し、再実行で失敗分だけを処理
- `range_summary.csv`と`range_summary.provenance.json`: 成功した指定rangeの集計と出典

集計後にβₑのみ作図:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/plot_environment_parameter_statistics.py \
  --start-month 202001 --end-month 202212 --parameter beta_e \
  --min-occupied-blocks 10 --min-bin-count 5
```

`--beta-e-summary /path/to/range_summary.csv`で別の追加集計を指定可能。
保存先は`min_blocks_10/beta_e_statistics/`。既定の4変数作図にはβₑを含めず、明示指定で作図する。
元masterのSHA256と追加集計CSVのSHA256を確認し、選択したphaseの対象rangeに不足があれば停止する。
1 rangeだけの確認結果では全期間の図を作れない。閾値を6へ下げる場合は追加集計も6で実行する。

注意: 過去のCDF全ファイルのchecksumは既存masterにないため、現在のlocal CDFと原解析入力の
byte単位の一致は保証できない。上記の保存済み環境統計との照合、Notebook・コード・時刻CSVのhashを記録する。
実データでのβₑ再構成は未実行であり、最初の1 rangeで入力読込みから出力まで確認してから全件へ進める。

## dDst/dt・dAE/dtと一括実行

中心差分（Dst ±1時間、AE・SYM-H・AL ±5分）の追加集計と、βₑを含む一括runnerを実装した。
詳細と実行コマンドは[地磁気変化率と一括実行](GEOMAG_DERIVATIVE_STATISTICS.md)を参照。
単独のplot scriptでは`--parameter d_dst_dt`、`--parameter d_ae_dt`、`--parameter d_sym_h_dt`、
`--parameter d_al_dt`を指定する。SYM-H・AL自体の保存値を使う図は`--parameter sym_h`・`--parameter al`。
単独plotの既定4変数は維持し、一括runnerの既定には今回追加したSYM-H・ALとその微分も含める。
