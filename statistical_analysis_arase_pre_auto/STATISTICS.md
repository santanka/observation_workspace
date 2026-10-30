# 統計解析の統一入口

通常は`run_statistics.py`だけを使用する。図の種類・横軸・期間・phase・閾値・並列数を指定できる。
内部は入力確認、環境値の追加集計、作図に分離し、従来の個別scriptは互換用・内部実装として保持する。
共通の目盛・gridと図/CSVの上書き判定は`statistics_plotting.py`へ集約した。
既存のfit、Notebook、βₑ worker、月別完了markerは変更しない。

## 202001–202212をβₑ込みで再作図

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/run_statistics.py \
  --start-month 202001 --end-month 202212 \
  --min-occupied-blocks 10 --min-bin-count 5 --workers 2 --overwrite
```

図の種類・パラメータを省略すると下記すべてを対象にする。
βₑの未完了分を再開し、必要な追加集計が成功した後で図を更新する。
成功済みβₑ JSONは従来のsignatureで再利用する。`--overwrite`は追加集計の強制再計算には作用しない。
既存の期間統合入力があれば期間・version・SHA256で確認し、月別状態の全検証を繰り返さない。
入力が未作成の期間だけ、元の月別結果を検証して期間統合入力を作成する。

計画確認だけなら同じコマンドに`--dry-run`を追加する。CDF読込み・実行・出力更新を行わず、
呼び出すコマンド一覧を表示する。重い全期間処理は利用者側で実行する。
βₑで過去に16並列のメモリ不足があったため、この入口の既定は2並列。
旧ジョブを止め、残っているworkerも終了してから開始する。

## 図の種類を限定

`--product`は反復指定可能。

| 指定 | 図 | 内容ディレクトリ |
|---|---|---|
| `spatial` | κ・X₀のL–MLAT、L–|MLAT|、L–MLTカラーマップ | `kappa_x0_statistics/` |
| `latitude` | L bin別のMLAT・|MLAT|散布点と中央値・分布幅 | `kappa_x0_latitude_profiles/` |
| `dependence` | 環境パラメータ依存性、各6パネル | 各パラメータの`*_statistics/` |
| `x0_kappa` | 横軸log₁₀X₀・縦軸κの散布図、E/B/Sの3パネル | `x0_kappa_relationship/` |
| `x0_kappa_fit` | 一次・二次回帰、月別交差検証残差、固定qの振幅変換 | `x0_kappa_fits/` |
| `spectra` | k⊥ρᵢに対する再構成振幅とイベント間の分布幅 | `reconstructed_spectra/` |
| `occurrence` | ORB滞在時間・KAWイベント時間・発生確率 | `occurrence_probability/` |
| `event_counts` | 解析range数・採用KAW range数 | `event_counts/` |

カラーマップ以外の図だけを更新する場合:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/run_statistics.py \
  --start-month 202001 --end-month 202212 \
  --product dependence --product latitude --product x0_kappa --workers 2 --overwrite
```

## 横軸・phaseを限定

`dependence`の`--parameter`は反復指定可能。省略すると全14種類:

`ae`, `ne`, `dst`, `beta_i`, `ti_te`, `beta_e`, `sym_h`, `al`,
`d_dst_dt`, `d_ae_dt`, `d_sym_h_dt`, `d_al_dt`, `vsys_x`, `abs_vsys_x`。

`vsys_x`は保存済み`range_perp_sys_speed_ms_median`をm/sからkm/sへ換算して使う。
本解析のtoroidal出力ではFACの符号付きx成分で、垂直速度の大きさではない。
現行v2の速度方針は観測partial ion momentを使い、衛星速度を差し引かない。
負値・0を保つ線形軸、初期bin幅10 km/s。`--vsys-x-bin-width-kms`で変更可能。
有効率による選別は行わず、他の環境依存性と同じrange中央値・occupied block条件を用いる。

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/run_statistics.py \
  --start-month 202001 --end-month 202212 --product dependence --parameter vsys_x \
  --min-occupied-blocks 10 --min-bin-count 5 --vsys-x-bin-width-kms 10 --plot-only --overwrite
```

保存先は`min_blocks_10/vsys_x_statistics/`。元データの再取得・追加集計は不要。
絶対値版は`--parameter abs_vsys_x`で指定し、`abs_vsys_x_statistics/`に保存する。
これは従来の横軸のrange中央値を絶対値にした`abs(median(V_sys_x))`であり、
時系列を先に絶対値にした`median(abs(V_sys_x))`とは区別する。
変換後の値でbin分け・中央値・分布幅・相関を再計算する。
bin幅は同じ`--vsys-x-bin-width-kms`で設定でき、元の符号付き図は保持する。

例：βₑとSYM-Hの依存性、allのみ:

```bash
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/run_statistics.py \
  --start-month 202001 --end-month 202212 --product dependence \
  --parameter beta_e --parameter sym_h --phase all --workers 2 --overwrite
```

`--parameter`は依存性の横軸だけに作用する。他の図も省く場合は`--product dependence`を付ける。
`--phase`は全作図に共通で、`all`, `toward_ionosphere`, `toward_magnetic_equator`, `standing`を反復指定できる。
追加集計の対象は全phaseで必要なrangeの和集合。`--phase`は追加集計のrangeを限定しない。

L別図は`--l-bin 4`で4≤L<5、`--latitude-coordinate l_abs_mlat`等で限定できる。
両方とも反復指定でき、座標省略時は符号付きMLATと|MLAT|の両方。
`--independent-y-limits`でL別図ごとの縦軸に切り替える。
L別図の集約点の最低数は既存の既定値1を維持し、`--latitude-min-bin-count 5`等で変更できる。

## 保存先とoverwrite

保存先は従来どおり:

```text
auto/statistics/202001-202212/
  inputs/                          # 元の期間統合入力
  cache/                           # ORB滞在時間cache
  environment_inputs/              # βₑ・地磁気変化率の独立追加集計
  min_blocks_10/
    statistics_run.json            # 統一入口の実行コマンド・段階別status
    kappa_x0_statistics/
    kappa_x0_latitude_profiles/
    x0_kappa_relationship/
    x0_kappa_fits/
    reconstructed_spectra/
    ae_statistics/
    ne_statistics/
    beta_e_statistics/
    ...
```

- `--overwrite`なし：PNG・PDF・CSVがすべて揃う図はスキップする。設定変更を自動判定して再作図はしない。
- `--overwrite`あり：選択した図のPNG・PDF・作図用CSVを同じパスへ上書きする。
- 一部ファイルだけある、空ファイルがある場合は、無断で上書きせず`--overwrite`を要求する。
- spatialのκ/X₀統合図はCSVを別の個別図と共有するため、PNG・PDFの組で判定する。
- 未選択の図は削除しない。元fit、期間入力、成功済み環境追加集計、ORB cacheも上書きの対象外。
- 完全スキップした図のprovenanceは更新しない。内容別provenanceは直近に新規生成した図について記録する。
- `--output-dir`は図の内容ディレクトリ群を置く親ディレクトリを変更する。追加集計の場所は変えない。
- `--statistics-dir`は既存の期間ディレクトリを指定する。

## 共通表示と条件

散布点・中央値・分布幅を描く図（環境依存性、L別緯度図、X₀–κ関係図）では共通に:

```python
ax.minorticks_on()
ax.grid(True, which="both", axis="both", alpha=0.3)
```

旧AE scriptも同じ共通設定を使用する。カラーマップの目盛・gridは変更しない。
環境依存性、L別緯度プロファイル、X₀–κの図にはSpearman順位相関ρₛとPearson相関rₚを併記する。
両者ともbin中央値ではなく、採用した個々のrangeの有限な組から計算する。
Pearsonは図の表示座標に対応させ、nₑ・βᵢ・βₑ・Tᵢ/Tₑの横軸はlog₁₀値を用いる。
図にもlog₁₀ xと明記し、AE・Dst等の線形軸には対数変換を適用しない。
縦軸は保存済みlog₁₀X₀またはκ。X₀–κ図ではlog₁₀振幅とκのPearsonを計算する。
CSVには`pearson_r`、`correlation_n`と計算座標の情報を追加し、`spearman_rho`も保持する。
緯度プロファイルでは各L bin・phase・成分・縦軸のパラメータごとに計算する。
符号付き図ではMLAT、南北をまとめた図では|MLAT|を横軸とし、18–06 MLT等の既存選別は維持する。
同じパネルの緯度bin行には同じ相関係数を記録し、bin表示最低数による相関用サンプルの除外はしない。
点数不足・一定値ではCSVをNaN、図をn/aとする。係数は記述統計で、fit共分散・時間相関は補正していない。
既存図への反映には、保存済み入力で`--plot-only --overwrite`を指定して再作図する。
有効率による選別と図への有効率表示は無効のまま。診断値は保存する。
occupied block≥10、空間統計・環境依存性のbin表示最低数n≥5を既定とする（変更可能）。
L別緯度図の集約点は既存どおりn≥1が既定で、上記の専用オプションで変更する。
イベント数・発生確率のカラーマップにはこの最低数条件を新たに課さず、既存の定義を維持する。
中央値が有限であること、対数軸の変数が正であることは要求する。
AEの統計CSVは旧列名も保持する。q16–q84は分布幅であり、信頼区間ではない。

追加集計済みで図だけ更新する場合は`--plot-only --overwrite`。
追加集計だけを進める場合は`--prepare-only`。両者は同時指定不可。
他所で用意したβₑ・微分の追加集計は`--beta-e-summary`・`--derivative-summary`で指定でき、
指定したものの準備段階は省略する。出典・対象rangeの整合性は作図scriptで確認する。

微分は既定でDst ±1時間、AE/SYM-H/AL ±5分の中心差分、単位nT/h。
差分幅や各パラメータのbin幅は`--help`にある従来のオプションで変更可能。
通常はoffline。βₑの不足CDF取得は`--allow-network`、ORBの不足日取得は`--download-missing-orbit`を明示した場合だけ許可する。
地磁気変化率はローカルOMNI CDFのみを読む。

## X₀とκ自体の関係

現在の処理が終了した後、保存済みfitからこの図だけを作る場合:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/run_statistics.py \
  --start-month 202001 --end-month 202212 \
  --product x0_kappa --plot-only \
  --min-occupied-blocks 10 --min-bin-count 5 \
  --x0-bin-width-dex 0.5 --overwrite
```

同一range・同一成分の保存済みlog₁₀X₀とκを1点とし、E/B/Sを3パネルで表示する。
all MLT、南北をまとめた方向分類を用い、phase省略時は全4種類。
両方が有限な点を成分ごとに採用し、有効率では選別しない。
散布点はbinの件数にかかわらず表示し、binの中央値・q16–q84・件数はn≥5で表示する。
`--x0-bin-width-dex`はこの図専用。成分ごとのbin境界と軸範囲は指定phase間で共通。
q16–q84はκの分布幅。表示するSpearman rhoは記述統計で、同一fitのパラメータ共分散は補正していない。

出力は`min_blocks_10/x0_kappa_relationship/`にphaseごとのPNG・PDF・集計CSV・
`.points.csv`（range_uid、元/変換後phase、成分、log₁₀X₀、κ）とprovenanceを保存する。
`--overwrite`はこの4ファイルを更新し、元fit・βₑ cacheは保持する。
βₑ・地磁気変化率の追加集計は必要ない。

## X₀–κの経験式とfit共分散の診断

X₀–κ散布図と同じ仕様で、振幅の基準位置だけをk⊥ρᵢ=3に変更する場合:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/run_statistics.py \
  --start-month 202001 --end-month 202212 --product x0_kappa --plot-only \
  --reference-kperp-rhoi 3 --min-occupied-blocks 10 --min-bin-count 5 --overwrite
```

横軸は`log10 X(3)=log10 X₀−κ log10 3`。同一range・成分の保存済み係数から計算する。
変換後の横軸を0.5 dexで再binningし、κの中央値・q16–q84・件数とSpearman相関も再計算する。
元のX₀は点CSVに保持し、`reference_kperp_rhoi`・`log10_X_reference`を追加する。
出力は`min_blocks_N/x0_kappa_relationship_kperp_rhoi_3/`に分け、元の図を上書きしない。
既定の`--reference-kperp-rhoi 1`では従来の図と保存先を維持する。
このオプションは`x0_kappa`だけに作用し、経験式fitや再構成スペクトルの条件は変えない。

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/run_statistics.py \
  --start-month 202001 --end-month 202212 \
  --product x0_kappa_fit --plot-only --min-occupied-blocks 10 \
  --bootstrap-months 300 --overwrite
```

`fit_x0_kappa_relationship.py`が同じ選別・方向分類を用いて、成分・phaseごとに
`kappa=a+b*(log10_X0-x_reference)+c*(log10_X0-x_reference)^2`を推定する。
一次式ではc=0。各rangeを等重みにし、通常最小二乗とsoft-L1 robust fitを比較する。
個々の点にfitするため、散布図のbin幅・min-bin-countはこの回帰には作用しない。
月を丸ごと1つずつ除外して予測し、全除外月を合わせたrange等重みのCV RMSE・MAEを保存する。
robustのscaleは学習データのOLS残差MADから求める。
月単位bootstrapでOLS係数の2.5–97.5 percentile区間も保存する（既定300回、seed=42）。
この区間は保存済み点群を条件とするもので、元のスペクトルfitの誤差共分散は補正していない。
時間的独立性を完全に保証する分割でもないため、CV誤差は記述的な汎化性能の評価として扱う。

上段にfit曲線（観測されたX₀範囲内のみ）、下段に一次・二次OLSの除外月予測残差を表示する。
別図では元のX₀と、`log10_X(q_p)=log10_X0-kappa*log10(q_p)`を比較する。
固定qは既定3・10。`--pivot-q 3 --pivot-q 5`等で変更可能。
元のfit下限はq=3、上限はイベントごとの温度比から変わる。
固定qが各rangeの実際の有効データ範囲内にあるかは未検証で、変換値はモデル振幅である。
相関の減少だけで推定共分散と物理的相関を分離できたとは判断しない。

保存先は`min_blocks_N/x0_kappa_fits/`。
phaseごとにfit図・残差図の組、固定q図（それぞれPNG/PDF）、係数CSV、
各点のfit/除外月予測CSV、月別CVスコアCSV、固定q相関CSV、変換点CSV、provenanceを保存する。
既存fit・βₑcacheを変更せず、入力CSVのchecksumだけを確認する。

## k⊥ρᵢに対する振幅・分布幅

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/run_statistics.py \
  --start-month 202001 --end-month 202212 --product spectra --plot-only \
  --min-occupied-blocks 10 --kperp-rhoi-min 1 --kperp-rhoi-max 100 --overwrite
```

`plot_reconstructed_spectra.py`は保存済みの同一range・成分のX₀とκから
`log10 X(k⊥ρᵢ)=log10 X₀−κ log10(k⊥ρᵢ)`を再構成する。
図の横軸は常にk⊥ρᵢ表記とし、qという略記は用いない（固定位置の診断図も同様）。
上段は個々のモデル曲線、全rangeの中央値・q16–q84、X₀の下位/上位四分位群の中央値。
縦軸はlog₁₀振幅、元fitと同じ振幅単位に基づく。四分位の所属はX₀で固定する。
下段はlog₁₀振幅の標準偏差とq84−q16の幅（dex）。信頼区間ではない。
個別fitの有効範囲は適用していないので、全rangeで件数は一定。
下限3未満を灰色で示す。3以上でも、個々のfit上限を超える外挿を含み得る。
これは実測スペクトルを重ねた図ではなく、保存済みfitの幾何学的関係の確認図として使う。

既定範囲は1–100、対数等間隔201点。
`--kperp-rhoi-min`、`--kperp-rhoi-max`、`--kperp-rhoi-points`で変更できる。
出力は`min_blocks_N/reconstructed_spectra/`にphaseごとのPNG/PDF・集計CSV・点CSV・provenance。
元fitやβₑ集計を再実行しない。表示には共通のminor ticks・major/minor grid alpha=0.3を使う。

## 内部・互換用script

個別のplot/prepare scriptは、計算・描画の実装と既存コマンド互換用に保持している。
個別scriptの従来の上書き動作は維持し、統一入口が`--skip-existing`または`--overwrite`を明示して出力を管理する。
通常の実行手順はこの文書に集約する。
