# AE指数に対するκ・X0統計の定義

## 解析単位と対象

`arase_phase_master.csv`の`status == "ok"`の1 rangeを1 sampleとして等重みで扱う。
対象量は`kappa_E/B/S`と`log10_X0_E/B/S`の6変数とする。phaseはall、north
traveling、south traveling、standingを混合せず、別々に集計する。

AEの代表値には`range_geomag_AE_nT_median`を使う。これはKAW検出時刻だけに依存する
`fit_geomag_AE_nT_median`ではなく、range全体の地磁気活動度を表す。既定では
`range_geomag_AE_nT_valid_fraction >= 0.8`かつ有限・非負のrangeを採用する。

## 図とbin統計

各phaseについて2行3列の図を作る。上段(a)--(c)は`log10_X0_E/B/S`、下段(d)--(f)は
`kappa_E/B/S`とする。灰色点は各range、色付き記号はAE bin内のmedian、error barは
range間のq16--q84である。q16--q84は中央値の信頼区間ではない。

AE binは既定で0 nTから200 nT刻みとし、上限は入力データを含むよう自動設定する。
各binについて次をCSVへ保存する。

- `n`
- `mean`, `std`
- `median`, `q16`, `q84`
- `min`, `max`
- 全rangeの`median`, `q16`, `q84`, `N`
- range単位のSpearman順位相関係数と、その通常のp値

通常のp値はCSVへ診断値として保存するが、隣接rangeや同一イベント内rangeの独立性を
仮定できない場合があるため、図上では有意性判定に使わない。図にはSpearman係数のみを
示す。

既定では`min_bin_count = 1`としてn > 0の全binを描く。設定は
`plot_ae_parameter_statistics.py`冒頭の`PLOT_CONFIG`で変更できる。

- `ae_bin_width_nT`: AE bin幅
- `ae_max_nT`: `None`なら上限を自動設定
- `min_bin_count`: 描画に必要なrange数
- `ae_min_valid_fraction`: AE有効率の下限
- `mlt_sector_hour`: `None`なら全MLT、`(18.0, 6.0)`なら夜側のみ
- `show_ae_range_spread`: range内AE q16--q84を横error barで表示

AE依存性は衛星位置依存性の代替ではない。AEとL、MLAT、MLTのサンプリングが相関する
場合、ここで得る関係は位置依存性を含む周辺分布になる。必要に応じて夜側限定や位置を
共変量とした解析を追加する。

## 入力上の注意

現在のmaster CSVのAEは、OMNI 1分値を解析時刻へnearestで整列した値から集約される。
将来、OMNIに長い欠測がある期間へ拡張する場合は、native 1分値からrange統計を再計算
するか、nearest整列に有限の許容幅を設定する必要がある。

## 出力

PNG、PDF、bin統計CSVを
`KAW_observation/auto/<dataset key>/ae_statistics/`へ保存する。
