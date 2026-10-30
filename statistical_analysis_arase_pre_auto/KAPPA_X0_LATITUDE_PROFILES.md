# L bin別の緯度依存性

`plot_kappa_x0_latitude_profiles.py`は保存済みの期間統合masterから、L binごとに
κ・log10 X0の緯度依存性と分布幅を描く。再解析・bootstrap再計算・データ取得は行わない。

1図は上段log10 X0 E/B/S、下段κ E/B/Sの2行3列。横軸はsigned MLATまたは|MLAT|。
灰色点を各range、色付き点・線を緯度binの中央値、error barをrange間のq16–q84とする。
q16–q84は中央値の信頼区間ではない。CSVにはmean、std、median、q16、q84、min、max、nを残す。
個々のrangeのbootstrap誤差はこの図のerror barには使わない。

各パネルにはSpearman順位相関ρₛとPearson相関rₚも表示する。
同じL bin・phase・成分に属する個々の有限なrangeを使い、横軸は表示どおりMLATまたは|MLAT|、
縦軸は保存済みlog₁₀X₀またはκとして計算する。bin中央値への相関ではない。
`--min-bin-count`は集約点の表示だけに作用し、相関用rangeには適用しない。
CSVには`spearman_rho`、`pearson_r`、`correlation_n`と計算座標を追加する。
同じパネルに属する緯度binの行には同じ係数を保存する。
2点未満や一定値では係数をNaN、図をn/aとする。時間相関・fit共分散の補正は行わない。

## 実行

2020–2022年、最低10 occupied block、全phase・両緯度座標・全L bin:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/plot_kappa_x0_latitude_profiles.py \
  --start-month 202001 --end-month 202212 --min-occupied-blocks 10
```

まずall、4≤L<5、|MLAT|の1図だけ作る場合:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/plot_kappa_x0_latitude_profiles.py \
  --start-month 202001 --end-month 202212 --min-occupied-blocks 10 \
  --phase all --l-bin 4 --coordinate l_abs_mlat
```

`--l-bin`は既存L binの下端を指定し、反復できる。既定は3,4,...,10。
各図は3≤L<4、4≤L<5、...、10≤L<11の範囲であり、Lが一点に固定された標本ではない。
`--phase`と`--coordinate`も反復できる。空のL binは図を作らず、provenanceへ記録する。
`--year 2022`にも対応する。

保存先:

```text
auto/statistics/202001-202212/min_blocks_10/
  kappa_x0_latitude_profiles/
    provenance.json
    l_mlat/
      L_4_5/                    # phase別のPNG、PDF、bin統計CSV
    l_abs_mlat/
      L_4_5/
```

全指定では最大4 phase × 2座標 × 8 L bin = 64図になる。`--statistics-dir`は期間入力の
ディレクトリ、`--output-dir`はlatitude profilesの出力ディレクトリそのものを上書きする。
同じphase・座標・L binを再実行すると対象図とCSVを上書きする。provenanceは直近の実行内容を保存する。

## 条件と表示

- range代表MLAT、90° pitch angleのL、代表MLTを使用
- 18–06 MLTの夜側に限定し、元のカラーマップと同じ5 degの緯度binを使用
- `status == ok`かつ`n_occupied_blocks >= N`。既定N=10、block duration=10 s
- 符号付きMLATでtoward ionosphere / toward magnetic equatorへ分類してから|MLAT|へ統合
- 1 rangeを1 sampleとして等重み。南北の中央値を平均せず、元のrangeをまとめて再集計
- |MLAT|は`abs(range_MLAT_deg_mean)`。range内の|MLAT|を平均した値とは異なる
- MLATは[-50,50)、|MLAT|は[0,50)。Lも下端を含み上端を含まない

既定では、各パラメータの縦軸を指定したL bin・phase・両座標で共通にし、全ての有限なraw値を
含む範囲を使う。細かい変化を各Lで拡大する場合は`--independent-y-limits`を追加する。
その場合は図ごとに縦軸が異なるので、L間の振幅比較では軸目盛を確認する。

`--min-bin-count 3`等を付ければ、中央値・分布幅の記号を表示する最低range数を変更できる。
閾値未満のbinでもraw点とCSVは残す。欠測binをまたぐ中央値の線は接続しない。
各panelのNはそのL binの有限なraw値の個数、緯度binごとのnは各記号の上に表示する。
L bin内でもLの分布が緯度で変わり得るため、曲線はそのL幅で平均化された条件付き分布となる。

X0は従来どおり保存済みlog10 X0を描く。E/B/Sでは元の物理単位が異なり、成分間の値を直接比較しない。
同じbinでの中央値等は既存カラーマップのCSVと一致する。
