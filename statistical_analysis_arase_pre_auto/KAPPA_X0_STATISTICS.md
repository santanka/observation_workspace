# κ・X0空間統計の定義

## 解析単位

`arase_phase_master.csv`の1 rangeを1 sampleとして等重みで扱う。長いrangeの影響を
時間長で再加重しない。各phaseで`status == "ok"`のrangeだけを使用する。

対象量は以下の6変数とする。

- `kappa_E`, `kappa_B`, `kappa_S`
- `log10_X0_E`, `log10_X0_B`, `log10_X0_S`

X0はmaster CSVに保存済みの`log10_X0_*`を直接使う。これは数値的に
`log10(X0_*)`と一致する。E、B、SのX0は元の物理単位が異なるため、成分間の絶対値を
直接比較せず、各成分内の空間分布とphase差を比較する。

## 空間bin

rangeを以下の代表位置で1つのbinへ入れる。

- L: `range_L_90deg_mean`
- MLAT: `range_MLAT_deg_mean`
- MLT: `range_MLT_circular_mean_hour`

binは下端を含み上端を含まない`[lower, upper)`とする。

- L: `[3, 11)`, 1刻み
- MLAT: `[-50, 50) deg`, 5 deg刻み
- MLT: `[0, 24) hour`, 1 hour刻み

L--MLTは全MLTを使用する。L--MLATはoccurrence probability図と同じく、
`18 <= MLT < 24`または`0 <= MLT < 6`のnightsideだけを使用する。

## bin統計

各変数・binについて次を保存する。

- `n`
- `mean`, `std`
- `median`
- `q16`, `q84`
- `min`, `max`

図の既定代表値はrange等重みの`median`とする。既定では`min_bin_count = 1`なので、
n > 0の全binを表示する。長期データでsample数条件を厳しくする場合は
`plot_kappa_x0_statistics.py`冒頭の`PLOT_CONFIG["min_bin_count"]`を3や5へ変更する。
条件未満のbinは図だけmaskし、CSVには全binの統計を残す。

統合図の各panelには、表示対象binに属するrangeを直接集計した全体の
`median [q16, q84], N`を示す。このq16--q84はrange間の分布幅であり、中央値の
推定誤差や信頼区間ではない。bin中央値を再集計しないため、binを不自然に等重み付け
しない。`min_bin_count`を変更した場合、全体統計からも非表示binのrangeを除外する。

## color scale

κとX0のどちらもE、B、Sごとに独立したlinear color limitsを設定する。自動範囲は、
表示条件を満たすbin中央値を全phase・両座標から集めたq5--q95とする。範囲外のbinは
color scaleの上下端で飽和表示し、colorbarの三角形で示す。q16--q84は全体の32%を
飽和させるため、空間分布の既定色域としては狭すぎる。

limitsは次の設定で固定値に変更できる。

- `kappa_color_limits`
- `log10_x0_color_limits`
- `color_percentiles`

`color_percentiles = (0, 100)`とすれば、各パラメータの全範囲を使える。

## 出力

all、north traveling、south traveling、standingについて、L--MLTとL--MLATを個別に
出力する。κとlog10(X0)それぞれのE、B、Sの3 panel図に加え、上段をlog10(X0)、
下段をκとする2行3列の統合図を保存する。統合図にはrow-major順で(a)--(f)を付ける。
PNG、PDF、およびbin統計CSVを
`KAW_observation/auto/<dataset key>/kappa_x0_statistics/`へ保存する。
