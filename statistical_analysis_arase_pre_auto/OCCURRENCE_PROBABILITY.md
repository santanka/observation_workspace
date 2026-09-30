# KAW occurrence probability の定義

## 目的と解析単位

この解析は、`batch_arase.py`が生成したrangeをKAW検出の観測単位とし、対象期間全体の
Arase衛星滞在時間で規格化したoccurrence probabilityをL--MLAT面およびL--MLT面に
集約する。KAW rangeは代表位置を1つのbinへ入れる。これにより、後続のκ・背景量・
プラズマ量のrange代表値による統計解析と観測単位を一致させる。

## 対象期間とORBデータ

対象期間はranges JSONの`time_range_input`を使う。ORB L2 `def`の日次fileを全期間に
ついて確認し、local cacheにない日だけdownloadする。downloadは既定で最大3回試行し、
全日が揃わない場合は部分的な分母で図を作らず停止する。

dwelling timeにはORB native cadenceの位置を使う。隣接時刻差を各sampleの時間重みとし、
30秒を超えるgapは滞在時間へ加算しない。分母は純粋な衛星滞在時間なので、EFD、MGF、
HFAがKAW検出に利用できなかった時間も含み得る。このため、得られる確率は全science
dataが利用可能な時間だけを分母にした確率より低くなる可能性がある。

## KAW rangeの代表位置

- L: `range_L_90deg_mean`
- MLAT: `range_MLAT_deg_mean`
- MLT: `range_MLT_circular_mean_hour`

KAW rangeの位置には`fit_*`を使用しない。binは下端を含み上端を含まない半開区間
`[lower, upper)`とする。

## Arase dwelling time

各ORB sampleについて、`erg_orb_l2_pos_Lm`の90度成分、
`erg_orb_l2_pos_rmlatmlt`のMLAT・MLTを用いてbinを決める。

```text
dwelling_duration_sec(i, j) = sum(valid ORB sample time weights in bin (i, j))
```

## KAW observation time

各phaseで`status == "ok"`のrangeをKAW検出rangeとし、range全体の時間を代表位置の
binへ加算する。

```text
kaw_duration_sec(i, j)
    = sum(range_duration_sec where status == "ok" in bin (i, j))
```

`status == "insufficient_data"`はKAW observation timeに含めない。現行Notebookの`ok`は、
共通E/B fit-valid sampleが64以上、かつoccupied 10秒blockが6以上であることを含む。

## Occurrence probability

```text
P_KAW(i, j) = kaw_duration_sec(i, j) / dwelling_duration_sec(i, j)
```

図では100倍したpercentを表示する。代表位置へまとめたKAW時間が同じbinのdwelling timeを
超えた場合は、binningの不整合として停止する。

## 座標と描画

- L--MLAT: `delta L = 1`, `delta MLAT = 5 deg`
- L--MLT: `delta L = 1`, `delta MLT = 1 hour`
- L範囲: `[3, 11)`
- MLAT範囲: `[-50, 50)` deg（符号付き）
- MLT範囲: `[0, 24)` hour

L--MLATは理想双極子関係`r = L cos^2(MLAT)`で夜側子午面へ写像する。plot側の地球半球を
黒、反対側を白とする。この写像は表示上の近似であり、ORBの実地心距離を表すものではない。
L--MLAT統計は`18 <= MLT < 24`または`0 <= MLT < 6`をnightsideとする。分母ではORB
sampleのMLT、分子ではrange代表値`range_MLT_circular_mean_hour`へ同じcutを適用する。
したがって06--18 MLTの滞在時間とKAW rangeはL--MLAT統計に含めない。

L--MLTはnoonを上、duskを左、midnightを下、dawnを右とする。中央の地球はmidnight側を
黒、noon側を白とする。
L--MLTは1 hourごとの放射状grid線を描き、MLT labelは00、06、12、18だけを表示する。
L--MLATはbin境界と同じL=1刻みの双極子曲線およびMLAT=5 deg刻みの線を描く。
MLAT=10 deg刻みをやや濃くし、L labelは2刻み、MLAT labelは10 deg刻みで最小限表示する。


color scaleは既定でlogとし、0のbinはgrayで示す。dwelling timeとKAW observation timeは
共通のLogNormを使う。正値の最小・最大値を包含する10の整数乗を自動範囲とし、確率の上限は
100%とする。固定範囲は`time_log_limits_hour`と`probability_log_limits_percent`で指定できる。

初期値ではdwelling timeが0より大きいbinをすべて表示する。長期統計では
`PLOT_CONFIG["min_dwelling_duration_min"]`でcoverageの少ないbinをmaskできる。

## 保存するbin統計

- `dwelling_duration_sec`, `kaw_duration_sec`
- `dwelling_duration_hour`, `kaw_duration_hour`
- `occurrence_probability`, `occurrence_percent`
- `batch_range_duration_sec`
- `n_dwelling_samples`, `n_batch_ranges`, `n_kaw_ranges`

図はdwelling time、KAW observation time、occurrence probabilityの3 panelとし、range数と
ORB sample数はCSVに残す。
