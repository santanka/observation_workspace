# `pyspedas.projects.erg.erg_lep_part_products` の moments 計算

## 対象

この文書は、ワークスペースの `.venv_pyspedas` にインストールされている PySPEDAS 2.1.0 の実装を調査した結果である。対象となる主なソースは次のとおり。

- `.venv_pyspedas/lib/python3.12/site-packages/pyspedas/projects/erg/satellite/erg/particle/erg_lep_part_products.py`
- `.venv_pyspedas/lib/python3.12/site-packages/pyspedas/projects/erg/satellite/erg/particle/erg_lepe_get_dist.py`
- `.venv_pyspedas/lib/python3.12/site-packages/pyspedas/projects/erg/satellite/erg/particle/erg_lepi_get_dist.py`
- `.venv_pyspedas/lib/python3.12/site-packages/pyspedas/projects/erg/satellite/erg/particle/erg_pgs_clean_data.py`
- `.venv_pyspedas/lib/python3.12/site-packages/pyspedas/projects/erg/satellite/erg/particle/erg_pgs_limit_range.py`
- `.venv_pyspedas/lib/python3.12/site-packages/pyspedas/projects/erg/satellite/erg/particle/erg_convert_flux_units.py`
- `.venv_pyspedas/lib/python3.12/site-packages/pyspedas/particles/moments/moments_3d.py`
- `.venv_pyspedas/lib/python3.12/site-packages/pyspedas/particles/moments/moments_3d_omega_weights.py`

`erg_lep_part_products(..., outputs="moments")` は、CDF に格納された既成の L2 moments を読み出す処理ではない。LEP-e または LEP-i の三次元 differential number flux を differential energy flux に変換し、有限のエネルギー・方位角・仰角ビンについて積分して moments を再構成する。

## 処理の依存関係

```text
erg_lep_part_products
 ├─ erg_lepe_get_dist / erg_lepi_get_dist
 │    └─ LEPデータを energy × phi × theta 配列へ整形
 ├─ erg_pgs_clean_data
 │    ├─ 単位変換
 │    └─ NaN・無効ビンの除外
 ├─ erg_pgs_limit_range
 │    └─ energy, theta, phi の指定範囲外を無効化
 ├─ erg_convert_flux_units(..., units="eflux")
 │    └─ differential energy flux に統一
 └─ spd_pgs_moments
      └─ moments_3d
           └─ 密度、流束、速度、圧力・温度テンソル、熱流束など
```

各観測時刻について分布を取り出し、無効ビンを除外してから `moments_3d()` を呼ぶ。通常の `moments` と、磁場座標へ変換してから計算する `fac_moments` の二経路がある。

## 入力分布の構築

### LEP-e

元の配列

```text
time × energy × anode × spin phase
```

を

```text
energy × spin phase(phi) × anode(theta)
```

へ並べ替える。質量と電荷は

$$
m_e=5.68566\times10^{-6}
\quad [\mathrm{eV}/(\mathrm{km/s})^2],
\qquad q=-1
$$

として格納される。

エネルギービン幅 $\Delta E$ は、有限なエネルギーチャンネルを並べ、隣接チャンネル間の対数中点から上下端を作って求める。LEP-e のエネルギー値は時刻依存配列として処理される。

### LEP-i

陽子質量を

$$
m_p=1.04535\times10^{-2}
\quad [\mathrm{eV}/(\mathrm{km/s})^2]
$$

とし、イオン種ごとに

$$
m_{\mathrm{He}^+}=4m_p,
\qquad
m_{\mathrm{O}^+}=16m_p
$$

を用いる。電荷はいずれも $q=+1$ である。

LEP-i の入力単位はコード上 `/keV/q/s/sr/cm²` と扱われ、

```python
data * 1e-3 / abs(charge)
```

によって `/eV/s/sr/cm²` に変換される。第0エネルギーチャンネルは明示的に無効化される。

## moments 計算前の処理

### 無効ビン

次のいずれかに該当するビンは `bins=0` となる。

- 元から無効なビン
- flux が NaN または inf
- energy が NaN または inf
- `energy`, `theta`, `phi` の指定範囲外

moments 計算時には、無効ビンの flux と PSD をゼロにし、energy と $\Delta E$ を形式上 1 eV に置換する。したがって無効ビンは積分値に寄与しない。

### 単位変換

計算直前に必ず

```python
erg_convert_flux_units(clean_data, units="eflux")
```

が呼ばれる。differential number flux を

$$
j(E,\Omega)
\quad
[\mathrm{cm^{-2}\,s^{-1}\,sr^{-1}\,eV^{-1}}]
$$

とすると、計算に使われる `data` は

$$
J_E(E,\Omega)=E j(E,\Omega)
$$

という differential energy flux である。関数引数の `units` が `flux` や `df` の場合でも、moments の積分直前には `eflux` に統一される。

## 立体角積分

粒子方向は

$$
\hat{\boldsymbol v}
=
(\cos\theta\cos\phi,\,
 \cos\theta\sin\phi,\,
 \sin\theta)
$$

で定義される。$\theta$ は余緯度ではなく、赤道面からの elevation/latitude である。

各ビンについて、ビン中心値だけを掛ける近似ではなく、

$$
\theta_\pm=\theta\pm\frac{\Delta\theta}{2},
\qquad
\phi_\pm=\phi\pm\frac{\Delta\phi}{2}
$$

を使い、次の角度積分を解析的に計算する。

$$
\int_{\Delta\Omega}d\Omega,
\qquad
\int_{\Delta\Omega}\hat v_i\,d\Omega,
\qquad
\int_{\Delta\Omega}\hat v_i\hat v_j\,d\Omega
$$

0次の立体角重みは

$$
W^{(0)}
=
[\sin\theta_+-\sin\theta_-]
[\phi_+-\phi_-]
$$

である。`moments_3d_omega_weights()` は0次1成分、1次3成分、2次6成分の積分重みを返す。

デフォルトの `no_ang_weighting=True` は、moments 積分から立体角重みを除く指定ではない。これは範囲制限時にビン幅を考慮するか、ビン中心だけで採否を決めるかを制御する。採用されたビンについては常に有限ビンの立体角積分が行われる。

## 宇宙機電位

一般の `moments_3d()` は

$$
E_\infty=E+q\Phi_{\mathrm{sc}}
$$

を用いる。また、低エネルギー側には

$$
w=
\operatorname{clip}
\left(
\frac{E+q\Phi_{\mathrm{sc}}}{\Delta E}+\frac12,
0,1
\right)
$$

という漸減重みを導入する。これは特に正の宇宙機電位下の電子について、低エネルギー光電子の寄与を滑らかに抑えるための処理である。

ただし、`erg_lep_part_products` は `spd_pgs_moments()` に `sc_pot` を渡していない。実際の LEP moments では常に

$$
\Phi_{\mathrm{sc}}=0,
\qquad
E_\infty=E,
\qquad
w=1
$$

となる。現状の出力には宇宙機電位補正が入っていない。

## 各出力の定義

以下では、全エネルギー・全角度ビンについての和を $\sum_b$ とする。

### `density`

コード上の密度は

$$
n=
10^{-5}\sqrt{\frac{m}{2}}
\sum_b
J_{E,b}
\frac{\Delta E_b}{E_b}
w_b
W^{(0)}_b
\frac{\sqrt{E_{\infty,b}}}{E_b}
$$

である。LEP の実際の呼び出しでは $E_\infty=E$、$w=1$ なので、

$$
n=
10^{-5}\sqrt{\frac{m}{2}}
\sum_b
J_{E,b}
\frac{\Delta E_b}{E_b^{3/2}}
W^{(0)}_b
$$

となる。出力単位は $\mathrm{cm^{-3}}$。

### `flux`

粒子流束は

$$
\boldsymbol\Gamma
=
\sum_b
J_{E,b}
\frac{\Delta E_b}{E_b}
w_b
\frac{E_{\infty,b}}{E_b}
\int_{\Delta\Omega_b}
\hat{\boldsymbol v}\,d\Omega
$$

である。$\Phi_{\mathrm{sc}}=0$ なら、

$$
\boldsymbol\Gamma
=
\sum_b
j_b\Delta E_b
\int_{\Delta\Omega_b}
\hat{\boldsymbol v}\,d\Omega
$$

となる。これは電流密度ではなく number flux $\boldsymbol\Gamma=n\boldsymbol U$ であり、単位は $\mathrm{cm^{-2}\,s^{-1}}$。電子についても電荷 $-e$ は掛けられない。

### `velocity`

$$
\boldsymbol U
=
\frac{\boldsymbol\Gamma}{n}\,10^{-5}
$$

で計算する。$10^{-5}$ は cm/s から km/s への変換で、出力単位は km/s。

### `mftens`

バルク流を差し引く前の生の2次モーメントであり、

$$
M_{ij}
=
m\int v_i v_j f(\boldsymbol v)\,d^3v
$$

に対応する。コード内部では

$$
M_{ij}\propto
\sum_b
J_{E,b}
\frac{\Delta E_b}{E_b}
w_b
\frac{E_{\infty,b}^{3/2}}{E_b}
\int_{\Delta\Omega_b}
\hat v_i\hat v_j\,d\Omega
$$

を計算し、質量と単位変換係数を掛ける。

成分順序は

```text
xx, yy, zz, xy, xz, yz
```

で、単位は eV/cm³。

### `ptens`

生の2次モーメントからバルク運動を差し引く。

$$
P_{ij}
=
M_{ij}-mU_i\Gamma_j\,10^{-5}
$$

これは通常の中心化された2次モーメント

$$
P_{ij}
=
m\int
(v_i-U_i)(v_j-U_j)
f(\boldsymbol v)\,d^3v
$$

に対応する。成分順序は `mftens` と同じで、単位は eV/cm³。

### `ttens`

温度テンソルは

$$
T_{ij}=\frac{P_{ij}}{n}
$$

で、出力は $3\times3$ 行列、単位は eV。

### `avgtemp`

$$
T_{\mathrm{avg}}
=
\frac{\operatorname{Tr}\boldsymbol T}{3}
=
\frac{T_{xx}+T_{yy}+T_{zz}}{3}
$$

である。異方性がある場合も、3方向の平均に縮約したスカラー量となる。

### `vthermal`

$$
v_{\mathrm{th}}
=
\sqrt{\frac{2T_{\mathrm{avg}}}{m}}
$$

で、単位は km/s。この実装の定義は1次元標準偏差 $\sqrt{T/m}$ ではなく $\sqrt{2T/m}$ である。

### `eflux`

$$
\boldsymbol F_E
=
\sum_b
J_{E,b}
\frac{\Delta E_b}{E_b}
w_b
\frac{E_{\infty,b}^{2}}{E_b}
\int_{\Delta\Omega_b}
\hat{\boldsymbol v}\,d\Omega
$$

で計算される。これはバルク流を差し引いていない lab-frame のエネルギー流束である。

### `qflux`

各ビンについて

$$
\boldsymbol w_b=\boldsymbol v_b-\boldsymbol U
$$

および

$$
E_{\mathrm{th},b}
=
\frac12m|\boldsymbol w_b|^2
$$

を計算し、

$$
\boldsymbol q
=
\int
E_{\mathrm{th}}\boldsymbol w
f(\boldsymbol v)\,d^3v
$$

に対応する積分を行う。`eflux` と異なり、バルク速度を差し引いた速度を用いる。出力単位はコード上 eV/(cm² s)。

## 温度テンソルの固有値関係

### `t3`

温度テンソル $T_{ij}$ の固有値3個。ただし単純な昇順ではない。

コードは最小・中間・最大固有値の異方性を比較して「対称軸らしい」固有方向を選び、その成分を第3要素へ巡回移動する。したがって

```text
t3 ≈ [T_perp1, T_perp2, T_symmetry]
```

という意図だが、第3成分が磁場平行温度である保証はない。

### `symm`

`t3` の第3成分に対応する固有ベクトル。温度楕円体の推定対称軸である。

### `symm_theta`, `symm_phi`

`symm` の極座標角。

### `symm_ang`

$$
\alpha_{\mathrm{symm}}
=
\cos^{-1}
\left(
\left|
\hat{\boldsymbol B}\cdot\hat{\boldsymbol s}
\right|
\right)
$$

で定義される、温度楕円体の対称軸と磁場のなす鋭角。

### `magt3`

磁場とバルク速度を使って、概ね

```text
z' = B方向
y' = B × U 方向
x' = y' × B 方向
```

という基底を作り、温度テンソルをこの座標へ回転した後の対角3成分を返す。このため

```text
magt3[2] ≈ T_parallel
magt3[0], magt3[1] ≈ 二つの垂直方向温度
```

となる。ただし、これは厳密な gyrotropic average ではない。通常の

$$
T_\perp=\frac{T_{\perp1}+T_{\perp2}}{2}
$$

が必要なら後処理で平均する必要がある。

## `moments` と `fac_moments`

通常の

```python
outputs="moments"
```

では、元の粒子方向座標で範囲制限した後に moments を計算する。

`pitch` または `gyro` がデフォルト以外なら、コードは `moments` を自動的に `fac_moments` に置き換える。FAC変換後に pitch-angle と gyrophase の範囲を適用し、限定された分布から moments を再計算する。

この場合の `density` は全粒子密度ではなく、指定した pitch/gyro 領域だけの部分密度である。速度や圧力も部分分布の moments なので、分布全体に対する通常の fluid moment とは意味が異なる。

## 生成される tplot 変数

入力が `erg_lepe_l2_3dflux_FEDU` なら、概ね次の変数が生成される。

```text
erg_lepe_l2_3dflux_FEDU_density
erg_lepe_l2_3dflux_FEDU_flux
erg_lepe_l2_3dflux_FEDU_velocity
erg_lepe_l2_3dflux_FEDU_mftens
erg_lepe_l2_3dflux_FEDU_ptens
erg_lepe_l2_3dflux_FEDU_ttens
erg_lepe_l2_3dflux_FEDU_vthermal
erg_lepe_l2_3dflux_FEDU_avgtemp
erg_lepe_l2_3dflux_FEDU_eflux
erg_lepe_l2_3dflux_FEDU_qflux
erg_lepe_l2_3dflux_FEDU_t3
erg_lepe_l2_3dflux_FEDU_magt3
erg_lepe_l2_3dflux_FEDU_symm
erg_lepe_l2_3dflux_FEDU_symm_theta
erg_lepe_l2_3dflux_FEDU_symm_phi
erg_lepe_l2_3dflux_FEDU_symm_ang
```

## 実装上の注意点

### 磁場データ

`density`, `velocity`, `ptens` の積分自体には磁場は不要だが、`magt3` や `symm_ang` の計算には必要となる。磁場がゼロまたは欠損すると、磁場の規格化や回転行列の計算で NaN が生じる。

`mag_name` がない場合、コードは `no_mag_for_moments=True` を設定するが、このフラグはその後使用されない。磁場依存出力を安全にスキップする実装にはなっていない。

### 使用されていない引数

- `datagap` は `erg_lep_part_products()` 内で使用されていない。ギャップ検出や補間抑制には効かない。
- `relativistic` も `erg_pgs_clean_data()` に渡されていないため、LEP moments の実際の処理には効いていない。

### 範囲境界

- `no_ang_weighting=True` でも、採用ビンの立体角積分自体は行われる。
- `energy=[Emin, Emax]` はエネルギービン中心値で完全採否を決める。境界ビンの部分エネルギー幅を積分する処理はない。
- `no_ang_weighting=False` の場合も、角度範囲と少しでも重なるビンを採用するだけで、範囲内に入った立体角の割合を連続的に掛ける処理ではない。

### ゼロ密度

`density=0` の時刻では

```python
velocity = flux / density
ttens = ptens / density
```

により NaN または inf が生じる。固有値計算の `LinAlgError` は捕捉されるが、全出力が安全に補正されるわけではない。

### FAC出力の座標メタデータ

`_mag` の付いた `fac_moments` でも、tplot メタデータ設定関数はベクトル座標を一律 `DSI` と設定する。数値はFAC変換後の成分である一方、メタデータが `DSI` となる可能性があり、少なくともコード上は整合していない。

## 解釈上の結論

この関数の moments は、観測された有限FOV・有限エネルギー範囲の3次元分布をビン積分して得た再構成量である。そのため、結果は次の条件に依存する。

- 観測されていない低・高エネルギー粒子
- 欠損した角度ビン
- 背景除去と負値処理
- `energy`, `theta`, `phi`, `pitch`, `gyro` による範囲制限
- 宇宙機電位補正の有無
- 磁場データの品質

特に、現行の `erg_lep_part_products` では宇宙機電位が実質ゼロとして扱われる。低エネルギー電子から求める密度や温度を定量的に利用する場合、この点を明示して評価する必要がある。

## 参考URL

- [PySPEDAS: ERG analysis tools](https://pyspedas.readthedocs.io/en/stable/erg_analysis.html)
- [公式 `erg_lep_part_products` ソース](https://pyspedas.readthedocs.io/en/stable/_modules/pyspedas/projects/erg/satellite/erg/particle/erg_lep_part_products.html)
- [公式 `moments_3d` ソース](https://pyspedas.readthedocs.io/en/stable/_modules/pyspedas/particles/moments/moments_3d.html)
