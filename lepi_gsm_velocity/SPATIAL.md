# LEP-i GSM流速の平均空間分布

`spatial.py`は変換済みGSM NPZを読み、KAW解析とは独立に流速そのものを集計する。ネットワーク通信、3D分布の読込み、moment再計算は行わない。

## 図と統計量

- H⁺ (`FPDU`)・He⁺ (`FHEDU`)・O⁺ (`FODU`)を混ぜず、夜側 `[18,24) ∪ [0,6)` MLTと昼側 `[6,18)` MLTを別々に出力する。
- 各図は2行3列。上段が各成分の `mean(v)`、下段が `mean(abs(v))`、列がGSM x・y・z。単位はkm/s。`abs(mean(v))`やベクトルの速さではない。
- パネル見出しは数式の `⟨v_x⟩`・`⟨|v_x|⟩`（x/y/zを下付き）で表示する。L–MLAT図は昼側のみ横軸を反転し、昼側のセルが中央円の左、夜側のセルが右になる。速度の符号・平均値は反転しない。中央円は画面上で常に左半分が白、右半分が黒。
- 各種に**L–MLT図も1図ずつ追加**する。L境界は同じ、MLT境界は0–24時の1時間刻み。MLATは−50°以上+50°以下を統合し、`--sectors`で選択した昼夜の和集合を使う（既定は全24時）。Lを半径とする円形模式図で、00 MLTを右、06を上、12を左、18を下に置く。中央円の白側が昼、黒側が夜に対応する。これはGSM赤道面の実位置図ではない。
- 符号付き平均はゼロ中心の発散色、絶対値平均は0始まりの連続色。各行について、3成分・3イオン種・昼夜で同じ色範囲を使う。既定は表示セルの最大絶対値までで、百分位による切捨てはしない。`--vmax-kms 100`のように明示した場合だけ表示色を飽和させる。CSV値は変わらない。
- **各有効レコードを等重みとする算術平均**。5分chunkごとの平均や日平均を再平均しない。密度・質量による重み付け、100秒平滑化、外れ値除去は行わない。一定cadenceでは観測時間平均に近いが、空間体積平均ではない。
- セルはL境界 `3,4,...,11`、MLAT境界 `−50,−45,...,+50°`。左端を含み右端を含まない。最終右端のみ含む。南北を統合しない。
- Lはdefinitive ORB `pos_Lm[:,0]`（90°ピッチ角のMcIlwain L）、MLAT・MLTは `pos_rmlatmlt[:,1:3]`。既存KAW空間分布と同じ変数・メッシュを採用する。KAW図のrange代表値とは異なり、各moment時刻へ座標を補間する。
- 表示位置は `ρ=L cos³(MLAT)`, `z=L cos²(MLAT) sin(MLAT)` による双極子模式表示。**GSM x–z平面上の実位置ではない**。GSMは色で示す速度ベクトルの座標系を指す。
- 既定では100レコード以上のセルを着色する（`--min-count`）。`--min-days`で独立したUTC観測日数の下限も指定できる。100点は表示上の暫定条件であり、独立標本100個を意味しない。CSVには閾値未満のセルも含む。

現行DSI入力は30–30,000 eV/qの部分モーメント。GSMへ回転しても衛星相対速度であり、衛星速度の加減算や低エネルギー成分の補正はしない。全エネルギーのプラズマbulk速度と同一視しない。低密度時の不安定なmomentや品質フラグに基づく追加選別は今回未実装。

## 入力・欠測・由来

GSM profileが一致する `complete`/`partial` stateと、そのSHA256に一致するNPZだけを読む。`moment_valid`・`rotation_valid`・種別の`velocity_valid`が真で、3成分が全て有限なレコードを使う。旧completeに`moment_valid`がない場合は全時刻有効とし、その件数を記録する。partialにマスクがなければ拒否する。

軌道CDFは共通cacheから日ごとの最新版を選択する。**各日CDFがその日00:00以上、翌日00:00未満のレコードを所有**する。日境界の重複endpointは次の日のCDFを採用し、同じ日内で矛盾する重複は拒否する。CDFのFILLVAL・VALIDMIN/MAX、非有限値、物理範囲外のMLAT/MLT/Lを無効化する。

軌道座標は最大120秒の両側支持点で線形補間する。MLTは0/24時をまたぐ最短方向で補間する。NaNを削除して隙間をつながず、外挿しない。隣接日CDFがない場合も外挿せず、当日CDF自体がない場合は集計を失敗させる。

各セルの `n_records`・`n_days` と符号付き/絶対値平均6成分をCSVに保存する。空セルはNaNで、流速ゼロとして埋めない。描画下限はCSV値を変更しない。`provenance.json`に期間・設定・state別件数・除外診断・profile・コード/軌道SHA256、`inputs.jsonl`にGSM NPZ/stateのSHA256を残す。

対象は起動時に存在するGSM出力であり、指定した3年間のDSI/GSM生成完了を保証しない。GSMに未変換のchunkや存在しないstateは、この処理の集計対象にならない。期間の網羅性はDSIとGSMのstate一覧で別途評価する。追加の科学的品質選別を入れる場合は、設定と出力先を分ける。

結果は一時ディレクトリで作成し、全処理成功後に出力ディレクトリへrenameする。既存の出力先には上書きしない。失敗・中断時は `.partial-*` と `INCOMPLETE.txt` を残す。途中の集計再開はなく、再実行時は全対象をストリーム集計し直す。GSM変換側のchunk単位再開は従来どおり。

## 実行

月・年の長期実行はユーザー側で行う。以下の長期コマンドは本実装の検証では未実行。

まず既存の独立処理でDSIからGSMへ変換する。元のmomentは変更しない。

長期間の変換は [PARALLEL.md](PARALLEL.md) のoffline並列版を使える。逐次版が実行中の場合はCtrl+Cで終了後に切り替える。同じ科学profileの完成済みchunkを再利用する。

```bash
.venv_pyspedas/bin/python lepi_gsm_velocity/convert.py \
  --moments-root /mnt/j/observation_data/Arase_LEPi_moments/native_dsi_1d86398b65a1 \
  --start 2020-01-01 --end 2023-01-01 --offline --dry-run

.venv_pyspedas/bin/python lepi_gsm_velocity/convert.py \
  --moments-root /mnt/j/observation_data/Arase_LEPi_moments/native_dsi_1d86398b65a1 \
  --start 2020-01-01 --end 2023-01-01 --offline
```

`--dry-run`が示す`output_root`を、次の`--gsm-root`に指定する。ローカル姿勢・軌道が不足する場合、offline変換は失敗を記録する。必要な取得を許す場合だけ変換コマンドの`--offline`を外す。変換の失敗/partial状態を確認してから集計する。

```bash
MPLCONFIGDIR=/tmp/lepi_spatial_matplotlib MPLBACKEND=Agg \
.venv_pyspedas/bin/python lepi_gsm_velocity/spatial.py \
  --gsm-root /mnt/j/observation_data/Arase_LEPi_GSM_velocity/gsm_observed_6597020f98c9 \
  --start 2020-01-01 --end 2023-01-01 \
  --output-dir /mnt/j/observation_data/Arase_LEPi_velocity_statistics/202001-202212_v2 \
  --min-count 100 --min-days 1
```

上記GSM rootは2026-11-04の現行環境・現行DSI profileについて、5分区間の変換dry-runで確認した値。変換コードやパッケージが変わるとhashも変わるため、その場合は改めてdry-runが示すパスを使う。

3種×昼夜のL–MLAT 6組（`FPDU_night.*`など）と、3種のL–MLT 3組（`FPDU_l_mlt.*`など）の合計9組についてPNG・PDF・CSV、由来JSON/JSONLを出力する。色範囲はL–MLATとL–MLTにも共通。L–MLT CSVの座標列は`mlt_left_hour`・`mlt_right_hour`。`--dry-run`を追加するとGSM stateの在庫確認だけを行い、NPZ/CDFは読まず、図も作らない。処理は1プロセス、nice=19で行い、全レコードをメモリへ蓄積しない。

## 限定検証

```bash
MPLCONFIGDIR=/tmp/lepi_spatial_matplotlib MPLBACKEND=Agg \
.venv_pyspedas/bin/python -m unittest discover -s lepi_gsm_velocity -p 'test_spatial.py'
```

既知の正負流速、絶対値を取る順序、マスク、セル端、日数、MLT wrap、支持点欠測・長いgap、合成GSM入力から全9図・CSV・由来の出力、SHA256不一致拒否を検証する。さらにMLT各方位・1時間bin、昼側軸反転、反転後も画面左白・右黒になる実描画pixelを確認する。実データの短区間検証と3年全体の解析は区別する。

2026-11-04の確認では、既存のcore/converter試験を含む19件が成功。過去のGSM検証出力 `Arase_LEPi_moments/_validation/gsm_observed_smoke/gsm_observed_b8721030d19a` の2022-09-29 05:05–05:10 UTC（37レコード×3種）から実軌道CDFを使って6図を出力した。夜側のL=[6,7), MLAT=[20,25)°の1セルに各37点が入り、3種×符号付き/絶対値×3成分の18平均値が独立したNumPy集計と一致した（rtol=atol=1e-14）。昼側は空であり、昼側の値を実データで検証したわけではない。昼夜の振り分け・全6図出力は合成入力でも検証した。

試験出力は `/mnt/j/observation_data/Arase_LEPi_velocity_statistics/_validation/spatial_20261104_smoke/`。表示下限はこの試験だけ `--min-count 1`。入力は過去の短区間検証用profileであり、現行DSI profileの3年出力を検証したものではない。

数式表記・白黒円・昼側反転・L–MLT追加後の試験出力は同じ親ディレクトリの `spatial_20261104_layout_v2/`。変更対象の6件のテストが成功し、実データのL–MLAT全6 CSVは変更前とbyte単位で一致、L–MLTには各種37点が入ることを確認した。3年間の処理は未実行。
