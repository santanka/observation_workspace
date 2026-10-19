# LEP-e温度の独立比較

`batch_arase.py`や既存ノートブックを実行・変更せず、既存候補イベントについて
PI提供bi-Maxwellian温度と `pyspedas.projects.erg.erg_lep_part_products` の直接積分温度を比較する。
対象は既存イベントフォルダがある区間すべて。KAW検出成立イベントだけに絞る追加選別は行わない。
イベントの秒以下の時刻は既存 `run_state/<dataset key>/ranges/range_*.json` から読む。
既存batchの状態・完了マーカー・解析結果を更新しない。

## 実行

ワークスペースのルートから実行する。日付はUTC。
日指定 `--date` と月指定 `--month` のどちらか一方を指定する。

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/lepe_temperature_comparison/compare.py \
  --date 2022-09-01 --dry-run

# 2022年9月: 対象イベント数とローカル入力ファイルの存在だけを確認
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/lepe_temperature_comparison/compare.py \
  --month 202209 --dry-run

# 2022年9月の既存候補イベントを順に計算し、各イベントフォルダへ保存
# 必要なローカルCDFが揃っている場合
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/lepe_temperature_comparison/compare.py \
  --month 202209

# 不足するLEP-e/MGF CDFを取得しながら実行する場合
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/lepe_temperature_comparison/compare.py \
  --month 202209 --download-missing

# YYYY-MM形式も使用可能。検証済みの完了結果は自動的に再利用
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/lepe_temperature_comparison/compare.py \
  --month 2022-09

# 1日だけ計算・描画
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/lepe_temperature_comparison/compare.py \
  --date 2022-09-01

# 別の場所へ試しに保存: <root>/<dataset key>/<date>/<event>/lepe_temperature_comparison/
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/lepe_temperature_comparison/compare.py \
  --date 2022-09-01 --output-root /tmp/lepe_temperature_comparison_dual_20220901

.venv_pyspedas/bin/python -m unittest discover \
  -s statistical_analysis_arase_pre_auto/lepe_temperature_comparison -p 'test_*.py'
```

同じ日/月のフォルダが複数datasetにある場合は `--dataset <auto/dataset key>` で指定する。
月指定ではその月に既存候補フォルダがある日だけを対象に、1日1プロセスで直列実行する。
各日のログを `<dataset>/lepe_temperature_comparison/logs/YYYY-MM/YYYY-MM-DD.log`、
日別成否を `<dataset>/lepe_temperature_comparison/lepe_temperature_comparison_dual_energy_month_YYYYMM_summary.csv` に保存する。
あるイベント/日で失敗しても他のイベント/日の処理を続け、失敗が1件でもあれば終了コード1とする。
日全体の入力読み込みが失敗した場合はその日を終了し、月ランナーが次の日へ進む。
月別summaryの `complete` はその日の処理が成功/検証済み再利用で終わったことを表す。

通常の再実行では入力ファイルSHA256、コードSHA256、パッケージ版、イベント時刻、選別条件、
出力5ファイルのSHA256が一致するイベントだけをスキップする。変更・破損・未完了時は再計算する。
`--force` を加えると検証済み結果も再計算する。再開単位はイベントであり、途中のスピンからは再開しない。
候補一覧は各プロセスの開始時点で既存の `run_state` から読み、実行中に増えた候補は次回対象となる。

入力は最新のローカルLEP-e L2 v05、MGF L2 8sec DSI、PI提供1min v2を使用する。
既定ではネットワーク取得を行わず、不足時はエラーとする。
`--download-missing` を付けると、不足しているLEP-e/MGFだけをergpyspedasの
`downloadonly=True` で直列取得してから入力を再確認する。既存CDFは更新せず再利用する。
PI fitの自動取得はなく、提供ファイルを指定先へ置く必要がある。
取得機能を使う場合、`--raw-root` は既定値と同様に `satellite/erg` で終わるパスとする。
ネットワーク失敗をデータ不存在と判定しない。取得後にも必要CDFがなければその日は失敗となる。
ATT/軌道読み込みは行わない。
`--dry-run` はファイルの存在と区間の対応の確認のみで、CDF内容や実時間被覆の検証ではない。
dry-runも不足入力があれば終了コード1となる。
CDFはscratchへストリーミング展開してgzip CRC/サイズを検査し、終了時に展開ファイルを削除する。
日をまたぐイベントはこのランナーでは未対応として当日を失敗扱いとする。

## 比較の定義と限界

- 積分は常に2条件を計算し、同じ図に重ねる。
  `all`: チャネル中心が `E >= 67 eV` の全観測チャネル、`energy=[67., np.inf]`。
  `le1000`: チャネル中心が `67 <= E <= 1000 eV`、`energy=[67., 1000.]`。
  1000 eVは含む。チャネル幅そのものは公開ルーチンの計算法を維持し、
  1 keVでビン幅を切り詰めない。
- PI fit: 提供値の `50 <= E <= 1000 eV` を使用する。
  全エネルギーIntegralとPI fitはエネルギー範囲が異なる。実チャネル中心の一覧/範囲は来歴に記録する。
  これはL1a PADとL2 3dfluxの較正差を除去するものではない。
- 直接積分はL2 3dfluxを公的ローダーと同じenergy/channel/sector順序でtplotへ登録し、
  公開 `erg_lep_part_products(outputs=['moments'])` を呼び出す。
  負フラックス・無効ビンの取り扱いも公開実装を維持する。
- `T_parallel = bhat^T T_DSI bhat`, `T_perp = (trace(T_DSI)-T_parallel)/2`。
  `T_effective = trace(T_DSI)/3 = (T_parallel+2*T_perp)/3`。すべてeV。
  バルク流を差し引いた中心モーメントで、衛星電位は0 V。
- 磁場はMGF L2 **8sec DSI**をスピン開始時刻へ線形補間する。
  外挿と16秒超の補間ギャップを拒否する。既存KAW解析で使う64 Hz背景磁場とは異なる。
  MGF品質フラグはNPZに保持し、追加のフラグ選別は行わない。
- PI生成コードと同じ `packet_info_bitmask` をL2にも適用する。
  正の密度・正の全温度成分・有限磁場/テンソルを満たすスピンを積分温度の平均に使用する。
  有効判定は2つの積分条件それぞれで行い、一方の欠損で他方を除外しない。
- PI温度は成分別に正・有限、対応する `eflux_*_mean > 1e4`、`num_valid_data > 3` を要求する。
  最後の条件は3パラメータに対する最低限の点数条件で、十分なフィット品質の保証ではない。
  effectiveは両成分が有効な場合だけ採用する。残差の閾値と温度上限は既定で設定しない。
  `--fit-flux-min` / `--fit-temperature-max` で変更できる。
  除外された有限正値のPI温度も灰色の×として表示し、生値をCSVに保存する。
- 積分側は**スピンごとの温度の算術1分平均**。PI側は**分布の1分平均後のフィット**。
  時間平均と推定の順序、部分積分とMaxwellianモデル推定、元データの較正が異なる。
  同じ物理量の推定誤差だけを比較しているとは解釈しない。
- イベントと重なる分区間はイベント外の部分も含む完全な分区間を使用する。
  CSVの時刻は左端、図の点は中心、横棒はPIの分区間。イベント外部分は灰色背景。
  ネイティブ時刻で区間を選ぶため、境界をまたぐスピンの露出時間を分割しない。
  各分の総数・有効数・有効率を保存するが最低カバレッジ条件は追加しない。
  PI元分布の各分の時間被覆率は提供ファイルからは分からない。

PI資料は本データを非公式・quick look用と位置付けている。差の原因や科学的妥当性は
この比較図だけでは確定できない。

## 出力

各イベントフォルダの下に `lepe_temperature_comparison/` を作り、
`lepe_temperature_comparison_dual_energy` を接頭辞とするPNG/PDF、
1分比較CSV、ネイティブ温度CSV、診断NPZ、来歴JSONを保存する。
図はeffective・parallel・perpendicularの3段。それぞれにIntegralの2条件とPI fitを表示する。
積分のネイティブ値は薄線、1分平均値は丸印付きの線、PI fitは四角印付きの線。
PIの横棒は1分平均分布の時間範囲であり、温度の誤差棒ではない。
NPZは条件別の元DSI温度テンソル、補間磁場、生温度、LEP-e/MGF品質フラグを含む。
1分CSVでは `moment_all_*` と `moment_le1000_*` を分け、PI fit列は共通。
native CSV/NPZでは `all__*` と `le1000__*` を区別する。
JSONには入力SHA256、コードSHA256、ライブラリ版、PI属性、時刻/品質/平均化条件を記録する。
日フォルダの `lepe_temperature_comparison/` にイベントごとの対応分数と積分/PI温度比の中央値のsummary CSVを保存する。
再計算が必要な場合は同名の比較出力を上書きする。
以前の `lepe_temperature_comparison_lt1keV*` と `lepe_temperature_comparison_ge67eV*` は
片方のIntegral条件だけを描いた結果。両条件の比較には `dual_energy` のファイルを参照する。
