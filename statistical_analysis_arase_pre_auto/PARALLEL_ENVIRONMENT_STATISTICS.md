# βₑ追加集計の並列数指定

実行中の直列版のコード・worker hashを変えずに利用できるよう、並列用の入口を追加した。
βₑの計算本体は従来の`prepare_electron_beta_statistics.py --worker-request`をそのまま使用する。
入力・コードのsignatureが一致する成功JSONは、直列版からも並列数変更後も再利用する。
有効率による選別は引き続き無効で、occupied blockとbin内標本数の条件は維持する。

## 一括実行で指定

**旧ジョブを停止し、そのβₑ workerの終了を確認してから**新しい入口で実行する。
旧workerは別process groupなので、親をCtrl+Cで止めても処理中のworkerが残る場合がある。
並列版は同じ出力先を使用中の旧親/workerを検出したら停止し、PIDを表示する。既存プロセスを勝手に終了しない。
旧一括ジョブと並列版を同時に動かさない。

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/run_environment_statistics_parallel.py \
  --start-month 202001 --end-month 202212 \
  --min-occupied-blocks 10 --min-bin-count 5 \
  --difference-method centered \
  --dst-lag-hours 1 --ae-lag-minutes 5 \
  --symh-lag-minutes 5 --al-lag-minutes 5 \
  --workers 2
```

`--workers`は同時に動かすβₑのrangeプロセス数。既定1、正の整数を指定する。
2→4へ変更する場合も、一旦止めて同じコマンドの値を変えて再開する。
最初は2を候補とし、メモリとディスクI/Oを確認して増やす。実データの速度倍率は未測定。
地磁気変化率集計、作図、初期/最終checksum確認は直列のまま。
一括版の他の引数（`--prepare-only`、`--plot-only`、`--parameter`等）は従来どおり使える。
旧`run_environment_statistics.py`には`--workers`を追加していないので、ファイル名の変更が必要。

## βₑだけを並列集計

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
  .venv_pyspedas/bin/python -u \
  statistical_analysis_arase_pre_auto/prepare_electron_beta_statistics_parallel.py \
  --start-month 202001 --end-month 202212 --workers 2
```

`--plan-only`は計画のみ、`--max-ranges 2`は先頭2 rangeでの動作確認に使える。
既存の入力を初期計画で検証し、完了後も元の集約コードでchecksumを再確認する。
最終確認で入力が変わっていた場合は追加計算を勝手に直列実行せず、再開を要求して停止する。

## 動作と保存先

- 各rangeを独立processで実行し、tplotのglobal状態を共有しない。
- 各processのOMP/OpenBLAS/MKLスレッド数を1に固定する。
- rangeごとに独立した一時ディレクトリ、request、log、結果JSONを使う。
- 元のreference照合、offline設定、timeoutを維持する。並列化のためにネットワークを有効化しない。
- offline接続要求・source変更での失敗は新規投入を止め、実行中の並列workerも終了する。
- Ctrl+Cでも新しい並列版が起動したworkerを終了する。完了済みJSONは保持する。
- 失敗が残れば集約CSVを更新しない。再実行時は一致する成功分を再利用する。
- 並列版同士の二重起動は出力ディレクトリのfile lockで防ぐ。lockファイルの存在だけでは実行中を意味せず、削除は不要。

保存先は従来と同じ`environment_inputs/beta_e/`。
`parallel_execution.json`に並列数、再利用数、新規成功数、失敗、中断を記録し、
成功時の`range_summary.provenance.json`にも並列数とcoordinatorのhashを追加する。
元Notebook、batch runner、直列入口、worker、既存fit、地磁気集計コードは変更しない。

検証では軽量な独立子processで同時実行数、成功cache再利用、timeout・fatal失敗時の終了を確認する。
実データでの並列batch・全期間処理はここでは実行しない。
