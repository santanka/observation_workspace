# FAC Vyの追加集計と依存性図

現在の期間統合masterはtoroidal側のVx代表値だけを保存している。
Vyは元Notebookで計算されているが、統計用の代表値は未保存だった。
`prepare_system_velocity_statistics.py`は、元の前処理を再利用してVyだけを独立集計する。
既存のtoroidal κ・X₀を条件づけるための処理であり、poloidal波動fitは行わない。
Notebook・batch runner・既存βₑ preparer/workerは書き換えず、成功済みβₑcacheを保持する。

## 通常の実行

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/run_statistics.py \
  --start-month 202001 --end-month 202212 \
  --product dependence --parameter abs_vsys_y \
  --min-occupied-blocks 10 --min-bin-count 5 \
  --vsys-y-bin-width-kms 10 --workers 2 --overwrite
```

未完了分の集計が全件成功した後で全4方向分類を作図する。
`--parameter vsys_y`を加えれば符号付きの図も同時に作る。
`--plot-only --overwrite`は追加集計済みの場合の再作図に使う。
`--prepare-only`は集計だけ、`--dry-run`は呼び出すコマンドの確認だけを行う。
`--overwrite`は図・作図用CSVの更新であり、成功済みVy JSONを無条件に再計算しない。

## 集計の定義と検証

- 現行v2の観測partial ion momentをそのまま使うFAC y成分［m/s］。衛星速度は差し引かない。
- βₑと同じNotebook前処理prefixを利用し、地磁気・CWT・κ/X₀ fit・bootstrapに進む前に停止する。
- 元の`build_plasma_parameter_products(..., smoothing_sec=100.0)`で作るpoloidal側の`perp_sys_speed`を使う。
- 保存済みκ時系列の時刻へ線形補間する。区間外への外挿は行わず、無効slotを維持する。
- 既存の符号付きVxも同じ方法で再構成し、保存済みrange集計値と照合する。
  n_total・n_validは完全一致、valid_fraction・mean・std・median・q16・q84はrtol=1e-6、atol=1e-10。
  一致しなければ失敗として保存を止める。
- Vyは符号付きのmean・std・median・q16・q84・n_total・n_valid・valid_fractionを保存する。
- 絶対値図は`abs(median(Vy))`を使う。`median(abs(Vy))`ではない。
- 元の範囲全体の代表値であり、検出時刻だけの代表値にはしない。有効率による選別は行わない。

sourceのNotebook・preparer・βₑ prefix helper・batch runnerとκ時系列CSVのSHA256を要求に記録し、
worker実行前後と集約時に再確認する。元CDFの当時のchecksumは残っていないため、
現在のCDF cacheを使った再構成であることとVx照合結果をprovenanceへ記録する。

## 保存先・再開・並列

```text
auto/statistics/202001-202212/
  environment_inputs/vsys_y/
    .run.lock                    # flock。ファイルが残っていても手で消さない
    plan.json                    # range_uid・入力hash・cache signature
    execution.json               # 再利用・新規完了・失敗数
    failures.json
    ranges/<uid-hash>.request.json
    ranges/<uid-hash>.log
    ranges/<uid-hash>.json        # 成功したVy集計と再構成Vxの照合値
    ranges/<uid-hash>.worker.lock
    range_summary.csv
    range_summary.provenance.json
  min_blocks_10/
    abs_vsys_y_statistics/        # 絶対値の図・CSV・provenance
    vsys_y_statistics/            # 符号付きも選択した場合
```

同じコマンドを再実行すると、入力・科学処理のsignatureが一致する成功JSONを再利用し、
失敗・未実行分だけを処理する。全planned rangeが成功するまで集約CSVを更新せず、
統一入口もその場合は作図に進まない。
`--workers`は独立したrangeプロセスの数で、各workerのBLAS/OpenMPは1 threadに固定する。
既定2並列。βₑと同様にデータ読込み・イオンmoment・座標変換は重いため、最初は1–2を使う。
Ctrl+C・致命的なoffline/source/referenceエラーでは、そのコマンドが起動したworkerを終了する。
workerも親のflockを継承するので、親が強制終了してworkerが残っている間は別の親を起動できない。

通常はsocket通信を禁止するoffline実行。不足データ取得を許可する場合だけ`--allow-network`を付ける。
ネットワーク許可の切替とworker数の変更は、既存成功cacheのsignatureに影響しない。

## 限定試験

まず計画だけ（CDFは読まない）:

```bash
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/prepare_system_velocity_statistics.py \
  --start-month 202001 --end-month 202212 --max-ranges 1 --plan-only
```

1 rangeを実際に再構成する場合:

```bash
MPLCONFIGDIR=/tmp/arase_matplotlib MPLBACKEND=Agg \
.venv_pyspedas/bin/python -u statistical_analysis_arase_pre_auto/prepare_system_velocity_statistics.py \
  --start-month 202001 --end-month 202212 --max-ranges 1 --workers 1
```

`--max-ranges 1`の集約CSVは限定試験用であり、全期間の作図には使えない。
その後max-rangesを省いて通常実行すれば、その1 rangeの成功JSONを再利用して残りを処理する。
限定CSVを本集計と分けたい場合は`--output-dir`を指定する。
