# GSM変換の並列実行

`parallel.py`は元の`convert.py`を変更せず、その`profile`・`Support`・`convert_chunk`を呼ぶ独立runner。実行中の逐次版に対して変換コードの変更検出を発生させない。

姿勢列形式エラーへの対応として、workerは`attitude_support.Support`を使う。正常日は従来の`Support`の経路をそのまま使用する。従来の姿勢読込みが数値変換・列数エラーになった場合だけ、raw TXTの専用パーサーへ切り替える。

専用パーサーは`26.2539-151515.3574`のような符号で隣接した数値を分離し、指数の符号（`1.2e-3`）は分離しない。FLAG_VI/Sensorがある16列、ない14列の両方を検証し、曖昧な連結・数値欄の文字・列数不整合は拒否する。使用する軸は従来どおりJ2000のI-Alpha/I-Delta。追加の品質選別は行わず、負値を含むADSGのraw値も保持する。

このfallbackでも**元TXTを変更しない**。profileと変換本体は従来と同じなので、正常に完了した既存chunkを再利用する。fallbackで生成したNPZのmanifestには、`inputs`のrole `attitude_reader_code:adjacent_signed_numbers_v1`として専用パーサーのパス・fingerprint・SHA256、読込みファイル中の修復行数、従来ローダーのエラーを記録する。空白分離だけに依存する元の`convert.py`単独実行にはこのfallbackを導入していないため、該当日の復旧には**parallel.pyを使う**（`--workers 1`も可）。

## 切替え

1. 現在の`convert.py`を実行している端末で **Ctrl+C** を押し、終了を待つ。Ctrl+Zではプロセスが残るので切替えにならない。`.run.lock`を削除しない。
2. 同じ入力・座標変換設定で並列版を起動する。

```bash
.venv_pyspedas/bin/python -u lepi_gsm_velocity/parallel.py \
  --moments-root /mnt/j/observation_data/Arase_LEPi_moments/native_dsi_1d86398b65a1 \
  --start 2020-01-01 --end 2023-01-01 \
  --offline --workers 4 --pause-seconds 0
```

現行環境では出力先は従来と同じ `/mnt/j/observation_data/Arase_LEPi_GSM_velocity/gsm_observed_6597020f98c9`。並列数や待機設定は科学profileに含めないので、worker数を変えても別profileにならない。入力の指紋・state・NPZ SHA256が一致する完成済みchunkを再利用し、壊れた出力は元の検証規則で再生成する。現在の実行がロックを保持していれば並列版は終了コード3で拒否する。

`--dry-run`を加えると、元stateの在庫・対象数・出力先・実worker数を表示して終了する。数十万stateの在庫確認自体にも時間はかかる。起動時の在庫確定後にworkerの変換が始まる。

## 高速化する部分

- 元の逐次版は1 CPU固定、5分chunkごとに既定2秒休止する。2026-11-04のDSI月別summaryではcomplete+partialが208,524 chunkで、全件を一巡する際の休止だけで417,048秒（約4.83日）。これはsummaryからの設定上の計算であり、実測所要時間や残り時間ではない。
- 並列版は**日単位で担当を分割**し、長寿命のworkerプロセスが複数日を処理する。日内で姿勢・軌道を再利用する。既定2 worker、上の例は4 worker。対象日が少なければ実worker数も減る。
- 各workerは利用可能CPUのうち別のCPUへ固定する（worker数が利用可能CPU数を超える場合はCPUを共有）。数値ライブラリは元の変換モジュールと同じ1 thread。nice=19、既定の空きメモリ下限は8 GiB。これはworkerごとのメモリ予約ではなく、システム全体の空きメモリによる開始待機条件。
- 休止は既定0秒。明示した休止があっても、検証済みskipでは休止しない。
- **取得workerは0**。全workerがofflineの支持データ読込みを使う。並列版には取得許可オプションを設けていない。支持データがなければ失敗を記録し、再取得は別途行う。
- `/mnt/j`の小ファイルI/O、SHA256検証、NPZ圧縮、初回の支持データ読込みは残る。4 workerで実所要時間が4分の1になるとは保証しない。まず2–4 workerで使い、I/Oが詰まるなら増やさない。

待機だけ外したい場合は、元の`convert.py`をCtrl+C後に同じコマンドへ `--pause-seconds 0` を加えて再開することもできる。この場合は1 CPUのまま。

## 状態・中断

`last_run.json`と`parallel_last_run.json`に実行範囲・実worker数・取得worker数0・全体count・各workerの最終chunkとcountを記録する。各workerのsnapshotは20 chunkごと・日終了時に更新するため、表示には若干の遅れがある。`rotation_incomplete`はpartialの追加診断数であり、chunk status数へ加算して総chunk数を求めない。

実行設定・worker status・専用SpacePy設定は `/tmp/lepi_gsm_parallel_*/` に置き、そのパスをreportに残す。科学profileには従来の変換コードとパッケージ情報を保持し、並列runnerのSHA256は実行reportへ別に記録する。

親は元の`.run.lock`を保持し、子も同じロックのfile descriptorを継承する。親が強制終了しても子が生きている間は別のrunnerが同じprofileを書けない。通常のCtrl+Cでは親が子を終了・回収してからロックを解放する。強制終了後は手動でロックファイルを削除せず、生存プロセスを確認する。

offline版では追加retry巡を行わない。失敗や未完了があれば終了コード2。再実行時は完成済みを再利用し、失敗分を再試行する。成功した元DSI由来のpartialをcompleteへ読み替えない。

## 検証

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/lepi_parallel_test_matplotlib \
.venv_pyspedas/bin/python -m unittest discover -s lepi_gsm_velocity -p 'test_parallel.py'
```

日単位の非重複分割、逐次版と全配列・manifestの一致、verified再開で出力SHA256が不変、実subprocess同時実行、親がロックを閉じた後も子がロックを保持し終了後に解放することを検証する。3年全体の並列運転と実データでの加速率は未測定。

2026-11-04: core・converter・spatial・parallelを合わせた24テストが成功。実データ2022-09-29 05:05–05:10 UTCを別出力先 `/tmp/lepi_parallel_real_20261104/gsm_observed_6597020f98c9` へ変換し、37レコードの回転行列・3種のGSM流速が過去の検証用出力と完全一致（最大差0）した。選択日が1日なので、この実データ試験は1 workerであり、実データの複数workerでの加速率を測ったものではない。実行中の元profileのconvert.py/core.py SHA256が不変であることも照合した。

同日の姿勢パーサー修正後は28テストが成功。1月27〜31日のraw TXTを検証し、1月29日の5,783行・1月30日の658行で連結数値を分離でき、I軸RA/Decは全レコードで元の数値と一致した。以前failedだった2020-01-28 00:00–00:05 UTCを別出力先 `/tmp/lepi_att_fixed_real_20261104/gsm_observed_6597020f98c9` に変換してcomplete（9レコード、全点回転成功、全3種で有限流速）を確認。回転前後の速さもrtol=atol=1e-12で一致した。fallbackのコードSHA256と修復行数はNPZ manifestのinputsへ記録済み。既存の本番failed stateは短区間試験で変更せず、その後のユーザー実行で再試行する。

停止後の本番stateでは同じ列形式のfailedが1月28日259件・29日220件・30日174件・31日90件、合計743件だった。この集計はその時点の保存済みstateであり、未着手の期間まで同じエラーがないと保証するものではない。
