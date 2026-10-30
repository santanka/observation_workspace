> 保存先を更新した。最新の構成・コマンドはlayout_migration.mdを参照。この文書は更新前の準備記録。

# 694区間のオフライン再計算準備と保存先整理

2026-10-21。再計算自体は未実行。新たなサーバ照会・ダウンロードはしていない。

監査の42計算例外と652入力検証失敗だけを固定リストにした（694個の5分区間、51日）。
旧invalid_mgf_fieldの11区間は含めない。
216個のローカルCDFが存在し、計画作成時にサイズ・mtime・パスを確認した。うち47個は旧規則で拒否されたMGF CDF。
過去の公式indexで欠落確認された7入力単位は、当時の欠落証跡を固定した。
CDFのSHA256と新方針での完全検証は実行時に行う。検証結果は出力root内の
verified_inputs/recovery_validations/<plan SHA256>/へ保存し、旧receiptは更新しない。
不足・不一致・判定不能を欠落確定には変更せず、通信へも切り替えない。

ワークスペースから実行：

```bash
.venv_pyspedas/bin/python lepi_moment_batch/batch.py \
  --recovery-plan /mnt/j/observation_data/Arase_LEPi_moments/2022_current/rerun_plan.json \
  --output-root /mnt/j/observation_data/Arase_LEPi_moments/2022_current/recovery_runs \
  --offline --compute-workers 4 --pause-seconds 0
```

同じコマンドで再開できる。`--dry-run`なら対象数・保存先の確認だけ。dry-run結果はplan_dry_run.json。
進捗の母数は694で、対象外の区間は入力prefetchも計算もしない。
計算条件は基準profileと照合し、選択リストとSHA256を出力のrecovery_plan.json・last_run.jsonに保存する。

入口は `/mnt/j/observation_data/Arase_LEPi_moments/LATEST_2022/`（2022_currentへのリンク）。

- baseline: native_dsi_eef86189baf8。最新の年次基準結果。
- recovery: recovery_runs/native_dsi_1d86398b65a1。今回の694区間だけの追加回収先。計算は未実行。
- history/01_original: native_dsi_7974a6054c9a。
- history/02_whole_chunk: native_dsi_e0b45f8d3c41。
- dataset.json: 基準・追加回収の構成と採用規則。

既存の出力本体・元パスは維持し、閲覧用のリンクとREADME・索引を追加した。
回収結果だけを年次結果として使わない。対象694区間では回収側のcomplete/partial/excludedを採用し、
未処理・失敗なら未回収として残す。他の区間はbaselineを使用する。個々の製品manifestは維持する。

49テスト成功。合成CDFによる実際のスケジューラ試験で、離れた2区間だけを処理し、中間区間を処理しないこと、
通信が呼ばれないこと、ローカルCDF不足・SHA不一致が失敗のまま残ること、再開・検証記録保存を確認した。
本番の694区間や年次全体の再計算は行っていない。
