# 年によらない現行プロファイルと旧結果の移動（2026-10-21）

現行保存先は `/mnt/j/observation_data/Arase_LEPi_moments/native_dsi_1d86398b65a1/`。
直下のnative_dsi_*はこの1個だけ。2022_current/LATEST_2022方式は廃止した。
現行native内の2022/MM、2021/MMという同一階層へ各年の結果を保存する。
コード・科学条件・ライブラリ版が同じなら、年指定によってprofile hashは変わらない。

旧native_dsi_7974a6054c9a、native_dsi_e0b45f8d3c41、native_dsi_eef86189baf8の
実体を `_archive/20261021_profiles/` へディレクトリ単位で移動した。
既存_archive内に同名プロファイルがあったため、それらを上書きせず、日付付きの置き場を作った。
旧年次2022結果は移動したeef86189baf8内に保持し、今回の694区間だけを現行nativeへ回収する。
異なる計算方針の製品本体を同一profileとしてコピー・改変する処理は行っていない。
構成は現行nativeの2022/dataset.jsonに記録した。

移動前に各プロファイルの実行ロックが空いていることを確認した。
DrvFSでは開いたファイルハンドルが親ディレクトリのrenameを妨げることを、独立した一時ディレクトリで確認し、
ロック確認用ハンドルを閉じてから移動した。
移動前後のprofile.jsonおよび既存last_run.jsonのSHA256一致を確認した。全NPZの再ハッシュは行っていない。
移動対応表は `_archive/migrations/20261021_layout.json`。
旧state/manifest内の絶対パスは当時の来歴として保持し、新旧パスを対応表で追跡できる。

2022年の再実行計画は `plans/2022_recovery.json` へ配置し、source_profileと694個のstate_pathを
移動先に更新した。元のパスもoriginal_*フィールドに保持し、plan SHA256を更新した。
入力CDFおよびverified_inputsは移動していない。

```bash
.venv_pyspedas/bin/python lepi_moment_batch/batch.py \
  --recovery-plan /mnt/j/observation_data/Arase_LEPi_moments/plans/2022_recovery.json \
  --offline --compute-workers 4 --pause-seconds 0
```

2021年の通常実行（必要な入力の取得を許可）：

```bash
.venv_pyspedas/bin/python lepi_moment_batch/batch.py \
  --year 2021 --download-workers 8 --compute-workers 4 --pause-seconds 0
```

両コマンドをdry-runだけで確認し、同じ現行nativeが出力先になることを確認。
2022年の対象は694区間のまま。移動後の694状態ファイル参照、現行ソースハッシュ、plan自己ハッシュも確認した。
実計算・ダウンロード・サーバ照会は行っていない。
