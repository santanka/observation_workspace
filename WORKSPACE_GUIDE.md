# Workspaceの案内と整理判断

2026-10-28のローカル調査。コード・入力・結果は移動も削除もしていない。
`observation_workspace.code-workspace` ではPython環境とキャッシュをExplorerから非表示にした。
再表示する場合は同ファイルの `settings.files.exclude` の該当項目を削除する。
これは表示設定であり、ファイルは引き続き利用できる。

## 通常使う入口

| 用途 | 入口・説明 | 補足 |
|---|---|---|
| Arase月別KAW解析 | [batchの説明](statistical_analysis_arase_pre_auto/README.md) | `batch_arase.py` |
| 期間統合統計・作図 | [統一入口の説明](statistical_analysis_arase_pre_auto/STATISTICS.md) | `run_statistics.py`。個別のplot/prepare scriptも内部から使用する |
| LEP-i部分モーメントのbatch | [batchの説明](lepi_moment_batch/README.md) | `batch.py`。入力準備・更新・復旧は同ディレクトリの別script |
| DSI→GSM観測流速 | [変換の説明](lepi_gsm_velocity/README.md) | 元のDSIモーメントとは別の処理 |
| LEP-e温度・密度比較 | [比較の説明](statistical_analysis_arase_pre_auto/lepe_temperature_comparison/README.md) | KAW batchとは独立 |
| LEP-i部分モーメントの検証 | [検証の説明](lepi_partial_moment_validation/README.md) | 合成データ、エネルギー境界別の結果 |
| 個別イベント・波動解析 | `KAW_observation/` | Arase/THEMISのNotebook群 |
| イベント前処理 | `statistical_analysis_arase/` | 月別前処理と解析Notebook |
| 共通処理 | `module_handmade/` | Notebook等から読み込むライブラリ |
| 外部磁場・座標変換ライブラリ | `geopack-vectorize/` | 別Git repository。現在の環境にeditable installされている |
| その他 | `20180922_Event/`, `TRACERS_observation/`, `tutorial/` | イベント解析、TRACERS試行、講習資料 |

## ファイル数と容量

以下は `/home/satanka/Documents/observation_workspace` 自体の値。
VS Codeに追加されている `/mnt/j/observation_data` と `/mnt/j/KAW_observation` は含まない。
容量は `du` による概数、ファイル数はsymlinkを除く通常ファイル数。

| 領域 | 容量 | 通常ファイル数 | 判断 |
|---|---:|---:|---|
| workspace全体 | 約2.9 GiB | 38,603 | 環境とGit履歴が大半 |
| `.venv_pyspedas/` | 約2.2 GiB | 36,168 | 現行の解析環境として保持 |
| `geopack-vectorize/` | 約559 MiB | 198 | うちGit履歴が約551 MiB。依存先なので保持 |
| workspaceの `.git/` | 約91 MiB | 1,773 | 履歴として保持 |
| `lepi_moment_batch/` | 約24 MiB | 80 | 大半は取得・復旧監査の記録 |
| `lepi_partial_moment_validation/` | 約20 MiB | 154 | 大半は検証結果 |
| `KAW_observation/` | 約9.9 MiB | 37 | Notebook内の図を含む |
| `statistical_analysis_arase_pre_auto/` | 約2.0 MiB | 119 | 本番解析、作図、共通処理、テスト、説明 |

この案内を追加する前の集計では、Python環境、全階層の `.git`、`__pycache__` を除くと441ファイル。
内訳はPython 147、Notebook 54、Markdown 41など。
この範囲の `.py`、`.md`、`.ipynb` をSHA256で比較したところ、空ファイル以外の完全一致はなかった。
これは意味や機能が重複していないこと、全ファイルが必要であることの証明ではない。

## 現時点で分かった整理候補と保持理由

- `__pycache__` は環境外で126ファイル、合計約2.7 MiB。再生成可能で、削除候補にはなるが容量効果は小さい。今回の変更では非表示にした。
- `.mplconfig` のフォントキャッシュは約1.1 MiB。再生成可能だが、今回の変更では非表示のみ。
- `lepi_moment_batch/audits/` は約24 MiB。古い実行記録を将来アーカイブする候補。ただし取得確認・失敗診断・復旧の証跡であり、一時ファイルとは扱わない。
- `lepi_partial_moment_validation/outputs/` は約19 MiB。エネルギー条件の異なる独立した検証結果なので、名前の類似だけでは削除しない。
- `lepi_moment_pilot/` は名前に反して本番依存先。`lepi_moment_batch/worker.py` が `pilot` をimportし、`batch.py` の由来管理にも含めている。移動・削除には依存関係とprofileへの影響確認が必要。
- `geopack-vectorize/` は仮想環境のeditable installと `module_handmade/tracers_conjugacy.py`、`tracers_t04_worker.py` の参照先。ディレクトリ単位での移動・削除は不可。
- 統計scriptは `run_statistics.py` を通常入口として使う。入口を統一しても内部実装のscriptは必要なので、個別scriptを一括削除しない。
- Git statusには変更済み25ファイル、未追跡75ファイルがあった。未追跡には新しい解析コード・テスト・説明・監査記録が含まれる。`git clean` による一括掃除は適さない。

## データ・結果領域の扱い

workspaceファイルには `/mnt/j/observation_data` と `/mnt/j/KAW_observation` も登録されている。
前者は生データ、解析用中間データ、モーメント等、後者は図・イベント別出力を含む。
本調査ではこれらの必要性・重複をファイル単位では判定していない。
PDF/PNGの組、`tmp` を含む名前、旧profileであることだけでは不要と判定できない。

また、コードはworkspaceに登録されていない `/mnt/j/statistical_analysis_arase/` の
解析結果・run stateも参照する。表示上の3領域だけで全依存先が閉じているわけではない。
生データ、manifest、run state、SHA256、profile、lockを削除する整理は別途確認が必要。

物理的にまとめる場合は、対象一覧と元パス・移動先・SHA256を記録し、依存先と再開判定を確認してから行う。
ファイル数の削減と、再計算・再取得コストの削減は別の目的として判断する。
