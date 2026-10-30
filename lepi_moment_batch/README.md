# 月・年単位のLEP-iモーメント計算と保存

KAW解析とは独立した、取得・検証・計算用ランナー。月または年を指定すると、UTCの5分区間を
既定では1ワーカーで処理する。`--compute-workers`で同時実行数を指定できる。
各区間は新しいプロセスで計算し、終了時にメモリとtplot状態を解放する。
未検証の入力は計算に使わない。必要なCDFを自動取得・検証して従来のSPEDASキャッシュへ保存する。
既存rawが破損している場合も、代替CDFの取得・検証が成功してからatomic renameで更新する。

## 実行

ワークスペースのルート `/home/satanka/Documents/observation_workspace` から実行する。
科学ライブラリが入った `.venv_pyspedas/bin/python` を使う。

```bash
# 対象日数・区間数・ローカルLEP-i CDFの存在を確認するだけ。計算・保存は行わない。
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --month 2022-09 --dry-run
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --year 2022 --dry-run

# 1か月を計算する。
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --month 2022-09

# 1年を月別ディレクトリへ保存する。
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --year 2022

# 小区間を試す。端点はUTCの5分境界に合わせる。
.venv_pyspedas/bin/python lepi_moment_batch/batch.py \
  --start 2022-09-29T05:05:00 --end 2022-09-29T05:15:00
```

日付・月・年の境界はUTC。時刻指定に `+09:00` 等のオフセットを付けてもUTCへ変換する。
中断後は同じコマンドで再開できる。`Ctrl+C` は実行中ワーカーを終了し、状態を記録する。
実行中のコードや科学パッケージ版の変更を検出すると停止する。
SIGTERMも中断として扱い、実行状態を記録する。

## 進捗表示

端末ではtqdmの進捗バーを既定で表示する。月・年の要求範囲全体を対象とし、
処理済み5分区間数、割合、経過時間、推定残り時間と、現在のUTC時刻・工程を表示する。
`ok`は全時刻で計算可能なチャンク、`partial`は部分計算とその再利用、
`skip`は検証済みcompleteの再利用、`excluded`は確定した除外とその再利用、
`failed` は失敗の件数。100%は全区間を試行したことを意味し、全区間の計算成功を意味しない。
展開失敗で当日の残りを打ち切る場合や中断時は、未試行区間を進捗に加えない。
初回の取得・展開・再試行で区間の所要時間は変わるため、残り時間は目安になる。

進捗バーはstderrへ出し、既存のstdoutのログ・JSONは維持する。
ログファイルへリダイレクトする場合は既定でバーを非表示にする。

```bash
# 強制表示（端末以外や出力をキャプチャする実行環境でも表示）。
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --month 2022-09 --progress on

# 非表示。
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --month 2022-09 --progress off
```

表示設定を変えるだけでは出力プロファイルは変わらない。
ただし今回のコード更新は、既存のソースハッシュ管理により新しいプロファイルとなり、
以前のプロファイルとは保存先が分かれる。数値計算の条件・処理には変更を加えていない。

## 欠損区間の高速化と旧結果からの再開（2026-10-16）

欠損区間（旧`no_data`・`missing_mgf`）では、ワーカー終了後の`--pause-seconds`の待機を省く。
科学ライブラリは磁場の補間条件を満たしてmoment計算へ進む場合だけ読み込む。
CDFの時刻列とUTC変換結果はscratch内の`*.epochs.npz`へ保存し、元の展開済みCDFの
パス・サイズ・更新時刻と時刻変数名が一致する場合だけ再利用する。scratch清掃時に索引も削除する。
検証済み入力のみを使う条件、日境界の隣接ファイル、重複時刻、磁場の補間・品質条件は維持する。
取得の並列数は`--download-workers`、計算の並列数は`--compute-workers`で指定する。
CPUの割り当てと計算ワーカーごとのメモリ解放を維持する。

ソースハッシュが変わるため、新しい出力プロファイルになる。従来の2022年処理を引き継ぐ例：

```bash
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --year 2022 \
  --reuse-results-from /mnt/j/observation_data/Arase_LEPi_moments/native_dsi_7974a6054c9a
```

`--reuse-results-from`は複数回指定でき、指定順に各チャンクの候補を確認する。
各プロファイルの互換性は起動時に検証し、チャンクがない・未完了・入力が異なる・NPZが破損している
場合は次のプロファイルを試す。最初に条件を満たしたcompleteだけを採用し、どの候補にもなければ再計算する。
指定プロファイル自体の条件が不適合なら起動時に停止する。元のファイルは変更せず、採用元を
新しいstateとNPZ manifestの`reused_from`に記録する。同じパスの重複指定は一度だけ確認する。
新しい結果を優先して2つの既存プロファイルを使う例：

```bash
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --year 2022 \
  --download-workers 8 --compute-workers 4 --pause-seconds 0 \
  --reuse-results-from /mnt/j/observation_data/Arase_LEPi_moments/native_dsi_e0b45f8d3c41 \
  --reuse-results-from /mnt/j/observation_data/Arase_LEPi_moments/native_dsi_7974a6054c9a
```

以前に磁場cadence・最大ギャップなどを指定していた場合は、同じ条件も指定する。
`--reuse-results-from`は旧profileのハッシュ、計算条件、パッケージ版、科学コード、
現在の入力証跡との一致を検証する。旧runnerの差分は今回確認した版に限定する。
検証できた旧`complete`のみ新プロファイルへ引き継ぐ。旧completeに有効時刻マスクがない場合は、
数値配列を保持したまま全時刻有効のマスクと理由0を追加する。旧`no_data`・`missing_mgf`・
`excluded`を含む、それ以外の状態は引き継がず、新しい分類で再判定する。
旧ファイルは変更しない。NPZの数値配列のNPYペイロードをそのまま保持し、manifestを新しい
プロファイルへ更新して別ファイルへ再保存する。旧profile・出力パス・SHA256を来歴として記録する。
引き継ぎにも読み書きとハッシュ検証の時間・追加ディスク容量が必要になる。
新プロファイルで一度引き継いだ結果は通常の再開検証でスキップされる。

今回のbatch.py更新で入力検証の識別子も変わるため、初回はキャッシュCDFの再検証が発生する。
この先行取得・検証の時間は、欠損チャンク自体の高速化とは別にかかる。
外部取得の方針は従来どおりで、`--refresh-downloads`を付けない限り有効な取得証跡を再利用する。

小さな合成CDFを用い各3回比較したワーカー起動から終了までの中央値は、`no_data`で
2.17秒→0.20秒、`missing_mgf`で2.66秒→0.24秒だった。さらに親側の既定2秒待機を省く。
これはローカルの合成データ測定であり、実データの全体処理速度を保証する値ではない。

## ダウンロードの並列数

先行取得は既定で最大8ワーカー。`--download-workers 8`で明示でき、
`--download-workers 1`で従来の直列取得に戻せる。`--offline`では直列で検証済み入力を参照する。

```bash
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --year 2022 \
  --download-workers 8 \
  --reuse-results-from /mnt/j/observation_data/Arase_LEPi_moments/native_dsi_7974a6054c9a
```

並列数はこのジョブ内の取得単位数の上限で、HTTP要求数や他ジョブを含めた上限ではない。
取得単位ごとに独立したプロセスとscratchを使い、HTTPの読み取り・全体タイムアウトを維持する。
親は同時に最大指定数だけを投入し、検証済み取得証跡・ファイルロック・atomic更新を維持する。
CDFの展開・検証・展開済みファイルの清掃は共有ロックで1取得単位ずつ処理するため、
ダウンロードを重ねても8個の日CDFを同時展開する構成にはならない。
計算との同時実行はなく、その月の取得が終わってからmoment計算へ進む。

`Ctrl+C`/SIGTERMでは実行中の取得プロセスも終了させ、一時ファイルとscratchを清掃する。
`--max-network-failures`の連続失敗数は、並列時は親が結果を受け取る順で数える。
閾値へ達すると追加投入を止めて実行中の取得を中断するが、その時点で別の取得が
既に完了している場合は、その取得証跡が残る。

今回もソースハッシュが変わるので新しいプロファイルになる。元のプロファイルに加え、
直前の欠損高速化版の結果も、`--reuse-results-from`へそのディレクトリを指定して検証・再利用できる。
ワーカー数の数値自体は計算プロファイルに含めないため、実装が同じなら並列数を変えても保存先は変わらない。

検証用コマンド（外部通信なし、127.0.0.1のみ）：

```bash
.venv_pyspedas/bin/python lepi_moment_batch/check_parallel_acquisition.py
```

8件の同時GET、CDF検証・共有キャッシュへの保存、メモリ内取得証跡の再利用、中断時の清掃を確認する。

## 粒子時刻ごとの部分計算（2026-10-16）

5分チャンク内の各LEP-i粒子時刻で、元のMGFサンプルによる補間範囲・端点の有限性・
最大ギャップ・補間した磁場の非ゼロ条件を判定する。条件とエネルギー範囲・座標系・
生qualityの保持方針は変更しない。条件を満たした時刻だけ、3イオン種共通のマスクで
公開moment関数を実行する。元の`epoch_tt2000`・`time_unix`は全時刻を保持する。
無効時刻の全48 moment系列と`*__b_dsi_at_moment`はNaNで保存し、元counts・quality・
粒子診断値・元ファイル／レコード番号は無効時刻についても保持する。

追加配列：

| 配列 | 内容 |
|---|---|
| `moment_valid` | 各粒子時刻で磁場条件を満たして計算したかを示すbool。全粒子分布binや全momentの有限性を保証するフラグではない |
| `moment_invalid_reason` | uint8の理由ビット。0は計算可能。複数ビットが立つ場合もある |
| `mgf_bracket_gap_seconds` | 元サンプルによるギャップ秒。範囲外はNaN |

理由ビットは1=磁場の時間範囲外、2=補間端点の非有限値/FILLVAL、4=最大ギャップ超過、
8=補間磁場がゼロ。対応はmanifestの`metadata.moment_selection.invalid_reason_bits`にも保存する。
全時刻有効なら`complete`、一部なら`partial`。全時刻無効ならNPZは作らず、時間範囲不足は
`excluded`。検証と隔離が完了して全時刻が磁場無効値等で計算不能と判定された場合も`excluded`とし、理由別件数を残す。構造不整合や判定不能は`failed_validation`のままとする。
`records_per_species`は元レコード数、`computed_records_per_species`・`invalid_records_per_species`は
それぞれ計算した／できなかった数。`valid_moment_fraction`はレコード数の割合で、滞在時間の割合ではない。
月別summaryにも、保存したcomplete/partial内の元・計算済み・無効レコード数を各種共通の数として保存する。

公開関数へは、元サンプルのTT2000補間端点で有効性を確定し、粒子時刻上に計算した有限磁場を渡す。
有効な粒子時刻が1個だけなら、公開補間器に定数ガード点を追加する。ガード点は観測値として
保存せず、無効と判定した粒子時刻を計算対象へ戻さない。

今回も新しいプロファイルになる。旧`partial`・失敗・除外は`--reuse-results-from`では引き継がず、
新しい規則で再判定する。新規プロファイルで作成したpartialは通常の再開でSHA256を確認して再利用する。
DSI→GSM変換はcomplete/partialを入力に取り、元時刻列と理由マスクを引き継いで無効時刻をNaNに保つ。
DSI入力がpartialという理由だけでは同じ変換を自動再試行しない。

外部通信なしの合成CDF検証：

```bash
.venv_pyspedas/bin/python lepi_moment_batch/check_partial_worker.py
```

降順エネルギー表と正の密度・有限な流速を持つ合成データで、正常時刻の48系列が参照計算と一致すること、
1時刻だけ計算可能な場合、範囲外・FILLVAL・ギャップ・ゼロ磁場の理由、全時刻不能を確認する。

## 計算の並列数と再判定

`--compute-workers N`で、5分チャンクを最大N個の独立プロセスで同時に計算する。既定は1。
呼び出し元のCPU affinityに含まれるCPUから各ワーカーへ異なるCPUを1つずつ割り当て、
NumPy等の内部スレッド数は1に保つ。指定数が利用可能CPU数を超える場合は起動しない。
request/result/logはチャンクごとに分離し、親が入力CDFを一度展開して読み取り専用で共有する。
時刻索引はatomic保存する。日内の全ワーカー終了を待ってから、その日以前のscratchを清掃する。
Ctrl+C/SIGTERMでは実行中の計算を終了させ、結果を回収して状態を記録する。
各プロセスの3 GiB仮想メモリ上限・CPU時間上限・wall timeoutは維持する。
`--pause-seconds`は成功チャンク後の各ワーカー枠の待機。0ならその待機を省く。

```bash
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --year 2022 \
  --download-workers 8 --compute-workers 4 --pause-seconds 0 \
  --reuse-results-from /mnt/j/observation_data/Arase_LEPi_moments/native_dsi_7974a6054c9a
```

このコマンドでは旧`complete`だけを検証して引き継ぎ、旧`no_data`・`missing_mgf`等は再判定する。
新プロファイルで既に確定した除外も再判定したい場合は`--recheck-excluded`を追加する。
実装が同じなら計算ワーカー数・待機時間の変更でプロファイルは変わらない。
取得と計算の段階は分けたままで、ダウンロード数を増やす操作とは独立している。

合成CDFの8チャンクを1・2・4・8並列で比較した所要時間は23.36・12.12・8.63・5.10秒。
全80配列（48 moment系列を含む）が一致し、最大RSSは各ワーカー約225 MiBだった。
これは少数レコードの合成データ試験で、実データの速度・メモリ消費を保証しない。
外部通信なしで検証を繰り返せる：

```bash
.venv_pyspedas/bin/python lepi_moment_batch/check_parallel_compute.py
```

## 保存構成

既定の保存先は `/mnt/j/observation_data/Arase_LEPi_moments`。

```text
Arase_LEPi_moments/
  native_dsi_<計算条件・コードのハッシュ>/
    profile.json
    last_run.json
    2022/
      09/
        summary.json
        data/
          20220929/
            20220929_050500_20220929_051000.npz
            20220929_051000_20220929_051500.npz
        state/
          20220929/
            20220929_050500_20220929_051000.json
        logs/
          20220929/
            20220929_050500_20220929_051000.log
        superseded/                  # 更新・破損等で再計算する前の保存ファイル
  verified_inputs/
    state/YYYY/MM/<granule>.json     # 取得証跡・入力SHA256・CDF検証結果
    receipts/YYYY/MM/<granule>.json # 検証成功の証跡。取得失敗で上書きしない
    rejected/<product>/YYYY/MM/      # 再取得で解決しない検証不整合の調査用CDF
    history/<product>/YYYY/MM/       # 同じ版名で内容が変わった旧入力
  _archive/                         # 以前の計算プロファイル。通常の集計対象外
  _validation/                      # 小規模な検証用の参照データと検証記録
```

年指定でも同じ月別構成になる。1月の全データを一括でメモリに載せる処理はない。
各月の `summary.json` は試行済み区間の状態一覧であり、その月全体の処理完了や観測網羅性を意味しない。
`last_run.json` には直近の要求範囲と、その実行の状態別件数を記録する。
開始時は `running`、終了時は `finished`、例外による停止は `stopped`、中断は `interrupted` を記録する。
停止時も今回の件数・原因を保存する。`running` の件数は開始時の値で、実行中の集計は月別状態を見る。

計算条件、主要ソース、パッケージ版が異なる場合は別のハッシュディレクトリへ分ける。
初回取得では公式サーバーのindexから版番号を数値で比較して選び、版・サイズ・更新時刻・SHA256を保存する。
既に検証済みの入力は同じ版を再利用する。新しい版を調べるには `--refresh-downloads` を指定する。
同じ版の入力が実行中に更新された場合も、ステージ済み入力と要求の整合性を確認する。

## 保存する量と現在の物理仕様

H⁺・He⁺・O⁺を種別に保存し、種間平均や座標変換はここでは作らない。
公開 `erg_lep_part_products` の16出力をすべて保持する。

`density`, `flux`, `mftens`, `velocity`, `ptens`, `ttens`, `vthermal`, `avgtemp`,
`eflux`, `t3`, `magt3`, `symm`, `symm_phi`, `symm_theta`, `symm_ang`, `qflux`

補助量として、TT2000/Unix時刻、sector時刻、energy表と選択チャンネル、raw counts、
raw quality、選択範囲の有限flux率・負fluxビン数、MGFとraw quality、補間の時間幅、
実際にモーメント計算へ供給した磁場、各レコードの入力ファイル・入力レコード番号を保存する。
LEP-i qualityは15 anode分を保存し、ビットの意味を仮定してマスクしない。

現在のエネルギー指定は暫定的な30–30000 eV/q、公開getterのwide anode 8方向を使う。
第0エネルギーチャンネルの無効化、負fluxのゼロ化は公開実装の仕様を引き継ぐ。
出力はネイティブDSI・衛星相対速度で、宇宙機電位は公開実装どおり0 V、衛星速度補正なし。
温度・圧力はテンソルも保存する。6成分テンソル順序は `xx, yy, zz, xy, xz, yz`。
`magt3` の内部磁場基底は地理的に固定したFACと同一視しない。
quality選別、電位補正、sub anode、座標変換、エネルギー下限の確定はPIとの議論事項。

各NPZの `manifest_json` に計算設定、入力・コードの由来、単位・成分順序、全数値配列のSHA256を保存する。
`np.load(..., allow_pickle=False)` で読み込める。

```python
import json
import numpy as np

with np.load(path_to_npz, allow_pickle=False) as saved:
    epoch = saved['epoch_tt2000']
    velocity_h = saved['FPDU__velocity']          # km/s, DSI
    temperature_h = saved['FPDU__ttens']         # eV, (time, 3, 3)
    pressure_h = saved['FPDU__ptens']            # eV/cm^3, (time, 6)
    manifest = json.loads(str(saved['manifest_json']))
```

## 展開・時間境界・再開

元CDF全体のgzip展開は小さなバッファでローカルscratchへ書き出す。
CRC・展開サイズを検証し、展開ファイルのSHA256を保持する。
取得検証では各CDFを一時展開して検証後に削除する。計算段階では日別CDFを展開して
その日の5分区間で再利用し、月全体の展開データは保持しない。
日付境界のスピンが隣の日別CDFにある場合に備えて、最初・最後の区間には隣接日のCDFも確認する。
同一ファイル内の重複TT2000時刻は、必要なflux・counts・quality・sector時刻（MGFなら磁場・quality）が
完全同一の場合だけ受け入れ、最後の元レコードを採用する。内容が異なる重複・時刻逆転は受け入れない。
隣接ファイル間の重複は日付順で後のファイルを優先し、元ファイル番号・元レコード番号を保存する。
前日・当日・翌日の展開キャッシュを必要な期間だけ保持し、不要になった分を順次削除する。
正常終了時は残った展開キャッシュも削除する。中断時の有効キャッシュは次の実行で再利用できる。

既定scratchは `/tmp/lepi_moments_work/<hash>`。入力の圧縮状態と時間境界により、
単一日だけでなく複数の日別CDFを一時的に保持するので、その容量に加えて4 GiBの空き容量を要求する。
出力とscratchのロックにより、同じ設定の多重起動を拒否する。

区間は `[start, end)` で、スピンレコードを丸ごと選択する。
保存は一時ファイルへ行い、全配列を再読み込みで照合してからatomic renameする。
完了済み区間は設定・入力の一致とNPZファイルのSHA256を確認してスキップする。
入力更新や保存ファイル破損で再計算する場合、以前のNPZを `superseded/` へ移して保持する。
科学解析で使えるのは **`complete`と`partial`の有効レコード**。`partial`では必ず
`moment_valid`と各物理量の有限性を確認し、`running`・失敗・旧保存物は含めない。

| 状態 | 意味・再実行 |
|---|---|
| `complete` | 全粒子時刻が磁場条件を満たし、計算・保存・照合完了。条件・入力・NPZ SHA256が一致すればスキップ |
| `partial` | 一部の粒子時刻だけ計算可能。有効時刻のmomentを保存し、無効時刻はNaN。同じ入力・SHA256なら再開時にスキップ |
| `excluded` | 必須ファイルの確認済み不在、検証済みCDF内のサンプル不在、または磁場の時間範囲・ギャップ条件不足。理由と根拠を記録 |
| `failed_download` | 通信・HTTP・転送長の失敗、未検証のoffline入力。確認済み不在とは区別 |
| `failed_authorization` | HTTP 401/403。認証・公開状況を確認後に再実行 |
| `failed_validation` | CDF構造不整合、隔離後に残る時刻逆転など、安全に入力を検証できない場合 |
| `failed_stage` | 展開失敗等。その日の残りを打ち切る |
| `failed` / `timeout` / `interrupted` | 計算・ワーカー管理の失敗、計算時間超過、中断。理由と再試行可否を記録 |

新しいチャンク状態では`no_data`・`missing_mgf`・`remote_absent`を出力しない。
取得単位の`remote_absent`は、公式indexの不在証跡として引き続き保存する。
終了コードは正常完了・`partial`・確定除外のみなら0、未取得・失敗区間があれば2、ロック競合3、中断130。
除外は観測された物理量がゼロであることや、衛星で観測されなかったことの証明ではない。
必要な隣接CDFの取得状況が不明なら、その区間は失敗として残す。

| `failure_class` | 分類 |
|---|---|
| `required_input_missing_remote` | 公式indexで必須LEP-i/MGFファイルの不在を確認。`excluded` |
| `empty_lepi_data` / `empty_mgf_data` | 検証済み入力に必要なサンプルがない。`excluded` |
| `insufficient_mgf_coverage` | 磁場が全粒子時刻を囲まない、または許容補間ギャップ超過。`excluded` |
| `no_valid_mgf_support` | 検証・隔離後、全粒子時刻で磁場補間条件を満たさない。`excluded` |
| `invalid_mgf_schema` | 必要変数不足。`failed_validation` |
| `catalog_timeout` / `download_timeout` | index取得／ファイル取得のタイムアウト。`failed_download` |
| `http_error` / `download_error` / `authorization_error` | HTTP・通信・認証の失敗 |
| `input_validation_error` / `unverified_input` | CDF検証失敗、検証済みoffline入力がない等 |
| `compute_timeout` / `worker_exit` / `calculation_error` | 計算時間超過・プロセス異常終了・計算中の例外 |

`retryable`は自動再試行の可否を表す。追加の自動再試行パスは、従来どおり取得失敗に関係する区間だけを対象とする。
`excluded`は`retryable: false`と`exclusion_basis`を保存し、月別summaryには理由別件数も保存する。
同じプロファイルの除外を再評価するには`--recheck-excluded`を指定する。
これは再計算・再判定を有効にする指定で、キャッシュ済みCDFの版や内容を更新する指定ではない。
新しい版・更新された内容も調べる場合は`--refresh-downloads`を併用する。
公式indexの不在は通常起動でも再照会され、入力証跡が変われば従来の除外を再利用しない。

## 取得の検証と再試行

`SPEDAS_DATA_DIR` を基準に、LEP-iとMGFの元CDFは
`<SPEDAS_DATA_DIR>/ergsc/satellite/erg/<product>/.../YYYY/MM/` に保存する。
環境変数が未指定の場合は `/mnt/j/observation_data/` を使う。`--input-root` はその
`ergsc/satellite/erg` 部分を直接指定する。モーメントNPZ・取得証跡は出力rootで別管理する。
旧 `verified_inputs/erg` の検証済み証跡は共通キャッシュへ移行して再検証できる。

月ごとに必要な日別LEP-i・時別MGFを先にすべて取得・検証し、その後で計算を開始する。
年指定でも各月の取得→計算を順に実行する。取得と計算は同時並行にしない。
取得失敗があっても、その部分を記録して残りの取得・計算を続ける。
要求範囲全体を一巡した後、取得失敗に関係した5分区間だけを再試行する。
既定は追加1巡・開始前30秒待機。`--retry-passes`、`--retry-pass-delay` で変更できる。
成功・確認済み不在・再取得で解決しない検証失敗は、追加巡で再ダウンロードしない。
完了済み区間は再計算しない。追加巡も失敗した部分は状態に残り、同じコマンドで再開できる。
最終件数は区間ごとの最終状態であり、初回失敗と回収成功を二重計上しない。
月別の取得結果は `acquisition_passes/pass_<N>_YYYY_MM.json` に保存する。
取得と計算それぞれにtqdmを表示できる（`--progress on`）。
既定では最大3回、再試行間隔30秒、HTTP読み取り60秒、転送全体600秒、CDF検証180秒で打ち切る。
少量ずつ送られ続ける応答もwall timeoutで打ち切る。
認証失敗は即座に失敗にし、404は**ディレクトリ**への応答の場合だけ不在の証拠とする。
indexに載っているファイルへの404、login画面、切れたindexは取得失敗とする。
失敗は同じファイルの全5分区間で再試行しないが、**次の日・時刻のファイルは独立に取得を試す**。
ディレクトリの失敗を月全体へ適用するキャッシュは廃止した。取得済みの正常indexは、
通常のファイル取得失敗では破棄せず、indexに載ったファイルが404/410の場合だけ照会し直す。
既定では通信失敗の連続を理由に全体停止しない（`--max-network-failures 0`）。
必要なら正の値で連続失敗時の停止を有効にできる。その場合も取得結果を保存する。
検証成功の証跡は `receipts/` に別途保持する。後の再取得が失敗しても消さず、
次の通常起動・offline起動では、SHA256が一致する検証済みの版を再利用できる。
旧実装で証跡が失われたファイルは、再度サーバー照合・検証が必要になる。

CDFは一時 `.part` に取得し、HTTP 200・正のContent-Length・転送バイト数を確認する。
同じ版の既存raw CDFを使う場合も、サーバーHEADとのサイズ一致を確認し、一時ファイルへコピーして
下記の完全読み取り検証を行う。既存rawの変更をコピー前後で検出する。
失敗後の再試行と `--refresh-downloads` は新しいHTTP GETを使う。

gzipのCRC・宣言された展開長、CDFの論理EOFとファイル長、TT2000時刻の順序（完全同一の重複を許可）と日付、
3種のflux・counts・quality・固定energy表、sector時刻、MGFの磁場・quality・レコード数を検証する。
必要変数の全レコードを1024レコードずつ読み、末尾の切断も検出する。
検証プロセスの終了コード、要求URL・HTTPメソッド・工程・時間制限も記録する。
内容が異なる重複・時刻逆転・日付範囲不整合などの再取得で解決しない問題は再ダウンロードを止め、
転送長を確認済みの元CDFを `verified_inputs/rejected/` に保存して調査に回す。計算には使わない。
レコードに空白時間やfill値があること自体は取得失敗にはしない。全日分の時刻があることは要求しない。
構造検証はメモリ3 GiB・CPU120秒の別プロセスで行う。
日付との整合性はLEP-iを前後64秒、MGFを前後32秒の範囲で確認する。
LEP-iのEpochは区間開始時刻であり、2020-06-01の先頭は前日23:59:23.596だった。
旧32秒の判定はこの正常な隣接日コンテキストを全日拒否していた。
`--lepi-boundary-tolerance-seconds` で許容幅を変更でき、計算プロファイルに記録する。
元時刻は移動せず、解析区間の `[start,end)` による選択は維持する。

検証成功後だけatomic renameで公開し、SHA256と検証結果を保存する。
取り直した内容が同一なら入力指紋を維持し、完了区間を再計算しない。
同じ版名で内容が変わった旧入力は `verified_inputs/history/` へコピー保存し、
検証成功後に共通キャッシュをatomic renameで更新する。開いている旧ファイルを直接書き換えない。
KAWが同時に同じrawファイルを更新する場合は、その指紋の変化を検出して当該計算を止める。
再起動時は入力ファイル全体のSHA256を再確認する。
検証コードの変更時も、以前の検証済みCDFのSHA256が一致すればローカルで再検証する。
そのための再取得は不要で、`--offline` でも再検証できる。同一実行中は指紋の変化も確認する。計算開始・保存直前にも確認し、
別プロファイルの再取得等で入力が途中変更された場合は結果を公開しない。
保存後に壊れたキャッシュや未検証入力は使用せず、再取得する。
旧`no_data`・`missing_mgf`は再開・旧結果の引き継ぎでスキップしない。
新しい`excluded`は、分類版・除外根拠・入力証跡が一致する場合だけ通常の再開でスキップする。

```bash
# サーバーの最新版本と取得内容を確認し直す。
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --month 2022-09 --refresh-downloads

# 通信を使わず、既に検証済みの共通キャッシュだけで計算する。
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --month 2022-09 --offline

# 通信条件を変更する例。
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --month 2022-09 \
  --download-attempts 4 --retry-delay 60 --http-timeout 60
```

`--offline` は未検証のraw CDFを信用せず、サーバー上の不在も再確認できないため失敗として残す。
`--offline` と `--refresh-downloads` は併用できない。
公式のリモートSHA256との照合は行っていない。既存rawの再利用では、同一サイズの別内容まで
サーバーの内容と同一と証明できない。サーバーから取り直すには `--refresh-downloads` を使う。
較正や物理的妥当性、科学的なquality選別はこの取得検証とは別に扱う。

## KAW解析との資源競合を抑える設定

CPU affinityを1論理CPU、nice=19、idle I/O priority、BLAS等を1スレッドに固定する。
ワーカーの仮想メモリ上限は3 GiB、CPU時間上限は120秒、wall timeoutは既定180秒。
使用可能メモリが8 GiB未満なら新しい処理の開始を待つ。
区間間には既定2秒の休止を置く。これらは既存KAWプロセスの設定を変えない。

```bash
# より休止を長くし、開始時のメモリ余裕も増やす例。
.venv_pyspedas/bin/python lepi_moment_batch/batch.py --month 2022-09 \
  --pause-seconds 10 --min-free-memory-gib 12
```

競合がゼロになる保証はない。特に入力・保存先が同じJドライブなのでI/Oは共有する。
低優先度でもKAW解析が増えれば待機・実行時間は変わる。

MGFは既定64 Hz DSI。隣接時・日ファイルを結合し、粒子時刻を挟む磁場だけで補間する。
既定の許容ギャップは1秒。8秒データを使う場合は `--mgf-cadence 8sec` と指定し、
その場合の許容ギャップは16秒になる。変更には `--max-mgf-gap-seconds` を使う。
これらの補間制限は暫定仕様として設定に記録し、異なる値は別プロファイルへ分ける。
qualityビットによる除外は行わず、raw qualityを保存する。

## 検証と現在の限界

2026-10-07の取得実装の失敗注入テスト21件とloopback HTTP/CDFの統合テストが通った。
公式サーバーからMGF実CDF（約5 MiB）をGETし、LEP-i実CDF全レコードの構造検証、
実データ5分の計算・保存、再開時のスキップ、独立試作76配列との完全一致を確認した。
実CDFのLEP-i部分は既存rawのコピーをサーバーHEADのサイズ照合と完全読み取りで検証した。
実取得の検証記録は保存先の `_validation/acquisition_verification.json` にある。
`check_real_acquisition.py` はこの小規模検証を再実行する（MGFの外部ダウンロードを含む）。

2026-10-07に実データの5分×2区間で、月別保存・各16モーメント・保存配列の照合・
再実行のスキップを検証した。5分区間は約4–5秒、最大RSSは約446 MiBだった。
既存の独立試作で計算し直した5分区間と全76数値配列が完全一致した。
初回の日別展開時間はこれとは別に必要。

テストは以下で実行できる。

```bash
.venv_pyspedas/bin/python -m unittest discover -s lepi_moment_batch -p 'test_*.py'
ionice -c 3 .venv_pyspedas/bin/python lepi_moment_batch/check_worker_edges.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ionice -c 3 nice -n 19 \
  .venv_pyspedas/bin/python lepi_moment_batch/check_missing_states.py
```

うるう年・月年境界・UTC変換・入力更新時の再開判定・保存ファイル破損検知・実行中コード変更検知をテストした。
小さな既知入力で磁場結合、重複時刻、欠測をまたぐ補間の拒否、非有限値の拒否、外挿の拒否も確認した。
HTTP/CDFの統合検証はloopbackサーバーと小さな完全CDF fixtureを使う。
通信失敗、認証失敗、途中切断、CDF変数欠落、破損キャッシュ、不在再照会、no_data・再開・timeoutを検証する。
月・年指定の計画作成も検証した。月・年の実運用で失敗が多かったため、
以下の取得・再試行方式へ調整している。全期間の科学的な網羅検証は未完了。

2022年のローカルLEP-i CDFを確認した時点では27日分だった（すべて9月）。
長期間実行前に `--dry-run` で現在の取得状況を確認する。
自動取得は上記の工程で行う。`--dry-run` は通信・検証・計算を行わず、ローカル存在状況だけを示す。
現段階では固定energy表・32×8×16幾何・3種そろった入力を想定し、異なる形式は失敗として残す。
8秒磁場の変数名は実CDFで確認したが、実データモーメントの比較は64 Hzの経路で行った。
物理的精度、較正の長期変化、品質ビットの解釈、多日・多モードの網羅検証は別途必要。

### 2026-10-08の調整

実行中の2020年batchを停止し、旧出力を保持した。2022-09-02のCDFでは5296レコード中に
1組の重複があり、3種の必要flux・counts・qualityとsector時刻まで完全同一と確認した。
月全体への失敗波及を廃止し、継続的な通信障害は停止、同一レコード重複は正しく扱うよう変更した。
スキーマとコードハッシュが変わるため、旧結果と新結果は別プロファイルへ保存する。
取得済みの正しいCDFは再検証して再利用する。年単位の処理は自動再開しない。

日付境界の追加確認：2020-06-01は先頭1レコードだけが旧32秒の範囲外（前日23:59:23.596）だった。
全4718レコードの必要変数を検証し、新しい64秒のファイル日付整合性判定で通ることを確認した。
検証成功の証跡を最新の取得試行状態から分離し、失敗したrefreshで正しいキャッシュの証跡が失われる問題も修正した。

### 共通キャッシュ・先行取得への変更

2026-10-08に、停止されていた2022-09のbatchを正常終了し、取得方式をv4へ変更した。
元CDFはSPEDAS共通キャッシュへ統合する。旧モーメントの固有結果は削除せず `_archive/` にまとめる。
共通キャッシュとSHA256が一致する重複CDF、および再生成できる専用scratchは削除する。
詳細は `_validation/cache_organization_20261008.json` に記録する。

31件のテスト、ワーカー境界条件、loopback HTTP/CDF統合検証が通った。
統合検証では、一時503で初回取得に失敗した入力を一巡後に回収し、最終件数を成功へ更新した。
共通保存先への公開、旧検証済みキャッシュのoffline移行、検証証跡の保持も確認した。
既定の連続失敗時停止を無効にし、要求全体の一巡後に追加1巡する方式へ置き換えた。
任意の `--max-network-failures 3` による停止も引き続き検証している。
旧出力はコードハッシュが異なるため、現行プロファイルの完了扱いへ自動移行しない。

共通キャッシュ移行後の実データ5分（2022-09-29 05:05–05:10 UTC）でも、
独立試作の76数値配列と完全一致した。記録は `_validation/shared_cache_verification_20261008.json`。
2022-09の月次処理を修正版でバックグラウンド再開し、PID・コマンド・進捗ログの所在を
現行プロファイルの `2022/09/batch_process.json` に保存した。2020年の通年処理は再開していない。

## GSM観測流速の別処理

DSIモーメントを保持してGSM観測流速を追加する後処理は
[lepi_gsm_velocity](../lepi_gsm_velocity/README.md) を使う。衛星速度補正は行わない。
既存DSIのbatchとは独立して実行し、変換結果は別rootへ保存する。


## MGF異常レコードの隔離と時刻精度（2026-10-21）

MGFのファイル構造・長さ・埋込チェックサム・必須変数・全レコード読み出しの検証は維持する。
公称時間範囲±32秒の外（NaT含む）のレコードは索引で隔離し、残りに時刻逆転があれば検証失敗。
LEP-iの入力検証方針は変更しない。元CDFには変更を加えない。

MGFの同一TT2000時刻に異なる磁場値またはqualityがあれば、その時刻群を曖昧な点として扱う。
同じ値の重複だけを統合し、矛盾する点への一致・その点を端点とする補間は無効にする。
異常点を削除して前後をつなぐことはしない。ファイル間の重複にもこの規則を適用する。
範囲外レコードを途中で隔離した箇所をまたぐ補間も無効にする。最大補間ギャップの条件は維持する。

粒子時刻での磁場は整数TT2000で補間端点を選び、整数の時間差から係数を計算する。
近接するMGF時刻が絶対Unix秒のfloat64で同じ値になっても、元の2点を保持して処理できる。
公開moment関数には有効な粒子時刻上の補間済み磁場を渡す。

`moment_invalid_reason`に16=`conflicting_mgf_epoch`、32=`quarantined_mgf_interval`を追加。
元の粒子時刻・counts・qualityを保持し、無効時刻のmomentをNaNにする規則は同じ。
NPZに`mgf_source_record_index`、`mgf_segment_index`、`mgf_conflicting_epoch`、
`mgf_b_dsi_at_particle`を保存し、manifestに隔離した元レコード番号・矛盾時刻を記録する。
検証済みreceiptにも入力全体の隔離索引を保存する。

全時刻有効は`complete`、一部有効は`partial`、全時刻無効と判定できた場合はNPZを作らず
`excluded`（通常再開ではスキップ、`--recheck-excluded`で再判定）。破損・構造不整合・
隔離後の時刻逆転などで判定できない場合は`failed_validation`、計算例外・timeoutも従来どおり失敗。
磁場有効性の判定であり、全moment量の有限性や未適用のquality条件まで保証するものではない。

処理方針・ソースハッシュが変わるため新規プロファイルになる。既存出力は変更しない。
旧方針のプロファイルは今回の`--reuse-results-from`互換対象に含めない。
rawデータは共有キャッシュから利用できるが、新しい検証IDでローカル再検証が必要になる。

ローカル合成CDFによる回帰テスト（外部通信・年次再実行なし）：

```bash
SPACEPY=/tmp/lepi_test_spacepy MPLCONFIGDIR=/tmp/lepi_test_mpl \
  .venv_pyspedas/bin/python -m unittest discover -s lepi_moment_batch -p 'test_*.py'
SPACEPY=/tmp/lepi_test_spacepy MPLCONFIGDIR=/tmp/lepi_test_mpl \
  .venv_pyspedas/bin/python lepi_moment_batch/check_partial_worker.py
```


## 失敗694区間だけのオフライン回収（2026-10-21）

2022年の監査で抽出した42区間（計算時刻精度）と652区間（MGFレコード検証）を、
`--recovery-plan`の固定リストで選択できる。11区間の旧invalid_mgf_fieldは今回の対象外。
月・年・startとは排他的な指定で、`--offline`が必須。計算条件が元プロファイルと異なる場合は停止する。

```bash
.venv_pyspedas/bin/python lepi_moment_batch/batch.py \
  --recovery-plan /mnt/j/observation_data/Arase_LEPi_moments/plans/2022_recovery.json \
  --offline --compute-workers 4 --pause-seconds 0
```

`--dry-run`を追加すれば対象数と出力先の確認だけを行う。同じコマンドで中断後に再開できる。
対象外の5分区間は計算・入力prefetchしない。リストには216個のローカルCDF（旧拒否CDF47個を含む）と、
当時の公式indexで欠落確認された7入力単位の証跡を固定した。ローカルCDFは実行時にSHA256と新規則で検証する。
不足・不一致・検証失敗は失敗として残し、サーバ照会やダウンロードには切り替えない。
indexの欠落証跡は過去の判定を利用するもので、現在のサーバ状況を再確認するものではない。

出力は別プロファイルで、`recovery_plan.json`とlast_runの対象件数・plan SHA256で選択範囲を記録する。
保存先は既定の`/mnt/j/observation_data/Arase_LEPi_moments/native_dsi_1d86398b65a1/`。
2022_current方式は廃止し、旧3プロファイルの実体は`_archive/20261021_profiles/`へ移動した。
ルート直下のnative_dsi_*は現行プロファイルのみ。移動対応表は`_archive/migrations/20261021_layout.json`。
旧state/manifestの絶対パスは履歴として保持しているので、旧パスをたどる場合は対応表を参照する。

年指定はprofile hashに含まれない。同じコード・科学条件・ライブラリ版なら、
`--year 2021`でも同じ現行プロファイルの`2021/MM/`以下へ保存する。

```bash
.venv_pyspedas/bin/python lepi_moment_batch/batch.py \
  --year 2021 --download-workers 8 --compute-workers 4 --pause-seconds 0
```

この通常実行例は必要な入力の取得を許可する。2022年694区間の回収コマンドは上記の`--offline`を使用する。
年次の既存2022結果は`_archive/20261021_profiles/native_dsi_eef86189baf8/2022/`に保持している。
今回回収する694区間以外は自動コピーしない。異なる処理方針の製品を同じprofileとして混ぜず、
現行プロファイルの`2022/dataset.json`に旧年次結果と新規回収分の構成を記録した。


## ダウンロード済み入力での通年再計算

2022年通年の固定入力リスト`plans/2022_full_local.json`を用意した。
105,120区間、6,955個のローカルCDF、2,174件の過去の欠落証跡を含む。
694区間限定の`2022_recovery.json`とは別リストで、archiveのmoment製品はコピー・再利用しない。

```bash
.venv_pyspedas/bin/python lepi_moment_batch/batch.py \
  --recovery-plan /mnt/j/observation_data/Arase_LEPi_moments/plans/2022_full_local.json \
  --offline --compute-workers 4 --pause-seconds 0
```

保存先は現行`native_dsi_1d86398b65a1/2022/MM/`。実計算は起動せずdry-runのみ確認した。
同じコマンドで中断後に再開できる。`--recovery-plan`というオプション名だが、
このファイルの選択範囲は2022年全体。単純な`--year 2022 --offline`では旧拒否CDFや欠落証跡を十分に引き継げないため、固定リストを指定する。


## プロダクトを指定した入力更新

更新取得専用コマンド（moment計算は起動しない）：

```bash
.venv_pyspedas/bin/python lepi_moment_batch/refresh_inputs.py \
  --year 2022 --products mgf64hz --download-workers 8
```

`--products`は`lepi`、`mgf64hz`、`mgf8sec`を複数指定できる。
`--month 2022-09`または`--start ... --end ...`にも対応する。
指定したプロダクトだけ公式indexの最新versionを選び直して再取得する。
補間のために必要な隣接グラニュールも含む。`--dry-run`なら計画表示だけで通信・書き込みをしない。
同じバイト列なら既存キャッシュのfingerprintを保持し、同名で内容が変われば旧データをhistoryへ保存してから更新する。

更新後は入力リストをローカル記録から作り直す（通信・moment計算なし）：

```bash
.venv_pyspedas/bin/python lepi_moment_batch/prepare_local_plan.py --year 2022
```

既定で`plans/2022_full_local.json`を置き換える。未確認・不一致の入力があれば旧リストを上書きせず停止する。
欠落確認済みは当時のindex証跡を利用し、未取得・タイムアウトは欠落確定に変更しない。
CDFの完全なSHA256・スキーマ検証は計算時に行う。更新と計算は別操作で順に実行する。
その後、上記の通年オフライン計算コマンドを実行する。
同じ現行profileで完了済みの区間は入力signature/SHA256で再開判定され、入力が変わった区間は再計算される。
