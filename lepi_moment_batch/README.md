# 月・年単位のLEP-iモーメント計算と保存

KAW解析とは独立した、取得・検証・計算用ランナー。月または年を指定すると、UTCの5分区間を
1ワーカーずつ順次処理する。各区間は新しいプロセスで計算し、終了時にメモリとtplot状態を解放する。
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
`ok` は計算完了、`skip` は検証済み結果の再利用、`empty` は `no_data` と確認済みファイル不在、
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
科学解析で使うのは **状態が `complete` の区間** とする。`running`、失敗、旧保存物は含めない。

| 状態 | 意味・再実行 |
|---|---|
| `complete` | 計算・保存・照合完了。互換性とファイル整合性が通ればスキップ |
| `no_data` | 検証済みLEP-i CDFに区間内のスピン時刻がない。入力と取得証跡が一致すればスキップ |
| `remote_absent` | 正常なサーバーindexに対象ファイルがない、またはディレクトリが404。次の起動で再照会 |
| `failed_download` | 通信・HTTP・転送長の失敗、未検証のoffline入力。再実行で再取得 |
| `failed_authorization` | HTTP 401/403。欠測にはしない。認証・公開状況を確認後に再実行 |
| `failed_validation` | CDF構造・必要変数・時刻等の検証失敗。再試行では新たにダウンロード |
| `missing_mgf` | 磁場不足、非有限値、補間ギャップ超過等。再実行で再確認 |
| `failed_stage` | 展開失敗等。同じ日を何百回も試さず、その日の残りを打ち切る |
| `failed` / `timeout` / `interrupted` | 未完了。再実行で再計算 |

終了コードは正常完了0、未取得・失敗区間があれば2、ロック競合3、中断130。
`no_data` はCDF内の時刻不在を意味し、プラズマが存在しないことを意味しない。
必要な隣接CDFの取得状況が不明なら、その区間は失敗として残す。
確認済みのサーバー上の不在は証跡を保存するが、衛星で観測されなかったことの証明ではない。

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
`no_data` の再開には検証済み入力の証跡が必要で、以前の未検証 `no_data` は引き継がない。

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
