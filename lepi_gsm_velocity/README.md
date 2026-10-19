# LEP-i GSM観測流速の後処理

既存DSIモーメントNPZを変更せず、H⁺・He⁺・O⁺のGSM観測流速を別NPZへ保存する。
DSI→J2000→GSMという[ERG-SCの変換経路](https://ergsc.isee.nagoya-u.ac.jp/assets/howto/ERG_Coordinate_System_202004.pdf)を使う。
**GSMへ回転した衛星相対流速**であり、衛星速度の加減算は行わない。温度・圧力の変換も行わない。
実行はユーザー側で行う。月・年の長期間処理を自動起動しない。

## 実行

ワークスペース `/home/satanka/Documents/observation_workspace` で、既存DSIプロファイルを明示する。

```bash
# 現在のcomplete区間数を確認するだけ。ダウンロード・変換はしない。
.venv_pyspedas/bin/python lepi_gsm_velocity/convert.py \
  --moments-root /mnt/j/observation_data/Arase_LEPi_moments/native_dsi_7974a6054c9a \
  --month 2022-09 --dry-run

# 月単位。姿勢・軌道を取得して変換する。
.venv_pyspedas/bin/python lepi_gsm_velocity/convert.py \
  --moments-root /mnt/j/observation_data/Arase_LEPi_moments/native_dsi_7974a6054c9a \
  --month 2022-09

# 年単位は --year 2022。短区間は --start と --end を使う。
.venv_pyspedas/bin/python lepi_gsm_velocity/convert.py \
  --moments-root /mnt/j/observation_data/Arase_LEPi_moments/native_dsi_7974a6054c9a \
  --start 2022-09-29T05:05 --end 2022-09-29T05:10 --offline
```

対象は起動時に存在する **verified complete/partialのDSI区間**。選択範囲に丸ごと入る5分区間だけを処理する。
DSIのbatchが進行中でも実行できるが、その後できた区間は次の同じコマンドで取り込む。
DSIのrunning・excluded・失敗区間は対象にしない。partialの有効レコードを変換し、
元の全時刻列、`moment_valid`、`moment_invalid_reason`を保持する。無効時刻のGSM流速はNaN。
旧completeにマスクがなければ全時刻有効とみなすが、partial入力のマスク欠落や状態との不整合は拒否する。
入力DSIはstateとNPZのSHA256・profile・座標表記を確認する。

既定出力は `/mnt/j/observation_data/Arase_LEPi_GSM_velocity/gsm_observed_<hash>/YYYY/MM/`。
`data/YYYYMMDD/<5分区間>.npz` と `state/YYYYMMDD/<5分区間>.json` を作る。
profileはDSI出力の由来、変換コード、パッケージ版、補間閾値などを含む。
同じコマンドで再開でき、入力証跡と出力SHA256が一致する結果は再計算しない。
壊れた出力は再生成する。入力が途中で変わった場合は公開しない。一時ファイルを照合してatomic renameする。
`last_run.json` は対象範囲と今回の状態別件数を記録する。件数は区間の最終状態で、追加巡を二重計上しない。

## 保存する配列

- `epoch_tt2000`, `time_unix`: 入力DSIと同じレコード・順序・時刻。
- `FPDU__velocity_gsm_observed`, `FHEDU__velocity_gsm_observed`, `FODU__velocity_gsm_observed`: `(N,3)`、km/s、GSM x/y/z。
- `<species>__velocity_valid`: 回転と入力3成分が有限であることを示す。科学的なquality選別を意味しない。
- `rotation_dsi_to_gsm`: `(N,3,3)`。`v_gsm[i] = R[i,j] v_dsi[j]`。
- `rotation_valid`, `rotation_quality`: 変換の有効性とbitフラグ。
- 姿勢・軌道の補間両端時刻と間隔。
- 姿勢の両端レコードのraw `FLAG_VI`, `Sensor`, `ADSG_interp_dt`, `ADSGt-iptime`、およびflags欠落フラグ。
- `manifest_json`: 入力ファイル・SHA256、設定、座標、単位、配列SHA256。

```python
import numpy as np
with np.load(path, allow_pickle=False) as data:
    valid = data['FPDU__velocity_valid']
    velocity_gsm = data['FPDU__velocity_gsm_observed'][valid]  # km/s, x/y/z
```

## 支持データと補間

公開pySPEDASでL2姿勢TXTとL2 definitive orbit CDFを取得し、ローカルからロードする。
`SPEDAS_DATA_DIR` の既定は `/mnt/j/observation_data/`。pySPEDASの `ERG_DATA_DIR` が設定されている場合は
その上書き規則にも従う。取得後は必要な変数・全レコードの読み取り・単位・SHA256・ロード前後の指紋を検証する。
取得・ロード失敗は欠測と断定せず、日単位で記録し、処理可能な別の日を続ける。
既定では取得最大3回、間隔30秒。全対象を一巡後、失敗区間と、有効な元moment時刻に回転不能が残るpartial区間を追加1巡する。
元DSIがpartialという理由だけでは再試行しない。
`--download-attempts`, `--retry-delay`, `--retry-passes` で変更できる。
`--offline` はローカル取得だけを使い、追加巡を行わない。

DSIのZ軸には姿勢のSGI-Z（J2000 RA/Dec）を使う。単位ベクトルを線形補間し、正規化する。
Y軸は Z×衛星から見た太陽方向、X軸はY×Z。各軸を公開cotransでGSMへ回転する。
L2 `pos_gse` のCDF単位は実例では `RE`。kmへ換算して公称太陽距離 `1.496e8 km` と整合させる。
RE換算係数は既定 `6371.2 km`（`--earth-radius-km` で変更可）、profileへ保存する。
公開 `dsi2j2000` のローカル版はこのRE換算を行わないため、そのままの出力とのビット単位一致は要求しない。
検証ではkmに揃えた太陽方向を公開ルーチンに渡して比較する。ライブラリ本体は変更しない。

外挿しない。補間両端が有限で、姿勢の間隔≤16秒、軌道の間隔≤120秒を満たす点だけ回転する。
閾値は暫定的な支持データの時間条件であり、`--max-attitude-gap`, `--max-orbit-gap` で変更できる。
大きな空白やNaNを削除して、その両側をつなぐことはしない。完全同一の重複時刻だけ最後を採用する。
回転行列の直交性・右手系・流速の大きさの保存を照合する。

実姿勢TXTにはFLAG_VI・Sensorが省略された14列の行もある。両者を空文字として保存し、
後続列を正しい位置から読む。16列行と区別して欠落を記録する。
raw品質列と姿勢データ生成時の補間間隔は保存するが、今回その意味を仮定した選別はしない。
8秒刻みの支持データが存在しても、その元の姿勢精度や較正精度を保証するものではない。

rotation_qualityのbit値: `1`=姿勢不足、`2`=軌道不足、`4`=DSI基底が退化、`8`=回転行列不整合。
変換不能点は時刻を保持してNaNを保存する。全点不能でもpartialとして診断を保存する。
`complete`は全レコードのDSI momentと回転が有効、`partial`は元momentまたは回転に無効な時刻がある場合。
`moment_valid`は元momentの計算可否、`rotation_valid`は回転の可否、各種の`*__velocity_valid`は
実際に有限なGSM流速を得られたかを表す。DSIや回転が有効でも元流速がNaNならvelocity_validはfalse。
再開したpartialは `skipped_partial` と数える。partialをcompleteへ読み替えない。
終了コードは0=元momentが有効な時刻の回転完了（元DSI由来のpartialを含む）、2=failedまたは必要な回転不能が残る、3=同一profileの実行競合、130=中断。
**対象区間の回転完了は、要求月全体のDSI生成完了や、3種すべての流速が有限であることを意味しない。**

1 CPU、nice=19、idle I/O、数値ライブラリ1スレッド。メモリ空き8 GiB未満で新しい区間を待機する。
姿勢・軌道は日ごとに保持し、LEP-i元CDFを読み直さない。区間間は既定2秒休止。
`--progress on/off/auto` でtqdm表示を制御する。ログへのリダイレクト時はautoで非表示。

## 検証

```bash
.venv_pyspedas/bin/python -m unittest discover -s lepi_gsm_velocity -p 'test_*.py'
.venv_pyspedas/bin/python lepi_gsm_velocity/check_real.py --output <短区間のNPZ> --report <検証JSON>
```

合成入力で既知軸・非自明な回転・逆変換・欠測・外挿拒否・NaN・fill値・重複・14列姿勢を検証する。
保存・入力不変・SHA256拒否・再開・壊れた出力の再生成・partial保持も検証する。
実データ確認はローカル支持データを使う5分区間に限定する。月・年の変換はユーザーが実行する。

2026-10-09の検証：13テストが通過した。2022-09-29 05:05–05:10 UTCの37レコード×3種で、
kmに揃えた太陽方向を使う公開変換との最大差は3.2×10⁻¹⁴ km/s、逆変換の最大差は3.6×10⁻¹⁴ km/s。
元DSIのSHA256不変、同時刻・同順序、再実行時のスキップを確認した。
記録は `/mnt/j/observation_data/Arase_LEPi_moments/_validation/gsm_observed_verification_20261009.json`。
月・年の変換は起動していない。
