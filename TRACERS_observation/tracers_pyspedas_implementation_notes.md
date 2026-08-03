# TRACERS観測データのPySPEDAS/SPEDAS対応状況とPython実装方針

調査日: 2026-07-31

## 1. 目的

TRACERS missionの公開観測データをPython上で取得・解析するため、以下を調査した。

- PySPEDASにTRACERS専用ローダーが存在するか
- IDL版SPEDASにTRACERS専用ローダーが存在するか
- NASA SPDF/CDAWeb経由でTRACERSデータを取得できるか
- Codex等を用いてPython用の簡便なローダーを実装する場合の推奨構成
- 各観測機器のデータ利用上の注意点

---

## 2. 結論

### 2.1 PySPEDAS

2026年7月31日時点で、PySPEDASには以下のようなTRACERS専用のmission-specific loaderは確認できない。

```python
pyspedas.projects.tracers.ace(...)
pyspedas.projects.tracers.aci(...)
pyspedas.projects.tracers.efi(...)
pyspedas.projects.tracers.msc(...)
pyspedas.projects.tracers.magic(...)
```

PySPEDAS 2.1.3の直接対応mission一覧には`tracers`が含まれておらず、公開GitHubリポジトリ内でも`TRACERS`または`tracers`に対応するプロジェクトモジュールは確認できなかった。

ただし、PySPEDASにはNASA CDAWebを利用する汎用インターフェースが実装されている。このため、CDAWebに登録済みのTRACERSデータは、PySPEDAS内で取得してtplot変数へ変換できる。

主要な入口は次の二つである。

```python
from pyspedas import find_datasets
from pyspedas import CDAWeb
```

したがって、現状は次のように整理できる。

- TRACERS専用の一行ローダー: 未実装
- CDAWebを介した汎用取得: 可能
- CDFからtplot変数への変換: 可能
- TRACERS固有の校正・quality flag処理: 原則として利用者側で実装が必要

### 2.2 IDL版SPEDAS

IDL版SPEDASについても、公式ドキュメント上では次のようなTRACERS専用ルーチンは確認できない。

```idl
tracers_load_ace
tracers_load_aci
tracers_load_efi
tracers_load_msc
tracers_load_magic
```

一方、NASA SPDFはIDL用の汎用CDAWebクライアント`spdfgetdata`を提供している。

```idl
d = spdfgetdata( $
    'TS2_L2_MSC_BAC', $
    ['ALL-VARIABLES'], $
    ['2026-01-01T00:00:00.000Z', $
     '2026-01-01T01:00:00.000Z'] $
)
```

これはNASA SPDFが“One-line Data Access”として提供している機能であり、結果はCDAWlib形式のIDL構造体に格納される。ただし、これはSPEDASのmission-specific loaderではなく、通常は自動的にtplot変数を生成しない。

注意点として、SPEDASの公開releaseは6.1（2024年5月）であり、nightly buildも配布されている。本調査ではnightly buildの全ソースを直接展開してgrepするところまでは完了していない。したがって、未文書化の試験的TRACERSルーチンがnightly buildに含まれる可能性を完全には排除できない。ただし、公式Wiki、公開コマンド一覧、検索可能な文書上にはTRACERS専用ローダーは見当たらない。

---

## 3. CDAWebに登録されているTRACERSデータセット

2026年7月31日時点で、CDAWebのdataset一覧に少なくとも以下が登録されている。

| Dataset ID | 内容 |
|---|---|
| `TS1_L2_ACI_IPD` | TRACERS Satellite 1, Analyzer for Cusp Ions, ion particle distribution |
| `TS2_L2_ACI_IPD` | TRACERS Satellite 2, Analyzer for Cusp Ions, ion particle distribution |
| `TS2_L2_ACE_DEF` | TRACERS Satellite 2, Analyzer for Cusp Electrons, differential energy flux |
| `TS2_L2_MSC_BAC` | TRACERS Satellite 2, Magnetic Search Coil, high-resolution magnetic waveform |

このリストは固定と考えるべきではない。TRACERS公式ページではEFI、MAG/MAGIC等のL2製品も説明されているが、CDAWebへの登録・公開は機器ごとに進行時期が異なる可能性がある。

したがって、実装ではdataset IDをすべてハードコードするのではなく、実行時にCDAWebを検索する設計が望ましい。

```python
from pyspedas import find_datasets

datasets = find_datasets(
    mission="TRACERS",
    label=True,
    quiet=False,
)
```

`find_datasets()`は内部でCDAS Web Servicesに問い合わせるため、PySPEDAS本体の更新前に新しいTRACERS datasetがCDAWebへ追加された場合でも検出できる。

---

## 4. PySPEDASによる現状の取得方法

### 4.1 Datasetの検索

```python
from pyspedas import find_datasets

datasets = find_datasets(
    mission="TRACERS",
    quiet=True,
)

print(datasets)
```

instrument名による簡易フィルタも可能である。

```python
msc_datasets = find_datasets(
    mission="TRACERS",
    instrument="MSC",
    quiet=True,
)
```

ただし、`instrument`引数はdataset IDに対する文字列フィルタとして実装されている。CDAWebの正式なinstrument type検索と完全に同じではない。

### 4.2 CDFを取得してtplot変数へ変換する

```python
from pathlib import Path

import pyspedas
from pyspedas import tplot

dataset_id = "TS2_L2_MSC_BAC"
trange = [
    "2026-01-01 00:00:00",
    "2026-01-01 01:00:00",
]

cda = pyspedas.CDAWeb()

remote_files = cda.get_filenames(
    [dataset_id],
    trange[0],
    trange[1],
)

if not remote_files:
    raise FileNotFoundError(
        f"No CDAWeb files found for {dataset_id} in {trange}"
    )

n_files, n_vars, tplot_vars = cda.cda_download(
    remote_files,
    local_dir=str(Path("./tracers_data")),
    prefix="ts2_",
    get_support_data=True,
    trange=trange,
    time_clip=True,
)

print(f"files: {n_files}")
print(f"variables: {n_vars}")
print(tplot_vars)

if tplot_vars:
    tplot(tplot_vars)
```

`get_support_data=True`は、エネルギーbin、look direction、pitch angle、quality flag、校正係数、周波数応答などを読む場合に重要となる。ただし、読み込まれる変数が多くなるため、実際の解析では`varnames`または`varformat`で対象を限定する方がよい。

---

## 5. xarrayを主とする場合の推奨経路

既存の解析がxarray中心である場合、tplotを経由せず、PySPEDASが依存しているNASA公式`cdasws`を直接利用する方が単純である。

`cdasws.CdasWs.get_data()`は、明示的に`DataRepresentation.XARRAY`を指定することで`xarray.Dataset`を返せる。

```python
from cdasws import CdasWs
from cdasws.datarepresentation import DataRepresentation

dataset_id = "TS2_L2_MSC_BAC"

cdas = CdasWs()

status, ds = cdas.get_data(
    dataset_id,
    ["ALL-VARIABLES"],
    "2026-01-01T00:00:00Z",
    "2026-01-01T01:00:00Z",
    dataRepresentation=DataRepresentation.XARRAY,
)

if isinstance(status, dict):
    status_code = status.get("http", {}).get("status_code")
else:
    # cdaswsの旧版との互換性
    status_code = status

if status_code != 200 or ds is None:
    raise RuntimeError(
        f"CDAWeb request failed: status={status!r}"
    )

print(ds)
```

明示的に`DataRepresentation.XARRAY`を指定しない場合、環境にSpacePyが入っているとSpacePy DataModelが優先される場合がある。そのため、再現性のあるxarray loaderでは明示指定すべきである。

CDAWebはUTCを前提とする。timezone-naiveな`datetime`もUTCとして扱われるが、実装上はISO 8601の`Z`付き文字列、またはUTC timezone-awareな`datetime`に統一する方が安全である。

---

## 6. 推奨するPythonローダーの設計

### 6.1 基本方針

最初からPySPEDAS本体へpatchを当てるより、独立した小規模モジュールとして実装・検証する方がよい。

例:

```text
tracers_loader/
├── __init__.py
├── datasets.py
├── load.py
├── ace.py
├── aci.py
├── efi.py
├── msc.py
├── magic.py
├── calibration.py
├── quality.py
└── tests/
```

初期実装では次の二つのbackendを分ける。

- `backend="tplot"`: `pyspedas.CDAWeb`を利用
- `backend="xarray"`: `cdasws.CdasWs.get_data`を利用

### 6.2 公開API案

```python
find_tracers_datasets(
    instrument: str | None = None,
    refresh: bool = False,
) -> list[str]
```

```python
load(
    dataset: str,
    trange: tuple[str, str] | list[str],
    *,
    variables: list[str] | None = None,
    backend: str = "xarray",
    prefix: str = "",
    suffix: str = "",
    get_support_data: bool = True,
    local_dir: str | None = None,
    time_clip: bool = True,
)
```

instrument-specific wrapper:

```python
ace(
    trange,
    *,
    probe: int = 1,
    datatype: str = "def",
    backend: str = "xarray",
    variables: list[str] | None = None,
)
```

```python
aci(
    trange,
    *,
    probe: int = 1,
    datatype: str = "ipd",
    backend: str = "xarray",
    variables: list[str] | None = None,
)
```

```python
msc(
    trange,
    *,
    probe: int = 2,
    datatype: str = "bac",
    backend: str = "xarray",
    variables: list[str] | None = None,
    apply_frequency_response: bool = False,
)
```

将来のdataset追加に備え、EFIおよびMAGICも同じ形式で用意する。ただし、CDAWeb dataset IDが見つからない場合には明示的な例外を返す。

```python
class DatasetNotAvailableError(RuntimeError):
    pass
```

### 6.3 Dataset IDの解決

既知のdataset IDだけを固定表にするのではなく、CDAWeb検索結果とaliasを組み合わせる。

```python
KNOWN_DATASETS = {
    ("aci", 1, "ipd"): "TS1_L2_ACI_IPD",
    ("aci", 2, "ipd"): "TS2_L2_ACI_IPD",
    ("ace", 2, "def"): "TS2_L2_ACE_DEF",
    ("msc", 2, "bac"): "TS2_L2_MSC_BAC",
}
```

解決手順:

1. `KNOWN_DATASETS`を確認する
2. 見つからなければ`find_datasets(mission="TRACERS")`を実行する
3. `TS{probe}_L2_{INSTRUMENT}_{DATATYPE}`に一致するdatasetを探索する
4. 複数候補の場合は候補一覧を表示して例外にする
5. 候補なしの場合は`DatasetNotAvailableError`を返す

曖昧なdatasetを黙って選択しないことが重要である。

### 6.4 返り値

backend間で返り値の型を混在させる場合、型を明示する。

- `backend="xarray"`: `xarray.Dataset`
- `backend="tplot"`: `list[str]`、すなわちtplot変数名
- `download_only=True`: ローカルCDFパスの一覧

より厳密には、共通のresult dataclassを用意してもよい。

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

@dataclass
class TracersLoadResult:
    dataset_id: str
    backend: Literal["xarray", "tplot", "download"]
    data: Any
    files: list[Path]
    variables: list[str]
```

ただし、既存PySPEDAS loaderと似た使用感を優先するなら、instrument wrapperはtplot変数名またはxarray.Datasetを直接返す方が簡便である。

---

## 7. 最小実装例

以下は独立モジュールとしての最小例である。実際にはログ、cache、変数選択、テストを追加する。

```python
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import xarray as xr
from cdasws import CdasWs
from cdasws.datarepresentation import DataRepresentation


Backend = Literal["xarray", "tplot"]


KNOWN_DATASETS: dict[tuple[str, int, str], str] = {
    ("aci", 1, "ipd"): "TS1_L2_ACI_IPD",
    ("aci", 2, "ipd"): "TS2_L2_ACI_IPD",
    ("ace", 2, "def"): "TS2_L2_ACE_DEF",
    ("msc", 2, "bac"): "TS2_L2_MSC_BAC",
}


class TracersLoadError(RuntimeError):
    pass


class DatasetNotAvailableError(TracersLoadError):
    pass


def resolve_dataset(
    instrument: str,
    probe: int,
    datatype: str,
) -> str:
    key = (instrument.lower(), int(probe), datatype.lower())

    try:
        return KNOWN_DATASETS[key]
    except KeyError as exc:
        raise DatasetNotAvailableError(
            "No known TRACERS dataset for "
            f"instrument={instrument!r}, "
            f"probe={probe}, datatype={datatype!r}. "
            "Query CDAWeb because new products may have been added."
        ) from exc


def _status_code(status: object) -> int | None:
    if isinstance(status, int):
        return status

    if isinstance(status, dict):
        http = status.get("http")
        if isinstance(http, dict):
            code = http.get("status_code")
            return int(code) if code is not None else None

    return None


def load_xarray(
    dataset_id: str,
    trange: Sequence[str],
    *,
    variables: Sequence[str] | None = None,
) -> xr.Dataset:
    if len(trange) != 2:
        raise ValueError("trange must contain exactly [start, stop].")

    requested_variables = (
        list(variables)
        if variables
        else ["ALL-VARIABLES"]
    )

    cdas = CdasWs()

    status, ds = cdas.get_data(
        dataset_id,
        requested_variables,
        trange[0],
        trange[1],
        dataRepresentation=DataRepresentation.XARRAY,
    )

    code = _status_code(status)

    if code != 200 or ds is None:
        raise TracersLoadError(
            f"CDAWeb request failed for {dataset_id}: "
            f"status={status!r}"
        )

    if not isinstance(ds, xr.Dataset):
        raise TypeError(
            f"Expected xarray.Dataset, got {type(ds)!r}"
        )

    ds.attrs.setdefault("cdaweb_dataset_id", dataset_id)
    ds.attrs.setdefault("tracers_loader_backend", "cdasws-xarray")

    return ds


def load_tplot(
    dataset_id: str,
    trange: Sequence[str],
    *,
    variables: Sequence[str] | None = None,
    local_dir: str | Path = "./tracers_data",
    prefix: str = "",
    suffix: str = "",
    get_support_data: bool = True,
    time_clip: bool = True,
) -> list[str]:
    import pyspedas

    if len(trange) != 2:
        raise ValueError("trange must contain exactly [start, stop].")

    cda = pyspedas.CDAWeb()

    remote_files = cda.get_filenames(
        [dataset_id],
        trange[0],
        trange[1],
    )

    if not remote_files:
        raise FileNotFoundError(
            f"No files found for {dataset_id} in {list(trange)!r}"
        )

    n_files, _, tplot_vars = cda.cda_download(
        remote_files,
        local_dir=str(local_dir),
        prefix=prefix,
        suffix=suffix,
        varnames=list(variables) if variables else None,
        get_support_data=get_support_data,
        trange=list(trange),
        time_clip=time_clip,
    )

    if n_files == 0:
        raise TracersLoadError(
            f"CDAWeb returned no downloadable files for {dataset_id}"
        )

    return list(tplot_vars)


def load(
    dataset_id: str,
    trange: Sequence[str],
    *,
    backend: Backend = "xarray",
    variables: Sequence[str] | None = None,
    **kwargs,
):
    if backend == "xarray":
        return load_xarray(
            dataset_id,
            trange,
            variables=variables,
        )

    if backend == "tplot":
        return load_tplot(
            dataset_id,
            trange,
            variables=variables,
            **kwargs,
        )

    raise ValueError(f"Unknown backend: {backend!r}")


def aci(
    trange: Sequence[str],
    *,
    probe: int = 1,
    datatype: str = "ipd",
    backend: Backend = "xarray",
    variables: Sequence[str] | None = None,
    **kwargs,
):
    dataset_id = resolve_dataset("aci", probe, datatype)

    return load(
        dataset_id,
        trange,
        backend=backend,
        variables=variables,
        **kwargs,
    )


def ace(
    trange: Sequence[str],
    *,
    probe: int = 2,
    datatype: str = "def",
    backend: Backend = "xarray",
    variables: Sequence[str] | None = None,
    **kwargs,
):
    dataset_id = resolve_dataset("ace", probe, datatype)

    return load(
        dataset_id,
        trange,
        backend=backend,
        variables=variables,
        **kwargs,
    )


def msc(
    trange: Sequence[str],
    *,
    probe: int = 2,
    datatype: str = "bac",
    backend: Backend = "xarray",
    variables: Sequence[str] | None = None,
    **kwargs,
):
    dataset_id = resolve_dataset("msc", probe, datatype)

    return load(
        dataset_id,
        trange,
        backend=backend,
        variables=variables,
        **kwargs,
    )
```

---

## 8. Codexに実装させる際のacceptance criteria

### 必須

1. `find_tracers_datasets()`がCDAWebを問い合わせる
2. 既知の4 datasetをaliasから解決できる
3. `backend="xarray"`で`xarray.Dataset`を返す
4. `backend="tplot"`でtplot変数名一覧を返す
5. start/stop時刻を検証する
6. CDAWebがHTTP 200以外を返した場合に明示的な例外を出す
7. データなしと通信失敗を区別する
8. `variables=None`では全変数を取得できる
9. support dataを取得できる
10. dataset ID、backend、取得時刻範囲をmetadataへ残す

### 推奨

1. `pytest`によるunit test
2. 実通信を行わないmock test
3. 少量のintegration test
4. loggingの導入
5. dataset検索結果の短時間cache
6. `SPEDAS_DATA_DIR`または専用`TRACERS_DATA_DIR`への対応
7. timezone-awareな時刻処理
8. `ruff`または`flake8`、`mypy`対応
9. API rate limitを考慮し、過剰な並列取得を避ける
10. 新しいdatasetが追加された場合に固定表なしでも探索できるfallback

### テスト例

```python
def test_resolve_known_dataset():
    assert (
        resolve_dataset("msc", 2, "bac")
        == "TS2_L2_MSC_BAC"
    )
```

```python
def test_unknown_dataset_raises():
    with pytest.raises(DatasetNotAvailableError):
        resolve_dataset("msc", 1, "bac")
```

```python
def test_invalid_trange():
    with pytest.raises(ValueError):
        load_xarray(
            "TS2_L2_MSC_BAC",
            ["2026-01-01T00:00:00Z"],
        )
```

integration testは、公開データが実際に存在する短い時間区間を確認してから追加する。mission初期段階ではファイル更新や欠測があり得るため、CIで毎回大容量データを取得する設計は避ける。

---

## 9. TRACERS固有の解析上の注意

汎用ローダーでCDFを読めても、物理解析に必要な補正が自動的に適用されるとは限らない。

### 9.1 ACE

ACE L2は、21 anode、49 energy stepについてraw countsとcalibrated differential energy fluxを含む。基本cadenceは50 msである。

背景推定値とcalibration matrixもCDFに含まれ、別の背景差し引きや校正を適用できる。

注意点:

- 背景差し引き後のfluxが負になる可能性がある
- spacecraft orientationによって低エネルギー電子が遮蔽される期間がある
- 低エネルギーACEデータを定量利用できない期間が存在する
- loaderはquality/caveatを自動判定せず、フラグまたは姿勢情報を利用者が確認する必要がある

### 9.2 ACI

ACIは約8 eV/eから20 keV/e、47 energy bin、16 look directionのion distributionを提供する。基本cadenceは約312 msである。

注意点:

- 2機のinstrument sensitivityに少なくともfactor of 2程度の差がある
- absolute flux uncertaintyもfactor of 2以上とされる
- 2機間のflux差をそのまま空間差または時間発展と解釈するのは危険
- 2026-01-15までにinstrument settingの変更が行われている
- calibration version、instrument setting、quality情報をmetadataとして保持すべき

### 9.3 EFI

EFIの低周波EDC製品は3成分電場を含むが、第3成分はMHD仮定

```text
E_parallel = 0
```

によって推定される。

これは、2成分観測から`E · B = 0`を仮定して第3成分を再構成する処理に相当する。磁場方向とboom planeの幾何によって誤差が増幅され得るため、再構成条件とquality maskが必要である。

EACおよびEHFには複素伝達関数がCDF内に含まれる。公開time seriesやspectrumには、周波数依存のgain/phase補正が完全には適用されていない場合がある。

### 9.4 MSC

MSC L2は3軸波形を2048 samples/sで提供する。

重要な校正条件:

- 振幅校正は100 Hz基準
- 周波数依存の振幅偏差が残る
- phase calibrationは未適用
- CDFに複素周波数応答tableが含まれる
- wave parameter、cross phase、Poynting flux、E/Bを評価する前に複素応答補正が必要

公式説明では、FFT後の複素スペクトルをMSC calibration tableで複素除算する。

概念的には、

```python
B_corrected_f = B_observed_f / H_msc_f
```

である。ここで`H_msc_f`は複素伝達関数である。

CDF内の例として`ts2_cal_freq_response`が説明されている。周波数tableは0.0625 Hzから1024 Hzまで0.0625 Hz刻み、32768-point FFTを前提としている。異なるFFT長では、周波数gridを一致させるためのdecimationまたはinterpolationが必要である。

FACデータが欠落する期間は、definitive ephemerisが不足している可能性がある。FACがNaNである場合にspacecraft-coordinate波形へ無条件にfallbackするのではなく、座標系をmetadataと変数名で明示するべきである。

### 9.5 MAGIC

MAGIC L2は、ROIで128 samples/s、back orbitで16 samples/sの3軸磁場を提供する。

quality flagの説明:

- `0`: 原則として良好
- `1`: magneto-torquer contamination
- `2`: definitive attitude kernelなし
- `3`: 上記二つの組合せ

mission進行に伴い、quality flagの値が追加される可能性がある。実装では`flag == 0`以外を一律NaNにするだけでなく、flag値と意味の対応表をmetadataとして保持する方がよい。

---

## 10. 実装で避けるべきこと

- TRACERS公式Webサーバのディレクトリ構造を推測してURLを組み立てる
- dataset IDを4個だけ固定し、CDAWeb検索を実装しない
- quality flagを無視して自動的に「calibrated data」とみなす
- MSCとEFIの複素周波数応答を振幅だけで補正する
- phase未校正のMSCをそのままE/B phaseまたはPoynting fluxに使う
- ACI 1とACI 2の絶対fluxを校正差なしに比較する
- EFIの第3成分を直接観測値として扱う
- `SpacePy DataModel`と`xarray.Dataset`の返り値を環境依存にする
- CDAWebへ多数の並列requestを送る

NASAの`cdasws`文書では、rate limitのため多数のthreadから同時requestを送ると性能が低下し得るとされ、5 thread以下が推奨されている。

---

## 11. 推奨する開発順序

### Phase 1: 汎用取得

- dataset discovery
- xarray backend
- tplot backend
- error handling
- metadata保持
- unit test

### Phase 2: instrument wrapper

- `ace()`
- `aci()`
- `msc()`
- 将来の`efi()`、`magic()`

### Phase 3: quality処理

- quality flagのdecode
- fill valueからNaNへの変換
- invalid interval mask
- attitude/ephemeris不足の識別

### Phase 4: 校正処理

- MSC complex frequency response
- EFI complex frequency response
- calibration versionの記録
- 補正前後の変数を別名で保持

### Phase 5: 解析補助

- energy-time spectrogram
- pitch-angle distribution
- FAC変換の検証
- E/B比
- cross spectrum、coherency、phase
- Poynting flux

---

## 12. Codex向けタスク記述例

```text
Implement a small Python package for loading NASA TRACERS mission data
from CDAWeb.

Requirements:

1. Use `pyspedas.find_datasets` or `cdasws.CdasWs.get_datasets` to
   discover current TRACERS datasets dynamically.
2. Provide `load_xarray()` using
   `cdasws.CdasWs.get_data(...,
   dataRepresentation=DataRepresentation.XARRAY)`.
3. Provide `load_tplot()` using `pyspedas.CDAWeb`.
4. Add convenience wrappers for ACE, ACI, and MSC.
5. Support the currently known datasets:
   - TS1_L2_ACI_IPD
   - TS2_L2_ACI_IPD
   - TS2_L2_ACE_DEF
   - TS2_L2_MSC_BAC
6. Do not silently guess an ambiguous dataset.
7. Handle both old integer and newer dictionary-style cdasws status
   return values.
8. Preserve CDAWeb dataset ID, requested time range, backend, and
   variable list in metadata.
9. Add typed exceptions for no dataset, no data, and request failure.
10. Add pytest tests using mocks; do not require large live downloads
    in the normal unit-test suite.
11. Keep the calibration routines separate from the loader.
12. Add placeholders and documented interfaces for MSC/EFI complex
    frequency-response correction, but do not apply a scientifically
    unverified correction automatically.
13. Use Python 3.10+ type hints and follow the public API described in
    this document.
```

---

## 13. 参考資料

### PySPEDAS

- PySPEDAS CDAWeb interface  
  https://pyspedas.readthedocs.io/en/stable/cdaweb.html

- PySPEDAS `find_datasets` source/documentation  
  https://pyspedas.readthedocs.io/en/stable/_modules/pyspedas/utilities/datasets.html

- PySPEDAS directly supported load routines  
  https://pyspedas.readthedocs.io/en/latest/projects.html

- PySPEDAS GitHub repository  
  https://github.com/spedas/pyspedas

### NASA SPDF/CDAWeb

- CDAWeb dataset list beginning with T  
  https://cdaweb.gsfc.nasa.gov/misc/NotesT.html

- CDAS Web Services  
  https://cdaweb.gsfc.nasa.gov/WebServices/

- Python `cdasws` API  
  https://cdaweb.gsfc.nasa.gov/WebServices/py/cdasws/

- `cdasws` PyPI  
  https://pypi.org/project/cdasws/

- IDL CDAWeb access and `spdfgetdata`  
  https://cdaweb.gsfc.nasa.gov/WebServices/REST/CdasIdlLibrary.html

### TRACERS

- TRACERS L2 public data products  
  https://tracers.physics.uiowa.edu/l2-public-data-products

- TRACERS MAGIC L2 data products  
  https://tracers.physics.uiowa.edu/magic-0

### IDL SPEDAS

- SPEDAS downloads and installation  
  https://spedas.org/wiki/index.php?title=Downloads_and_Installation

---

## 14. 調査上の留保

- CDAWebのTRACERS datasetは今後追加される可能性が高い。
- Dataset ID、変数名、quality flag、calibration variableはCDF versionにより変更され得る。
- 本文中の既知dataset一覧は2026年7月31日時点のCDAWeb公開一覧に基づく。
- IDL版SPEDASについて、公式ドキュメント上では専用ローダーを確認できなかったが、nightly build全ソースの直接検査は未完了である。
- 校正処理はinstrument teamの最新文書およびCDF metadataを優先すること。
