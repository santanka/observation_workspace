# イオンモーメントを系の速度として用いる再解析

解析versionは `lepe_ge67eV_vion_v2`。`v_sys_fac` は `v_ion_fac` と同じFACベクトル [m/s] を使う。LEP-iの30 eV/q--30 keV/q partial moment、LEP-eの67 eV以上、toroidal成分と100秒平滑化の定義は維持する。

新versionの既定出力・stateを使い、旧 `--state-dir` / `--output-root` は指定しない。初回は `--noncomplete-only` を付けない。旧complete/excluded rangeも再評価する。CDFはそのまま再利用できる。再解析はこの変更作業では実行していない。

## 停止していた3か月

以下は各terminalで必要な月を実行するためのコマンド。3本同時ならprefetch workerは各2にする。

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py run \
  --ranges /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20210101_000000_to_20210201_000000.json \
  --download-workers 2 --plot-event-orbits
```

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py run \
  --ranges /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20210201_000000_to_20210301_000000.json \
  --download-workers 2 --plot-event-orbits
```

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py run \
  --ranges /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20210301_000000_to_20210401_000000.json \
  --download-workers 2 --plot-event-orbits
```

```bash
.venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py run \
  --ranges /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20210401_000000_to_20210501_000000.json \
  --download-workers 2 --plot-event-orbits
```

## 旧結果のある全17か月を順番に処理する場合

途中失敗した月でloopを止める。全rangeの実行状態は各月のmarkerに保存されるため、同じコマンドで再開できる。

```bash
for range_manifest in \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20200101_000000_to_20200201_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20200201_000000_to_20200301_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20200301_000000_to_20200401_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20200401_000000_to_20200501_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20200501_000000_to_20200601_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20200601_000000_to_20200701_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20200701_000000_to_20200801_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20200801_000000_to_20200901_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20200901_000000_to_20201001_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20201001_000000_to_20201101_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20201101_000000_to_20201201_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20201201_000000_to_20210101_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20210101_000000_to_20210201_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20210201_000000_to_20210301_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20210301_000000_to_20210401_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20210401_000000_to_20210501_000000.json \
  /mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/Arase_valid_time_ranges_20220901_000000_to_20221001_000000.json \
; do
  .venv_pyspedas/bin/python statistical_analysis_arase_pre_auto/batch_arase.py run \
    --ranges "$range_manifest" --download-workers 2 --plot-event-orbits || break
done
```

既存入力が揃っている月でprefetchを省略する場合は `--skip-prefetch` を追加できる。ただし2022年9月の旧stateには `failed_download` が4 rangeあったため、その月はprefetchを推奨する。
