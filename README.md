# キーボード打鍵時の手指動画像を用いた継続認証方法の検討-プログラム

## フォルダ構成

```
research/
├── data/
│   ├── collected/
│   ├── training/
│   ├── validation/
│   │   ├── enrollment/
│   │   └── probe/
│   └── test/
│       └── probe/
├── experiments/
│   ├── exp001/
│   │   ├── config.json
│   │   ├── samples/
│   │   │   ├── training/
│   │   │   ├── validation/
│   │   │   │   ├── enrollment/
│   │   │   │   └── probe/
│   │   │   └── test/
│   │   │       └── probe/
│   │   ├── sample_indices/
│   │   │   ├── training/
│   │   │   ├── validation/
│   │   │   │   ├── enrollment/
│   │   │   │   └── probe/
│   │   │   └── test/
│   │   │       └── probe/
│   │   ├── models/
│   │   │   ├── bow.npy
│   │   │   ├── bop.npy
│   │   │   └── bomp/
│   │   │       ├── left.npy
│   │   │       └── right.npy
│   │   ├── histograms/
|   │   |   ├── validation/
|   │   |   │   ├── enrollment/
|   │   |   │   └── probe/
|   │   |   └── test/
|   │   |       └── probe/
│   │   ├── threshold/
│   │   ├── offset/
│   │   └── results/
│   │
│   ├── exp002/
│   │   ├── config.json
│   │   ├── samples/
│   │   ├── sample_indices/
│   │   ├── models/
│   │   ├── histograms/
│   │   ├── threshold/
│   │   ├── offset/
│   │   └── results/
│   │
│   └── exp003/
│       ├── config.json
│       ├── samples/
│       ├── sample_indices/
│       ├── models/
│       ├── histograms/
│       ├── threshold/
│       ├── offset/
│       └── results/
│
└── src/
    ├── experiment_utils.py
    ├── video_recorder.py
    ├── prompt_input.py
    ├── prompt_capture.py
    ├── extract_sample.py
    ├── build_codebook.py
    ├── encode_histogram.py
    ├── calculate_offset.py
    ├── calibrate_threshold.py
    ├── authenticate.py
    └── evaluate.py
```

## データの保存と処理の流れ

実験は、実験番号ごとに設定と生成物を分離して管理します。
被験者番号は`XX`、プロンプト番号は`YY`、実験番号は`EE`として、
ファイル名やディレクトリ名に以下の形式を使用します。

- 被験者番号: `sXX`
- プロンプト番号: `pYY`
- 実験番号: `expEE`

主な処理と生成物は、次の順序で利用します。

| 順序 | 処理 | 入力 | 出力 |
| --- | --- | --- | --- |
| 1 | `prompt_capture.py`または`video_recorder.py`で録画 | 被験者番号、プロンプト番号、保存先 | `data/<分割>/<サブセット>/sXX_pYY.mp4` |
| 2 | `extract_sample.py`で特徴量抽出 | 録画済み動画と実験設定の`L` | `experiments/expEE/samples/`と`experiments/expEE/sample_indices/` |
| 3 | `build_codebook.py`でコードブック学習 | training入力サンプルと実験設定の`L`、`D`、`seed` | `experiments/expEE/models/codebook.npy` |
| 4 | `encode_histogram.py`でヒストグラム作成 | 入力サンプルとコードブック | `experiments/expEE/histograms/<分割>/<サブセット>/sXX_pYY.npy` |
| 5 | `calibrate_threshold.py`で閾値決定 | 被験者ごとのvalidationヒストグラム | `experiments/expEE/threshold/sXX_threshold.npy`、`experiments/expEE/threshold/sXX_threshold.json` |
| 6 | `authenticate.py`で本人認証 | test/probeヒストグラムと被験者閾値 | `experiments/expEE/results/authentication.csv` |
| 7 | `calculate_offset.py`でオフセット空間を計算 | validation/enrollment・probeの入力サンプルとコードブック | `experiments/expEE/offset/validation/` |

### 実験設定

各実験の設定は、次のファイルに保存します。

```text
experiments/expEE/config.json
```

設定には、連結するフレーム数`L`、コードワード数`D`、乱数シード`seed`、
および使用する手法を記述します。実験番号を指定すると、対応する設定から
入力サンプル、コードブック、ヒストグラムの保存先を決定します。

### 保存先とファイル名

録画動画は、収集目的に応じて次のいずれかへ保存します。

| 保存先 | 用途 |
| --- | --- |
| `data/collected/` | 一時的な収集データ用（既定） |
| `data/training/` | コードブック学習用 |
| `data/validation/enrollment/` | 被験者登録用 |
| `data/validation/probe/` | 閾値較正用 |
| `data/test/probe/` | 認証性能評価用 |

各生成ファイルの対応関係は、次のとおりです。

| 種類 | 保存形式 |
| --- | --- |
| 録画動画 | `data/<分割>/<サブセット>/sXX_pYY.mp4` |
| 入力サンプル | `experiments/expEE/samples/<分割>/<サブセット>/sXX_pYY.npy` |
| サンプルのフレーム番号 | `experiments/expEE/sample_indices/<分割>/<サブセット>/sXX_pYY.npy` |
| コードブック | `experiments/expEE/models/codebook.npy` |
| ヒストグラム | `experiments/expEE/histograms/<分割>/<サブセット>/sXX_pYY.npy` |
| オフセット空間 | `experiments/expEE/offset/validation/sXX_pYY__sAA_pBB.csv` |
| 閾値 | `experiments/expEE/threshold/sXX_threshold.npy` |
| 閾値メタデータ | `experiments/expEE/threshold/sXX_threshold.json` |

例えば、被験者3、プロンプト7、実験1の場合、
validation/enrollmentから作成される主なファイルは次のとおりです。

```text
data/validation/enrollment/s03_p07.mp4
experiments/exp01/samples/validation/enrollment/s03_p07.npy
experiments/exp01/sample_indices/validation/enrollment/s03_p07.npy
experiments/exp01/histograms/validation/enrollment/s03_p07.npy
```

## 実行方法

以下では、プロジェクトルートをカレントディレクトリとして実行します。

### プログラム一覧

| プログラム | 概要 | 主な用途 |
| --- | --- | --- |
| `video_recorder.py` | Webカメラ映像を単独で録画します。 | カメラ動作の確認や、プロンプト入力を伴わない録画 |
| `prompt_input.py` | `prompts.json`からプロンプトを選択し、キーボード入力を表示・判定します。 | プロンプトと入力画面の単独確認 |
| `prompt_capture.py` | プロンプト入力とWebカメラ録画を同時に実行します。 | 被験者データの収集 |
| `extract_sample.py` | 録画済み動画からMediaPipeランドマーク特徴量と時間窓の開始フレーム番号を保存します。 | 動画から入力サンプルへの変換 |
| `build_codebook.py` | 実験設定とtrainingサンプルからBoWコードブックを作成します。 | コードブックの学習 |
| `encode_histogram.py` | 入力サンプルをコードブックへ割り当て、L1正規化したヒストグラムを作成します。 | validation/testヒストグラムの作成 |
| `calculate_offset.py` | validation/enrollmentとvalidation/probeの全ペアについて、同一コードワードのフレーム差をCSVへ保存します。 | BoP用オフセット空間の確認 |
| `calibrate_threshold.py` | validationの本人・他人ヒストグラムから被験者ごとの閾値とメタデータを決定します。 | 被験者ごとの閾値較正 |
| `authenticate.py` | test/probeヒストグラムを登録被験者と照合し、本人・他人判定をCSVへ保存します。 | 1対1照合と認証 |

各スクリプトの引数は、次のようにヘルプ表示できます。

```powershell
python src\<プログラム名>.py --help
```

### `video_recorder.py`

Webカメラ映像を単独で録画します。保存先は`data/`です。

| 引数 | 型 | 概要 | 既定値 |
| --- | --- | --- | --- |
| `filename` | 位置引数 | 保存するファイル名（例: `session01.mp4`） | なし |
| `--camera-index` | int | 使用するカメラ番号 | `0` |

ファイル名に拡張子を指定しない場合は`.mp4`を付加します。
録画中は`q`キーまたは`Esc`キーで終了できます。

```powershell
python src\video_recorder.py session01.mp4
python src\video_recorder.py session01.mp4 --camera-index 1
```

### `prompt_input.py`

`prompts.json`から指定したプロンプトを表示し、キーボード入力を判定します。
録画は行いません。

| 引数 | 型 | 概要 | 既定値 |
| --- | --- | --- | --- |
| `prompt_index` | 位置引数 | 使用するプロンプト番号（1以上） | `1` |

入力中の動作は次のとおりです。

- 改行位置でEnterを押すと改行を挿入します。
- 改行位置以外でEnterを押すと無視します。
- プロンプト末尾でEnterを押すと入力を確定します。
- Escキーを押すと、入力が一致しなくても終了します。

```powershell
python src\prompt_input.py 7
```

### `prompt_capture.py`

プロンプト入力とWebカメラ映像を同時に収集します。保存名は
`sXX_pYY.mp4`で、被験者番号とプロンプト番号をゼロ埋めした形式です。

| 引数 | 型 | 概要 | 既定値 |
| --- | --- | --- | --- |
| `subject_index` | 位置引数 | 被験者番号 | なし |
| `prompt_index` | 位置引数 | プロンプト番号 | なし |
| `--camera-index` | int | 使用するカメラ番号 | `0` |
| `--save-dir` | choice | 保存先（`collected`、`training`、`validation/enrollment`、`validation/probe`、`test/probe`） | `collected` |
| `--force` | flag | 同名の既存動画を上書きする | 指定なし |

保存対象の動画が既に存在する場合は警告を表示して録画を開始しません。
既存動画を明示的に上書きする場合のみ`--force`を指定します。

```powershell
# 被験者3のプロンプト7をcollectedへ保存（既定）
python src\prompt_capture.py 3 7

# validation/enrollmentへ保存
python src\prompt_capture.py 3 7 --save-dir validation/enrollment

# カメラ番号1を使用
python src\prompt_capture.py 3 7 --camera-index 1

# 同名の既存動画を上書き
python src\prompt_capture.py 3 7 --save-dir validation/enrollment --force
```

### `extract_sample.py`

実験設定`experiments/expXX/config.json`の`L`を読み込み、録画済み動画から
Lフレームの時間窓サンプルを作成します。時間窓には50%オーバーラップを使用し、
動画ごとの時間窓特徴量を`experiments/expXX/samples/`へ、各時間窓の開始フレーム番号を
`experiments/expXX/sample_indices/`へ保存します。フレーム番号は特徴量配列の行と同じ順序で対応します。
既存のサンプルとフレーム番号が両方存在する場合は既定でスキップします。

| 引数 | 型 | 概要 | 既定値 |
| --- | --- | --- | --- |
| `experiment` | 位置引数 | 実験番号（例: `1`は`exp01`） | なし |
| `--data-dir` | Path | 入力動画のルートディレクトリ | `data` |
| `--subject` | int | 被験者番号でフィルタ | 指定なし |
| `--prompt` | int | プロンプト番号でフィルタ | 指定なし |
| `--force` | flag | 既存の入力サンプルを上書きする | 指定なし |

```powershell
# すべての動画から入力サンプルを作成
python src\extract_sample.py 1

# 被験者1のプロンプト2だけ処理
python src\extract_sample.py 1 --subject 1 --prompt 2

# 別の入力ディレクトリを使用
python src\extract_sample.py 1 --data-dir data

# 既存サンプルを再処理して上書き
python src\extract_sample.py 1 --force
```

### `build_codebook.py`

実験設定の`L`、`D`、`seed`と`training`サンプルを読み込み、BoWコードブックを
作成します。

| 引数 | 型 | 概要 | 既定値 |
| --- | --- | --- | --- |
| `experiment` | 位置引数 | 実験番号（例: `1`は`exp01`） | なし |
| `--force` | flag | 既存のコードブックを上書きする | 指定なし |

既存のコードブックがある場合は既定で処理を停止し、上書きには
`--force`を指定します。

```powershell
python src\build_codebook.py 1
python src\build_codebook.py 1 --force
```

実行後は`experiments/exp01/models/codebook.npy`へ保存されます。

### `encode_histogram.py`

作成済みのコードブックを読み込み、validation/testの入力サンプルを
L2正規化したD次元ヒストグラムへ変換します。既存のヒストグラムは既定で
スキップします。

| 引数 | 型 | 概要 | 既定値 |
| --- | --- | --- | --- |
| `experiment` | 位置引数 | 実験番号（例: `1`は`exp01`） | なし |
| `--split` | choice | 処理対象（`all`、`validation`、`test`） | `all` |
| `--subset` | choice | 処理対象（`all`、`enrollment`、`probe`） | `all` |
| `--force` | flag | 既存のヒストグラムを上書きする | 指定なし |

```powershell
# validation/testをすべて処理
python src\encode_histogram.py 1

# validation/enrollmentだけ処理
python src\encode_histogram.py 1 --split validation --subset enrollment

# test/probeだけ処理
python src\encode_histogram.py 1 --split test --subset probe

# 既存のヒストグラムを上書き
python src\encode_histogram.py 1 --force
```

### `calculate_offset.py`

validation/enrollmentとvalidation/probeの入力サンプルをそれぞれコードワードへ割り当て、
同一コードワードとなる全ての時間窓の組について、開始フレーム番号の差と出現回数を計算します。
各動画ペアのCSVは`experiments/expEE/offset/validation/`へ保存され、列は`offset,count`です。
既存サンプルにフレーム番号ファイルがない場合は、`extract_sample.py`を再実行して作成してください。

```powershell
# validationのenrollment × probe全ペアを処理
python src\calculate_offset.py 1

# 既存のオフセットCSVを上書き
python src\calculate_offset.py 1 --force
```

### `calibrate_threshold.py`

validationの本人・他人ヒストグラムを比較し、被験者ごとの閾値を決定します。
対象被験者を省略した場合は、`validation/enrollment`に登録されている全被験者を処理します。

閾値は本人類似度の最小値と他人類似度の最大値を比較して決定します。
分離可能な場合は2値の平均、分離できない場合は本人類似度の最小値を閾値とします。

閾値とメタデータは次のパスへ保存されます。

```text
experiments/expEE/threshold/sXX_threshold.npy
experiments/expEE/threshold/sXX_threshold.json
```

JSONには閾値、決定規則、本人・他人類似度の最小値または最大値、
各類似度の組み合わせ、および類似度の一覧を保存します。

| 引数 | 型 | 概要 | 既定値 |
| --- | --- | --- | --- |
| `experiment` | 位置引数 | 実験番号（例: `1`は`exp01`） | なし |
| `subject` | 位置引数 | 対象被験者番号（省略時は全被験者） | 指定なし |
| `--force` | flag | 既存の閾値とメタデータを上書きする | 指定なし |

```powershell
# exp01の全被験者を処理
python src\calibrate_threshold.py 1

# exp01の被験者3だけ処理
python src\calibrate_threshold.py 1 3

# 既存の出力を上書き
python src\calibrate_threshold.py 1 --force
```

### `authenticate.py`

test/probeのヒストグラムと、validation/enrollmentに登録されたヒストグラムを比較し、
被験者ごとの閾値によって本人または他人を判定します。

登録ヒストグラムごとの判定を多数決し、同数の場合は判定保留とします。
認証対象ファイル名の被験者番号は正解ラベルとしてCSVに保存しますが、照合判定には使用しません。

| 引数 | 型 | 概要 | 既定値 |
| --- | --- | --- | --- |
| `experiment` | 位置引数 | 実験番号（例: `1`は`exp01`） | なし |
| `--probe-subject` | int | 認証対象の被験者番号（省略時は全被験者） | 指定なし |
| `--candidate-subject` | int | 照合対象の被験者番号（省略時は全被験者） | 指定なし |
| `--output` | Path | 認証結果CSVの出力先 | `experiments/expEE/results/authentication.csv` |

```powershell
# test/probeの全被験者を全登録被験者と照合
python src\authenticate.py 1

# 被験者3の認証対象を全登録被験者と照合
python src\authenticate.py 1 --probe-subject 3

# 被験者3の認証対象を被験者1だけと照合
python src\authenticate.py 1 --probe-subject 3 --candidate-subject 1

# 全認証対象を被験者1だけと照合
python src\authenticate.py 1 --candidate-subject 1
```

既定の出力先は次のとおりです。

```text
experiments/expEE/results/authentication.csv
```

CSVには認証対象ファイル名、正解ラベル、照合対象被験者、登録ヒストグラム、
類似度、閾値、各比較の判定、本人・他人判定数、多数決による最終判定を記録します。

