# PLaMo 0.8B Metal Runtime

`pfnet/plamo-3-translate-0.8b` のBF16モデルを、Apple Silicon Mac向けに最適化したSwift + Metal製の軽量・高速翻訳エンジンです。

推論時に **Python、PyTorch、MLX、llama.cpp などの外部ランタイムに一切依存せず**、単一のネイティブバイナリとして動作します。直感的に使えるモダンなローカルWeb UIも同梱しています。

---

## 主な特長

- 🚀 **推論時依存ゼロ**: SwiftとMetalのみで実装。推論実行時はPython環境すら不要で超高速に起動します。
- ⚡ **Apple Silicon ネイティブ最適化**:
  - Unified Memoryを活用した重みファイルの直接メモリマップ（ゼロコピー）
  - 入力埋め込み（約330MB）をCPUへ逃がし、GPUフットプリントを最小化
  - 局所Attention（14層）と大域Attention（2層）に特化した専用ハイブリッドKVキャッシュ
  - BF16ベクトルロードとSIMD-groupタイル行列積による高効率GPUカーネル
- 💾 **可逆圧縮モード (`--packed`)**:
  - 重みビットを完全に保持したままファイルサイズとMetal確保量を削減（短文時約1.1GB）。
- 📜 **スマートな長文区間分割 (`--segment-tokens`)**:
  - 自然文境界（改行・文末）で区間を分割し、長文翻訳時の反復暴走を防止。
- ⚡ **実験的4区間GPU並列バッチ (`--batch4`)**:
  - 複数区間をGPU上で最大4本同時に生成し、長文処理時間を大幅短縮。
- 🌐 **プライベートWeb UI同梱**:
  - Bunで動くデジタル庁デザインシステム（DADS）準拠のローカルWeb画面。
  - ストリーミング翻訳、完全ローカル実行（外部通信ゼロ）、翻訳ごとの自動メモリ解放。
- 🛠️ **Unixパイプライン連携**:
  - 標準入出力・ストリーミング（`--stream`）対応。シェルスクリプトや他ツールとシームレスに結合可能。

---

## 動作要件

- **OS**: macOS 14.0 (Sonoma) 以降
- **ハードウェア**: Apple Silicon (M1 / M2 / M3 / M4 シリーズ)
- **ビルドツール**: Xcode または Command Line Tools (Swift 5.9+)
- **モデル取得時のみ**: Python 3.11+
- **Web UI実行時のみ**: [Bun](https://bun.sh)

---

## クイックスタート

### 1. モデルのダウンロード

初回のみ、Hugging Faceからモデル重みとトークナイザーを取得します。

```sh
python3 scripts/download.py
```

※ 高速起動用の検索木キャッシュ `tokenizer.jsonl.trie` も自動生成されます。

### 2. ビルド

```sh
swift build -c release
```

### 3. CLIで翻訳する

```sh
.build/release/plamo --to en '会議は来週の月曜日に変更されました。'
```

```text
The meeting has been rescheduled for next Monday.
```

標準入力からのパイプライン渡しやストリーミング出力も可能です。

```sh
echo "This system is convenient." | .build/release/plamo --to ja --stream
```

---

## ブラウザで使う (Web UI)

デジタル庁デザインシステムを採用した、視認性の高いWebインターフェースが付属しています。

```sh
# 可逆圧縮重みを準備（推奨）
.venv/bin/python scripts/pack_weights.py

# サーバー起動
bun run web
```

ブラウザで `http://127.0.0.1:3000` を開きます。（ポート変更は `PORT=3001 bun run web`）

### Web UIのセキュリティ & 仕様
- **完全ローカル**: `127.0.0.1` のみで待ち受け、同一オリジンからのリクエストのみ受理。外部フォントやCDN、外部APIへの通信は一切行いません。
- **データ非保持**: 入力文・訳文をディスクやアクセスログに一切保存しません。
- **自動メモリ解放**: 翻訳リクエストごとにCLIプロセスを呼び出し、完了または中止時にモデルを即時解放します。
- **排他制御**: 複数タブからの多重実行を防ぎ、同時実行は1件に制限されます。

---

## CLI 使い方 & オプション

### 基本構文

```sh
.build/release/plamo [オプション] [翻訳したいテキスト]
```

※ テキストを引数で渡さない場合は、標準入力（UTF-8）から読み込みます。  
※ 訳文は **標準出力** に、処理時間やメモリ等のメトリクスJSONは **標準エラー** に出力されます。

### 主要オプション一覧

| オプション | 既定値 | 説明 |
|---|:---:|---|
| `--to <lang>` | `ja` | 翻訳先言語コード（`en`, `ja` など53言語対応） |
| `--stream` | 無効 | トークン生成と同時に標準出力へ文字を逐次ストリーミング出力 |
| `--packed` | 無効 | 可逆圧縮重み `models/plamo/model.p12` を使用（要packスクリプト実行） |
| `--segment-tokens <N>` | 無効 | 入力を改行・文境界で最大Nトークン（64〜8,192）ごとに区間分割して順次翻訳 |
| `--batch4` | 無効 | 最大4区間をGPU上で並列生成する実験機能（`--segment-tokens` 併用必須） |
| `--max-tokens <N>` | `256` | 生成トークン数の上限（区間分割時は1区間ごとの上限） |
| `--context <N>` | 自動 | 入出力合計コンテキスト長の上限（最大32,768） |
| `--model <dir>` | `models/plamo` | モデルディレクトリのパス |
| `--repeat <N>` | `1` | モデルを保持したまま指定回数繰り返し計測 |

### 反復制御オプション（安全弁）

| オプション | 既定値 | 説明 |
|---|:---:|---|
| `--repetition-penalty <F>` | `1.0` | 文脈内の既出トークンのlogitをFで除算して抑制 |
| `--frequency-penalty <F>` | `0.0` | 出現頻度に比例してlogitを減算 |
| `--no-repeat-ngram <N>` | `0` (無効) | 文脈内のN-gramと同一の並びを生成することを禁止 |
| `--loop-limit <N>` | `0` (無効) | 同一パターンのループがN回続いた時点で安全に打ち切り (`stop_reason: "repeat"`) |

---

## 実践ユースケース

### 1. 長文・複数段落テキストの翻訳

長文をそのまま翻訳すると、モデルの性質上途中で反復ループに陥ることがあります。`--segment-tokens` を指定することで、段落や文境界で自然に分割し、安定した翻訳が可能です。

```sh
.build/release/plamo --packed --stream --segment-tokens 512 --max-tokens 2048 --to en < documents.txt
```

### 2. 4区間GPUバッチで長文を高速化

`--batch4` を併用すると、モデル重みを共有しながら最大4区間をGPUで一括生成します。

```sh
.build/release/plamo --packed --stream --segment-tokens 512 --batch4 --max-tokens 2048 --to en < documents.txt
```

※ 複数区間生成時の総所要時間を大幅に短縮できます（Web UIでもこの設定が有効になっています）。

### 3. 可逆圧縮モードの活用

BF16の精度を1ビットも捨てずに、16bit整数オフセットを活用してファイルサイズとメモリを抑えます。

```sh
# 準備 (NumPy環境が必要)
.venv/bin/python scripts/pack_weights.py

# 実行
.build/release/plamo --packed --to en 'Apple Siliconの性能を最大限に引き出します。'
```

---

## アーキテクチャと最適化

本ランタイムは、PLaMo 0.8Bモデルの構造的特徴に徹底的に特化して設計されています。

1. **ゼロコピー重み参照 (Unified Memory)**
   - 通常モードはSafetensors、圧縮モードは `.p12` ファイルを読み取り専用でmmapし、Metalバッファから直接GPUが参照します。
   - モデル重み全体のホストコピーやFP32展開を一切行いません。
2. **入力埋め込みのGPUオフロード回避**
   - 約330MBを占める入力トークン埋め込みテーブルをGPUの常駐バッファから除外。
   - CPUが必要な行だけを読み出し、少量の活性値としてGPUへ転送することでVRAMを節約。
3. **専用ハイブリッドKVキャッシュ**
   - PLaMoの全16層のうち、14層の局所Attentionには直近2,048トークンのリングバッファを割り当て。
   - 残り2層の大域Attentionのみ全履歴を保持することで、メモリ使用量を大幅に抑制。
4. **Metal GPUシェーダー最適化**
   - 生成時はBF16の4要素ベクトルロードと32レーン行列ベクトル積を活用。
   - プロンプト処理（Prefill）は16トークン単位のSIMD-groupタイル行列積で高速化。
   - 回転済みキーとMLP中間活性値のメモリ領域を共有し、無駄な確保を排除。
5. **高速トークナイザー**
   - 事前生成されたTrie木キャッシュ（`tokenizer.jsonl.trie`）を直接読み込み、起動レイテンシを最小化。

---

## ベンチマーク・実測値

M2 MacBook Air（GPU 8コア、メモリ16GB）における実測値の概要です。

| 条件 | モード | 完了時間 (目安) | Metal確保量 (短文) |
|---|:---:|:---:|:---:|
| 短文（約30文字） | 通常BF16 | 約0.65秒 (起動込) / 生成約45〜55 tok/s | 約1.46 GB |
| 短文（約30文字） | 可逆圧縮 (`--packed`) | 約0.64秒 (起動込) / 生成約45〜55 tok/s | **約1.13 GB** |
| ニュース文（800文字） | 順次翻訳 | 英→日 約3.3秒 / 日→英 約7.0秒 | 約1.14 GB |
| 複数段落（3,200文字） | 逐次分割 (`--segment-tokens`) | 約13.8秒 | 約1.14 GB |
| 複数段落（3,200文字） | 4区間バッチ (`--batch4`) | **約6.1秒** (2倍以上高速) | 約1.28 GB |

> [!NOTE]
> 各種最適化の詳細な比較データ、長文検証、および実験的機能（QSA / MTP）の計測ログは **[BENCHMARK.md](BENCHMARK.md)** および **[RESEARCH.md](RESEARCH.md)** を参照してください。

---

## 開発 & テスト

### 単体テスト

```sh
swift test -c release
```

区間分割テストを実行する場合：
```sh
PLAMO_SEGMENTS=1 swift test -c release --filter SegmentTests
```

### Python公式実装との数値照合

```sh
# 検証環境のセットアップ
python3 scripts/download.py --reference
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-verify.txt

# トークナイザーおよび次トークンLogitsのRMSE検証
.venv/bin/python scripts/verify.py
.venv/bin/python scripts/verify.py --packed
```

### Web UI テスト

```sh
# サーバー単体テスト
bun run test:web

# 実ブラウザ (Chromium) E2Eテスト
bun install
bun run test:browser
```

---

## 注意事項・制限事項

- **対応言語**: モデル定義の53言語コードを受け付けますが、実用に耐えうる翻訳品質の検証は**日英・英日翻訳**を中心に行っています。
- **同時並行処理**: 単一GPUインスタンスに最適化されているため、複数プロセスからの並行同時推論は想定していません（Web UI側で1件ごとの排他制御を実施）。
- **長文生成時の注意**: 単一区間で極端に長い文章（数千トークン以上）を一括生成すると、モデルの特性上ループや破綻が生じやすくなります。長文には必ず `--segment-tokens` の併用を推奨します。

---

## ライセンス & クレジット

- **ランタイム実装**: [GNU General Public License v3.0 or later (GPL-3.0-or-later)](LICENSE)
- **モデル重み**: [pfnet/plamo-3-translate-0.8b](https://huggingface.co/pfnet/plamo-3-translate-0.8b)
  - モデルの利用・再配布には [PLaMo Community License](https://huggingface.co/pfnet/plamo-3-translate-0.8b/tree/main/LICENSE) が適用されます。商用利用条件はモデル配布元のライセンスをご確認ください。
- **デザインシステム**: [デジタル庁デザインシステム (DADS)](https://design.digital.go.jp/) (MIT License, `web/vendor/dads/` に同梱)
