# kotonia-cli セットアップ & 運用手順書

このドキュメントは kotonia-cli を**ゼロから動かす**ための手順書です。
「ビルド → バックエンドを1つ選ぶ → 動作確認」の順で進めます。

> kotonia-cli は LLM を駆動する**エージェント本体**にすぎません。実際に賢く動くには
> 「どのモデルに繋ぐか（バックエンド）」を必ず1つ用意する必要があります。ここが
> 最初の関門です。

---

## 0. 前提条件

| 必須 | 用途 | 確認コマンド |
|---|---|---|
| Rust toolchain (1.70+) | ビルド | `cargo --version` |
| git | worktree 隔離（既定動作） | `git --version` |
| **bash** | kotonia-cli は全コマンドを `bash -c` で実行する | `bash --version` |

- **Linux / macOS**: bash は標準で入っている。
- **Windows**: `bash` が PATH に必要。**Git for Windows（Git Bash）** か **WSL** を入れる。
  入っていないと、エージェントが出す最初の bash コマンドで `failed to spawn bash` になる。

加えて、下の「2. バックエンド」で**最低1つ**を用意する必要がある。

---

## 1. ビルド & インストール

```sh
git clone https://github.com/zhener562/kotonia-cli
cd kotonia-cli
cargo install --path .
```

`~/.cargo/bin/kotonia-cli` にバイナリが入る（`~/.cargo/bin` が PATH 上にあること）。

開発中で都度実行したいだけなら：

```sh
cargo run -- "your task here"
```

---

## 2. バックエンドを1つ選ぶ（必須）

`--model <id>` で接続先が決まる。**用途とハードに応じて1つ選ぶ。**

| 経路 | `--model` | 推論場所 | GPU | 鍵 | tool calling |
|---|---|---|---|---|---|
| **A. DeepSeek API** | `deepseek-chat` / `deepseek-reasoner` | DeepSeek社 | 不要 | `DEEPSEEK_API_KEY` | native |
| **B. kotonia ホスト** | `kotonia-gemma4-26b` / `kotonia-v4-flash` | あなたの kotonia.ai | 不要(ユーザー側) | `KOTONIA_API_KEY` | native / delimiter |
| **C. 完全ローカル** | `gemma4-26b-uncensored` / `deepseek-v4-flash` | 自分のGPU | **必須** | 不要 | native / delimiter |

**最短で動かしたいなら A**。**プライバシー/無料が目的なら C**（ただしGPUとサーバ構築が要る）。

---

### 経路 A: DeepSeek API（GPU不要・最速）

```sh
export DEEPSEEK_API_KEY=sk-...        # Windows PowerShell: $env:DEEPSEEK_API_KEY="sk-..."

kotonia-cli --model deepseek-chat "explain what this repo does"
# 推論を効かせたい難タスクは reasoner:
kotonia-cli --model deepseek-reasoner "find and fix the failing test"
```

- 鍵は DeepSeek のダッシュボードで発行。
- コード/プロンプトは DeepSeek（外部）へ送信される点に留意。

---

### 経路 B: kotonia.ai ホスト（GPU不要・自分のサーバ経由）

推論はあなたの kotonia.ai サーバーで走る。クライアント側は鍵だけでよい。

```sh
export KOTONIA_API_KEY=...            # kotonia.ai/api-manager で発行
# 自前サーバが別URLなら:
export KOTONIA_API_BASE=https://kotonia.ai   # 既定値。SSH先やlocalhostに向けるとき上書き

kotonia-cli --model kotonia-gemma4-26b "summarise the README"
```

- `kotonia-gemma4-26b` = native tools で安定（推奨）。
- `kotonia-v4-flash` = 256K の超ロングコンテキストだが delimiter で不安定・低速。長文読解専用枠。

---

### 経路 C: 完全ローカル（GPU必須・本命）

自分のGPUで vLLM / llama.cpp を立て、そこへ繋ぐ。kotonia-cli は固定ポートを見る：

| `--model` | 接続先（kotonia-cli が叩く） | サーバ |
|---|---|---|
| `gemma4-26b-uncensored` | `http://127.0.0.1:8899/v1` | vLLM（native tools） |
| `deepseek-v4-flash` | `http://127.0.0.1:8898/v1` | llama.cpp（delimiter） |

> ⚠️ **モデル配信サーバの起動スクリプト本体はこのリポジトリには含まれない**。
> 別途あなたの推論基盤（`llm_server/` 等）で立てる前提。下記は kotonia-cli が
> 期待する「最低条件」。

#### C-1. Gemma 4 26B Uncensored を vLLM :8899 に

native tool calling を使うため、vLLM は **tool-call parser を有効化**して起動する必要がある
（kotonia-cli の `local_vllm.rs` が前提にしている）。概形：

```sh
vllm serve <gemma4-26b-uncensored の重み> \
  --served-model-name gemma4-26b-uncensored \
  --port 8899 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4
```

- `--served-model-name` は **`gemma4-26b-uncensored` に一致させる**（kotonia-cli がこの名で投げる）。
- vLLM は IPv4 で bind すること（kotonia-cli は `127.0.0.1` 固定。`localhost` が `::1` に解決すると繋がらない）。

#### C-2. DeepSeek V4-Flash を llama.cpp :8898 に（任意）

CPU MoE offload の遅い超ロングコンテキスト枠。tool-call parser は無い前提なので
kotonia-cli は自動で delimiter モードにフォールバックする。OpenAI 互換サーバを
`127.0.0.1:8898` で立て、`--served-model-name` を `deepseek-v4-flash` にする。

#### 起動確認

```sh
curl -s http://127.0.0.1:8899/v1/models   # 8899 が応答すれば OK
kotonia-cli --model gemma4-26b-uncensored "list the files here and summarise the repo"
```

---

## 3. 周辺ツール（任意 — 無くても動く）

エージェントの `web_search` / `fetch_url` ツールは、PATH 上の小さな Python ラッパーに
shell out する。**無くても kotonia-cli は動く**（該当ツール呼び出しだけがエラーを返す）。

```sh
# scripts/ を PATH に通す
ln -s "$(pwd)/scripts/web-search" ~/.local/bin/web-search
ln -s "$(pwd)/scripts/fetch-url"  ~/.local/bin/fetch-url
```

- **web-search**: ローカル Searxng（既定 `http://127.0.0.1:8888`、`SEARXNG_URL` で上書き）が要る。
  最短は searxng-docker（<https://github.com/searxng/searxng-docker>）。
- **fetch-url**: `trafilatura` CLI が要る。
  ```sh
  uv tool install trafilatura   # または: pipx install trafilatura
  ```

---

## 4. 使い方

```sh
# ワンショット
kotonia-cli --model deepseek-chat "explain src/agent/agent.rs"

# 対話 REPL（引数なし）
kotonia-cli --model deepseek-chat

# パイプ入力
echo "summarise the README" | kotonia-cli --model deepseek-chat

# 過去セッション
kotonia-cli --list-sessions
kotonia-cli --resume 20260624-1530-ab12
```

### 主なフラグ

| フラグ | 既定 | 説明 |
|---|---|---|
| `--model <id>` | `deepseek-v4-flash` | バックエンド選択（**既定はローカル :8898。立ってないと接続エラー**） |
| `--approval all\|allowlist\|auto` | `allowlist` | コマンド承認の厳しさ |
| `--in-place` | off | worktree を作らず cwd で直接作業（既定は `/tmp/kotonia-agent-*` に隔離） |
| `--keep-worktree` | off | 終了後も worktree を残す（手動 merge 用） |
| `--no-history` | off | セッションログを書かない |
| `--max-iterations <n>` | 30 | 1ターンのループ上限 |

### 承認モード

- **all** … 毎コマンド `[y/N]` 確認。最も安全。
- **allowlist**（既定）… 読み取り/ビルド/テスト系は無確認、破壊的操作は確認。
  ⚠️ `python`/`node`/`cargo`/`make` 等は「安全」扱いで**無確認実行**になる（＝任意コード実行）。
  信用できないタスクでは `--approval all` を使うこと。
- **auto** … 一切確認しない。使い捨て環境専用。

### worktree の流れ（既定）

1. 起動時に現在ブランチから `/tmp/kotonia-agent-<id>` に git worktree を作成。
2. エージェントはそこで作業。**あなたの作業コピーは触られない**。
3. 終了時に worktree は自動削除（`--keep-worktree` で残せる）。残した場合は
   `git worktree remove` / `git merge kotonia-agent/<id>` で手動回収。

---

## 5. Windows 固有メモ

- **bash 必須**: Git Bash か WSL を入れ、`bash` を PATH に通す。
- 環境変数は PowerShell では `$env:DEEPSEEK_API_KEY="..."`。
- worktree の既定パスは `std::env::temp_dir()`（Windows では `%TEMP%`）下に作られる。
- 起動は Git Bash から行うと PATH 周りが素直。

---

## 6. トラブルシュート

| 症状 | 原因 / 対処 |
|---|---|
| `local LLM error: ... connection refused` | バックエンドが立っていない。経路Cなら :8899/:8898 を起動。または `--model` をAPI経路に変更 |
| `missing DEEPSEEK_API_KEY` / `missing KOTONIA_API_KEY` | 該当の鍵を export してから再実行 |
| `failed to spawn bash` | bash が PATH に無い（特に Windows）。Git Bash / WSL を導入 |
| `failed to create worktree` | git リポジトリ外で起動した。リポジトリ内で実行するか `--in-place` を付ける |
| web_search / fetch_url だけ失敗 | Searxng 未起動 / trafilatura 未導入。3章参照。無視しても本体は動く |
| 知らぬ間に Grok に送信されている | `XAI_API_KEY` が環境にあるとローカル落ち時に xAI へ自動フォールバックする。意図しないなら unset |
| native tools が効かず delimiter になる | V4-Flash 系は仕様（tool-call parser 無し）。native が要るなら gemma4 系を使う |

---

## 7. 最小動作確認チェックリスト

```sh
# 1. ビルドが通る
cargo build --release

# 2. バックエンド疎通（経路Aの例）
export DEEPSEEK_API_KEY=sk-...
kotonia-cli --model deepseek-chat "print exactly: hello-kotonia"
#   → "hello-kotonia" を含む final answer が返れば end-to-end OK

# 3. ツール実行（git リポジトリ内で）
kotonia-cli --model deepseek-chat "list the files in this directory and count them"
#   → bash ツールが ls/wc を実行し、件数を答えれば OK
```

3つ通れば「動くモノ」として最低限の実用ラインに乗っている。
