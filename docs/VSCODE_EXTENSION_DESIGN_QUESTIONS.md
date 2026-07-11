# kotonia-cli VS Code 拡張化 — 設計確認リスト（推奨回答入り）

> 目的: 実装着手前に固めるべき設計論点を洗い出し、回答を集約する。
> **状態: 全項目を推奨（たたき台）で仮確定済み（2026-07-04）**。異論のある項目だけ `→ 回答:` を上書きしてください。
>
> 前提: Rust 製 kotonia-cli を「エンジン」として温存し、TypeScript の VS Code 拡張を別プロセスのフロントとして被せる（Claude Code for VS Code と同構造）。エージェント本体（ReAct ループ / ツール / worktree / 履歴）は無改変が原則。継ぎ目は `EventSink` trait と `ApprovalHandler` trait。
>
> 参照コード:
> - `src/agent/agent.rs` — `Event` enum (L50) / `EventSink` (L63) / `ApprovalHandler` (L44) / `run_turn` (L217)
> - `src/main.rs` — `StdoutSink` (L423) / `StdioApproval` (L490) / worktree デフォルト (L138) / `print_banner` (L333)
> - `src/agent/worktree.rs` — worktree 生成 / `cleanup` (L123)
> - `src/agent/provider.rs` — `Provider` enum (L47) / キー env 読み (L104) / `complete` 非ストリーミング (L166)

---

## 確定した3つのアーキ規定（全体の土台）

1. **#18 engine 実行環境 = Linux 側（WSL / Remote-SSH）**。Windows native engine は追わない。
2. **#13 worktree デフォルト + 差分ビュー + Merge ボタン**（in_place は設定で切替可）。
3. **#16 既定モデル = `kotonia-gemma4-26b`（hosted・GPU不要）、キーは SecretStorage → spawn 時 env 注入**。

この3つの帰結として A グループ（プロトコル封筒）は下記のとおり確定。

---

## A. プロトコル設計（Phase 1 の核・最優先）

### 1. フレーミング方式
- たたき台: **JSONL 推奨**
- → 回答: **JSONL 採用**。1行1 JSON オブジェクト。engine は改行を含む出力も `serde_json` エスケープで1行に収める。空行は無視。

### 2. メッセージ封筒（type タグ）の全集合
- → 回答: 下記セットで開始。全メッセージに `"type"` 必須。
  - **上り（拡張→engine, stdin）**: `user_turn` / `approval_response` / `cancel`
    （※ `resume` は protocol ではなく **spawn 引数**で渡す＝#24）
  - **下り（engine→拡張, stdout）**: 既存 `Event` 8種（`iteration_start` / `llm_thinking` / `bash` / `bash_skipped` / `observation` / `final` / `malformed` / `error` / `done`）+ `approval_request` + `hello`
  - Event の enum tag は serde の `#[serde(tag = "type", rename_all = "snake_case")]` で自動導出。不足は追って追加（プロトコル version を上げる）。

### 3. 相関ID
- → 回答: **`turn_id`（u64、ターン開始ごとに engine が採番）を全下りメッセージに付与**。承認は **`approval_id`（u64）を `approval_request` に付け、`approval_response` で echo** して突合。native tool の `id` は当面 observation に載せない（後で拡張可）。

### 4. ハンドシェイク（hello）の中身
- → 回答: engine 起動直後に1回 `hello` を emit。フィールド:
  `protocol_version`(int) / `model` / `backend`(local|deepseek-api|kotonia-api) / `tool_mode`(native|delimiter) / `approval_mode` / `workspace_root` / `is_worktree`(bool) / `session_id` / `kotonia_api`(bool)。
  = 現 `print_banner` の情報の機械可読版。

### 5. stdout / stderr の分離規約
- → 回答: **stdout = プロトコル JSONL 専用、stderr = 人間向けログ**で固定。JsonSink はプロトコル以外を stdout に出さない。既存の `eprintln!` 系（履歴書込失敗・retry 通知等）は stderr のまま流用。

### 6. プロトコルのバージョニング
- → 回答: `hello.protocol_version` は整数（**v1 開始**）。拡張は major 不一致で**エラー表示して接続拒否**、minor 差は許容（前方互換の追加のみ）。lockstep はモノレポ（#25）で担保。

---

## B. 承認フローの非同期境界

### 7. `ApprovalHandler::ask` が同期ブロッキング
- → 回答: **trait は無改変（同期のまま）**。JsonApproval は「`approval_request` を stdout に emit → `std::sync::mpsc` の recv で対応する `approval_response` を待つ」。ブロッキング recv は現行 `current_thread` ランタイム＋直列ループなので許容。

### 8. stdin の多重化
- → 回答: **専用の stdin リーダースレッド1本**を spawn。各行を parse し `type` で振り分け:
  - `user_turn` / `cancel` → ターン制御用の `mpsc` チャネルへ
  - `approval_response` → 承認待ち用チャネルへ（JsonApproval が recv）
  serve ループ本体は非同期、リーダーは同期スレッドで橋渡し。**Phase 1 実装の中核**。

### 9. 承認の「記憶」
- → 回答: **セッション内メモリのみの "allow for session" を採用**。`approval_response` に `remember: bool` を追加。拡張側で「コマンドの leading token（コマンド系列）」単位に記憶し、次回同系列は自動 approve。**永続化はしない**（プロセス終了で消える）。engine 側の allowlist（`approval.rs`）は無改変。

---

## C. ターン／プロセスのライフサイクル

### 10. 同時実行モデル
- → 回答: **直列1ターン**。実行中に届いた `user_turn` は**拒否**（`error` で "busy" を返す）。UI 側は実行中は入力欄を無効化。キューは持たない（将来必要なら深さ1で追加）。

### 11. キャンセル
- → 回答: **粗いキャンセルを Phase 1 に入れる**。`cancel` 受信で `AtomicBool` を立て、`run_turn` の **iteration 境界でチェックして中断**（`Event::Done{success:false}` で終了）。in-flight の bash/LLM の即時 hard-abort は後回し（tokio の future drop での中断は Phase 3 で検討）。※これは `run_turn` に cancel token を通す最小改修が必要（継ぎ目の例外的改修として許容）。

### 12. engine プロセスの単位
- → 回答: **VS Code ウィンドウ1つに engine 1プロセス**。**起動は遅延（初回プロンプト時）** — activate 時に起こさない（使わないウィンドウで worktree を作らないため）。終了は deactivate / ウィンドウ閉じで SIGTERM → engine が worktree cleanup。

---

## D. ワークスペース／worktree の IDE 化

### 13. worktree デフォルト vs in_place ★確定
- → 回答: **(a) worktree デフォルト + 差分ビュー + Merge ボタン**。CLI と一貫、レビュー後マージの安全モデル。**in_place は設定 `kotonia.workspaceMode` でワンクリック切替可**。worktree の編集はエディタに映らないので、Phase 3 で「差分を VS Code の diff ビューに出す」導線を必須実装とする。

### 14. worktree のリーク処理
- → 回答: **engine 起動時に best-effort でスタレ掃除**。`git worktree prune` +
  `kotonia-agent/*` ブランチのうち対応ディレクトリが消えているものを削除。加えて `/tmp/kotonia-agent-*` の孤児ディレクトリを一定期間（例: 24h）超で削除。正常終了時の `cleanup`（既存）はそのまま。

### 15. 非 git ワークスペース／マルチルート
- → 回答: **非 git → 自動で in_place にフォールバック**し、UI に「git 外なので隔離なし」バナー。マルチルート → **アクティブエディタのファイルが属するフォルダを repo root**、無ければ先頭ワークスペースフォルダ。曖昧時は quick-pick でユーザーに選ばせる。

---

## E. モデル・APIキー・実行環境

### 16. キー注入 ★確定
- → 回答: 既定モデル **`kotonia-gemma4-26b`**（hosted・native tools・GPU不要・鍵だけ）。キーは **VS Code SecretStorage**（`KOTONIA_API_KEY` / `DEEPSEEK_API_KEY`）に保存し、**spawn 時に子プロセス env へ注入**。`KOTONIA_API_BASE` は通常設定（既定 `https://kotonia.ai`）。`XAI_API_KEY` 等の暗黙フォールバックがあれば UI で明示表示。

### 17. モデル切替＝engine 再起動
- → 回答: **セッション開始時に model 確定。切替 = engine プロセス再起動**（Provider は Agent 構築時固定のため）。UI のモデル選択は次ターンから新プロセスで反映。

### 18. engine をどこで動かすか ★確定
- → 回答: **Linux 側で動かす**。dev = **WSL2**、GPU 箱 = **Remote-SSH**。VS Code を Remote 接続し、拡張＝engine はリモート（Linux）側で動作。理由:
  - この Windows 機は **SAC で native Rust ビルド不可**（memory `env_windows_smart_app_control_blocks_rust`）
  - engine は bash / git / ローカル LLM サーバー・`web-search`/`fetch-url` ラッパに依存（すべて Linux 側）
  - **Windows-native engine は追わない**。

### 19. バイナリの探索・配布
- → 回答: 拡張設定 **`kotonia.enginePath`** で解決。優先順:
  1. `kotonia.enginePath`（明示）→ 2. PATH 上の `kotonia-cli` → 3. .vsix 同梱バイナリ（Phase 4）。
  Phase 2 は `kotonia.enginePath` にデバッグビルド（`target/debug/kotonia-cli`、WSL パス）を指す運用。Phase 4 でプラットフォーム別バイナリを .vsix 同梱。

---

## F. UX・体験の粒度

### 20. トークンストリーミングの有無
- → 回答: **Phase 2 は粗いイベントのみ**（`llm_thinking` スピナー → `final` 一括表示）。逐次表示は `Provider::complete` のストリーミング化が要るため**後回し**（Phase 3+）。ローカルモデルの体感遅延対策は将来課題として記録。

### 21. エディタコンテキストの同梱形
- → 回答: `user_turn` に任意の **`context` フィールド**を持たせる:
  `active_file`（repo 相対パス）/ `selection`（`{start_line, end_line}`）/ `selection_text`（上限 ~2KB で truncate）。**ファイル全文は載せない**（大きすぎ・agent が bash で読める）。選択がある時のみ system 直後に「ユーザーは <path>:<range> を選択中」の短い注記を注入。

### 22. bash 出力のレンダリング
- → 回答: observation は **exit code バッジ + monospace**。長い出力（例 > 40行）は折り畳み＋展開。ANSI はストリップ/正規化。`timed_out` / `truncated` はバッジ表示（`ExecutionResult` のフラグをそのまま反映）。

---

## G. 堅牢性・運用

### 23. engine クラッシュ検知と復帰
- → 回答: 拡張が child の **exit / stdin パイプ切れを監視**。ターン中の異常終了 → UI にエラー表示 + **「再起動して session `<id>` を resume」** ボタン（セッションログはディスク上に残るため復帰可）。自動再起動はしない（ユーザー確認を挟む）。

### 24. セッション／履歴の IDE 露出
- → 回答: セッション一覧は **`~/.kotonia/sessions/` を読む**（or engine の `--list-sessions` を叩く）。**resume は spawn 引数 `--resume <id>`**（CLI 準拠、protocol メッセージにしない）。`no_history` は設定 `kotonia.noHistory` として露出。

---

## H. リポ構成・ビルド動線

### 25. モノレポ確定
- → 回答: **モノレポ**。同一 repo に `vscode-extension/`（TS）を追加。`Cargo.toml` に `exclude = ["vscode-extension/"]`。CI は Rust（cargo build/test）と TS（vsce package）を並列ジョブで。dev ループは `kotonia.enginePath` → `target/debug/kotonia-cli`（WSL パス）。protocol の共有型は engine を SoT に、TS 側は手書き or 生成した型定義を追従。

---

## 実装順（推奨回答からの帰結）

- **Phase 1（Rust）**: `Event` に `Serialize` 付与 → `JsonSink`（Event→JSONL/stdout）→ stdin リーダースレッド（#8）→ `JsonApproval`（channel 待ち, #7）→ `run_turn` に cancel token（#11）→ `serve` サブコマンド（`--protocol json` 分岐、既存 TTY モード温存）→ `hello` emit（#4）。検証: `echo '{"type":"user_turn",...}' | kotonia-cli serve`。
- **Phase 2（TS）**: `yo code` 雛形 → engine spawn（`kotonia.enginePath`, SecretStorage→env）→ JSONL パーサ + Webview 描画（粗イベント）→ 承認 UI（`approval_request`→`approval_response`）→ `user_turn` に editor context（#21）。
- **Phase 3**: worktree 差分ビュー + Merge（#13）、file:line ジャンプ、設定 UI、（余力で）ストリーミング。
- **Phase 4**: クロスビルド + .vsix 同梱（#19）、CI マトリクス、`vsce publish`。

---

## 自由記入欄（上記に収まらない論点・懸念）

- （追記があればここへ）
