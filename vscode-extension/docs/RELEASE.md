# Release and Sync

`kotonia-vscode` is published from the `vscode-extension/` subtree. The
extension bundles a released `kotonia-cli` helper binary and falls back to
`kotonia.enginePath`/PATH when no bundled helper is present.

## One-time setup

Create the public extension repository:

```bash
gh repo create zhener562/kotonia-vscode --public \
  --description "Kotonia Agent VS Code extension"
git remote add kotonia-vscode git@github.com:zhener562/kotonia-vscode.git
```

Create or verify the Visual Studio Marketplace publisher id `kotonia`, then add
`VSCE_PAT` as a GitHub Actions secret in `zhener562/kotonia-vscode`.

## Sync from the monorepo

From the repository that owns `vscode-extension/`:

```bash
git subtree split --prefix=vscode-extension -b split/kotonia-vscode
git push kotonia-vscode split/kotonia-vscode:main
git branch -D split/kotonia-vscode
```

Do this after the subtree has absorbed the intended `kotonia-cli` changes.

## CLI release first

`kotonia-cli` publishes helper binaries from its own release workflow on `v*`
tags. The VS Code extension pins the CLI release tag in `kotonia-cli.version`.
Update that file intentionally when taking a new engine into the extension.

## Extension release

In `kotonia-vscode`, tag the synced commit:

```bash
git tag v0.0.16
git push origin v0.0.16
```

The extension workflow downloads the pinned CLI release, places it at
`bin/kotonia-cli`, builds platform-specific VSIX packages for `linux-x64` and
`linux-arm64`, attaches them to the GitHub release, and publishes them to the
Marketplace when `VSCE_PAT` is configured.
