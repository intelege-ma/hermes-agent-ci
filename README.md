# Hermes Agent CI Mirror

> ⚠️ **This is NOT the Hermes Agent project.** This is a sanitised, fresh-history
> CI mirror used exclusively to run GitHub Actions on a private fork without
> exposing private repository history.

For the real project, see **[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)**.

## What this repo is

A public mirror of a [private Hermes Agent fork](https://github.com/intelege-ma/hermes-agent)
(not publicly visible). It receives sanitised file snapshots — no git history,
no private paths, no secrets. Used to run CI (lint, test, build) on free
GitHub Actions runners.

## What this repo is NOT

- **Not** the canonical Hermes Agent source — that's at NousResearch/hermes-agent
- **Not** a fork you can contribute to — the private repo is not public
- **Not** guaranteed to be up to date — exports happen on demand for CI verification

## Branches

- `main` — sanitised baseline export
- `pr-N-*` — CI-only verification branches for private PRs (may be stale/deleted)

## License

MIT — inherited from NousResearch/hermes-agent.
