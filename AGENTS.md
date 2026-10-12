# AGENTS.md — rules for AI coding agents

Read this before changing PRISM-Endo.

## Ground rules

1. **Honesty over polish.** Never report a metric that was not measured on held-out patients or clips. Never
   fabricate labels. If data leaked into training, say so in the docs.
2. **No clinical claims.** Keep the research-prototype disclaimer. Distances are pixels unless marked approximate;
   the risk score is a heuristic, not a probability of injury.
3. **Don't overwrite** trained models, labels or other people's results. New experiments write to new folders.
4. **Never commit** datasets, trained weights, videos, virtual environments or credentials (see `.gitignore`).
5. **One heavy GPU job at a time.**

## Environments

| Env | Use |
|---|---|
| `.venv` | CPU/CUDA, tests, Streamlit, DeepLabCut CLI |
| `.venv-dml` | AMD/Intel GPU via DirectML — training and the web API |
| `web/` (Node 20+) | React app and Playwright tests |

## Before you finish a change

```powershell
.venv\Scripts\python -m pytest -q          # must stay green
cd web; npm run build                       # type-check + build
npm run test:e2e                            # with the API running on :8510
```

## Conventions

- Python: match the surrounding style; configs in `configs/*.yaml`; new options get a config key and a test.
- Web: design tokens from `web/src/index.css`; colour only for meaning ([DESIGN_SYSTEM.md](DESIGN_SYSTEM.md));
  every control keyboard-reachable and labelled.
- Windows paths: split on both `/` and `\`. Beware shells eating backslashes in regexes.
- Long jobs: run detached (`scripts/supervisor.py start()`), log to `outputs/logs/`.

## Where things are

See [ARCHITECTURE.md](ARCHITECTURE.md) for components and API, and
[docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md) for every command and measurement.
