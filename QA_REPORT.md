# QA Report — PRISM-Endo

Date: 2026-10-12 · Scope: pipeline, API and web app.

## 1. Summary

| Suite | Tool | Tests | Result |
|---|---|---|---|
| Unit / integration | pytest | 117 | all pass |
| End-to-end browser | Playwright + axe-core | 20 | all pass (also 2× repeated) |
| Accessibility | axe (WCAG 2 A/AA), light + dark | 2 scans | no violations |

Run:

```powershell
.venv\Scripts\python -m pytest -q
cd web; npm run test:e2e        # server on :8510 must be running
```

## 2. End-to-end coverage (`web/e2e/app.spec.ts`)

| # | Area | What is checked |
|---|---|---|
| 01 | Load | KPI cards, all sections, video ready, no console errors |
| 02 | Dropdowns ×2 themes | open, themed popup (not white in dark), keyboard, select, click-outside |
| 03 | Header | theme toggle, shortcuts dialog (button, `?`, Esc), downloads (HTTP 200), compare |
| 04 | Timeline | click-to-seek, drag-zoom + reset, milestones, thumbnails |
| 05 | Signals | 5 tabs (click + arrows), hover, click-to-inspect, flat-line note |
| 06 | Events · summary | filters, jump, "View" links, print |
| 07 | Reviewer flags | buttons, note, key `1`, remove, persisted (user flags restored) |
| 08 | Inspector · keys | ±1 frame / ±1 s, slider, D/F/J/K, video sync, Space |
| 09 | Upload | wrong type rejected with message, real video accepted |
| 10 | Stop | progress, Stop, toast, earlier results untouched, no temp folder left |
| 11 | Auto-open | finished analysis opens with highlight |
| 12 | Mobile | menu opens/closes the sidebar sheet |
| 13 | Accessibility ×2 | axe WCAG 2 AA |
| 14 | Tissue annotation | targets, untested warning, outline-all, overlay strength reach the analysis |
| 15 | Layout ×4 sizes | fills window, no horizontal scroll, no wrapped buttons (1100–2000 px wide) |

An `afterEach` hook stops any analysis a test left running.

## 3. Defects found and fixed

| Severity | Defect | Fix |
|---|---|---|
| High | Page froze during playback (60 Hz full re-renders, 25 frame decodes/s) | memoised charts/timeline, 10 Hz playhead, debounced inspector frame |
| High | Static route allowed reading project files (`..` path) | serve only files inside `web/dist` |
| High | Stop waited for model loading / DeepLabCut | DLC in a stoppable subprocess; stop checks between phases |
| Medium | Native dropdown popup white in dark mode (Windows) | themed accessible Select |
| Medium | Shortcuts dialog failed to reopen after fast Esc + `?` | explicit open requests; ignore stale close events |
| Medium | Tip source "Auto" skipped DeepLabCut | default Hybrid; DLC-only also auto-analyses |
| Medium | Settings change reloaded all models (10–20 s) | model cache → 0.3 s |
| Medium | Windows paths shown as `Videos\…` | split on both slash types |
| Low | "Why this risk" box changed height every frame | fixed two-line box |
| Low | Flat all-zero chart line invisible on the axis | "All values are 0" note |
| Low | App shorter than window (dark band), wrapped buttons | full-height root, no-wrap buttons |
| Low | Broken thumbnails / endless loader for runs without video | graceful fallbacks |

## 4. Known limitations

- Model accuracy limits are listed in [README.md](README.md#limitations); clinical validation is pending.
- One analysis at a time (single GPU).
- Public sharing uses a temporary tunnel; a permanent URL needs an account.
