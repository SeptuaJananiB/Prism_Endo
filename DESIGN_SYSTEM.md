# Design System — PRISM-Endo web app

A calm, neutral review tool. **Colour is reserved for meaning** — risk level, reliability, status — never for
decoration.

| Principle | In practice |
|---|---|
| **Honesty first** | every model output shows how far to trust it (tested / not tested, low-confidence hatching) |
| **Calm by default** | neutral surfaces, 1 px hairlines, colour only for status |
| **Motion answers actions** | entrances, count-ups and highlights confirm what changed; nothing loops for decoration |
| **Keyboard-first review** | every action reachable without a mouse |

Source: [`web/src/index.css`](web/src/index.css).

## 1. Colour tokens

### Neutrals

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#f0f0ee` | `#0f0f10` | page canvas |
| `--surface` | `#ffffff` | `#161617` | cards, inputs |
| `--surface-2` | `#f5f5f3` | `#1c1c1e` | subtle fills |
| `--sidebar` | `#e8e8e5` | `#131314` | left panel |
| `--line` / `--line-strong` | `#e1e1dd` / `#cfcfca` | `#2a2a2d` / `#3a3a3e` | hairlines |
| `--ink` / `--ink-2` | `#111111` / `#2f3437` | `#f2f2f0` / `#d6d6d3` | text |
| `--muted` | `#5f5e5a` | `#a1a09c` | secondary text (AA on every surface) |

Light mode layers grey canvas → white cards → deeper grey sidebar so panels never blend into white.

### Semantic

| Token | Meaning |
|---|---|
| `--ok` | tested, clear, saved |
| `--warn` | elevated, barely tested |
| `--err` | bleeding, high risk, errors |
| `--info` | focus, links, highlights |

### Risk scale

| Category | Token |
|---|---|
| Unavailable | `--risk-unavailable` grey |
| Minimal | `--risk-minimal` green |
| Low | `--risk-low` yellow-green |
| Medium | `--risk-medium` amber |
| High | `--risk-high` red |

The risk chart colours line and fading area by the category at each moment.

## 2. Typography

| Role | Style |
|---|---|
| Body / UI | Geist Variable, 14 px / 1.55 |
| Section label | 11 px, 600, uppercase, 0.08 em tracking, muted |
| KPI figure | 20–22 px, 600, tabular numbers |
| Meta / times | 12 px, muted, tabular numbers |
| Code | Geist Mono |

## 3. Shape and spacing

- Cards 12 px radius, inputs and buttons 8 px, chips fully rounded.
- 4 px spacing grid; sidebar groups 28 px apart, fields 14 px apart.
- One content column, max 1280 px; cards align to the same grid at every width.

## 4. Components

| Component | Notes |
|---|---|
| **Card** | header with uppercase label and optional aside; hover lifts with a soft shadow |
| **KPI card** | label · count-up number · optional status chip · caption |
| **Select** | themed listbox (native popups ignore dark mode); arrows, Home/End, type-ahead, Esc |
| **Tissue chip** | toggle with reliability dot (green tested, amber barely tested, grey not tested) |
| **Switch** | "Outline all tissues"; real checkbox with `role="switch"` |
| **Stacked timeline** | one row per signal, shared playhead, drag-zoom, diamond milestones, flag pins |
| **Toast** | bottom-right, auto-dismiss, optional action |
| **Dialog** | native `<dialog>`, focus-trapped, Esc to close |

## 5. Motion

| Motion | Duration |
|---|---|
| Card entrance (rise, staggered 70 ms) | 600 ms |
| KPI count-up | 900 ms |
| Chart line draw / timeline grow | ~1 s |
| Fresh-result glow on cards | 3 s, once |
| Dropdown pop | 160 ms |

All motion is disabled under `prefers-reduced-motion`.

## 6. Accessibility

- WCAG 2 AA contrast, verified by an automated axe scan in light and dark themes.
- Visible focus ring on every control; skip link to results.
- Live regions for progress, toasts and errors; labelled form fields; meaningful button names.
- Keyboard: `Space` play/pause · `D/F` frame (`Shift` 1 s) · `J/K` events · `1/2/3` flags · `T` theme · `?` help.
