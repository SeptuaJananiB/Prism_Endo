import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

const ROOT = path.resolve(HERE, "../..");
const SHOTS = path.join(HERE, "../e2e-shots");
const RUN = "appendicectomy_bleed"; // stable 1,122-frame run with events
const SHORT = "Videos/video01_00240.mp4"; // 80 frames, ~20 s to analyse
fs.mkdirSync(SHOTS, { recursive: true });

const errors: string[] = [];
async function open(page: Page, theme: "light" | "dark" = "light") {
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error" && !/Failed to load resource/.test(m.text())) errors.push(m.text()); });
  await page.addInitScript((t) => localStorage.setItem("pe-theme", t), theme);
  await page.goto("/#app");
  await expect(page.locator("#run-pick")).toBeVisible();
}
async function pickRun(page: Page, name: string) {
  await page.locator("#run-pick").click();
  await page.getByRole("option", { name, exact: false }).filter({ hasText: new RegExp(`^${name}`) }).first().click();
  await expect(page.locator("h1")).toHaveText(name);
  await expect(page.locator("input#frame")).toBeVisible();
}
const frameVal = (page: Page) => page.locator("input#frame").inputValue();
const shot = (page: Page, n: string) => page.screenshot({ path: path.join(SHOTS, `${n}.png`), fullPage: false });

test.afterAll(() => { if (errors.length) console.log("Console errors:\n" + errors.join("\n")); });

// never leave an analysis running after a test (a failed test must not block the next ones)
test.afterEach(async ({ request }) => {
  const a = await (await request.get("/api/active")).json();
  if (a.id) {
    await request.post(`/api/jobs/${a.id}/cancel`);
    await expect.poll(async () => (await (await request.get("/api/active")).json()).id ?? null, { timeout: 30_000 }).toBeNull();
  }
});

test("01 page loads: header, KPIs, videos, sections, no console errors", async ({ page }) => {
  await open(page);
  await pickRun(page, RUN);
  for (const l of ["Frames", "Speed", "Tip source", "Medium / high risk", "Bleeding", "CVS met"])
    await expect(page.locator(".card .label", { hasText: l }).first()).toBeVisible();
  for (const s of ["Timeline", "Summary", "Events", "Reviewer flags", "Signals", "Frame inspector", "Analysis settings", "Why this risk"])
    await expect(page.getByRole("region", { name: s })).toBeVisible();
  await expect(page.locator("video").first()).toHaveJSProperty("readyState", 4, { timeout: 20_000 });
  await shot(page, "01-light-overview");
  expect(errors, errors.join("\n")).toEqual([]);
});

for (const theme of ["light", "dark"] as const) {
  test(`02 every dropdown opens themed, keyboard works, selects (${theme})`, async ({ page }) => {
    await open(page, theme);
    const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    for (const id of ["src", "surg", "kp", "run-pick"]) {
      const btn = page.locator(`#${id}`);
      await btn.click();
      const list = page.locator(`#${id}-list`);
      await expect(list).toBeVisible();
      const listBg = await list.evaluate((e) => getComputedStyle(e).backgroundColor);
      // the native-select bug: a white popup in dark mode
      if (theme === "dark") expect(listBg, `${id} popup is white in dark mode`).not.toBe("rgb(255, 255, 255)");
      expect(await list.locator("[role=option]").count()).toBeGreaterThan(0);
      await shot(page, `02-${theme}-dropdown-${id}`);
      await page.keyboard.press("Escape");
      await expect(list).toBeHidden();
      // keyboard: open, move, choose
      await btn.focus();
      await page.keyboard.press("ArrowDown");
      await expect(list).toBeVisible();
      await page.keyboard.press("Escape");
    }
    // choose a different tip source with the keyboard
    await page.locator("#kp").focus();
    await page.keyboard.press("Enter");
    await page.keyboard.press("End");
    await page.keyboard.press("Enter");
    await expect(page.locator("#kp")).toContainText("Geometry only");
    // click outside closes
    await page.locator("#surg").click();
    await page.mouse.click(900, 500);
    await expect(page.locator("#surg-list")).toBeHidden();
    expect(bg).toBeTruthy();
  });
}

test("03 header buttons: theme, shortcuts dialog, download menu, compare", async ({ page }) => {
  await open(page);
  await pickRun(page, RUN);
  const html = page.locator("html");
  await page.getByRole("button", { name: /Switch to dark theme/ }).click();
  await expect(html).toHaveAttribute("data-theme", "dark");
  await shot(page, "03-dark-overview");
  await page.getByRole("button", { name: /Switch to light theme/ }).click();
  await expect(html).toHaveAttribute("data-theme", "light");

  await page.getByRole("button", { name: "Keyboard shortcuts", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await shot(page, "03-shortcuts");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toBeHidden();
  await page.locator("body").press("?");
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("dialog").getByRole("button", { name: "Close", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeHidden();

  await page.getByRole("button", { name: "Download" }).click();
  const items = page.getByRole("menuitem");
  await expect(items).toHaveCount(3);
  for (const href of await items.evaluateAll((els) => els.map((e) => (e as HTMLAnchorElement).href))) {
    const r = await page.request.get(href);
    expect(r.status(), href).toBe(200);
  }
  await page.keyboard.press("Escape");
  await expect(items.first()).toBeHidden();

  await page.getByRole("button", { name: "Compare" }).click();
  const cmp = page.getByRole("region", { name: "Compare analyses" });
  await expect(cmp).toBeVisible();
  await page.locator("#cmp").click();
  await page.locator("#cmp-list [role=option]").nth(1).click();
  await expect(cmp).toContainText("of frames agree");
  await shot(page, "03-compare");
  await cmp.getByRole("button", { name: "Close" }).click();
  await expect(cmp).toBeHidden();
});

test("04 timeline: click seeks, drag zoom + reset, milestones, thumbnails", async ({ page }) => {
  await open(page);
  await pickRun(page, RUN);
  const track = page.getByRole("img", { name: "Risk track" });
  await track.scrollIntoViewIfNeeded();
  const box = (await track.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.75, box.y + box.height / 2);
  const f1 = Number(await frameVal(page));
  expect(f1).toBeGreaterThan(700);
  await page.mouse.move(box.x + box.width * 0.2, box.y + 10);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, box.y + 10, { steps: 8 });
  await page.mouse.up();
  await expect(page.getByRole("button", { name: "Reset zoom" })).toBeVisible();
  await shot(page, "04-timeline-zoomed");
  await page.getByRole("button", { name: "Reset zoom" }).click();
  await expect(page.getByRole("button", { name: "Reset zoom" })).toBeHidden();
  const ms = page.getByRole("button", { name: /^Peak risk/ });
  await ms.click();
  expect(Number(await frameVal(page))).not.toBe(f1);
  const thumbs = page.getByRole("button", { name: /^Jump to .* seconds$/ });
  await expect(thumbs).toHaveCount(10);
  await thumbs.nth(2).click();
  const f3 = Number(await frameVal(page));
  expect(f3).toBeGreaterThan(150);
  expect(f3).toBeLessThan(400);
});

test("05 signals tabs (click + arrow keys) all draw a chart", async ({ page }) => {
  await open(page);
  await pickRun(page, RUN);
  for (const t of ["Risk", "Distance", "Tip speed", "CVS", "Bleeding"]) {
    await page.getByRole("tab", { name: t }).click();
    await expect(page.getByRole("tab", { name: t })).toHaveAttribute("aria-selected", "true");
    // a chart is drawn, or the panel says why not (e.g. all values 0)
    await expect(page.locator("[role=tabpanel] svg path.draw, [role=tabpanel] .flat-note, [role=tabpanel] p").first()).toBeAttached();
    expect(await page.locator("[role=tabpanel] svg path.draw").count() + await page.locator("[role=tabpanel] .flat-note").count()).toBeGreaterThan(0);
  }
  await page.getByRole("tab", { name: "Bleeding" }).press("ArrowRight");
  await expect(page.getByRole("tab", { name: "Risk" })).toHaveAttribute("aria-selected", "true");
  const svg = page.locator("[role=tabpanel] svg");
  await svg.scrollIntoViewIfNeeded();
  const b = (await svg.boundingBox())!;
  await page.mouse.move(b.x + b.width * 0.5, b.y + b.height * 0.5);
  await expect(page.locator("[role=tabpanel] figcaption")).toContainText("click to inspect");
  await page.mouse.click(b.x + b.width * 0.5, b.y + b.height * 0.5);
  expect(Number(await frameVal(page))).toBeGreaterThan(400);
});

test("06 events filter + click, summary View links, Print", async ({ page }) => {
  await open(page);
  await pickRun(page, RUN);
  const ev = page.getByRole("region", { name: "Events" });
  const rows = ev.locator("ol button");
  expect(await rows.count()).toBeGreaterThan(0);
  const before = await frameVal(page);
  await rows.nth(1).click();
  expect(await frameVal(page)).not.toBe(before);
  const chips = ev.getByRole("group", { name: "Filter events" }).getByRole("button");
  for (let i = 0; i < (await chips.count()); i++) { await chips.nth(i).click(); await expect(chips.nth(i)).toHaveAttribute("aria-pressed", "true"); }
  await chips.first().click();
  const sum = page.getByRole("region", { name: "Summary" });
  const view = sum.getByRole("button", { name: "View" }).first();
  const b2 = await frameVal(page);
  await view.click();
  expect(await frameVal(page)).not.toBe(b2);
  await page.evaluate(() => { (window as unknown as { __printed: number }).__printed = 0; window.print = () => { (window as unknown as { __printed: number }).__printed++; }; });
  await sum.getByRole("button", { name: "Print / PDF" }).click();
  expect(await page.evaluate(() => (window as unknown as { __printed: number }).__printed)).toBe(1);
});

test("07 reviewer flags: buttons, note, keyboard 1/2/3, remove, persisted", async ({ page, request }) => {
  await open(page);
  await pickRun(page, RUN);
  const orig = await (await request.get(`/api/runs/${RUN}/flags`)).json();
  try {
    const fp = page.getByRole("region", { name: "Reviewer flags" });
    await fp.locator("#flag-note").fill("e2e test note");
    await fp.getByRole("button", { name: "Agree" }).click();
    await fp.getByRole("button", { name: "AI wrong" }).click();
    await fp.getByRole("button", { name: "Important" }).click();
    await page.locator("body").press("1");
    await expect(fp.locator("ol li")).toHaveCount(orig.length + 4);
    await expect.poll(async () => (await (await request.get(`/api/runs/${RUN}/flags`)).json()).length).toBe(orig.length + 4);
    await expect(page.getByRole("button", { name: /Reviewer flag agree at/ }).first()).toBeVisible();
    await shot(page, "07-flags");
    for (let i = 0; i < 4; i++) {
      await fp.locator("ol li").first().hover();
      await fp.getByRole("button", { name: /Remove flag/ }).first().click();
    }
    await expect.poll(async () => (await (await request.get(`/api/runs/${RUN}/flags`)).json()).length).toBe(orig.length);
  } finally {
    await request.put(`/api/runs/${RUN}/flags`, { data: orig });
    const f = path.join(ROOT, "outputs/runs", RUN, "review_flags.json");
    if (!orig.length && fs.existsSync(f)) fs.unlinkSync(f);
  }
});

test("08 inspector buttons, slider, keyboard D/F/J/K, video sync", async ({ page }) => {
  await open(page);
  await pickRun(page, RUN);
  const ins = page.getByRole("region", { name: "Frame inspector" });
  await page.locator("input#frame").fill("500");
  expect(await frameVal(page)).toBe("500");
  await ins.getByRole("button", { name: "Next frame" }).click();
  expect(await frameVal(page)).toBe("501");
  await ins.getByRole("button", { name: "Previous frame" }).click();
  await ins.getByRole("button", { name: "Forward one second" }).click();
  expect(await frameVal(page)).toBe("525");
  await ins.getByRole("button", { name: "Back one second" }).click();
  expect(await frameVal(page)).toBe("500");
  await expect(ins.locator("img")).toHaveJSProperty("complete", true);
  await page.locator("body").press("f");
  expect(await frameVal(page)).toBe("501");
  await page.locator("body").press("d");
  await page.locator("body").press("k");
  expect(Number(await frameVal(page))).toBeGreaterThan(500);
  await page.locator("body").press("j");
  // the annotated video follows the playhead
  const t = await page.locator("video").first().evaluate((v: HTMLVideoElement) => v.currentTime);
  expect(t).toBeGreaterThan(5);
  // play / pause with Space
  await page.locator("body").press(" ");
  await page.waitForTimeout(1200);
  await page.locator("body").press(" ");
  const paused = await page.locator("video").first().evaluate((v: HTMLVideoElement) => v.paused);
  expect(paused).toBe(true);
});

test("09 upload: wrong type rejected with message, valid video accepted", async ({ page }) => {
  await open(page);
  const input = page.locator('input[type=file]');
  await input.setInputFiles({ name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("x") });
  await expect(page.getByRole("alert")).toContainText(/unsupported/i);
  const vid = path.join(ROOT, "Videos/video01_00080.mp4");
  const dest = path.join(ROOT, "outputs/uploads/e2e_upload.mp4");
  try {
    await input.setInputFiles({ name: "e2e_upload.mp4", mimeType: "video/mp4", buffer: fs.readFileSync(vid) });
    await expect(page.getByRole("status").filter({ hasText: "Uploaded e2e_upload.mp4" })).toBeVisible();
    await expect(page.locator("#src")).toContainText("e2e_upload.mp4");
  } finally {
    if (fs.existsSync(dest)) fs.unlinkSync(dest);
  }
});

test("10 analyse + stop: progress, Stop, toast, existing results untouched", async ({ page, request }) => {
  await open(page);
  const before = fs.statSync(path.join(ROOT, "outputs/runs/appendicectomy_max/summary.json")).mtimeMs;
  await page.locator("#src").click();
  await page.getByRole("option", { name: "appendicectomy_max.mp4" }).click();
  await page.getByRole("button", { name: "Analyse video" }).click();
  await expect(page.getByRole("progressbar")).toBeVisible();
  await expect(page.getByText(/Processing appendicectomy_max/)).toBeVisible({ timeout: 60_000 });
  await shot(page, "10-running");
  await page.getByRole("button", { name: "Stop analysis" }).click();
  await expect(page.getByRole("status").filter({ hasText: "Analysis stopped" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("button", { name: "Analyse video" })).toBeEnabled();
  expect(fs.statSync(path.join(ROOT, "outputs/runs/appendicectomy_max/summary.json")).mtimeMs).toBe(before);
  expect(fs.readdirSync(path.join(ROOT, "outputs/runs")).filter((d) => d.startsWith(".partial"))).toEqual([]);
  expect(await (await request.get("/api/active")).json()).toEqual({});
});

test("11 analyse short video: auto-opens result with highlight", async ({ page }) => {
  await open(page);
  await pickRun(page, RUN);
  await page.locator("#src").click();
  await page.getByRole("option", { name: path.basename(SHORT) }).click();
  await page.getByRole("button", { name: "Analyse video" }).click();
  await expect(page.locator(".fresh")).toBeVisible({ timeout: 110_000 });
  await expect(page.locator("h1")).toHaveText("video01_00240");
  await expect(page.getByRole("status").filter({ hasText: "Showing the results" })).toBeVisible();
  await shot(page, "11-fresh-result");
});

test("12 mobile layout: menu opens sidebar sheet, closes", async ({ page }) => {
  await page.setViewportSize({ width: 420, height: 860 });
  await page.goto("/#app");
  await expect(page.getByRole("button", { name: "Open menu" })).toBeVisible();
  await page.getByRole("button", { name: "Open menu" }).click();
  await expect(page.getByRole("complementary", { name: "Controls" })).toBeInViewport({ ratio: 0.9 });
  await shot(page, "12-mobile-menu");
  await page.getByRole("button", { name: "Close menu" }).click();
  await expect(page.getByRole("complementary", { name: "Controls" })).not.toBeInViewport();
});

for (const theme of ["light", "dark"] as const) {
  test(`13 accessibility scan (axe, WCAG 2 AA) - ${theme}`, async ({ page }) => {
    await open(page, theme);
    await pickRun(page, RUN);
    await page.waitForTimeout(2500); // let entrance animations finish (axe measures real colours, not mid-fade)
    const res = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).exclude("video").analyze();
    const v = res.violations.map((x) => `${x.id} (${x.impact}): ${x.nodes.length} - ${x.nodes.slice(0, 2).map((n) => n.target.join(" ")).join(" | ")}`);
    expect(v, v.join("\n")).toEqual([]);
  });
}

test("14 tissue annotation: all tissues by default, deselect, untested warning, reaches the analysis", async ({ page }) => {
  await open(page);
  const tissue = page.locator('section[aria-label="3 · Tissue annotation"]');
  const chips = tissue.locator("button[aria-pressed]");
  const n = await chips.count();
  expect(n).toBeGreaterThan(5);
  await expect(tissue.locator('button[aria-pressed="true"]')).toHaveCount(n); // all selected by default
  await expect(tissue.getByRole("switch")).toHaveCount(0);                    // outline switch removed
  await expect(page.locator("#alpha")).toHaveCount(0);                        // overlay slider removed
  await expect(tissue.getByRole("note")).toHaveCount(0);                      // untested-tissue warning removed
  await tissue.getByRole("button", { name: "Clear all" }).click();
  await expect(tissue.locator('button[aria-pressed="true"]')).toHaveCount(0);
  const chip = (name: string) => chips.filter({ hasText: new RegExp(`^${name},`) });
  await chip("Liver").click();
  await chip("Gallbladder").click();
  await shot(page, "14-tissue-fields");
  const reqP = page.waitForRequest((r) => r.url().endsWith("/api/run") && r.method() === "POST");
  await page.locator("#src").click();
  await page.getByRole("option", { name: "video01_00240.mp4" }).click();
  await page.getByRole("button", { name: "Analyse video" }).click();
  const body = (await reqP).postDataJSON();
  expect(body.targets).toEqual(["liver", "gallbladder"]);
  expect(body.show_all_tissue).toBe(true);
  await expect(page.locator(".fresh")).toBeVisible({ timeout: 110_000 });
  const sum = JSON.parse(fs.readFileSync(path.join(ROOT, "outputs/runs/video01_00240/summary.json"), "utf-8"));
  expect(Object.keys(sum.proximity_targets).sort()).toEqual(["gallbladder", "liver"]);
});

for (const vp of [{ width: 1280, height: 720 }, { width: 1600, height: 900 }, { width: 2000, height: 1125 }, { width: 1100, height: 800 }]) {
  test(`15 layout ${vp.width}x${vp.height}: fills window, no overflow, no wrapped buttons`, async ({ page }) => {
    await page.setViewportSize(vp);
    await open(page, "dark");
    await pickRun(page, RUN);
    await page.waitForTimeout(1200);
    const r = await page.evaluate(() => {
      const root = document.querySelector("#root > div")!.getBoundingClientRect();
      const wrapped = [...document.querySelectorAll("button.btn, .chip")]
        .filter((b) => (b as HTMLElement).offsetParent && b.getBoundingClientRect().height > 46)
        .map((b) => (b.textContent ?? "").trim().slice(0, 30));
      const overflowX = document.documentElement.scrollWidth > window.innerWidth + 1;
      const cards = [...document.querySelectorAll("main .card")].filter((c) => {
        const b = c.getBoundingClientRect(); return b.width > 0 && b.right > window.innerWidth + 1;
      }).length;
      return { rootH: Math.round(root.height), winH: window.innerHeight, wrapped, overflowX, cardsOffscreen: cards };
    });
    expect(r.rootH, "app does not fill the window height").toBe(r.winH);
    expect(r.wrapped, "buttons wrapped onto two lines").toEqual([]);
    expect(r.overflowX, "horizontal scroll").toBe(false);
    expect(r.cardsOffscreen).toBe(0);
    for (const y of [0, 900, 1800]) {
      await page.locator("main").evaluate((m, yy) => m.scrollTo(0, yy), y);
      await page.waitForTimeout(250);
      await shot(page, `15-layout-${vp.width}x${vp.height}-${y}`);
    }
  });
}


test("16 loading screen, landing page, AI explanation card", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("status", { name: "Loading PRISM-Endo" })).toBeVisible();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("See every instrument");
  await shot(page, "16-landing");
  for (const s of ["What it does", "How it works", "Recent analyses"]) await expect(page.getByRole("heading", { name: s })).toBeVisible();
  await page.getByRole("button", { name: "Start analysing" }).click();
  await expect(page.locator("#run-pick")).toBeVisible();
  await expect(page).toHaveURL(/#app$/);
  const ai = page.getByRole("complementary", { name: "AI explanation" });
  if (!(await ai.isVisible())) await page.getByRole("button", { name: "AI", exact: true }).click();
  await expect(ai).toBeVisible();
  // resizable: drag the left edge, keyboard on the handle, close / reopen
  const handle = ai.getByRole("separator", { name: "Resize panel" });
  const w0 = Number(await handle.getAttribute("aria-valuenow"));
  await handle.focus();
  await page.keyboard.press("ArrowLeft");
  await expect(handle).toHaveAttribute("aria-valuenow", String(Math.min(760, w0 + 24)));
  const hb = (await handle.boundingBox())!;
  await page.mouse.move(hb.x + hb.width / 2, hb.y + hb.height / 2);
  await page.mouse.down();
  await page.mouse.move(hb.x - 120, hb.y + hb.height / 2, { steps: 6 });
  await page.mouse.up();
  expect(Number(await handle.getAttribute("aria-valuenow"))).toBeGreaterThan(w0 + 24);
  await ai.getByRole("button", { name: "Close AI explanation" }).click();
  await expect(ai).toBeHidden();
  await page.getByRole("button", { name: "AI", exact: true }).click();
  await expect(ai).toBeVisible();
  const status = await (await page.request.get("/api/llm/status")).json();
  if (status.configured) {
    await ai.getByRole("button", { name: "Explain this analysis" }).click();
    await expect(ai.getByRole("heading", { name: /What happened in the video/i })).toBeVisible({ timeout: 60_000 });
    await expect(ai.getByRole("group", { name: "Read aloud" })).toBeVisible();
    await expect(ai.getByRole("button", { name: "Read explanation aloud" })).toBeVisible();
    await expect(ai.getByRole("radiogroup", { name: "Reading speed" })).toBeVisible();
  }
  else await expect(ai).toContainText("GROQ_API_KEY");
  await page.getByRole("button", { name: "Home" }).click();
  await expect(page.getByRole("button", { name: "Open workspace" })).toBeVisible();
});
