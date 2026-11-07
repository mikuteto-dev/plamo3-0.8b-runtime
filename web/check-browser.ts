import { chromium } from "playwright-core";
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { createHandler } from "./server";

const server = Bun.serve({
  hostname: "127.0.0.1", port: 0, idleTimeout: 255,
  fetch: request => handle(request),
});
const origin = `http://127.0.0.1:${server.port}`;
const handle = createHandler(origin);
const browser = await chromium.launch({
  executablePath: Bun.env.CHROME_PATH ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
});
const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, locale: "ja-JP", reducedMotion: "reduce" });
const page = await context.newPage();
const errors: string[] = [], external: string[] = [];
page.on("pageerror", error => errors.push(error.message));
page.on("request", request => { if (!request.url().startsWith(origin+"/")) external.push(request.url()); });
const passed: string[] = [];
const source = page.locator("#source"), result = page.locator("#result");
const submit = page.getByRole("button", { name: "翻訳する", exact: true });
async function complete() {
  await page.waitForFunction(() => document.querySelector("#status")?.textContent?.startsWith("翻訳しました。"), null, { timeout: 60_000 });
}
try {
  await mkdir("results", { recursive: true });
  await page.goto(origin, { waitUntil: "networkidle" });
  assert.equal(await page.title(), "翻訳 | PLaMo Translate");
  assert.equal(await source.inputValue(), "");
  assert.equal(await page.locator("#copy").isDisabled(), true);
  assert.equal(await page.locator("#cancel").isVisible(), false);
  assert.equal(await submit.evaluate(element => getComputedStyle(element).backgroundColor), "rgb(0, 23, 193)");
  await page.keyboard.press("Tab");
  assert.equal(await page.locator(".skip-link").evaluate(element => element === document.activeElement), true);
  await page.keyboard.press("Enter");
  assert.equal(await page.locator("#main").evaluate(element => element === document.activeElement), true);
  passed.push("Official DDS styles, labels, initial state and keyboard skip link");

  await source.fill("会議は来週の月曜日に変更されました。");
  await submit.click();
  await complete();
  const english = "The meeting has been rescheduled for next Monday.";
  assert.equal(await result.textContent(), english);
  assert.equal(await result.getAttribute("lang"), "en");
  const desktopInput = await source.boundingBox(), desktopOutput = await result.boundingBox();
  const desktopHeading = await page.locator("#result-heading").boundingBox();
  assert(desktopInput && desktopOutput && desktopHeading);
  assert(Math.abs(desktopInput.width-desktopOutput.width) < 1);
  assert(Math.abs(desktopInput.y-desktopOutput.y) < 1);
  assert(desktopHeading.y < desktopOutput.y);
  await context.grantPermissions(["clipboard-read", "clipboard-write"], { origin });
  await page.getByRole("button", { name: "コピー", exact: true }).click();
  assert.equal(await page.evaluate(() => navigator.clipboard.readText()), english);
  await page.screenshot({ path: "results/translation-desktop.png", fullPage: true });
  passed.push("Actual Japanese-to-English inference and clipboard copy");

  await source.fill(Array(5).fill("会議は来週の月曜日に変更されました。").join("\n\n"));
  await submit.click();
  await complete();
  assert.equal(await result.textContent(), Array(5).fill(english).join("\n\n"));
  passed.push("Five paragraphs span two GPU batches with document order and blank-line separators preserved");

  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#target").selectOption("ja");
  assert.equal(await result.textContent(), "");
  await source.fill(english);
  await submit.click();
  await complete();
  assert.match(await result.textContent() ?? "", /月曜日/);
  assert.equal(await result.getAttribute("lang"), "ja");
  const inputBox = await source.boundingBox(), outputBox = await result.boundingBox();
  const mobileHeading = await page.locator("#result-heading").boundingBox();
  assert(inputBox && outputBox && mobileHeading && inputBox.y < outputBox.y && mobileHeading.y < outputBox.y);
  assert(Math.abs(inputBox.width-outputBox.width) < 1);
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.screenshot({ path: "results/translation-mobile.png", fullPage: true });
  passed.push("Actual English-to-Japanese inference, stacked mobile layout and no horizontal overflow");

  await source.fill("<|plamo:tag|>");
  await submit.click();
  await page.locator("#error").waitFor({ state: "visible" });
  assert.match(await page.locator("#error").textContent() ?? "", /予約文字/);
  assert.equal(await source.inputValue(), "<|plamo:tag|>");
  await source.fill("a".repeat(12001));
  await submit.click();
  assert.equal(await source.getAttribute("aria-invalid"), "true");
  assert.equal((await source.inputValue()).length, 12001);
  assert.equal(await page.locator("#input-error").isVisible(), true);
  passed.push("Server errors and over-limit source remain visible without input loss");

  await page.route("**/api/translate", route => route.fulfill({
    contentType: "application/x-ndjson",
    body: '{"type":"delta","text":"途中の訳文"}\n{"type":"done","seconds":1,"truncated":true}\n',
  }));
  await source.fill("A test.");
  await submit.click();
  await page.locator("#warning").waitFor({ state: "visible" });
  assert.match(await page.locator("#warning").textContent() ?? "", /途中で終了/);
  assert.equal(await result.textContent(), "途中の訳文");
  assert.doesNotMatch(await page.locator("#status").textContent() ?? "", /翻訳しました/);
  await page.unroute("**/api/translate");
  passed.push("Explicit warning for truncated output (controlled API response)");

  await source.fill("Check the configuration before starting. ".repeat(250));
  const started = page.waitForRequest(request => request.url().endsWith("/api/translate"));
  await submit.click();
  await started;
  await Bun.sleep(100);
  const post = () => fetch(origin+"/api/translate", {
    method: "POST", headers: { Origin: origin, "Content-Type": "application/json" },
    body: JSON.stringify({ text: "Hello.", target: "ja" }),
  });
  assert.equal((await post()).status, 409);
  await page.getByRole("button", { name: "中止", exact: true }).click();
  await page.waitForFunction(() => document.querySelector("#status")?.textContent === "翻訳を中止しました。");
  let response: Response | undefined;
  for (let attempt = 0; attempt < 40; attempt++) {
    response = await post();
    if (response.status !== 409) break;
    await Bun.sleep(50);
  }
  assert.equal(response?.status, 200);
  assert.match(await response!.text(), /"type":"done"/);
  assert.equal(await source.isDisabled(), false);
  assert.equal(await page.locator("#cancel").isVisible(), false);
  passed.push("Live inference cancellation kills the job and releases the single-inference slot");

  await page.getByRole("button", { name: "消去", exact: true }).click();
  assert.equal(await source.inputValue(), "");
  assert.equal(await result.textContent(), "");
  assert.equal(await page.locator("#copy").isDisabled(), true);
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  passed.push("Clear resets the form; no page exceptions or external network requests");
  await Bun.write("results/web-browser-check.json", JSON.stringify({ passed, pageErrors: errors, externalRequests: external }, null, 2)+"\n");
  console.log(`PASS: ${passed.length} browser checks\n${passed.join("\n")}`);
} finally {
  await browser.close();
  server.stop(true);
}
