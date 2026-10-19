const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL, fileURLToPath } = require("node:url");
const { chromium } = require(
  "C:/Users/xyx/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"
);

async function main() {
  const output = path.join(__dirname, "converted");
  const qa = path.join(__dirname, "qa");
  fs.mkdirSync(qa, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  });
  const checks = [];
  try {
    for (const viewport of [{ width: 1440, height: 1050 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport });
      const errors = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await page.goto(pathToFileURL(path.join(output, "research_index.html")).href);
      assert.equal(await page.locator("article:visible").count(), 10);
      assert.equal(await page.evaluate(() =>
        document.documentElement.scrollWidth > innerWidth + 2), false);
      const links = await page.locator("a").evaluateAll(nodes => nodes.map(a => a.href));
      for (const href of links) {
        const url = new URL(href);
        if (url.protocol !== "file:") continue;
        const fragment = decodeURIComponent(url.hash.slice(1));
        url.hash = "";
        const target = fileURLToPath(url);
        assert.ok(fs.existsSync(target), target);
        if (fragment && target.endsWith(".html")) {
          const source = fs.readFileSync(target, "utf8");
          assert.ok(source.includes(`id="${fragment}"`), href);
        }
      }
      await page.locator("#topic").selectOption("多智能体与协作");
      assert.equal(await page.locator("article:visible").count(), 3);
      await page.locator("#query").fill("zero-intelligence");
      assert.equal(await page.locator("article:visible").count(), 1);
      assert.equal(await page.locator("article:visible").getAttribute("data-number"), "05");
      assert.ok((await page.locator(".match:visible").innerText()).toLowerCase().includes("zero-intelligence"));
      await page.locator("#topic").selectOption("");
      await page.locator("#query").fill("11.14%");
      assert.equal(await page.locator("article:visible").getAttribute("data-number"), "10");
      await page.locator("#query").fill("10.1016/j.apenergy.2026.127561");
      assert.equal(await page.locator("article:visible").getAttribute("data-number"), "02");
      await page.locator("#query").fill("不存在的关键词-abcdef123456");
      assert.equal(await page.locator("article:visible").count(), 0);
      assert.equal(await page.locator("#empty").isVisible(), true);
      await page.locator("#query").fill("");
      assert.equal(await page.locator("article:visible").count(), 10);
      await page.locator("#paper-01 summary").click();
      assert.equal(await page.locator("#paper-01 details").getAttribute("open"), "");
      assert.deepEqual(errors, []);
      await page.screenshot({
        path: path.join(qa, `research_index_${viewport.width}.png`),
        fullPage: false,
      });
      checks.push({ viewport, papers: 10, links: links.length,
        filtering: "passed", fulltext: "passed", emptyState: "passed",
        anchors: "passed", overflow: false, errors });
      await page.close();
    }
    fs.writeFileSync(path.join(qa, "research_index_checks.json"),
      JSON.stringify(checks, null, 2));
    console.log(JSON.stringify(checks, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
