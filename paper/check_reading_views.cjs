const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL, fileURLToPath } = require("node:url");
const { chromium } = require(
  "C:/Users/xyx/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"
);

async function main() {
  const root = __dirname;
  const output = path.join(root, "converted");
  const report = JSON.parse(fs.readFileSync(path.join(output, "conversion_report.json"), "utf8"));
  const edge = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
  const browser = await chromium.launch({
    headless: true,
    ...(fs.existsSync(edge) ? { executablePath: edge } : {}),
  });
  const checks = [];
  try {
    for (const viewport of [{ width: 1440, height: 1050 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport });
      for (const item of report) {
        await page.goto(pathToFileURL(path.join(output, item.folder, "paper.html")).href);
        await page.locator("img").evaluateAll(async (images) => {
          await Promise.all(images.map((img) => img.decode().catch(() => null)));
        });
        const state = await page.evaluate(() => ({
          title: document.querySelector("h1")?.textContent,
          characters: document.body.innerText.length,
          images: document.images.length,
          brokenImages: [...document.images].filter((img) => !img.complete || !img.naturalWidth).length,
          mathml: document.querySelectorAll("math").length,
          overflow: document.documentElement.scrollWidth > window.innerWidth + 2,
        }));
        const localLinks = await page.locator("a").evaluateAll((links) =>
          links.map((link) => link.href).filter((href) => href.startsWith("file:"))
        );
        const missingLinks = localLinks.filter((href) => !fs.existsSync(fileURLToPath(href)));
        if (missingLinks.length) {
          throw new Error(JSON.stringify({ paper: item.number, missingLinks }));
        }
        if (!state.title || state.characters < 3000 || state.brokenImages || state.overflow) {
          const overflowElements = await page.evaluate(() => [...document.querySelectorAll("main *")]
            .filter((node) => node.getBoundingClientRect().right > innerWidth + 2)
            .filter((node) => {
              for (let parent = node.parentElement; parent; parent = parent.parentElement) {
                if (["auto", "scroll", "hidden", "clip"].includes(getComputedStyle(parent).overflowX)) return false;
              }
              return true;
            })
            .slice(0, 12).map((node) => ({
              tag: node.tagName, width: node.getBoundingClientRect().width,
              text: node.textContent.slice(0, 90),
            })));
          throw new Error(JSON.stringify({ paper: item.number, viewport, ...state, overflowElements }));
        }
        checks.push({ paper: item.number, viewport, ...state });
        if (["01", "09"].includes(item.number)) {
          await page.screenshot({
            path: path.join(root, "qa", `reading_${item.number}_${viewport.width}.png`),
          });
          const math = page.locator('math[display="block"]').first();
          if (await math.count()) {
            await math.scrollIntoViewIfNeeded();
            await page.screenshot({
              path: path.join(root, "qa", `equations_${item.number}_${viewport.width}.png`),
            });
          }
        }
      }
      await page.goto(pathToFileURL(path.join(output, "index.html")).href);
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: path.join(root, "qa", `index_${viewport.width}.png`) });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  fs.writeFileSync(path.join(root, "qa", "reading_checks.json"), JSON.stringify(checks, null, 2));
  console.log(`Verified ${checks.length} document/viewport combinations.`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
