#!/usr/bin/env node
// Render an HTML diagram to PNG using puppeteer (headless Chrome).
// Usage: node render_diagram.js <input.html> <output.png> [--width 1200] [--scale 2]
//
// One-time setup: cd /tmp && npm install puppeteer

const puppeteer = require('puppeteer');
const path = require('path');

const args = process.argv.slice(2);
if (args.length < 2) {
  console.error('Usage: node render_diagram.js <input.html> <output.png> [--width 1200] [--scale 2]');
  process.exit(1);
}

const inputHtml = path.resolve(args[0]);
const outputPng = path.resolve(args[1]);

let width = 1200;
let scale = 2;

for (let i = 2; i < args.length; i++) {
  if (args[i] === '--width' && args[i + 1]) width = parseInt(args[i + 1]);
  if (args[i] === '--scale' && args[i + 1]) scale = parseInt(args[i + 1]);
}

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width, height: 600, deviceScaleFactor: scale });
  await page.goto('file://' + inputHtml);
  await new Promise((r) => setTimeout(r, 500));
  await page.screenshot({ path: outputPng, fullPage: true });
  console.log(`Saved: ${outputPng}`);
  await browser.close();
})();
