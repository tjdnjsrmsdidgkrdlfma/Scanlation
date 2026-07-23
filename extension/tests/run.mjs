// Dependency-free test runner for the extension's pure JS (no bundler, no npm).
// Mirrors the Python suites' spirit: plain asserts, one file, non-zero exit on
// failure. Run with:  node extension/tests/run.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";
import assert from "node:assert/strict";

const srcDir = join(dirname(fileURLToPath(import.meta.url)), "..", "src");

// Load a browser script (no module system) into a fresh V8 global and hand back
// that global. md5.js et al. declare top-level functions, which in a vm context
// bind onto the context's global object — so ctx.md5 is the function under test.
function loadGlobals(file, seed = {}) {
  const code = readFileSync(join(srcDir, file), "utf8");
  const ctx = vm.createContext({ ...seed });
  vm.runInContext(code, ctx, { filename: file });
  return ctx;
}

let passed = 0;
let failed = 0;
function test(name, fn) {
  try {
    fn();
    console.log(`  O ${name}: PASSED`);
    passed++;
  } catch (e) {
    console.log(`  X ${name}: FAILED\n      ${e.message}`);
    failed++;
  }
}

console.log("=".repeat(60));
console.log("extension.md5");
console.log("=".repeat(60));

// md5.js declares a top-level `function md5` (a content-script global that
// content.js calls after it in the manifest js array). The expected digests are
// Python hashlib.md5(x.encode("utf-8")).hexdigest() ground truth, so a pass
// proves md5.js is byte-equal to the server's hashing — the wire cache-key
// contract. content.js hashes over the base64 STRING (see md5.js header).
const { md5 } = loadGlobals("md5.js");

// RFC 1321 §A.5 known-answer vectors (ASCII — the shape of a base64 string).
const RFC1321 = [
  ["", "d41d8cd98f00b204e9800998ecf8427e"],
  ["a", "0cc175b9c0f1b6a831c399e269772661"],
  ["abc", "900150983cd24fb0d6963f7d28e17f72"],
  ["message digest", "f96b697d7cb7938d525a2f31aaf161d0"],
  ["abcdefghijklmnopqrstuvwxyz", "c3fcd3d76192e4007dfb496cca67e13b"],
  ["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", "d174ab98d277d9f5a5611c2c9f419d9f"],
  ["12345678901234567890123456789012345678901234567890123456789012345678901234567890", "57edf4a22be3c955ac49da2e2107b67a"],
];
for (const [input, want] of RFC1321) {
  test(`md5 rfc1321 (${input.length} chars)`, () => assert.equal(md5(input), want));
}

// UTF-8 path: the encoder (unescape(encodeURIComponent(str))) must yield the same
// bytes Python hashes for non-ASCII input too, not just for the ASCII base64.
test("md5 utf-8 japanese", () => assert.equal(md5("こんにちは"), "c0e89a293bd36c7a768e4e9d2c5475a8"));
test("md5 utf-8 astral (emoji)", () => assert.equal(md5("🚀 rocket"), "0c11577563d11e6d8fa70018ea8d7470"));

// Production shape: md5 of a base64 string of image bytes. base64(bytes 0..255)
// is a fixed 344-char ASCII string spanning several 64-byte MD5 blocks.
test("md5 base64-of-image-bytes (production shape)", () => {
  const b64 = Buffer.from(Array.from({ length: 256 }, (_, i) => i)).toString("base64");
  assert.equal(md5(b64), "22b393fe586838478742ce7fa899d897");
});

console.log("\n" + "=".repeat(60));
console.log("extension.util");
console.log("=".repeat(60));

// util.js publishes pure helpers on globalThis.SCANUTIL, shared by content/
// background/popup. They touch no DOM/extension API at load, so they run here.
// Their return objects come from the vm realm, so spread into a main-realm plain
// object (`{ ...x }`) before a strict deep-equal (which else trips on prototype).
const { SCANUTIL } = loadGlobals("util.js");

test("boxFractions maps [l,b,r,t] bounds to fractions of natural size", () => {
  // bounds is [x_min, y_min, x_max, y_max]; nw=200, nh=400
  assert.deepEqual({ ...SCANUTIL.boxFractions([10, 20, 110, 120], 200, 400) },
    { left: 0.05, top: 0.05, w: 0.5, h: 0.25 });
});

test("trimEndpoint strips a single trailing slash (and tolerates null)", () => {
  assert.equal(SCANUTIL.trimEndpoint("http://127.0.0.1:4010/"), "http://127.0.0.1:4010");
  assert.equal(SCANUTIL.trimEndpoint("http://127.0.0.1:4010"), "http://127.0.0.1:4010");
  assert.equal(SCANUTIL.trimEndpoint(null), "");
});

test("authHeaders adds X-Auth-Token only for a non-empty token", () => {
  const json = { "Content-Type": "application/json" };
  assert.deepEqual({ ...SCANUTIL.authHeaders(json, "") }, { "Content-Type": "application/json" });
  assert.deepEqual({ ...SCANUTIL.authHeaders(json, "tok") },
    { "Content-Type": "application/json", "X-Auth-Token": "tok" });
  assert.deepEqual({ ...SCANUTIL.authHeaders(undefined, "tok") }, { "X-Auth-Token": "tok" });
});

console.log("\n" + "=".repeat(60));
console.log("extension.i18n");
console.log("=".repeat(60));

const { SCANI18N } = loadGlobals("i18n.js");

test("defaults to en; a known key resolves", () => {
  assert.equal(SCANI18N.lang, "en");
  assert.equal(SCANI18N.t("btn.translate"), "Translate");
});

test("setLang('ko') localizes the same key", () => {
  SCANI18N.setLang("ko");
  assert.equal(SCANI18N.t("btn.translate"), "번역");
});

test("setLang normalizes an unknown/absent code to en", () => {
  assert.equal(SCANI18N.setLang("xx"), "en");
  assert.equal(SCANI18N.setLang(undefined), "en");
});

test("t() fills {v} vars and falls back to the raw key", () => {
  SCANI18N.setLang("en");
  assert.equal(SCANI18N.t("status.connected", { v: "0.1.0" }), "connected · v0.1.0");
  assert.equal(SCANI18N.t("no.such.key"), "no.such.key");
});

test("en and ko tables have identical key sets (no drift)", () => {
  assert.deepEqual([...Object.keys(SCANI18N.STRINGS.en)].sort(),
    [...Object.keys(SCANI18N.STRINGS.ko)].sort());
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
