// Makes listen.json: what Listen's own code (the Nitya Upasana site, beside this repository) answers for a few
// cases, so tests/test_follow.py can check that shravan/follow.py answers the same. After a change to the site's
// js/core/align.js or js/core/follow.js: node tests/fixtures/listen.mjs > tests/fixtures/listen.json
import { align, logProbs, target } from "../../../nitya-upasana/js/core/align.js";
import { letters, match, makeDoc, locate, pieceLines } from "../../../nitya-upasana/js/core/follow.js";
import { buildContent } from "../../../nitya-upasana/js/text/content.js";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
// a fixed pseudo-random generator, so the cases are the same every time
let seed = 7;
const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const cases = [];
for (let c = 0; c < 8; c++) {
  // the last two: a window that hears only the middle of a long stretch, as Listen's does
  const outputs = 6 + (c % 6), steps = 20 + 9 * (c % 6), L = c < 6 ? 3 + 2 * c : 40;
  const ids = Array.from({ length: L }, () => 2 + Math.floor(rnd() * (outputs - 2)));
  // scores that follow the letters roughly, with noise, so the path is not trivial
  const scores = new Float32Array(steps * outputs);
  for (let t = 0; t < steps; t++) {
    const k = c < 6 ? Math.floor((t / steps) * L * 1.3) - 1 : 12 + Math.floor((t / steps) * 9);
    for (let o = 0; o < outputs; o++) scores[t * outputs + o] = Math.round((rnd() * 3 - 1.5) * 1000) / 1000;
    if (k >= 0 && k < L) scores[t * outputs + ids[k]] += c % 2 ? 1.2 : 3; else scores[t * outputs + 0] += c % 2 ? 1.2 : 3;
  }
  const lp = logProbs(scores, outputs);
  cases.push({ outputs, ids, scores: Array.from(scores), log_probs: Array.from(lp), blank: 0, word: 1, expected: align(lp, outputs, ids, 0, 1) });
}
const text = "स्वादिष्ठया मदिष्ठया पवस्व सोम धारया। इन्द्राय पातवे सुतः॥ ९.००१.०१ ॥ रक्षोहा विश्वचर्षणिः";
const doc = makeDoc([{ text: "स्वादि॒ष्ठया॒ मदि॑ष्ठया॒" }, { text: "पव॑स्व सोम॒ धार॑या ।" }, { text: "धृपद ध्रुव पद", refrain: true }, { text: "इन्द्रा॑य॒ पात॑वे सु॒तः ॥ ९.००१.०१ ॥" }, { text: "र॒क्षो॒हा वि॒श्वच॑र्षणिः" }]);
const matches = [["पवस्वसोमधारया", 0, undefined], ["इन्द्रायपातवेसुत", 5, 60], ["रक्षहाविश्व", 0, undefined]].map(([h, f, t]) => ({ heard: h, from: f, to: t ?? null, expected: match(h, doc.text, f, t) }));
const locates = [
  { heard: "इन्द्राय पातवे सुतः रक्षोहा", here: null },
  { heard: "पवस्व सोम धारया इन्द्राय", here: { doc: 0, line: 1 } },
  { heard: "धृपद ध्रुव पद धृपद", here: { doc: 0, line: 3 } },
  { heard: "सोम", here: null },
].map((x) => ({ ...x, expected: locate(x.heard, [doc], x.here) }));
// every text of this repository's copy (texts/) as the site lays it out for Listen: a digest of its letters and which
// are the dhrupad, so a test can check shravan/follow.py lays every text out the same
const here = (p) => new URL(p, import.meta.url);
const lib = JSON.parse(readFileSync(here("../../texts/library.json"), "utf8"));
const files = Object.fromEntries(lib.texts.map((x) => [x.id, readFileSync(here("../../texts/" + x.file.split("/").pop()), "utf8")]));
const site = buildContent(lib, files, JSON.parse(readFileSync(here("../../../nitya-upasana/content/categories.json"), "utf8")));
const docs = {};
for (const c of site.categories)
  for (const p of c.pieces) {
    const d = makeDoc(pieceLines(p));
    docs[p.text] = createHash("sha256").update(d.text + "|" + d.refrain.map((r) => (r ? 1 : 0)).join("")).digest("hex");
  }
console.log(JSON.stringify({
  texts_commit: readFileSync(here("../../texts/SOURCE"), "utf8").match(/^commit: (\S+)/m)[1],
  docs,
  from: "nitya-upasana js/core/align.js and js/core/follow.js",
  letters: [text].map((t) => ({ line: t, expected: letters(t) })),
  doc, align: cases, match: matches, locate: locates, target: { letters: "कमलक", ids: { "क": 3, "म": 4 }, from: 10, expected: target("कमलक", { "क": 3, "म": 4 }, 10) },
}));
