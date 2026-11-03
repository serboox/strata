import assert from "node:assert/strict";
import { test } from "node:test";
import { NumberedReads, numberLines, readView, splitReadNotice, stripLineNumbers } from "../src/code-view.ts";

const file = Array.from({ length: 300 }, (_, i) => (i % 50 === 0 ? `def function_${i}(x):` : `    value_${i} = x + ${i}`)).join("\n");

test("lines are numbered from the first line of the read", () => {
  assert.equal(numberLines("a\nb", 10), "10\ta\n11\tb");
});

test("pi's continuation notice is split off and put back", () => {
  const parts = splitReadNotice("a\nb\n\n[Showing lines 1-2 of 90. Use offset=3 to continue.]");
  assert.equal(parts.body, "a\nb");
  assert.equal(parts.notice, "[Showing lines 1-2 of 90. Use offset=3 to continue.]");
  assert.equal(splitReadNotice("plain").notice, "");
  assert.match(splitReadNotice("x\n\n[7 more lines in file. Use offset=4 to continue.]").notice, /7 more lines/);
});

test("a small file is only numbered", () => {
  const out = readView("one\ntwo\n", { firstLine: 1, lineNumbers: true, maxChars: 1000 });
  assert.equal(out, "1\tone\n2\ttwo\n");
});

test("numbering can be switched off", () => {
  assert.equal(readView("one\ntwo", { firstLine: 1, lineNumbers: false, maxChars: 1000 }), "one\ntwo");
});

test("the notice stays at the end and is not numbered", () => {
  const out = readView("a\nb\n\n[Showing lines 5-6 of 90. Use offset=7 to continue.]", { firstLine: 5, lineNumbers: true, maxChars: 1000 });
  assert.equal(out, "5\ta\n6\tb\n\n[Showing lines 5-6 of 90. Use offset=7 to continue.]");
});

test("a big file keeps both ends, original numbers and an outline of what was left out", () => {
  const out = readView(file, { firstLine: 1, lineNumbers: true, maxChars: 2500 });
  assert.ok(out.length < 3300, `length ${out.length}`);
  assert.ok(out.startsWith("1\tdef function_0(x):"));
  assert.ok(out.endsWith("300\t    value_299 = x + 299"));
  const marker = /\[\.\.\. lines (\d+)-(\d+) omitted \(~(\d+) tokens\)\.(.*) Read the part you need with offset=(\d+) and limit\.\]/.exec(out);
  assert.ok(marker, out);
  assert.equal(marker[5], marker[1]);
  assert.match(marker[4], /Definitions in the omitted part: \d+ def function_\d+/);
  const shownBefore = out.split("\n").filter((l) => /^\d+\t/.test(l));
  const numbers = shownBefore.map((l) => Number(l.split("\t")[0]));
  assert.ok(numbers.every((n, i) => i === 0 || n > numbers[i - 1]));
  assert.ok(!numbers.some((n) => n >= Number(marker[1]) && n <= Number(marker[2])));
});

test("stripLineNumbers removes the prefix only when every filled line has one", () => {
  assert.equal(stripLineNumbers("12\tfoo\n13\t  bar\n"), "foo\n  bar\n");
  assert.equal(stripLineNumbers("12\tfoo\nbar"), "12\tfoo\nbar");
  assert.equal(stripLineNumbers("plain text"), "plain text");
  assert.equal(stripLineNumbers("5\t\n6\tx"), "\nx");
});

test("an empty file stays empty", () => {
  assert.equal(readView("", { firstLine: 1, lineNumbers: true, maxChars: 100 }), "");
});

test("stripLineNumbers needs consecutive numbers when nothing is known about the file", () => {
  assert.equal(stripLineNumbers("1\ta\n1\tb"), "1\ta\n1\tb");
  assert.equal(stripLineNumbers("7\ta\n8\tb"), "a\nb");
});

test("NumberedReads strips only lines a read really showed", () => {
  const reads = new NumberedReads();
  reads.remember("/p/a.py", readView("alpha\nbeta\ngamma\n", { firstLine: 1, lineNumbers: true, maxChars: 1000 }));
  assert.equal(reads.strip("/p/a.py", "2\tbeta\n3\tgamma"), "beta\ngamma");
  assert.equal(reads.strip("/p/a.py", "2\tbeta\n3\tCHANGED"), "2\tbeta\n3\tCHANGED");
  assert.equal(reads.strip("/p/other.py", "2\tbeta"), "2\tbeta");
  reads.forget();
  assert.equal(reads.strip("/p/a.py", "2\tbeta"), "2\tbeta");
});

test("a file whose own lines start with a number and a tab keeps them", () => {
  const reads = new NumberedReads();
  reads.remember("/p/data.tsv", readView("1\tfoo\n2\tbar\n", { firstLine: 1, lineNumbers: true, maxChars: 1000 }));
  assert.equal(reads.strip("/p/data.tsv", "1\tfoo\n2\tbar"), "1\tfoo\n2\tbar");
  assert.equal(reads.strip("/p/data.tsv", "1\t1\tfoo\n2\t2\tbar"), "1\tfoo\n2\tbar");
});

test("remembered lines accumulate across reads of one file and follow the offset", () => {
  const reads = new NumberedReads();
  reads.remember("/p/a.py", readView("a\nb\n", { firstLine: 1, lineNumbers: true, maxChars: 1000 }));
  reads.remember("/p/a.py", readView("x\ny\n", { firstLine: 41, lineNumbers: true, maxChars: 1000 }));
  assert.equal(reads.strip("/p/a.py", "41\tx"), "x");
  assert.equal(reads.strip("/p/a.py", "2\tb"), "b");
});
