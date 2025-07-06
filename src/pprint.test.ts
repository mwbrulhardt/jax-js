import { expect, suite, test } from "vitest";

import { PPrint } from "./pprint";

suite("PPrint.toString()", () => {
  test("single line", () => {
    const pp = PPrint.pp("hello");
    expect(pp.toString()).toBe("hello");
  });

  test("multi-line", () => {
    const pp = PPrint.pp("hello\nworld");
    expect(pp.toString()).toBe("hello\nworld");
  });
});

suite("PPrint.indent()", () => {
  test("increase", () => {
    const pp = PPrint.pp("hello\nworld").indent(2);
    expect(pp.toString()).toBe("  hello\n  world");
  });

  test("multiple calls", () => {
    const pp = PPrint.pp("test").indent(3).indent(2);
    expect(pp.toString()).toBe("     test");
  });
});

suite("PPrint.concat()", () => {
  test("two instances", () => {
    const pp1 = PPrint.pp("hello");
    const pp2 = PPrint.pp("world");
    expect(pp1.concat(pp2).toString()).toBe("hello\nworld");
  });

  test("multiple instances", () => {
    const pp1 = PPrint.pp("a");
    const pp2 = PPrint.pp("b");
    const pp3 = PPrint.pp("c");
    expect(pp1.concat(pp2, pp3).toString()).toBe("a\nb\nc");
  });

  test("prototype call", () => {
    expect(PPrint.prototype.concat().toString()).toBe("");
    expect(PPrint.prototype.concat(PPrint.pp("a")).toString()).toBe("a");
    expect(
      PPrint.prototype.concat(PPrint.pp("a"), PPrint.pp("b")).toString(),
    ).toBe("a\nb");
  });
});

suite("PPrint.stack()", () => {
  test("basic", () => {
    const pp1 = PPrint.pp("foo");
    const pp2 = PPrint.pp("bar");
    expect(pp1.stack(pp2).toString()).toBe("foobar");
  });

  test("indentation", () => {
    const pp1 = PPrint.pp("hello").indent(2);
    const pp2 = PPrint.pp("world").indent(4);
    expect(pp1.stack(pp2).toString()).toBe("  hello    world");
  });

  test("empty input", () => {
    const pp1 = PPrint.pp("hello");
    const pp2 = new PPrint([], []);
    expect(pp1.stack(pp2).toString()).toBe("hello");
    expect(pp2.stack(pp1).toString()).toBe("hello");
  });

  test("let and assignments", () => {
    const pp1 = PPrint.pp("let ");
    const pp2 = PPrint.pp("x=3\ny=4");
    expect(pp1.stack(pp2).toString()).toBe("let x=3\n    y=4");
  });
});
