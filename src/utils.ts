/** @file Generic programming utilities with no dependencies on library code. */

export const DEBUG: boolean = false;

export function unzip2<T, U>(pairs: Iterable<[T, U]>): [T[], U[]] {
  const lst1: T[] = [];
  const lst2: U[] = [];
  for (const [x, y] of pairs) {
    lst1.push(x);
    lst2.push(y);
  }
  return [lst1, lst2];
}

export function zip<T, U>(xs: T[], ys: U[]): [T, U][] {
  return xs.map((x, i) => [x, ys[i]]);
}

export function rep<T>(
  length: number,
  value: T,
): (T extends (...args: any) => infer R ? R : T)[] {
  if (value instanceof Function) {
    return new Array(length).fill(0).map((_, i) => value(i));
  }
  return new Array(length).fill(value);
}

/** Shorthand for integer division, like in Python. */
export function idiv(a: number, b: number): number {
  return Math.floor(a / b);
}

/** Check if two objects are deep equal. */
export function deepEqual(a: any, b: any): boolean {
  if (a === b) {
    return true;
  }
  if (typeof a !== "object" || typeof b !== "object") {
    return false;
  }
  if (a === null || b === null) {
    return false;
  }
  if (Object.keys(a).length !== Object.keys(b).length) {
    return false;
  }
  for (const key of Object.keys(a)) {
    if (!deepEqual(a[key], b[key])) {
      return false;
    }
  }
  return true;
}

export function range(start: number, stop: number, step: number = 1): number[] {
  const result = [];
  for (let i = start; i < stop; i += step) {
    result.push(i);
  }
  return result;
}

export function isPermutation(axis: number[], n: number): boolean {
  if (axis.length !== n) return false;
  const seen = new Set<number>();
  for (const x of axis) {
    if (x < 0 || x >= n) return false;
    seen.add(x);
  }
  return seen.size === n;
}

/** Topologically sort a DAG, given terminal nodes and an ancestor function. */
export function toposort<T>(terminals: T[], parents: (node: T) => T[]) {
  const childCounts: Map<T, number> = new Map();

  // First iteartion counts the number of children for each node.
  const stack = [...new Set(terminals)];
  while (true) {
    const node = stack.pop();
    if (!node) break;
    for (const parent of parents(node)) {
      if (childCounts.has(parent)) {
        childCounts.set(parent, childCounts.get(parent)! + 1);
      } else {
        childCounts.set(parent, 1);
        stack.push(parent);
      }
    }
  }
  for (const node of terminals) {
    childCounts.set(node, childCounts.get(node)! - 1);
  }

  // Second iteration produces a reverse topological order.
  const order: T[] = [];
  const frontier = terminals.filter((n) => !childCounts.get(n));
  while (true) {
    const node = frontier.pop();
    if (!node) break;
    order.push(node);
    for (const parent of parents(node)) {
      const c = childCounts.get(parent)! - 1;
      childCounts.set(parent, c);
      if (c == 0) {
        frontier.push(parent);
      }
    }
  }

  return order.reverse();
}
