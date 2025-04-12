export enum DType {
  Float32 = "float32",
  Int32 = "int32",
  Bool = "bool",
  Complex64 = "complex64",
}

/**
 * Mathemtical expression on scalar values.
 *
 * This is similiar to and based on tinygrad's UOp class, but it's more specific
 * to just math on scalars. We're doing this to avoid the complexity of a full
 * graph rewrite engine.
 */
export class AluExp {
  #simplified?: AluExp;

  constructor(
    readonly op: AluOp,
    readonly dtype: DType,
    readonly src: AluExp[],
    readonly arg: any = undefined,
  ) {}

  static add(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Add, a.dtype, [a, b]);
  }
  static sub(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Sub, a.dtype, [a, b]);
  }
  static mul(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Mul, a.dtype, [a, b]);
  }
  static idiv(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Idiv, a.dtype, [a, b]);
  }
  static min(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Min, a.dtype, [a, b]);
  }
  static max(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Max, a.dtype, [a, b]);
  }
  static mod(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Mod, a.dtype, [a, b]);
  }
  static cast(dtype: DType, a: AluExp): AluExp {
    if (a.dtype === dtype) return a;
    return new AluExp(AluOp.Cast, dtype, [a]);
  }
  static cmplt(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Cmplt, DType.Bool, [a, b]);
  }
  static cmpne(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Cmpne, DType.Bool, [a, b]);
  }
  static where(cond: AluExp, a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Where, a.dtype, [cond, a, b]);
  }
  static const(dtype: DType, value: any): AluExp {
    return new AluExp(AluOp.Const, dtype, [], value);
  }
  static special(dtype: DType, name: string, n: number): AluExp {
    return new AluExp(AluOp.Special, dtype, [], [name, n]);
  }
  static variable(dtype: DType, name: string): AluExp {
    return new AluExp(AluOp.Variable, dtype, [], name);
  }
  static globalIndex(dtype: DType, gid: number, bufidx: AluExp): AluExp {
    return new AluExp(AluOp.GlobalIndex, dtype, [bufidx], gid);
  }

  static i32(value: number): AluExp {
    return AluExp.const(DType.Int32, value);
  }
  static f32(value: number): AluExp {
    return AluExp.const(DType.Float32, value);
  }
  static bool(value: boolean): AluExp {
    return AluExp.const(DType.Bool, value);
  }

  not(): AluExp {
    if (this.dtype !== DType.Bool) {
      throw new Error("not() can only be called on boolean expressions");
    }
    return AluExp.cmpne(this, AluExp.const(DType.Bool, true));
  }

  /** Simplify the expression by replacing any known patterns. */
  simplify(): AluExp {
    // Cache this to avoid recomputing (especially exponential blowup).
    if (this.#simplified !== undefined) return this.#simplified;
    return (this.#simplified = this.simplifyInner());
  }

  simplifyInner(): AluExp {
    const src = this.src.map((x) => x.simplify());
    const { op } = this;

    // Folding with one item being a no-op constant.
    if (AluGroup.Binary.has(op)) {
      for (let i = 0; i < 2; i++) {
        if (src[i].op !== AluOp.Const) continue;
        const x = src[i].arg;
        if (op === AluOp.Add && x === 0) return src[1 - i];
        if (op === AluOp.Sub && i === 1 && x === 0) return src[1 - i];
        if (op === AluOp.Mul && x === 1) return src[1 - i];
        if (op === AluOp.Mul && x === 0) return AluExp.const(this.dtype, 0);
        if (op === AluOp.Idiv && i === 1 && x === 1) return src[1 - i];
      }
    }

    // x + (-1 * y) => x - y
    // x - (-1 * y) => x + y
    if ((op === AluOp.Add || op === AluOp.Sub) && src[1].op === AluOp.Mul) {
      const [a, b] = src[1].src;
      const opNeg = op === AluOp.Add ? AluOp.Sub : AluOp.Add;
      if (a.op === AluOp.Const && a.arg === -1) {
        return new AluExp(opNeg, this.dtype, [src[0], b]);
      } else if (b.op === AluOp.Const && b.arg === -1) {
        return new AluExp(opNeg, this.dtype, [src[0], a]);
      }
    }

    // Select statement.
    if (op === AluOp.Where) {
      if (src[0].op === AluOp.Const) return src[src[0].arg ? 1 : 2];
    }

    // If any src was simplified, should construct a new expression.
    const newExp = src.every((s, i) => s === this.src[i])
      ? this
      : new AluExp(op, this.dtype, src, this.arg);

    // Constant folding.
    if (src.every((x) => x.op === AluOp.Const) && !AluGroup.Variable.has(op)) {
      return AluExp.const(this.dtype, newExp.evaluate({}));
    }

    return newExp;
  }

  /** Resolve this to a value, or `undefined` if not possible. */
  resolve(): any | undefined {
    const x = this.simplify();
    if (x.op === AluOp.Const) return x.arg;
    return undefined;
  }

  /**
   * Evaluate the expression on CPU, returning the result.
   *
   * Typically you would compile the AluExp as a representation to a lower-level
   * language. This is just to define the semantics and help debug.
   */
  evaluate(
    context: Record<string, any>,
    globals?: (gid: number, bufidx: number) => any,
  ): any {
    if (AluGroup.Binary.has(this.op) || AluGroup.Compare.has(this.op)) {
      const x = this.src[0].evaluate(context, globals);
      const y = this.src[1].evaluate(context, globals);
      switch (this.op) {
        case AluOp.Add:
          return this.dtype === DType.Bool ? x || y : x + y;
        case AluOp.Sub:
          return x - y;
        case AluOp.Mul:
          return this.dtype === DType.Bool ? x && y : x * y;
        case AluOp.Idiv:
          return Math.floor(x / y);
        case AluOp.Mod:
          return x % y;
        case AluOp.Min:
          return Math.min(x, y);
        case AluOp.Max:
          return Math.max(x, y);
        case AluOp.Cmplt:
          return x < y;
        case AluOp.Cmpne:
          return x != y;
        default:
          throw new Error(`Missing implemementation for ${this.op}`);
      }
    }

    if (AluGroup.Unary.has(this.op)) {
      const x = this.src[0].evaluate(context, globals);
      switch (this.op) {
        case AluOp.Sin:
          return Math.sin(x);
        case AluOp.Cos:
          return Math.cos(x);
        case AluOp.Cast:
          if (this.dtype === DType.Int32) return Math.floor(Number(x));
          else if (this.dtype === DType.Float32) return Number(x);
          else if (this.dtype === DType.Bool) return Boolean(x);
          else throw new Error(`Unsupported cast to ${this.dtype}`);
        default:
          throw new Error(`Missing implemementation for ${this.op}`);
      }
    }

    switch (this.op) {
      case AluOp.Where:
        return this.src[0].evaluate(context, globals)
          ? this.src[1].evaluate(context, globals)
          : this.src[2].evaluate(context, globals);
      case AluOp.Const:
        return this.arg;
      case AluOp.Special:
        return context[this.arg[0]];
      case AluOp.Variable:
        return context[this.arg];
      case AluOp.GlobalIndex:
        if (!globals) throw new Error("Missing globals function");
        const gid: number = this.arg;
        const bufidx = this.src[0].evaluate(context, globals);
        return globals(gid, bufidx);
      default:
        throw new Error(`Missing implemementation for ${this.op}`);
    }
  }
}

/** Symbolic form for each mathematical operation. */
export enum AluOp {
  Add = "Add",
  Sub = "Sub",
  Mul = "Mul",
  Idiv = "Idiv",
  Mod = "Mod",
  Min = "Min",
  Max = "Max",

  Sin = "Sin",
  Cos = "Cos",
  Cast = "Cast",

  Cmplt = "Cmplt",
  Cmpne = "Cmpne",
  Where = "Where",

  // Const is a literal constant, while GlobalIndex takes data from an array
  // buffer. Special and Variable are distinguished since the former is for
  // indices like the global invocation, while the latter is a value.
  Const = "Const", // arg = value
  Special = "Special", // arg = [variable, n]
  Variable = "Variable", // arg = variable
  GlobalIndex = "GlobalIndex", // arg = gid; src = [bufidx]
}

export const AluGroup = {
  Binary: new Set([
    AluOp.Add,
    AluOp.Sub,
    AluOp.Mul,
    AluOp.Idiv,
    AluOp.Mod,
    AluOp.Min,
    AluOp.Max,
  ]),
  Unary: new Set([AluOp.Sin, AluOp.Cos, AluOp.Cast]),
  Compare: new Set([AluOp.Cmplt, AluOp.Cmpne]),
  Variable: new Set([AluOp.Special, AluOp.Variable, AluOp.GlobalIndex]),
  Reduce: new Set([AluOp.Add, AluOp.Mul, AluOp.Min, AluOp.Max]),
};

/**
 * Description of a kernel to be compiled.
 *
 * Each of these can be processed by a backend into some lower-level
 * representation. It consists of one or more fused operations, optionally
 * indexing into a buffer.
 */
export class Kernel {
  constructor(
    /** Number of global arguments / arrays. */
    readonly nargs: number,
    /** Size of the result array in element count. */
    readonly size: number,
    /** Expression to be evaluated. */
    readonly exp: AluExp,
    /** Optional reduction to be performed. */
    readonly reduction?: Reduction,
  ) {}

  /** Unique string representation of this object, usable as a map key. */
  toString() {
    JSON.stringify(this, (_key, value) => {
      // Make sure that we sort the keys alphabetically for determinism.
      if (value instanceof Object && !(value instanceof Array)) {
        return Object.fromEntries(
          Object.entries(value).sort(([a], [b]) => a.localeCompare(b)),
        );
      }
      return value;
    });
  }
}

/**
 * Description of a reduction.
 *
 * The strategy of jax-js backends is to either handle a standard operation that
 * is dispatched in a vectorized way over an array, or to reduce over one axis
 * of some computation. This is a description of the reduction.
 *
 * Reduction only supports a few operations, and only over one axis. Users can
 * always `flatten()` the array before reducing if needed.
 *
 * The backend is responsible for implementing the reduction in a way that
 * minimizes the number of global memory loads, for efficiency. This involves
 * passing through some optimization strategy. But optimizations are not coded
 * at this level since they depend on GPU, versus CPU or Wasm.
 */
export class Reduction {
  constructor(
    /** Data type of the values being reduced over. */
    readonly dtype: DType,
    /** Operation to perform. Only a few reduction ops are supported. */
    readonly op: AluOp.Add | AluOp.Mul | AluOp.Min | AluOp.Max,
    /** Axis to reduce over. */
    readonly axis: number,
    /** Size of the reduction axis. */
    readonly size: number,
    /** Follow-up expression defined with the "reduced" variable, defaults to identity. */
    readonly fusion: AluExp = AluExp.variable(dtype, "reduced"),
  ) {}
}
