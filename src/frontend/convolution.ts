// Implementation of the Conv primitive (lax.conv_general_dilated).
//
// This handles both forward and transposed convolutions.
//
// Reference:
//  - https://openxla.org/xla/operation_semantics#conv_convolution
//  - https://github.com/jax-ml/jax/blob/main/jax/_src/lax/convolution.py

/** Definition of a general dilated convolution. Should be valid on creation. */
export interface ConvParams {
  strides: number[];
  padding: [number, number][];
  lhsDilation: number[];
  rhsDilation: number[];
}

/*
Rules for transposing a convolution:

Backprop of activations:
  y = conv(x, filter) -> x’ = conv(y’, filter), where

- in_channels <-> out_channels
- stride <-> lhs_dilation
- rhs_dilation stays the same
- left_padding -> (kernel_size - 1) - left_padding
- right_padding -> (kernel_size - 1) - right_padding
- kernel -> flip(kernel)

Backprop of filter:
  y = conv(x, filter) -> filter’ = conv1x1(x, y’), where

- in_channels & out_channels are transposed with batch size
- stride <-> rhs_dilation
- lhs_dilation stays the same
- padding stays the same
*/

/** Check that the parameters passed to convolution are valid. */
export function checkConvShape(
  lhsShape: number[],
  rhsShape: number[],
  { strides, padding, lhsDilation, rhsDilation }: ConvParams,
): void {
  if (lhsShape.length !== rhsShape.length) {
    throw new Error(
      `conv() requires inputs with the same number of dimensions, got ${lhsShape.length} and ${rhsShape.length}`,
    );
  }
  const n = lhsShape.length - 2;
  if (n < 0) throw new Error("conv() requires at least 2D inputs");
  if (strides.length !== n) throw new Error("conv() strides != spatial dims");
  if (padding.length !== n) throw new Error("conv() padding != spatial dims");
  if (lhsDilation.length !== n)
    throw new Error("conv() lhsDilation != spatial dimensions");
  if (rhsDilation.length !== n)
    throw new Error("conv() rhsDilation != spatial dimensions");
  if (lhsShape[1] !== rhsShape[1])
    throw new Error(`conv() input channels: ${lhsShape[1]} != ${rhsShape[1]}`);
}

/**
 * Calculate the shape of the output of a convolution.
 * Does not check that the parameters are valid, use `checkConvShape()` first.
 */
export function convShapeOut(
  lhsShape: number[],
  rhsShape: number[],
  params: ConvParams,
): number[] {
  const outShape = [lhsShape[0], rhsShape[0]]; // Batch size and out_channels
  for (let i = 2; i < lhsShape.length; i++) {
    // Each spatial dimension is computed based on strides, padding, and dilation.
    const x = lhsShape[i];
    const k = rhsShape[i];
    const s = params.strides[i - 2];
    const p = params.padding[i - 2];
    const ld = params.lhsDilation[i - 2];
    const rd = params.rhsDilation[i - 2];
    const kernelSize = Math.max(0, (k - 1) * rd + 1);
    const inSize = Math.max(0, (x - 1) * ld + 1) + p[0] + p[1];
    outShape.push(Math.ceil((inSize - kernelSize + 1) / s));
  }
  return outShape;
}
