import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops";
import "@tensorflow/tfjs-core/dist/register_all_gradients";
import "@tensorflow/tfjs-backend-cpu";

export function array() {
  tf.tensor([
    [1, 2],
    [3, 4],
    [5, 0],
  ]).print();

  tf.grad((x) => tf.mul(x, tf.mul(x, x)))(10).print();

  return 42;
}
