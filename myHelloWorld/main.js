var tf = require("@tensorflow/tfjs");

// y= ax^3 + bx^2 + c^x + d

const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

const x = tf.scalar(1);


function predict(x) {
  return tf.tidy(
    () => {
      return a.mul(x.pow(tf.scalar(3))) // a * x ^ 3
        .add(b.mul(x.square()))   //+ b * x^2
        .add(c.mul(x))
        .add(d);
    }
  );
}

/** loss function */

function loss(predictions, labels) {
  const meanSquareError = predictions.sub(labels).square().mean()
}




/** training loop */
function train(xs, ys, numIteration = 75) {

  /** Optimize: SGD */
  const learningRate = 0.5;
  const optimizer = tf.train.sgd(learningRate);

  for (let index = 0; index < numIteration; index++) {
    optimizer.minimize(
      () => {
        const predsYs = predict(xs);
        return loss(predsYs, ys);
      }
    );

  }

}
