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
