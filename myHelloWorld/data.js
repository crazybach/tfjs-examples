var tf = require("@tensorflow/tfjs")

module.exports = {
  generateData: function (numOfPoints, coeff, sigma = 0.04) {
    return tf.tidy(
      () => {
        // weight init
        const [a, b, c, d] = [
          tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
          tf.scalar(coeff.d)
        ];

        const xs = tf.randomUniform([numPoints], -1, 1);

        const three = tf.scalar(3, 'int32');
        //y = a * x^3 + b * x^2 + c * x + d
        const ys = a.mul(xs.pow(three))
          .add(b.mul(xs.square()))
          .add(c.mul(xs))
          .add(d)
          //random noise
          .add(tf.randomNormal([numPoints], 0, sigma));

        // Normalize the y value
        const ymin = ys.min();
        const ymax = ys.max();
        const yrange = ymax.sub(yin);
        const ysNormalized = ys.sub(ymin).div(yrange);

        return {
          xs,
          ys: ysNormalized
        };
      }
    );
  }
}
/*
export function generateData(numOfPoints, coeff, sigma = 0.04) {
  return tf.tidy(
    () => {
      // weight init
      const [a, b, c, d] = [
        tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
        tf.scalar(coeff.d)
      ];

      const xs = tf.randomUniform([numPoints], -1, 1);

      const three = tf.scalar(3, 'int32');
      //y = a * x^3 + b * x^2 + c * x + d
      const ys = a.mul(xs.pow(three))
        .add(b.mul(xs.square()))
        .add(c.mul(xs))
        .add(d)
        //random noise
        .add(tf.randomNormal([numPoints], 0, sigma));

      // Normalize the y value
      const ymin = ys.min();
      const ymax = ys.max();
      const yrange = ymax.sub(yin);
      const ysNormalized = ys.sub(ymin).div(yrange);

      return {
        xs,
        ys: ysNormalized
      };
    }
  );

}
*/
