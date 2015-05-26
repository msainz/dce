    package es.valcarcelsainz.dce;

    import smile.math.MultivariateFunction;

    /**
     * http://en.wikipedia.org/wiki/Rosenbrock_function
     * 2-dim support for now.
     */
    public class Rosenbrock implements MultivariateFunction {

        public double[] getSolution() {
            return solution;
        }

        private final double[] solution =
                new double[] { 1d, 1d };

        @Override
        public double f(double [] xy) {
            double x = xy[0]; double y = xy[1];
            return -(Math.pow((1 - x), 2) + 100 * Math.pow((y - x * x), 2));
        }
    }
