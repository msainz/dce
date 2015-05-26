    package es.valcarcelsainz.dce;

    import smile.math.MultivariateFunction;

    /**
     * http://en.wikipedia.org/wiki/Rosenbrock_function
     * 2-dim support for now.
     */
    public class Rosenbrock implements MultivariateFunction {

        private final double[] solution =
                new double[] { 1d, 100d };

        @Override
        public double f(double [] xy) {
            double x = xy[0]; double y = xy[1];
            double a = solution[0]; double b = solution[1];
            return Math.pow((a - x), 2) + b * Math.pow((y - x * x), 2);
        }
    }
