    package es.valcarcelsainz.dce;

    import smile.math.MultivariateFunction;

    /**
     * http://en.wikipedia.org/wiki/Rosenbrock_function
     * 2-dim support for now.
     */
    public class Rosenbrock implements MultivariateFunction {

        @Override
        public double f(double [] x) {
            double x0 = x[0];
            double x1 = x[1];
            return Math.pow((1 - x0), 2) + 100 * Math.pow((x1 - x0 * x0), 2);
        }
    }
