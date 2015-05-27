package es.valcarcelsainz.dce;

import static smile.math.Math.*;

/**
 * Implementation of 2-dimensional Rosenbrock function as per
 * http://en.wikipedia.org/wiki/Rosenbrock_function
 *
 * @author Marcos Sainz
 */
public class Rosenbrock extends GlobalSolutionFunction  {

    @Override
    public double[] getSoln() {
        return new double[]{1., 1.};
    }

    @Override
    public double f(double[] xy) {
        double x = xy[0];
        double y = xy[1];
        return -(pow((1. - x), 2.) + 100. * pow((y - x * x), 2.));
    }
}
