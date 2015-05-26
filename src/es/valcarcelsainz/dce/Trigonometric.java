package es.valcarcelsainz.dce;

import smile.math.MultivariateFunction;
import static smile.math.Math.*;
import java.util.Arrays;

/**
 * http://www.isr.umd.edu/~marcus/docs/MRAS_OR.pdf
 * page 559, eqn. H_5(x)
 */
public class Trigonometric implements MultivariateFunction {

    public static final int M = 20;
    public static final double[] SOLUTION = new double[M];
    static {
        Arrays.fill(SOLUTION, .9);
    }

    @Override
    public double f(double [] x) {
        double sum = 1.;
        for (int i=0; i<M; i++) {
            sum += 8. * pow(sin(pow( 7. * (x[i] - .9), 2.)), 2.) +
                   6. * pow(sin(pow(14. * (x[i] - .9), 2.)), 2.) +
                    pow(x[i] - .9, 2.);
        }
        return -sum;
    }
}
