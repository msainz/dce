package es.valcarcelsainz.dce;

import static smile.math.Math.*;
import java.util.Arrays;

/**
 * Implementation of M-dimensional Trigonometric function as per
 * http://www.isr.umd.edu/~marcus/docs/MRAS_OR.pdf
 * page 559, eqn. H_5(x)
 *
 * @author Marcos Sainz
 */
public class Trigonometric extends GlobalSolutionFunction {

    @Override
    public double[] getSoln() {
        double [] soln = new double[20];
        Arrays.fill(soln, .9);
        return soln;
    }

    @Override
    public double f(double [] x) {
        int M = getDim();
        double sum = 1.;
        for (int i=0; i<M; i++) {
            sum += 8. * pow(sin(pow( 7. * (x[i] - .9), 2.)), 2.) +
                   6. * pow(sin(pow(14. * (x[i] - .9), 2.)), 2.) +
                    pow(x[i] - .9, 2.);
        }
        return -sum;
    }

}
