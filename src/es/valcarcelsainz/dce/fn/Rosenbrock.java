package es.valcarcelsainz.dce.fn;

import static smile.math.Math.*;
import java.util.Arrays;

/**
 * Implementation of M-dimensional Rosenbrock function as per
 * http://www.isr.umd.edu/~marcus/docs/MRAS_OR.pdf
 * page 559, eqn. H_3(x)
 *
 * @author Marcos Sainz
 */
public class Rosenbrock extends GlobalSolutionFunction  {

    @Override
    public double [] getSoln() {
        double [] soln = new double[4];
        Arrays.fill(soln, 1.);
        return soln;
    }

    @Override
    public double f(double [] x) {
        checkDim(x);
        int M = getDim();
        double sum = 0.;
        for (int i = 0; i < M-1; i++) {
            sum += 100.*pow(x[i+1]-pow(x[i],2.),2.) + pow(x[i]-1,2.);
        }
        return -sum;
    }
}
