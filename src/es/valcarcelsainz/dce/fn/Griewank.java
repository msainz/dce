package es.valcarcelsainz.dce.fn;

import java.util.Arrays;

import static smile.math.Math.*;

/**
 * Implementation of M-dimensional Griewank function as per
 * http://www.isr.umd.edu/~marcus/docs/MRAS_OR.pdf
 * page 559, eqn. H_6(x)
 *
 * @author Marcos Sainz
 */
public class Griewank extends GlobalSolutionFunction {

    public final int M;

    public Griewank(Integer M) {
        this.M = M;
    }

    @Override
    public double [] getSoln() {
        double [] soln = new double[M];
        Arrays.fill(soln, 0.);
        return soln;
    }

    @Override
    public double getOptVal() {
        double optVal = 0;
        return optVal;
    }


    @Override
    public double f(double [] x) {
        checkDim(x);
        double sum = 0.;
        double product = 1.;
        for (int i = 0; i < M; i++) {
            sum += pow(x[i],2.);
            product *= cos(x[i] / sqrt(i+1));
        }
        return -((1./4000.)*sum - product + 1);
    }

}
