package es.valcarcelsainz.dce.fn;

import java.util.Arrays;

import static smile.math.Math.pow;

/**
 * Implementation of M-dimensional Powell function as per
 * http://www.isr.umd.edu/~marcus/docs/MRAS_OR.pdf
 * page 559, eqn. H_4(x)
 *
 * @author Marcos Sainz
 */
public class Powell extends GlobalSolutionFunction {

    public final int M;

    public Powell(Integer M) {
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
        for (int i = 1; i < M-2; i++) {
            sum += pow(x[i-1]+10.*x[i],2.) + 5.*pow(x[i+1]-x[i+2],2.) +
                    pow(x[i]-2.*x[i+1],4.) + 10.*pow(x[i-1]-x[i+2],4.);
        }
        return -sum;
    }

}
