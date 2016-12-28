package es.valcarcelsainz.dce.fn;

import static smile.math.Math.*;

/**
 * Implementation of M-dimensional Zakharov function as per https://www.sfu.ca/~ssurjano/zakharov.html
 * but with opposite sign for maximization, instead of minimization.
 * See also H_3(x) http://ieeexplore.ieee.org/abstract/document/5779703/
 *
 * @author Sergio Valcarcel Macua
 */
public class Zakharov extends GlobalSolutionFunction  {

    public final int M;

    public Zakharov(Integer M) {
        this.M = M;
    }

    @Override
    public double[] getSoln() {
        return new double[M];
    }

    @Override
    public double getOptVal() {
        double optVal = 0.0;
        return optVal;
    }

    @Override
    public double f(double[] x) {
        checkDim(x);
        double sum1 = 0;
        double sum2 = 0;
        for (int i=0; i<M; i++) {
            sum1 += pow(x[i],2.);
            sum2 += 0.5*i*x[i];
        }
        return -(sum1 + pow(sum2, 2.) + pow(sum2, 4.));
    }
}
