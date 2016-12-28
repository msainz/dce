package es.valcarcelsainz.dce.fn;

import static smile.math.Math.PI;
import static smile.math.Math.cos;
import static smile.math.Math.pow;

/**
 * Implementation of M-dimensional Rastrigin function as per https://www.sfu.ca/~ssurjano/rastr.html
 * but with opposite sign for maximization, instead of minimization.
 * See also H_4(x) http://ieeexplore.ieee.org/abstract/document/5779703/
 *
 * @author Sergio Valcarcel Macua
 */
public class Rastrigin extends GlobalSolutionFunction  {

    public final int M;

    public Rastrigin(Integer M) {
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
        double sum = 0;
        for (int i=0; i<M; i++) {
            sum += pow(x[i],2.) - 10*cos(2.*PI*x[i]);
        }
        return -(10*M + sum);
    }
}

