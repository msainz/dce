package es.valcarcelsainz.dce.fn;

import static smile.math.Math.*;

/**
 * Implementation of M-dimensional Ackley function as per https://www.sfu.ca/~ssurjano/ackley.html
 * but with opposite sign for maximization, instead of minimization.
 * See also H_5(x) in http://ieeexplore.ieee.org/abstract/document/5779703/
 *
 * @author Sergio Valcarcel Macua
 */
public class Ackley extends GlobalSolutionFunction  {

    public final int M;

    public Ackley(Integer M) {
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
        final double a = 20.;
        final double b = 0.2;
        final double c = 2.*PI;
        double sum1 = 0;
        double sum2 = 0;
        for (int i=0; i<M; i++) {
            sum1 += pow(x[i],2.);
            sum2 += cos(c*x[i]);
        }
        return a*exp(-b*sqrt(1./M*sum1)) +exp(1./M*sum2) - a - exp(1.);
    }
}