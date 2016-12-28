package es.valcarcelsainz.dce.fn;

import static smile.math.Math.*;

/**
 * Implementation of M-dimensional Ackley function as per https://www.sfu.ca/~ssurjano/ackley.html
 * but with opposite sign for maximization, instead of minimization.
 * See also H_6(x) in http://ieeexplore.ieee.org/abstract/document/5779703/
 *
 * @author Sergio Valcarcel Macua
 */
public class Levy extends GlobalSolutionFunction  {

    public final int M;

    public Levy(Integer M) {
        this.M = M;
    }

    @Override
    public double[] getSoln() {
        double[] sol = new double[M];
        for (int i=0; i<M; i++) {
            sol[i] = 1.;
        }
        return sol;
    }

    @Override
    public double getOptVal() {
        double optVal = 0.0;
        return optVal;
    }

    @Override
    public double f(double[] x) {
        checkDim(x);
        double sum = 0.;
        double wi;
        for (int i=0; i<M-1; i++){
            wi = 1. + (x[i] - 1.)/4.;
            sum += pow(wi - 1, 2.)*(1 + 10*pow(sin(PI*wi + 1), 2));
        }
        double w0 = 1. + (x[0] - 1.)/4.;
        double wd = 1. + (x[M-1] - 1.)/4.;
        return - (pow(sin(PI*w0), 2.) + sum + pow(wd - 1, 2.)*(1. + pow(sin(2.*PI*wd), 2)));
    }
}