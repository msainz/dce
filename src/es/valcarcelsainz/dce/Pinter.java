package es.valcarcelsainz.dce;

import static smile.math.Math.*;

/**
 * Implementation of M-dimensional Pintér function as per
 * http://www.isr.umd.edu/~marcus/docs/MRAS_OR.pdf
 * page 559, eqn. H_7(x)
 *
 * @author Marcos Sainz
 */
public class Pinter extends GlobalSolutionFunction {

    @Override
    public double[] getSoln() {
        return new double[20];
    }

    @Override
    public double f(double[] _x) {
        int M = getDim();
        double [] x = new double[M+2];
        System.arraycopy(_x, 0, x, 1, M);
        x[0] = x[M];
        x[M+1] = x[1];
        double sum = 0.;
        for (int i=1; i<=M; i++) {
            sum += i*pow(x[i],2.) + 20.*i*pow(sin(x[i-1]*sin(x[i]-x[i]+sin(x[i+1]))),2.) +
                i*log10(1.+i*pow(pow(x[i-1]-2.*x[i]+3.*x[i+1]-cos(x[i])+1.,2.),2.));
        }
        return -sum;
    }

}
