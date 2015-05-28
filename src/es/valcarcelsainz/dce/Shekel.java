package es.valcarcelsainz.dce;

import java.util.Arrays;

import static smile.math.Math.*;

/**
 * Implementation of 4-dimensional Shekel function as per
 * http://www.isr.umd.edu/~marcus/docs/MRAS_OR.pdf
 * page 559, eqn. H_2(x)
 *
 * @author Marcos Sainz
 */
public class Shekel extends GlobalSolutionFunction {

    @Override
    public double [] getSoln() {
        return new double[] {4.,4.,4.,4.};
    }

    @Override
    public double f(double [] x) {
        if (x.length != getDim()) {
            throw new RuntimeException(
                getName() + "only supports " + getDim() + " dimensions.");
        }
        double [][] a = new double [][] {
                {4.,4.,4.,4.},
                {1.,1.,1.,1.},
                {8.,8.,8.,8.},
                {6.,6.,6.,6.},
                {3.,7.,3.,7.},
        };
        double [] c = new double [] {.1, .2, .2, .4, .4};
        double sum = 0.;
        for (int i = 0; i < 5; i++) {
            double innersum = 0.;
            for (int j = 0; j < 4; j++) {
                innersum += pow(x[j] - a[i][j],2.);
            }
            sum += 1. / (innersum + c[i]);
        }
        return sum;
    }

}
