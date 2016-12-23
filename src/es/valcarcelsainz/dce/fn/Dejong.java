package es.valcarcelsainz.dce.fn;

import static smile.math.Math.pow;

/**
 * Implementation of 2-dimensional Dejong's 5th function as per
 * http://www.isr.umd.edu/~marcus/docs/MRAS_OR.pdf
 * pp. 558-559, eqn. H_1(x)
 *
 * @author Marcos Sainz
 */
public class Dejong extends GlobalSolutionFunction {

    public Dejong(Integer M) {
        if(M != 2)
            throw new IllegalArgumentException("Dejong only supports 2 dimensions.");
    }

    @Override
    public double [] getSoln() {
        return new double[] {-32.,-32.};
    }

    @Override
    public double getOptVal() {
        double optVal = -0.998003838818649;
        return optVal;
    }


    @Override
    public double f(double [] x) {
        checkDim(x);
        double[][] a = {
                {-32, -16,   0,  16,  32,
                 -32, -16,   0,  16,  32,
                 -32, -16,   0,  16,  32,
                 -32, -16,   0,  16,  32,
                 -32, -16,   0,  16,  32 },
                {-32, -32, -32, -32, -32,
                 -16, -16, -16, -16, -16,
                   0,   0,   0,   0,   0,
                  16,  16,  16,  16,  16,
                  32,  32,  32,  32,  32 }};
        double sum = .002;
        for (int i = 0; i < 25; i++) {
            sum += 1./ ((i+1) + pow(x[0]-a[0][i],6.) + pow(x[1]-a[1][i],6.));
        }
        return -1./sum;
    }

}
