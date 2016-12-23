package es.valcarcelsainz.dce;

import smile.math.Random;

/**
 * Created by love on 23/12/16.
 */
public class UniformDistribution implements RandomDistribution {

    int M;

    double lowerBound;
    double upperBound;
    private Random rng;

    public UniformDistribution (double lowerBound, double upperBound, int M) {
        this.M = M;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
    }

    /**
     * Generate a random multivariate Gaussian sample.
     */
    public double[] rand(final long seed) {
        double[] pt = new double[M];

        for (int i = 0; i < M; i++) {
            pt[i] = rng.nextDouble();
        }
        return pt;
    }

    public double[] rand() {
        return rand(System.currentTimeMillis() +
                Thread.currentThread().getId()
        );
    }
}
