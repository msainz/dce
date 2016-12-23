package es.valcarcelsainz.dce;

import smile.math.Math;
import smile.math.Random;
import smile.math.matrix.CholeskyDecomposition;

public class MultivariateGaussianDistribution implements RandomDistribution {

    double[] mu;
    double[][] sigma;
    //private myRandom rng;
    private Random rng;
    private double[][] sigmaL;

    public MultivariateGaussianDistribution(double[] mean, double[][] cov, Random uniformRandom) {
    //public MultivariateGaussianDistribution(double[] mean, double[][] cov, myRandom uniformRandom) {
        if (mean.length != cov.length) {
            throw new IllegalArgumentException("Mean vector and covariance matrix have different dimension");
        }

        rng = uniformRandom;
        mu = new double[mean.length];
        sigma = new double[mean.length][mean.length];
        for (int i = 0; i < mu.length; i++) {
            mu[i] = mean[i];
            System.arraycopy(cov[i], 0, sigma[i], 0, mu.length);
        }

        CholeskyDecomposition cholesky = new CholeskyDecomposition(sigma);
        sigmaL = cholesky.getL();
    }

    /**
     * Generate a random multivariate Gaussian sample.
     */
    public double[] rand(final long seed) {
        double[] spt = new double[mu.length];

        for (int i = 0; i < mu.length; i++) {
            double u, v, q;
            do {
                u = rng.nextDouble();
                v = 1.7156 * (rng.nextDouble() - 0.5);
                double x = u - 0.449871;
                double y = Math.abs(v) + 0.386595;
                q = x * x + y * (0.19600 * y - 0.25472 * x);
            } while (q > 0.27597 && (q > 0.27846 || v * v > -4 * Math.log(u) * u * u));

            spt[i] = v / u;
        }

        double[] pt = new double[sigmaL.length];

        // pt = sigmaL * spt
        for (int i = 0; i < pt.length; i++) {
            for (int j = 0; j <= i; j++) {
                pt[i] += sigmaL[i][j] * spt[j];
            }
        }

        Math.plus(pt, mu);

        return pt;
    }

    public double[] rand() {
        return rand(System.currentTimeMillis() +
                Thread.currentThread().getId()
        );
    }

}
