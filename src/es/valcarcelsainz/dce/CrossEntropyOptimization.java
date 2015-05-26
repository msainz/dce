package es.valcarcelsainz.dce;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import smile.math.*;
import smile.math.Math;
import smile.sort.IQAgent;
import smile.stat.distribution.MultivariateGaussianDistribution;
import java.util.Arrays;

/**
 * Cross-Entropy Optimization algorithm to minimize a black-box cost function.
 *
 */
public final class CrossEntropyOptimization {

    private static final Logger LOG = LogManager.getLogger(CrossEntropyOptimization.class);

    private final double gammaQuantile;
    private final double epsilon;

    public CrossEntropyOptimization(double gammaQuantile, double epsilon) {
        this.gammaQuantile = gammaQuantile;
        this.epsilon = epsilon;
    }

    private static double smoothIndicator(double y, double gamma, double epsilon) {
        return 1 / (1 + Math.exp(-epsilon * (y - gamma)));
    }

    private static int computeNumSamplesForIteration(int iteration) {
        return (int) Math.max(50, Math.pow(iteration, 1.01));
    }

    private static double computeSmoothingForIteration(int iteration) {
        return 2.0 / Math.pow(iteration + 100, 0.501);
    }

    private void scalarMult(double scalar, double [] vect) {
        for(int i=0; i<vect.length; i++) {
            vect[i] *= scalar;
        }
    }

    public void minimize(MultivariateFunction J, double [] mu, double [][] sigma, int maxIter) {

        final int M = mu.length;
        LOG.info("initial mu: " + Arrays.toString(mu));

        for (int iter = 0; iter < maxIter; iter++) {

            int numSamples = computeNumSamplesForIteration(iter);
            double alpha = computeSmoothingForIteration(iter);

            MultivariateGaussianDistribution f =
                    new MultivariateGaussianDistribution(mu, sigma);

            double [][] xs = new double[numSamples][M];
            double [] ys = new double[xs.length];
            IQAgent gammaFinder = new IQAgent(M);

            // sample from latest surrogate belief distribution f
            // and compute new elite threshold gamma
            for(int xsInd=0; xsInd < numSamples; xsInd++) {
                double [] x = f.rand();
                double y = J.f(x);
                xs[xsInd] = x;
                ys[xsInd] = y;
                gammaFinder.add(y);
            }
            double gamma = gammaFinder.quantile(gammaQuantile);

            // mu update
            double [] numer = new double[M];
            double denom = 0d;
            for(int xsInd=0; xsInd < numSamples; xsInd++) {
                double [] x = xs[xsInd];
                double y = ys[xsInd];
                double I = smoothIndicator(y, gamma, epsilon);
                scalarMult(I, x);
                Math.plus(numer, x);
                denom += I;
            }
            scalarMult(alpha / denom, numer);
            scalarMult(1d - alpha, mu);
            Math.plus(mu, numer);

            // sigma update
            // ...

            LOG.info("i: " + iter + " | mu: " + Arrays.toString(mu));
        }
    }

    public static void main(String [] args) {

        MultivariateFunction J = new Rosenbrock();
        final int M = 2;
        final int maxIter = 500;
        final double gammaQuantile = 0.75;
        final double epsilon = 1e6;

        double [] mu = new double[M];
        new Random().nextDoubles(mu, -100d, 100d);
        double [][] sigma = new double[M][M];
        for (int i = 0; i < M; i++) {
            sigma[i][i] = 1000;
        }

        new CrossEntropyOptimization(gammaQuantile, epsilon)
                .minimize(J, mu, sigma, maxIter);
    }

}