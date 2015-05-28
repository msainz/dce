package es.valcarcelsainz.dce;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import smile.math.*;
import static smile.math.Math.*;
import smile.sort.IQAgent;
import smile.stat.distribution.MultivariateGaussianDistribution;
import java.util.Arrays;

/**
 * Cross-Entropy Optimization algorithm to minimize a black-box cost function.
 *
 */
public final class CrossEntropyOptimization {

    private static final Logger LOG =
            LogManager.getLogger(CrossEntropyOptimization.class);

    private final double gammaQuantile;
    private final double epsilon;

    public CrossEntropyOptimization(double gammaQuantile, double epsilon) {
        this.gammaQuantile = gammaQuantile;
        this.epsilon = epsilon;
    }

    private static double smoothIndicator(double y, double gamma, double epsilon) {
        return 1 / (1 + exp(-epsilon * (y - gamma)));
    }

    private static int computeNumSamplesForIteration(int iteration) {
        return (int) max(500., pow(iteration, 1.01));
    }

    private static double computeSmoothingForIteration(int iteration) {
        return 2. / pow(iteration + 100., 0.501);
    }

    /**
     * A = a * A
     * Scale each element of a matrix by a constant.
     */
    private static double [][] scaleA(double a, double [][] A) {
        for(int i=0; i<A.length; i++) {
            for(int j=0; j<A[0].length; j++) {
                A[i][j] *= a;
            }
        }
        return A;
    }

    public void init(double [] mu, double [][] sigma, final long seed) {
        new Random(seed).nextDoubles(mu, -100d, 100d);
        for (int i = 0; i < mu.length; i++) {
            sigma[i][i] = 1000;
        }
    }

    public void maximize(MultivariateFunction J, double [] mu, double [][] sigma, int maxIter) {

        final int M = mu.length;
        LOG.info("initial mu: " + Arrays.toString(mu));

        for (int iter = 0; iter < maxIter; iter++) {

            int numSamples = computeNumSamplesForIteration(iter);
            double alpha = computeSmoothingForIteration(iter);

            MultivariateGaussianDistribution f =
                    new MultivariateGaussianDistribution(mu, sigma);

            // sample from latest surrogate belief distribution f
            // and compute new elite threshold gamma
            double [][] xs = new double[numSamples][M];
            double [] ys = new double[xs.length];
            IQAgent gammaFinder = new IQAgent(M);
            for(int xsInd=0; xsInd < numSamples; xsInd++) {
                double [] x = f.rand();
                double y = J.f(x);
                xs[xsInd] = x;
                ys[xsInd] = y;
                gammaFinder.add(y);
            }

            double gamma = gammaFinder.quantile(gammaQuantile);

            // Arrays.sort(ys);
            // LOG.debug("gamma: " + gamma + " | ys: " + Arrays.toString(ys));

            // mu update -- eqn. (32)
            double [] mu_prev = new double[M];
            copy(mu, mu_prev); // deep copy mu -> mu_prev
            double [] numer = new double[M];
            double denom = 0d;
            for(int xsInd=0; xsInd < numSamples; xsInd++) {
                double [] x = xs[xsInd];
                double y = ys[xsInd];
                double I = smoothIndicator(y, gamma, epsilon);
                scale(I, x);
                plus(numer, x);
                denom += I;
            }
            scale(alpha / denom, numer);
            scale(1d - alpha, mu);
            plus(mu, numer);

            // sigma update -- eqn. (33)
            double [][] numer2 = new double[M][M];
            for(int xsInd=0; xsInd < numSamples; xsInd++) {
                double [] x = xs[xsInd];
                double y = ys[xsInd];
                double I = smoothIndicator(y, gamma, epsilon);
                minus(x, mu);
                // scale(sqr(alpha * I / denom), x); // faster alternative? or numeric issues?
                double [][] A = new double[][] { x }; // 1 x M
                plus(numer2, scaleA(I, atamm(A)) ); // M x M
            }
            minus(mu_prev, mu);
            double [][] A = new double[][] { mu_prev }; // 1 x M
            plus(sigma, atamm(A)); // M x M
            scaleA(1d - alpha, sigma);
            plus(sigma, scaleA(alpha / denom, numer2) );

            LOG.debug("i: " + iter + " | alpha: " + alpha + " | J(mu): " + J.f(mu) + " | mu: " + Arrays.toString(mu));
        }
    }

    public static void main(String [] args) throws InterruptedException {

        final int maxIter = 500;
        final double gammaQuantile = 0.96;
        final double epsilon = 1e12;

        double [] mu;
        double [][] sigma;

        final CrossEntropyOptimization SACE =
                new CrossEntropyOptimization(gammaQuantile, epsilon);

        GlobalSolutionFunction [] funs = new GlobalSolutionFunction[] {
            new Rosenbrock(),
            new Trigonometric(),
            new Pinter(),
            new Powell(),
            new Griewank(),
            new Shekel(),
        };

        for (GlobalSolutionFunction J : funs) {
            int M = J.getDim();
            mu = new double[M];
            sigma = new double[M][M];
            SACE.init(mu, sigma, 0L);
            SACE.maximize(J, mu, sigma, maxIter);
            logSolnAndPause(J);
        }
    }

    private static void logSolnAndPause(GlobalSolutionFunction J) throws InterruptedException {
        Logger LOG = LogManager.getLogger(CrossEntropyOptimization.class);
        LOG.debug(J.getName() + " | max: " + J.getMax() + " | solution: " + Arrays.toString(J.getSoln()));
        Thread.sleep(3000);
    }
}