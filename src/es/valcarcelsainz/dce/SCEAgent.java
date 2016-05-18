package es.valcarcelsainz.dce;

import com.google.gson.Gson;
import es.valcarcelsainz.dce.fn.*;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import smile.math.*;
import static smile.math.Math.*;
import smile.sort.IQAgent;

/**
 * Cross-Entropy Optimization algorithm to minimize a black-box cost function.
 * MAVEN_OPTS="-ea" mvn clean install exec:java -Dexec.mainClass="es.valcarcelsainz.dce.SCEAgent"
 */
public final class SCEAgent {

    private static final Logger LOG =
            LogManager.getLogger(SCEAgent.class);

    public static final Gson GSON = new Gson();

    private final double gammaQuantile;
    private final double epsilon;

    public SCEAgent(double gammaQuantile, double epsilon) {
        this.gammaQuantile = gammaQuantile;
        this.epsilon = epsilon;
    }

    public static double smoothIndicator(double y, double gamma, double epsilon) {
        return 1. / (1. + exp(-epsilon * (y - gamma)));
    }

    public static int computeNumSamplesForIteration(int iteration) {
        return (int) max(500., pow(iteration, 1.01));
    }

    public static double computeSmoothingForIteration(int iteration) {
        return 2. / pow(iteration + 100., 0.501);
    }

    /**
     * A = a * A
     * Scale each element of a matrix by a constant.
     */
    public static double [][] scaleA(double a, double [][] A) {
        for(int i = 0; i < A.length; i++) {
            for(int j = 0; j < A[0].length; j++) {
                A[i][j] *= a;
            }
        }
        return A;
    }

    public static double rmse(double [] x, double [] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException(String.format(
                    "Arrays have different length: x[%d], y[%d]", x.length, y.length)
            );
        }
        double rmse = 0.;
        for (int j = 0; j < x.length; j++) {
            rmse += pow(y[j] - x[j], 2.); // squared residuals
        }
        rmse /= x.length; // MSE
        rmse = pow(rmse, .5); // RMSE
        return rmse;
    }

    // TODO: split into initMu and initSigma
    public static void init(double[] mu, double[][] sigma, final long seed) {
        new Random(seed).nextDoubles(mu, -100d, 100d);
        for (int i = 0; i < mu.length; i++) {
            sigma[i][i] = 1000;
        }
    }

    // see eqn. (32)(top)
    // sample from latest surrogate belief distribution f, evaluate function J
    // at the samples, and compute new elite threshold gamma
    public static double sample(MultivariateGaussianDistribution f,
                           MultivariateFunction J,
                           double [][] xs, double [] ys,
                           double gammaQuantile) {
        int numSamples = xs.length;
        int M = xs[0].length;
        IQAgent gammaFinder = new IQAgent(M);
        for(int xsInd=0; xsInd < numSamples; xsInd++) {
            double [] x = f.rand();
            double y = J.f(x);
            xs[xsInd] = x;
            ys[xsInd] = y;
            gammaFinder.add(y);
        }
        double gamma = gammaFinder.quantile(gammaQuantile);
        return gamma;
    }

    // see eqn. (32)(top)
    // performs mu <- ((1 - alpha) * mu) + ((alpha / denom) * numer)
    public static void updateMu(double [] mu, double[][] xs, double[] ys,
                                double alpha, double gamma, double epsilon) {
        int numSamples = ys.length;
        assert xs.length == ys.length;
        assert xs.length > 0;
        int M = xs[0].length;
        double [] numer = new double[M];
        double denom = 0d;
        for(int xsInd = 0; xsInd < numSamples; xsInd++) {
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
    }

    // see eqn. (32)(bottom)
    // performs sigma <- (1 - alpha)(sigma + (mu_prev - mu)(mu_prev - mu)^T) + ((alpha / denom) * numer)
    public static void updateSigma(double [][] sigma,
                                   double [] mu, double [] mu_prev,
                                   double[][] xs, double[] ys,
                                   double alpha, double gamma, double epsilon) {
        int numSamples = ys.length;
        assert xs.length == ys.length;
        assert xs.length > 0;
        int M = xs[0].length;
        double[][] numer = new double[M][M];
        double denom = 0d;
        for (int xsInd = 0; xsInd < numSamples; xsInd++) {
            double[] x = xs[xsInd];
            double y = ys[xsInd];
            double I = smoothIndicator(y, gamma, epsilon);
            minus(x, mu);
            // scale(sqr(alpha * I / denom), x); // faster alternative? or numeric issues?
            double[][] A = new double[][]{x}; // 1 x M
            plus(numer, scaleA(I, atamm(A))); // M x M
            denom += I;
        }
        minus(mu_prev, mu);
        double[][] A = new double[][]{mu_prev}; // 1 x M
        plus(sigma, atamm(A)); // M x M
        scaleA(1d - alpha, sigma);
        plus(sigma, scaleA(alpha / denom, numer));
    }

    public void maximize(GlobalSolutionFunction J, double [] mu, double [][] sigma, int maxIter) {

        final int M = mu.length;
        LOG.info("initial mu: " + GSON.toJson(mu));

        for (int iter = 1; iter <= maxIter; iter++) {

            int numSamples = computeNumSamplesForIteration(iter);
            double alpha = computeSmoothingForIteration(iter);

            MultivariateGaussianDistribution f =
                    new MultivariateGaussianDistribution(mu, sigma, new Random(iter));

            double [][] xs = new double[numSamples][M];
            double [] ys = new double[xs.length];
            double gamma = sample(f, J, xs, ys, gammaQuantile);

            LOG.trace(String.format("i: %s | alpha: %s | gamma: %s | J(mu): %s | mu: %s | sigma: %s",
                    iter, alpha, gamma, J.f(mu), GSON.toJson(mu), GSON.toJson(sigma)));

            // deep copy mu -> mu_prev
            double [] mu_prev = new double[M];
            copy(mu, mu_prev);

            // mu update -- eqn. (32)
            updateMu(mu, xs, ys, alpha, gamma, epsilon);
            LOG.trace(String.format("i: %s | mu_hat: %s", iter, GSON.toJson(mu)));

            // sigma update -- eqn. (33)
            updateSigma(sigma, mu, mu_prev, xs, ys, alpha, gamma, epsilon);
            LOG.trace(String.format("i: %s | sigma_hat: %s", iter, GSON.toJson(sigma)));
            LOG.info(String.format("completed iteration(%s), rmse: %s", iter, rmse(mu, J.getSoln())));
        }

        LOG.info("final sigma: " + GSON.toJson(sigma));
        LOG.info("final mu: " + GSON.toJson(mu));
        LOG.info("final rmse: " + rmse(mu, J.getSoln()));
    }

    // MAVEN_OPTS="-ea" mvn clean install exec:java -Dexec.mainClass="es.valcarcelsainz.dce.SCEAgent"
    public static void main(String [] args) throws InterruptedException {
        final int maxIter = 500;
        final double gammaQuantile = 0.98;
        final double epsilon = 1e12;

        double [] mu;
        double [][] sigma;

        final SCEAgent SACE =
                new SCEAgent(gammaQuantile, epsilon);

        GlobalSolutionFunction [] funs = new GlobalSolutionFunction[] {
            //new Dejong(),
            //new Shekel(),
            //new Rosenbrock(),
            //new Powell(),
            new Trigonometric(200),
            //new Griewank(),
            //new Pinter(),
        };

        for (GlobalSolutionFunction J : funs) {
            int M = J.getDim();
            mu = new double[M];
            sigma = new double[M][M];
            init(mu, sigma, System.currentTimeMillis());
            SACE.maximize(J, mu, sigma, maxIter);
            logSolnAndPause(J);
        }
    }

    private static void logSolnAndPause(GlobalSolutionFunction J) throws InterruptedException {
        Logger LOG = LogManager.getLogger(SCEAgent.class);
        LOG.debug(J.getName() + " | max: " + J.getMax() + " | solution: " + GSON.toJson(J.getSoln()));
        Thread.sleep(3000);
    }
}