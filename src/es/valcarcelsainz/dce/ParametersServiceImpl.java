package es.valcarcelsainz.dce;

import smile.stat.distribution.MultivariateGaussianDistribution;

import java.util.Arrays;

import static smile.math.Math.copy;

public class ParametersServiceImpl implements IParametersService {

    // agent who we are serving
    private final DCEAgent clientAgent;

    // mu_i and mu_i+1
    private double[] mu[] = new double[2][];

    // sigma_i and sigma_i+1
    private double[][] sigma[] = new double[2][][];

    // synchronization locks for mu/sigma_i and mu/sigma_i+1
    private Object[] lock = new Object[] {
            new Object(),
            new Object()
    };

    protected ParametersServiceImpl(DCEAgent clientAgent) {
        this.clientAgent = clientAgent;
    }

    @Override
    public double[][] getMu() {
        return mu;
    }

    @Override
    public double[][][] getSigma() {
        return sigma;
    }

    @Override
    public Object[] getLock() {
        return lock;
    }

    @Override
    public void initParameters() {
        for (int j = 0; j < 2; j++) {
            synchronized (lock[j]) {
                int M = M();
                mu[j] = new double[M];
                sigma[j] = new double[M][M];
                CrossEntropyOptimization.unifRandInit(
                        mu[j], sigma[j], System.currentTimeMillis()
                );
            }
        }
    }

    private int M() {
        return clientAgent.getTargetFn().getDim();
    }

    @Override
    public void clearParameters(int i) {
        synchronized (lock[i]) {
            Arrays.fill(mu[i], 0);
            for (int j = 0; j < M(); j++) {
                Arrays.fill(sigma[i][j], 0);
            }
        }
    }

    @Override
    public void updateMu(int i, double weight, String mu_hat) {
        synchronized (lock[currInd(i)]) {
            // TODO
        }
    }

    @Override
    public void updateSigma(int i, double weight, String sigma_hat) {
        synchronized (lock[currInd(i)]) {
            // TODO
        }
    }

    @Override
    public String computeMuHat(int i) {
        // mu_hat depends on mu from previous iteration
        // see eqn. (32)(top)

        int M = M();
        int numSamples = CrossEntropyOptimization.computeNumSamplesForIteration(i);
        double alpha = CrossEntropyOptimization.computeSmoothingForIteration(i);

        // create new surrogate gaussian with latest estimates of mu and sigma
        MultivariateGaussianDistribution f = null;
        synchronized (lock[currInd(i)]) {
            // this constructor deep copies mu[i] and sigma[i]
            f = new MultivariateGaussianDistribution(mu[currInd(i)], sigma[currInd(i)]);
        }

        // sample from surrogate and evaluate target at the samples
        double [][] xs = new double[numSamples][M];
        double [] ys = new double[xs.length];
        double gamma = CrossEntropyOptimization.sample(f, clientAgent.getTargetFn(),
                xs, ys, clientAgent.getGammaQuantile());

        return "";

    }

    @Override
    public String computeSigmaHat(int i) {
        // sigma_hat depends on current mu and
        // mu and sigma from previous iteration
        // see eqn. (33)(top)
        String sigma_hat = "";
        String currMu, prevMu, prevSigma  = null;
        synchronized (lock[currInd(i)]) {

        }
        synchronized (lock[prevInd(i)]) {

        }
        return sigma_hat;
    }

    @Override
    public int currInd(int i) {
        return i % 2;
    }

    @Override
    public int prevInd(int i) {
        return (i + 1) % 2;
    }
}
