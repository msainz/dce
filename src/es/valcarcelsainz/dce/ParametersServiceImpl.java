package es.valcarcelsainz.dce;

import com.google.gson.reflect.TypeToken;
import es.valcarcelsainz.dce.fn.GlobalSolutionFunction;
import smile.math.Random;

import java.lang.reflect.Type;
import java.util.Arrays;

import static smile.math.Math.plus;
import static smile.math.Math.scale;

public class ParametersServiceImpl implements IParametersService {

    // agent who we are serving
    private final DCEAgent clientAgent;

    // mu_i and mu_i+1
    private double[] mus[] = new double[2][];

    // sigma_i and sigma_i+1
    private double[][] sigmas[] = new double[2][][];

    // synchronization locks for mu/sigma_i and mu/sigma_i+1
    private Object[] locks = new Object[] {
            new Object(),
            new Object()
    };

    protected ParametersServiceImpl(DCEAgent clientAgent) {
        this.clientAgent = clientAgent;
    }

    @Override
    public double[][] getMus() {
        return mus;
    }

    @Override
    public double[][][] getSigmas() {
        return sigmas;
    }

    @Override
    public Object[] getLocks() {
        return locks;
    }

    @Override
    public void initParameters() {
        for (int j = 0; j < 2; j++) {
            synchronized (locks[j]) {
                int M = M();
                mus[j] = new double[M];
                sigmas[j] = new double[M][M];
                CrossEntropyOptimization.init(mus[j], sigmas[j],
                        System.currentTimeMillis() + Thread.currentThread().getId()
                );

//                mus[j][0] = 66.3948655128479;
//                mus[j][1] = -8.485877513885498;
//                mus[j][2] = 86.19184494018555;
//                mus[j][3] = 0.46983957290649414;
            }
        }
    }

    protected int M() {
        return clientAgent.getTargetFn().getDim();
    }

    @Override
    public void clearParameters(int i) {
        synchronized (locks[i]) {
            Arrays.fill(mus[i], 0);
            for (int j = 0; j < M(); j++) {
                Arrays.fill(sigmas[i][j], 0);
            }
        }
    }

    @Override
    // create new surrogate gaussian with mu and sigma from prev iteration
    // which won't receive more updates by the time this method is called
    public Object[] sampleNewGaussian(int i, double[][] xs, double[] ys, GlobalSolutionFunction J) {
        MultivariateGaussianDistribution f = new MultivariateGaussianDistribution(
                // this constructor deep-copies mu[i] and sigma[i]
                // later accessible via f.mean() and f.cov() respectively
                mus[prevInd(i)], sigmas[prevInd(i)],
                new Random(System.currentTimeMillis() + Thread.currentThread().getId()) // new Random(i)
        );
        double gamma = CrossEntropyOptimization.sample(
                f, J, xs, ys, clientAgent.getGammaQuantile()
        );
        return new Object[] {f, gamma};
    }

    @Override
    public void updateMu(int i, double weight, double [] mu_hat) {
        synchronized (locks[currInd(i)]) {
            // using a+bc == b(a/b+c) to avoid
            // allocating a new array
            double [] mu = mus[currInd(i)];
            double inverse_weight = 1./weight;
            for (int j = 0; j < mu.length; j++) {
                mu[j] *= inverse_weight;
                mu[j] += mu_hat[j];
                mu[j] *= weight;
            }
        }
    }

    @Override
    public void updateMu(int i, double weight, String mu_hat) {
        double [] muHat = clientAgent.GSON.fromJson(mu_hat, double[].class);
        updateMu(i, weight, muHat);
    }

    @Override
    public void updateSigma(int i, double weight, double[][] sigma_hat) {
        synchronized (locks[currInd(i)]) {
            // using a+bc == b(a/b+c) to avoid
            // allocating a new array
            double [][] sigma = sigmas[currInd(i)];
            double inverse_weight = 1./weight;
            for(int j = 0; j < sigma.length; j++) {
                for(int k = 0; k < sigma[0].length; k++) {
                    sigma[j][k] *= inverse_weight;
                    sigma[j][k] += sigma_hat[j][k];
                    sigma[j][k] *= weight;
                }
            }
        }
    }

    @Override
    public void updateSigma(int i, double weight, String sigma_hat) {
        Type sigmaType = new TypeToken<double[][]>() {}.getType();
        double [][] sigmaHat = clientAgent.GSON.fromJson(sigma_hat, sigmaType);
        updateSigma(i, weight, sigmaHat);
    }

    @Override
    public void computeMuHat(int i, double[] mu_hat, double [][] xs, double [] ys,
                             double alpha, double gamma, double epsilon) {
        // see eqn. (32)(top)
        CrossEntropyOptimization.updateMu(
                mu_hat, xs, ys, alpha, gamma, epsilon
        );
    }

    @Override
    public void computeSigmaHat(int i, double [][] sigma_hat, double [][] xs, double [] ys,
                                double alpha, double gamma, double epsilon) {

        // neither will receive updates when this method is called
        double [] mu_prev = mus[prevInd(i)];
        double [] mu_curr = mus[currInd(i)];

        // see eqn. (33)(top)
        CrossEntropyOptimization.updateSigma(
                sigma_hat, mu_curr, mu_prev, xs, ys, alpha, gamma, epsilon
        );
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
