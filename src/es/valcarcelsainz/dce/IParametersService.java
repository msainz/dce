package es.valcarcelsainz.dce;

import es.valcarcelsainz.dce.fn.GlobalSolutionFunction;

public interface IParametersService {

    double[][]   getMus();
    double[][][] getSigmas();
    Object[]     getLocks();

    void initParameters();
    void clearParameters(int i);

    /**
     * Create new surrogate gaussian with mu and sigma from previous iteration
     * (which won't receive more updates by the time this method gets called)
     * and draw samples (xs) and evaluate the target function J at these points
     * and store in ys.
     *
     * @return new Object[] { f, gamma }
     * f: new gaussian distribution where samples xs were drawn from
     * gamma: new threshold that drives the response of the soft indicator function I
     */
    Object[] sampleNewGaussian(int i, double[][] xs, double[] ys, GlobalSolutionFunction J);

    void updateMu(int i, double weight, double[] mu_hat);
    void updateMu(int i, double weight, String mu_hat);

    void updateSigma(int i, double weight, double[][] sigma_hat);
    void updateSigma(int i, double weight, String sigma_hat);

    void computeMuHat(int i, double[] mu_hat, double[][] xs, double[] ys,
                      double alpha, double gamma, double epsilon);

    void computeSigmaHat(int i, double[][] sigma_hat, double[][] xs, double[] ys,
                         double alpha, double gamma, double epsilon);

    int currInd(int i);
    int prevInd(int i);
}
