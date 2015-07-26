package es.valcarcelsainz.dce;

public interface IParametersService<U,S> {

    U[] getMu();

    S[] getSigma();

    Object[] getLock();

    void initParameters();

    void clearParameters(int i);

    void updateMu(int i, double weight, String mu_hat);

    void updateSigma(int i, double weight, String sigma_hat);

    String computeMuHat(int i);

    String computeSigmaHat(int i);

    int currInd(int i);

    int prevInd(int i);
}
