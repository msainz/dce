package es.valcarcelsainz.dce;

public class ParametersService {

    // id of client agent using this parameters service
    private final int clientDCEAgentId;

    // mu_i and mu_i+1
    private String[] mu = new String[] {"", ""}; // TODO: soon will be an array of double[]

    // sigma_i and sigma_i+1
    private String[] sigma = new String[] {"", ""}; // TODO: soon will be an array of double[][]

    // synchronization locks for mu/sigma_i and mu/sigma_i+1
    private Object[] lock = new Object[] {
            new Object(),
            new Object()
    };

    protected ParametersService(int clientDCEAgentId) {
        this.clientDCEAgentId = clientDCEAgentId;
    }

    protected String[] getMu() {
        return mu;
    }

    protected String[] getSigma() {
        return sigma;
    }

    protected Object[] getLock() {
        return lock;
    }

    protected void clearPrev(int i) {
        synchronized (lock[prevInd(i)]) {
            mu[prevInd(i)] = "";
            sigma[prevInd(i)] = "";
        }
    }

//    private void concurrentWeightedIncrement(Object lock, String mutableTarget, String increment, double weight) {
//        synchronized (lock) {
//            int len = mutableTarget.length();
//            mutableTarget += (len > 0 ? ", " : "") +
//                String.format("\"%.2f_%s", weight, increment.substring(1));
//        }
//    }

    protected void updateMu(int i, double weight, String mu_hat) {
        synchronized (lock[currInd(i)]) {
            int len = mu[currInd(i)].length();
            mu[currInd(i)] += (len > 0 ? ", " : "");
            mu[currInd(i)] += String.format("\"%.2f_%s",
                    weight, mu_hat.substring(1)
            );
        }
    }

    protected void updateSigma(int i, double weight, String sigma_hat) {
        synchronized (lock[currInd(i)]) {
            int len = sigma[currInd(i)].length();
            sigma[currInd(i)] += (len > 0 ? ", " : "");
            sigma[currInd(i)] += String.format("\"%.2f_%s",
                    weight,
                    sigma_hat.substring(1)
            );
        }
    }

    protected String computeMuHat(int i) {
        // mu_hat depends on mu from previous iteration
        // see eqn. (32)(top)
        String prevMu = null;
        synchronized (lock[prevInd(i)]) {
            prevMu = String.format("{%s}", mu[prevInd(i)]);
        }
        String mu_hat = null;
        try {
            mu_hat = String.format("\"muhat_%1$s_%2$s\": {\"mu_%1$s_%3$s\": %4$s}",
                    clientDCEAgentId, i, i - 1, prevMu);
            Thread.sleep(Math.round(Math.random() * 1000)); // sleep up to 1 sec
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        return mu_hat;
    }

    protected String computeSigmaHat(int i) {
        // sigma_hat depends on current mu and
        // mu and sigma from previous iteration
        // see eqn. (33)(top)
        String currMu, prevMu, prevSigma  = null;
        synchronized (lock[currInd(i)]) {
            currMu = String.format("{%s}", mu[currInd(i)]);
        }
        synchronized (lock[prevInd(i)]) {
            prevMu = String.format("{%s}", mu[prevInd(i)]);
            prevSigma = String.format("{%s}", sigma[prevInd(i)]);
        }
        String sigma_hat = null;
        try {
            sigma_hat = String.format(
                    "\"sigmahat_%1$s_%2$s\": {\"mu_%1$s_%2$s\": %4$s, \"mu_%1$s_%3$s\": %5$s, \"sigma_%1$s_%3$s\": %6$s}",
                    clientDCEAgentId, i, i - 1, currMu, prevMu, prevSigma);
            Thread.sleep(Math.round(Math.random() * 1000)); // sleep up to 1 sec
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        return sigma_hat;
    }

    protected int currInd(int i) {
        return i % 2;
    }

    protected int prevInd(int i) {
        return (i + 1) % 2;
    }
}
