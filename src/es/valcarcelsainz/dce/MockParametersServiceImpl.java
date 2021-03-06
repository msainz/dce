package es.valcarcelsainz.dce;

public class MockParametersServiceImpl {

    // agent who we are serving
    private final DCEAgent clientAgent;

    // mu_i and mu_i+1
    private StringBuilder[] mu = new StringBuilder[2]; // mutable, non-synchronized

    // sigma_i and sigma_i+1
    private StringBuilder[] sigma = new StringBuilder[2]; // mutable, non-synchronized

    // synchronization locks for mu/sigma_i and mu/sigma_i+1
    private Object[] lock = new Object[] {
            new Object(),
            new Object()
    };

    protected MockParametersServiceImpl(DCEAgent clientAgent) {
        this.clientAgent = clientAgent;
    }

    public StringBuilder[] getMus() {
        return mu;
    }

    public StringBuilder[] getSigmas() {
        return sigma;
    }

    public Object[] getLocks() {
        return lock;
    }

    public void initParameters() {
        for (int j = 0; j < 2; j++) {
            synchronized (lock[j]) {
                mu[j] = new StringBuilder();
                sigma[j] = new StringBuilder();
            }
        }
    }

    public void clearParameters(int i) {
        synchronized (lock[i]) {
            mu[i].setLength(0);
            sigma[i].setLength(0);
        }
    }

//    private void concurrentWeightedIncrement(Object lock, StringBuilder mutableTarget, String increment, double weight) {
//        synchronized (lock) {
//            int len = mutableTarget.length();
//            mutableTarget += (len > 0 ? ", " : "") +
//                String.format("\"%.2f_%s", weight, increment.substring(1));
//        }
//    }

    public void updateMu(int i, double weight, String mu_hat) {
        synchronized (lock[currInd(i)]) {
            int len = mu[currInd(i)].length();
            mu[currInd(i)].append(len > 0 ? ", " : "");
            mu[currInd(i)].append(String.format("\"%.2f_%s",
                    weight, mu_hat.substring(1)
            ));
        }
    }

    public void updateSigma(int i, double weight, String sigma_hat) {
        synchronized (lock[currInd(i)]) {
            int len = sigma[currInd(i)].length();
            sigma[currInd(i)].append(len > 0 ? ", " : "");
            sigma[currInd(i)].append(String.format("\"%.2f_%s",
                    weight,
                    sigma_hat.substring(1)
            ));
        }
    }

    public String computeMuHat(int i) {
        // mu_hat depends on mu from previous iteration
        // see eqn. (32)(top)
        String prevMu = null;
        synchronized (lock[prevInd(i)]) {
            prevMu = String.format("{%s}", mu[prevInd(i)]);
        }
        String mu_hat = null;
        try {
            mu_hat = String.format("\"muhat_%1$s_%2$s\": {\"mu_%1$s_%3$s\": %4$s}",
                    clientAgent.getAgentId(), i, i - 1, prevMu);
            Thread.sleep(Math.round(Math.random() * 1000)); // sleep up to 1 sec
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        return mu_hat;
    }

    public String computeSigmaHat(int i) {
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
                    clientAgent.getAgentId(), i, i - 1, currMu, prevMu, prevSigma);
            Thread.sleep(Math.round(Math.random() * 1000)); // sleep up to 1 sec
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        return sigma_hat;
    }

    public int currInd(int i) {
        return i % 2;
    }

    public int prevInd(int i) {
        return (i + 1) % 2;
    }
}
