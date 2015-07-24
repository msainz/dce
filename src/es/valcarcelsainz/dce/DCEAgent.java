package es.valcarcelsainz.dce;

import com.google.gson.Gson;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPubSub;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Phaser;

/**
 * @author Marcos Sainz
 */
public class DCEAgent {

    private static final Logger logger =
            LoggerFactory.getLogger(DCEAgent.class);

    private final int agentId; // k

    // TODO: use an immutable map instead
    // http://google-collections.googlecode.com/svn/trunk/javadoc/com/google/common/collect/ImmutableMap.html
    private final Map<Integer, Double> neighWeights;

    private final int maxIter;
    private final String redisHost;
    private final int redisPort;

    // pace multi-threaded computation
    private final Phaser muPhaser = new Phaser();
    private final Phaser sigmaPhaser = new Phaser();

    // target to optimize
    private final GlobalSolutionFunction targetFn;

    // mu_i and mu_i+1
    private String[] mu = new String[] {"", ""}; // TODO: soon will be an array of double[]

    // sigma_i and sigma_i+1
    private String[] sigma = new String[] {"", ""}; // TODO: soon will be an array of double[][]

    // synchronization locks for mu/sigma_i and mu/sigma_i+1
    private Object[] lock = new Object[] {
            new Object(),
            new Object()
    };

    public DCEAgent(Integer agentId, Map<Integer, Double> neighWeights, Integer maxIter,
                    String redisHost, Integer redisPort, GlobalSolutionFunction targetFn) {

        this.agentId = agentId;
        this.neighWeights = neighWeights;
        this.maxIter = maxIter;
        this.redisHost = redisHost;
        this.redisPort = redisPort;
        this.targetFn = targetFn;

        subscribeToBroadcast();
        subscribeToNeighbors();

        // TODO: jedisPubSub.unsubscribe();
    }

    // TODO: prevent multiple undesired calls to start()
    public void start() {
        logger.info("agent({}) started", agentId);
        logger.info("connecting to redis at {}:{}", redisHost, redisPort);

        Gson gson = new Gson(); // TODO: make static since it's thread-safe
        Jedis jedis = new Jedis(redisHost, redisPort);

        for (int i = 1; i <= maxIter; i++) {

            // compute mu_hat of current iteration i, eqn. (32)(top)
            String mu_hat = computeMuHat(i);
            updateMu(i, agentId, mu_hat);

            // broadcast mu_hat to neighbors
            publish(jedis, gson, mu_hat, i, Message.PayloadType.MU);

            // compute mu, eqn. (32)(bottom)
            // wait for all neighbors' mu_hat for current iteration i
            muPhaser.awaitAdvance(i - 1); // phase is 0-based

            // compute sigma_hat of current iteration i, eqn. (33)(top)
            String sigma_hat = computeSigmaHat(i);
            updateSigma(i, agentId, sigma_hat);

            // broadcast sigma_hat to neighbors
            publish(jedis, gson, sigma_hat, i, Message.PayloadType.SIGMA);

            // compute sigma, eqn. (33)(bottom)
            // wait for all neighbors' sigma_hat for current iteration i
            sigmaPhaser.awaitAdvance(i - 1); // phase is 0-based

            // at this point it's safe to clear mu_i-1 and sigma_i-1
            clearPrev(i);
            logger.trace("System.gc()");
            System.gc();

            if (logger.isTraceEnabled()) {
                synchronized (lock[currInd(i)]) {
                    logger.trace("{\"mu_{}_{}\": {{}}}", agentId, i, mu[currInd(i)]);
                    logger.trace("{\"sigma_{}_{}\": {{}}}", agentId, i, sigma[currInd(i)]);
                }
            }
            // TODO: compute and display iteration error
            logger.info("completed iteration({})", i);
        }

        logger.debug("final solution (mu): {\"mu_{}_{}\": {{}}}", agentId, maxIter, mu[maxIter % 2]);
        logger.debug("final solution (sigma): {\"sigma_{}_{}\": {{}}}", agentId, maxIter, sigma[maxIter % 2]);
    }

    private void clearPrev(int i) {
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

    private void updateMu(int i, int fromAgentId, String mu_hat) {
        synchronized (lock[currInd(i)]) {
            int len = mu[currInd(i)].length();
            mu[currInd(i)] += (len > 0 ? ", " : "");
            mu[currInd(i)] += String.format("\"%.2f_%s",
                    neighWeights.get(fromAgentId),
                    mu_hat.substring(1)
            );
        }
    }

    private void updateSigma(int i, int fromAgentId, String sigma_hat) {
        synchronized (lock[currInd(i)]) {
            int len = sigma[currInd(i)].length();
            sigma[currInd(i)] += (len > 0 ? ", " : "");
            sigma[currInd(i)] += String.format("\"%.2f_%s",
                    neighWeights.get(fromAgentId),
                    sigma_hat.substring(1)
            );
        }
    }

    private String computeMuHat(int i) {
        // mu_hat depends on mu from previous iteration
        // see eqn. (32)(top)
        String prevMu = null;
        synchronized (lock[prevInd(i)]) {
            prevMu = String.format("{%s}", mu[prevInd(i)]);
        }
        String mu_hat = null;
        try {
            mu_hat = String.format("\"muhat_%1$s_%2$s\": {\"mu_%1$s_%3$s\": %4$s}",
                    agentId, i, i - 1, prevMu);
            Thread.sleep(Math.round(Math.random() * 1000)); // sleep up to 1 sec
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        return mu_hat;
    }

    private String computeSigmaHat(int i) {
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
                    agentId, i, i - 1, currMu, prevMu, prevSigma);
            Thread.sleep(Math.round(Math.random() * 1000)); // sleep up to 1 sec
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        return sigma_hat;
    }

    private int currInd(int i) {
        return i % 2;
    }

    private int prevInd(int i) {
        return (i + 1) % 2;
    }

    private void publish(Jedis jedis, Gson gson, String mu_hat, int i, Message.PayloadType type) {
        String out = gson.toJson(new Message(i, agentId, mu_hat, type));
        jedis.publish(Integer.toString(agentId), out);
    }

    public void quit() {
        logger.info("quit called!");
    }

    class Subscriber implements Runnable {

        private String channel;
        private JedisPubSub jedisPubSub;

        public Subscriber(JedisPubSub jedisPubSub, String channel) {
            this.jedisPubSub = jedisPubSub;
            this.channel = channel;
        }

        @Override
        public void run() {
            logger.info("connecting to redis at {}:{}", redisHost, redisPort);
            Jedis jedis = new Jedis(redisHost, redisPort);
            logger.info("agent({}) subscribing to channel({})", agentId, channel);
            jedis.subscribe(jedisPubSub, channel);
            logger.info("channel({}) subscribe returned for agent({})", channel, agentId);
            jedis.quit();
        }
    }

    public JedisPubSub subscribeToBroadcast() {
        final JedisPubSub broadcastPubSub = new JedisPubSub() {
            @Override
            public void onMessage(String channel, String message) {
                logger.trace("channel: {}, message: {}", channel, message);
                DCEAgent enclosingAgent = DCEAgent.this;
                try {
                    Method method = enclosingAgent.getClass().getMethod(message);
                    method.invoke(enclosingAgent);
                } catch (NoSuchMethodException | InvocationTargetException | IllegalAccessException e) {
                    logger.error(e.getMessage(), e);
                }
            }
        };
        new Thread(new Subscriber(broadcastPubSub, "broadcast"),
                String.format("mainThread(%s)", agentId)).start();
        return broadcastPubSub;
    }

    static class Message {
        enum PayloadType { MU, SIGMA }
        Message(int i, int fromAgentId, String payload, PayloadType type) {
            this.i = i;
            this.fromAgentId = fromAgentId;
            this.payload = payload;
            this.type = type;
        }
        int i;
        int fromAgentId;
        String payload;
        PayloadType type;
    }

    public List<JedisPubSub> subscribeToNeighbors() {
        final List<JedisPubSub> neighPubSubs = new LinkedList<>();
        for (Map.Entry<Integer,Double> neighWeight : neighWeights.entrySet()) {

            Integer neighId = neighWeight.getKey();
            if (neighId == agentId) {
                // do not subscribe to self
                continue;
            }

            final JedisPubSub neighPubSub = new JedisPubSub() {
                @Override
                public void onMessage(String channel, String msg) {
                    logger.trace("channel: {}, message: {}", channel, msg.substring(0, Math.min(msg.length(), 1024)));
                    Gson gson = new Gson();
                    Message in = gson.fromJson(msg, Message.class);
                    switch (in.type) {
                        case MU:
                            updateMu(in.i, in.fromAgentId, in.payload);
                            muPhaser.arrive();
                            break;
                        case SIGMA:
                            updateSigma(in.i, in.fromAgentId, in.payload);
                            sigmaPhaser.arrive();
                            break;
                        default:
                            assert false;
                    }

                }
            };

            new Thread(new Subscriber(neighPubSub, neighId.toString()),
                    String.format("listenThread(%s)<-(%s)", agentId, neighId))
                    .start();

            neighPubSubs.add(neighPubSub);
            muPhaser.register();
            sigmaPhaser.register();
        }
        return neighPubSubs;
    }
}
