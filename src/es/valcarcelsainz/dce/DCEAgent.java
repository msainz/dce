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

        for (int i = 0; i < maxIter; i++) {

            // compute mu_hat of the current iteration i, eqn. (32)(top)
            String mu_hat = computeMuHat(i);
            updateMu(i, mu_hat);

            // broadcast my mu_hat to my neighbors
            publishMuHat(jedis, gson, mu_hat, i);

            // compute mu, eqn. (32)(bottom)
            // wait for all neighbors' mu_hat for current iteration i
            muPhaser.awaitAdvance(i);

            // at this point it's safe to clear mu_i-1
            clearOldMu(i);

            // compute and publish sigma_hat of current iteration i, eqn. (33)(top)

            // compute sigma, eqn. (33)(bottom)
            // wait for all neighbors' sigma_hat for current iteration i
            // sigmaPhaser.awaitAdvance(i);

            // at this point it's safe to clear sigma_i-1
            // clearOldSigma(i);

            if (logger.isTraceEnabled()) {
                synchronized (lock[i % 2]) {
                    logger.trace("{\"mu_{}_{}\": {{}}}", agentId, i, mu[i % 2]);
                }
            }
            logger.info("completed iteration({})", i);
        }

        logger.debug("final solution: {\"mu_{}_{}\": {{}}}", agentId, maxIter - 1, mu[(maxIter - 1) % 2]);
    }

    private void clearOldMu(int i) {
        // TODO: perhaps simply call update
        synchronized (lock[(i + 1) % 2]) {
            mu[(i + 1) % 2] = "";
        }
        System.gc(); // TODO: don't call this on *every* iteration
    }

    private void updateMu(int i, String mu_hat) {
        synchronized (lock[i % 2]) {
            int len = mu[i % 2].length();
            mu[i % 2] += ((len > 0) ? ", " : "") + mu_hat;
        }
    }

    private String computeMuHat(int i) {
        String prevMu = null;
        synchronized (lock[(i + 1) % 2]) {
            prevMu = String.format("{%s}", mu[(i + 1) % 2]);
        }
        String mu_hat = null;
        try {
            mu_hat = String.format("\"muhat_%s_%s\": %s", agentId, i, prevMu);
            Thread.sleep(Math.round(Math.random() * 1000)); // sleep up to 1 sec
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        return mu_hat;
    }

    private void publishMuHat(Jedis jedis, Gson gson, String mu_hat, int i) {
        String out = gson.toJson(new Message(i, mu_hat, Message.PayloadType.MU));
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
        enum PayloadType { MU, SIGMA;};
        Message(int i, String payload, PayloadType type) {
            this.i = i;
            this.payload = payload;
            this.type = type;
        }
        int i;
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
                    updateMu(in.i, in.payload); // TODO: in.type
                    muPhaser.arrive();
                }
            };

            new Thread(new Subscriber(neighPubSub, neighId.toString()),
                    String.format("listenThread(%s)<-(%s)", agentId, neighId))
                    .start();

            neighPubSubs.add(neighPubSub);
            muPhaser.register();
        }
        return neighPubSubs;
    }
}
