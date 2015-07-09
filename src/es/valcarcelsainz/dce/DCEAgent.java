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
    private final Map<Integer, Double> neighWeights;
    private final int maxIter;
    private final String redisHost;
    private final int redisPort;

    // pace multi-threaded computation
    private final Phaser muPhaser = new Phaser();
    private final Phaser sigmaPhaser = new Phaser();
    private final GlobalSolutionFunction targetFn; // target to optimize

    // TODO: store the two mu's (mu_i and mu_i+1) in array of length 2
    // then as neighbor updates arrive, can do (i % 2) --but how to check for >1 gaps
    // in neighbor updates due to incorrect (e.g. un-reciprocated) hasting weights?
    private StringBuilder mu_i = new StringBuilder("");

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

        Gson gson = new Gson();
        Jedis jedis = new Jedis(redisHost, redisPort);

        for (int k = agentId, i = 0; i < maxIter; i++) {

            // compute and publish mu_hat of the current iteration i, eqn. (32)(top)
            String mu_hat = computeMuHat(k, i);
            synchronized (mu_i) {
                mu_i.append(mu_hat);
            }

            publishMuHat(jedis, gson, mu_hat, k, i);

            // wait for all neighbors' mu_hat for current iteration i
            muPhaser.awaitAdvance(i);

            // compute mu, eqn. (32)(bottom)

            // compute and publish sigma_hat of current iteration i, eqn. (33)(top)

            // wait for all neighbors' sigma_hat for current iteration i
            // sigmaPhaser.awaitAdvance(i);

            // compute sigma, eqn. (33)(bottom

            logger.trace("mu[{}][{}] = \"{}\"", k, i, mu_i);
            logger.info("completed iteration({})", i);
        }

    }

    private void publishMuHat(Jedis jedis, Gson gson, String mu_hat, int k, int i) {
        String out = gson.toJson(new Message(i, mu_hat, Message.PayloadType.MU));
        jedis.publish(Integer.toString(k), out);
    }

    private String computeMuHat(int k, int i) {
        String mu_hat = null;
        try {
            mu_hat = String.format("mu_hat[%s][%s] ", k, i);
            Thread.sleep(Math.round(Math.random() * 3000)); // sleep between 0 and 3 sec
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        return mu_hat;
    }

    public void stop() {
        logger.info("stop called!");
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
                    logger.trace("channel: {}, message: {}", channel, msg.substring(0, Math.min(msg.length(), 100)));
                    Gson gson = new Gson();
                    Message in = gson.fromJson(msg, Message.class);

                    // update mu_i (or mu_i+1) being mindful
                    // of concurrent neighbor updates
                    synchronized (mu_i) {
                        mu_i.append(in.payload);
                    }

                    // notify that the neighbor update is done
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
