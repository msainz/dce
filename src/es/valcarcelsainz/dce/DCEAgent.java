package es.valcarcelsainz.dce;

import com.google.gson.Gson;
import es.valcarcelsainz.dce.fn.GlobalSolutionFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPubSub;
import smile.math.Math;

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
    private final double gammaQuantile = 0.98;
    private final double epsilon = 1e12;

    private final String redisHost;

    private final int redisPort;
    private final IParametersService ps;
    // pace multi-threaded computation
    private final Phaser muPhaser = new Phaser();

    private final Phaser sigmaPhaser = new Phaser();
    // target to optimize
    private final GlobalSolutionFunction targetFn;

    // thread-safe, so we make static to share among agents
    public static final Gson GSON = new Gson();

    // public getters/setters
    public int getAgentId() {
        return agentId;
    }

    public double getGammaQuantile() {
        return gammaQuantile;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public GlobalSolutionFunction getTargetFn() {
        return targetFn;
    }

    // constructor
    public DCEAgent(Integer agentId, Map<Integer, Double> neighWeights, Integer maxIter,
                    String redisHost, Integer redisPort, GlobalSolutionFunction targetFn) {

        this.agentId = agentId;
        this.neighWeights = neighWeights;
        this.maxIter = maxIter;
        this.redisHost = redisHost;
        this.redisPort = redisPort;
        this.targetFn = targetFn;
        this.ps = new ParametersServiceImpl(this);
//        this.ps = new MockParametersServiceImpl(this);

        subscribeToBroadcast();
        subscribeToNeighbors();

        // TODO: jedisPubSub.unsubscribe();
    }

    // TODO: prevent multiple undesired calls to start()
    public void start() {
        logger.info("agent({}) started", agentId);
        logger.info("connecting to redis at {}:{}", redisHost, redisPort);
        Jedis jedis = new Jedis(redisHost, redisPort);

        ps.initParameters();
        ps.clearParameters(1);

        for (int i = 1; i <= maxIter; i++) {

            // weight given to our own estimates
            double selfWeight = neighWeights.get(agentId);

            int M = targetFn.getDim();
            int numSamples = CrossEntropyOptimization.computeNumSamplesForIteration(i);
            double alpha = CrossEntropyOptimization.computeSmoothingForIteration(i);

            // array of samples and array of target function evaluations
            double [][] xs = new double[numSamples][M];
            double [] ys = new double[xs.length];

            // sample from surrogate and evaluate target at the samples
            Object[] fAndGamma = ps.sampleNewGaussian(i, xs, ys, targetFn);
            MultivariateGaussianDistribution f = (MultivariateGaussianDistribution) fAndGamma[0];
            double gamma = (double) fAndGamma[1];

            logTraceParameters(i, "before-computeMuHat");
            logger.trace("i: {} | alpha: {} | gamma: {}", i, alpha, gamma);

            // ugly optimization, but since we don't need to draw any more
            // samples from f, we'll reuse it's internal register as buffer
            // instead of making another deep copy of mu (and sigma)
            double [] mu_hat = f.mu; // effectively, mus[prevInd(i)]

            // compute mu_hat of current iteration i, eqn. (32)(top)
            ps.computeMuHat(i, mu_hat, xs, ys, alpha, gamma, epsilon);
            ps.updateMu(i, selfWeight, mu_hat);

            // broadcast mu_hat to neighbors
            publish(jedis, GSON.toJson(mu_hat), i, Message.PayloadType.MU);

            // compute mu, eqn. (32)(bottom)
            // wait for all neighbors' mu_hat for current iteration i
            muPhaser.awaitAdvance(i - 1); // phase is 0-based

            logTraceParameters(i, "before-computeSigmaHat");

            // compute sigma_hat of current iteration i, eqn. (33)(top)
            double [][] sigma_hat = f.sigma; // same ugly optimization
            ps.computeSigmaHat(i, sigma_hat, xs, ys, alpha, gamma, epsilon);
            ps.updateSigma(i, selfWeight, sigma_hat);

            // at this point it's safe to clear mu_i-1 and sigma_i-1
            ps.clearParameters(ps.prevInd(i));

            // broadcast sigma_hat to neighbors
            publish(jedis, GSON.toJson(sigma_hat), i, Message.PayloadType.SIGMA);

            // compute sigma, eqn. (33)(bottom)
            // wait for all neighbors' sigma_hat for current iteration i
            sigmaPhaser.awaitAdvance(i - 1); // phase is 0-based

            synchronized (ps.getLocks()[ps.currInd(i)]) {
                double rmse = CrossEntropyOptimization.rmse(ps.getMus()[ps.currInd(i)], targetFn.getSoln());
                logger.info("completed iteration({}), rmse: {}", i, rmse);
            }
        }
        logger.info("final sigma: {\"sigma_{}_{}\": {{}}}", agentId, maxIter, ps.getSigmas()[maxIter % 2]);
        logger.info("final mu: {\"mu_{}_{}\": {{}}}", agentId, maxIter, ps.getMus()[maxIter % 2]);
    }

    private void logTraceParameters(int i, String msg) {
        if (logger.isTraceEnabled()) {
            synchronized (ps.getLocks()[ps.currInd(i)]) {
                logger.trace("{} {\"mu_{}_{}\": {{}}}", msg, agentId, i, ps.getMus()[ps.currInd(i)]);
                logger.trace("{} {\"sigma_{}_{}\": {{}}}", msg, agentId, i, ps.getSigmas()[ps.currInd(i)]);
            }
        }
    }

    private void publish(Jedis jedis, String mu_hat, int i, Message.PayloadType type) {
        String out = GSON.toJson(new Message(i, agentId, mu_hat, type));
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

    // TODO: alternatively, subscribe to a MU channel and SIGMA channel for each neighbor
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
                    double neighWeight = neighWeights.get(in.fromAgentId);
                    switch (in.type) {
                        case MU:
                            ps.updateMu(in.i, neighWeight, in.payload);
                            muPhaser.arrive();
                            break;
                        case SIGMA:
                            ps.updateSigma(in.i, neighWeight, in.payload);
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
