package es.valcarcelsainz.dce;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.Phaser;
import java.util.function.Function;
import java.util.function.Supplier;

import com.codahale.metrics.Gauge;
import com.codahale.metrics.JmxReporter;
import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import es.valcarcelsainz.dce.fn.GlobalSolutionFunction;
import org.apache.log4j.FileAppender;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.SimpleLayout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPubSub;
import smile.math.Math;
import smile.math.Random;
import smile.sort.IQAgent;

import static com.codahale.metrics.MetricRegistry.name;
import static smile.math.Math.*;

/**
 * @author Marcos Sainz
 */
public class DCEAgent {

    static final MetricRegistry registry = new MetricRegistry();
    static final Logger logger =
            LoggerFactory.getLogger(DCEAgent.class);

    // thread-safe, so we make static to share among agents
    static final Gson GSON = new Gson();

    // list of neighbor-subscriber threads
    final List<JedisPubSub> neighPubSubs;

    // subscriber to broadcast channel
    JedisPubSub broadcastPubSub;

    final Map<Integer, Double> neighWeights; // TODO: use immutable map
    final int agentId; // k
    final int maxIter;
    final double gammaQuantile;
    final double lowerBound;
    final double upperBound;
    final double epsilon;
    final double initSamples;
    final boolean increaseSamples;
    final double regNoise;
    final String redisHost;
    final int redisPort;
    final Path resultsDirPath;

    // pace multi-threaded computation
    final Phaser muPhaser = new Phaser();
    final Phaser sigmaPhaser = new Phaser();

    // target to optimize
    final GlobalSolutionFunction targetFn;

    // instance fields to allow async polling
    // e.g. jmx gauge reporter
    double rmse;
    double alpha;
    double gamma;
    int numSamples;

    // mu/sigma_i and mu/sigma_i+1
    double[] mus[] = new double[2][];
    double[][] sigmas[] = new double[2][][];
    double[][] cov_mat;
    double[][] identity_mat;

    // synchronization locks for mu/sigma_i and mu/sigma_i+1
    Object[] locks = new Object[]{
            new Object(),
            new Object()
    };

    // constructor
    DCEAgent(Integer agentId, Map<Integer, Double> neighWeights, Integer maxIter, double gammaQuantile,
             double lowerBound, double upperBound, double epsilon, double initSamples, boolean increaseSamples,
             double regNoise,
             String redisHost, Integer redisPort,
             GlobalSolutionFunction targetFn, String resultsDirPath) {

        this.agentId = agentId;
        this.neighWeights = neighWeights;
        this.maxIter = maxIter;
        this.gammaQuantile = gammaQuantile;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.epsilon = epsilon;
        this.initSamples = initSamples;
        this.increaseSamples = increaseSamples;
        this.regNoise = regNoise;
        this.redisHost = redisHost;
        this.redisPort = redisPort;
        this.targetFn = targetFn;
        this.resultsDirPath = Paths.get(resultsDirPath);

        // initialize mus[0] to M uniformly-random numbers in [lowerBound, upperBound]
        // and sigmas[0] to a M x M diagonal matrix with 1000 along the diagonal; and
        // initialize mus[1] to M zeros and sigmas[0] to M x M zeros
        initParameters();
        clearParameters(1);

        this.broadcastPubSub = subscribeToBroadcast();
        this.neighPubSubs = subscribeToNeighbors();
    }

    static double smoothIndicator(double y, double gamma, double epsilon) {
        return 1. / (1. + exp(-epsilon * (y - gamma)));
    }

    static int computeNumSamplesForIteration(int iteration, double initSamples) {
        return (int) max(initSamples, pow(iteration, 1.01));
    }

    static double computeSmoothingForIteration(int iteration) {
        return 2. / pow(iteration + 100., 0.501);
    }

    // scale each element of a matrix by a constant.
    static double[][] scaleA(double a, double[][] A) {
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                A[i][j] *= a;
            }
        }
        return A;
    }


    static double mse(double[] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException(String.format(
                    "Arrays have different length: x[%d], y[%d]", x.length, y.length)
            );
        }
        double mse = 0.;
        for (int j = 0; j < x.length; j++) {
            mse += pow(y[j] - x[j], 2.); // squared residuals
        }
        mse /= x.length; // MSE
        return mse;
    }

    static double rmse(double[] x, double[] y) {
        return pow(mse(x, y), .5);
    }

    static int currInd(int i) {
        return i % 2;
    }

    static int prevInd(int i) {
        return (i + 1) % 2;
    }

    void initParameters() {
        for (int j = 0; j < 2; j++) {
            synchronized (locks[j]) {
                int M = targetFn.getDim();
                mus[j] = new double[M];
                sigmas[j] = new double[M][M];

                long seed = System.currentTimeMillis() + Thread.currentThread().getId();
                new Random(seed).nextDoubles(mus[j], this.lowerBound, this.upperBound);

                cov_mat = new double[M][M];
                for (int i = 0; i < M; i++) {
                    //cov_mat[i][i] = this.upperBound - this.lowerBound;
                    cov_mat[i][i] = 1000;
                }

                double[][] A = new double[][]{mus[j]}; // 1 x M
                copy(cov_mat, sigmas[j]);
                plus(sigmas[j], atamm(A)); // M x M

                identity_mat = new double[M][M];
                for (int i = 0; i < M; i++) {
                    identity_mat[i][i] = regNoise;
                }
            }
        }
    }

    void clearParameters(int i) {
        synchronized (locks[i]) {
            Arrays.fill(mus[i], 0);
            for (int j = 0; j < targetFn.getDim(); j++) {
                Arrays.fill(sigmas[i][j], 0);
            }
        }
    }

    // see eqn. (32)(top)
    // create new surrogate gaussian with mu and sigma from prev iteration
    // which won't receive more updates by the time this method is called
    Object[] sampleNewGaussian(int i, double[][] xs, double[] ys, GlobalSolutionFunction J) {
        final MultivariateGaussianDistribution f;
        long seed = System.currentTimeMillis() + Thread.currentThread().getId();
        plus(cov_mat, identity_mat); // Regularize with with small diagonal noise to force positive definite covariance
                                     // and be able to reduce the number of samples per node.
        try {
                f = new MultivariateGaussianDistribution(
                        // this constructor deep-copies mu[i] and cov_mat
                        // later accessible via f.mean() and f.cov() respectively.
                        mus[prevInd(i)], cov_mat,
                        new Random(seed) // TODO: reuse Random instance
                );
        } catch (IllegalArgumentException e) {
            //logger.info("{\"mu_{}_{}\": {{}}}", agentId, i - 1, mus[prevInd(i)]);
            //logger.info("{\"sigma_{}_{}\": {{}}}", agentId, i - 1, sigmas[prevInd(i)]);
            logger.info("i: {} | id: {}  singular matrix", i, agentId);
            throw e;
        }
        double gamma = computeGamma2(xs, ys, () -> f.rand(), (double[] x) -> J.f(x), gammaQuantile);

        /*
        // First sampling distribution must be uniform in the searching box
        RandomDistribution f;
        if (i == 0) {
            int M = targetFn.getDim();
            f = new UniformDistribution(this.lowerBound, this.upperBound, M);
        } else {
            long seed = System.currentTimeMillis() + Thread.currentThread().getId();
            try {
                    f = new MultivariateGaussianDistribution(
                            // this constructor deep-copies mu[i] and cov_mat
                            // later accessible via f.mean() and f.cov() respectively.
                            mus[prevInd(i)], cov_mat,
                            //new myRandom(System.currentTimeMillis() + Thread.currentThread().getId())
                            new Random(seed)
                    );
            } catch (IllegalArgumentException e) {
                //logger.info("{\"mu_{}_{}\": {{}}}", agentId, i - 1, mus[prevInd(i)]);
                //logger.info("{\"sigma_{}_{}\": {{}}}", agentId, i - 1, sigmas[prevInd(i)]);
                logger.info("i: {} | id: {}  singular matrix", i, agentId);
                throw e;
            }
        }
        double gamma = computeGamma2(xs, ys, () -> f.rand(), (double[] x) -> J.f(x), gammaQuantile);
        */

        return new Object[]{f, gamma};
    }

    static double computeGamma2(double[][] xs, double[] ys, Supplier<double[]> f, Function<double[], Double> jfn, double gammaQuantile) {
        int numSamples = xs.length;
        int M = xs[0].length;
        int maximumSize = (int) Math.round(numSamples * gammaQuantile);
        TreeSet<Double> pq = new TreeSet(Comparator.<Double>naturalOrder());

        // for potential speedup
        // using Guava's bounded priority queue:
        // MinMaxPriorityQueue<Double> pq = MinMaxPriorityQueue
        //        .orderedBy(Comparator.<Double>naturalOrder())
        //        .maximumSize(maximumSize)
        //        .create();
        //
        // for further speedup, see
        // https://github.com/davidmoten/rtree/issues/39
        // https://github.com/davidmoten/rtree/blob/master/src/main/java/com/github/davidmoten/rtree/internal/util/BoundedPriorityQueue.java

        for (int xsInd = 0; xsInd < numSamples; xsInd++) {
            double[] x = f.get();
            double y = jfn.apply(x);
            xs[xsInd] = x;
            ys[xsInd] = y;
            pq.add(y);
            if (pq.size() > maximumSize)
                pq.remove(pq.last());
        }
        return pq.last();
    }

    /*static double computeGamma(double[][] xs, double[] ys, Supplier<double[]> f, Function<double[], Double> jfn, double gammaQuantile) {
        int numSamples = xs.length;
        int M = xs[0].length;

        IQAgent gammaFinder = new IQAgent(M);
        for (int xsInd = 0; xsInd < numSamples; xsInd++) {
            double[] x = f.get();
            double y = jfn.apply(x);
            xs[xsInd] = x;
            ys[xsInd] = y;
            gammaFinder.add(y);
        }
        return gammaFinder.quantile(gammaQuantile);
    }*/

    // see eqn. (32)(top)
    // performs mu <- ((1 - alpha) * mu) + ((alpha / denom) * numer)
    static void computeMuHat(double[] mu_hat, double[][] xs, double[] ys,
                             double alpha, double gamma, double epsilon) {
        int numSamples = ys.length;
        assert xs.length == ys.length;
        assert xs.length > 0;
        int M = xs[0].length;
        double[] numer = new double[M];
        double denom = 0d;
        for (int xsInd = 0; xsInd < numSamples; xsInd++) {
            double[] x = new double[M];
            copy(xs[xsInd], x);
            double y = ys[xsInd];
            double I = smoothIndicator(y, gamma, epsilon);
            // TODO: Fix this bug: scale(I, x) --> xs[xsInd] = scale(I, x)
            scale(I, x);
            plus(numer, x);
            denom += I;
        }
        scale(alpha / denom, numer);
        scale(1d - alpha, mu_hat);
        plus(mu_hat, numer);
    }

    void updateMu(int i, double weight, double[] mu_hat) {
        synchronized (locks[currInd(i)]) {
            // using a+bc == b(a/b+c) to avoid
            // allocating a new array
            double[] mu = mus[currInd(i)];
            for (int j = 0; j < mu.length; j++) {
                mu[j] += weight*mu_hat[j];
            }
        }
    }

    void updateMu(int i, double weight, String mu_hat) {
        double[] muHat = GSON.fromJson(mu_hat, double[].class);
        updateMu(i, weight, muHat);
    }

    // TODO add eqn. number reference <old: see eqn. (32)(bottom)>
    // old: performs sigma <- (1 - alpha)(sigma + (mu_prev - mu)(mu_prev - mu)^T) + ((alpha / denom) * numer)
    // new: performs Rx_hat <- (1 - alpha)(sigma_prev + mu_prev * mu_prev^T) + ((alpha / denom) * numer)
    static void computeSigmaHat(double[][] sigma_hat, double[][] xs, double[] ys, double alpha, double gamma, double epsilon) {

        int numSamples = ys.length;
        assert xs.length == ys.length;
        assert xs.length > 0;
        int M = xs[0].length;

        double[][] numer = new double[M][M];
        double denom = 0d;
        for (int xsInd = 0; xsInd < numSamples; xsInd++) {
            double[] x = new double[M];
            copy(xs[xsInd], x);
            double y = ys[xsInd];
            double I = smoothIndicator(y, gamma, epsilon);
            // scale(sqr(alpha * I / denom), x); // faster alternative? or numeric issues?
            double[][] A = new double[][]{x}; // 1 x M
            plus(numer, scaleA(I, atamm(A))); // M x M
            denom += I;
        }
        scaleA(1d - alpha, sigma_hat);
        plus(sigma_hat, scaleA(alpha / denom, numer));
    }

    void updateSigma(int i, double weight, double[][] sigma_hat) {
        synchronized (locks[currInd(i)]) {
            double[][] sigma = sigmas[currInd(i)];
            // alternatively we could use a+bc == b(a/b+c) to avoid allocating a new array
            for (int j = 0; j < sigma.length; j++) {
                for (int k = 0; k < sigma[0].length; k++) {
                    sigma[j][k] += weight*sigma_hat[j][k];
                }
            }
        }
    }

    void updateSigma(int i, double weight, String sigma_hat) {
        Type sigmaType = new TypeToken<double[][]>() {
        }.getType();
        double[][] sigmaHat = GSON.fromJson(sigma_hat, sigmaType);
        updateSigma(i, weight, sigmaHat);
    }


    void updateCovMat(int i) {
        copy(sigmas[currInd(i)], cov_mat);
        double[][] A = new double[][]{mus[currInd(i)]}; // 1 x M
        minus(cov_mat, atamm(A)); // M x M
    }


    public void start() throws IOException {

        // terminate subscription to broadcast channel
        // so as to avoid multiple calls to start()
        broadcastPubSub.unsubscribe();

        logger.info("agent({}) started", agentId);
        logger.info("connecting to redis at {}:{}", redisHost, redisPort);
        Jedis jedis = new Jedis(redisHost, redisPort);

        String agentIdStr = String.format("%05d", agentId);

        String timerName = name(DCEAgent.class, agentIdStr, "iteration", "timer");
        Timer iteration_timer = registry.timer(timerName);

        registry.register(name(DCEAgent.class, agentIdStr, "rmse", "gauge"), (Gauge<Double>) () -> rmse);
        registry.register(name(DCEAgent.class, agentIdStr, "alpha", "gauge"), (Gauge<Double>) () -> alpha);
        registry.register(name(DCEAgent.class, agentIdStr, "gamma", "gauge"), (Gauge<Double>) () -> gamma);
        registry.register(name(DCEAgent.class, agentIdStr, "numSamples", "gauge"), (Gauge<Integer>) () -> numSamples);

        JmxReporter jmxReporter = JmxReporter.forRegistry(registry).build();
        jmxReporter.start();

        org.apache.log4j.Logger csvLogger = org.apache.log4j.Logger.getLogger(agentIdStr);
        csvLogger.setLevel(Level.INFO);
        resultsDirPath.toFile().mkdirs(); // create any necessary parent directories
        Path csvPath = Paths.get(resultsDirPath.toString(), agentIdStr + ".csv");
        csvLogger.addAppender(new FileAppender(new SimpleLayout(), csvPath.toString(), /* append */ false, /* bufferedIO */ true, /* bufferSize */ 1024));


        int M = targetFn.getDim();

        for (int i = 1; i <= maxIter; i++) {
            Timer.Context iteration_timer_context = iteration_timer.time();

            // weight given to our own estimates
            double selfWeight = neighWeights.get(agentId);

            if (increaseSamples) {
                numSamples = computeNumSamplesForIteration(i, initSamples);
            } else {
                numSamples =  (int) initSamples;
            }
            alpha = computeSmoothingForIteration(i);
            //logger.info("i: {} | numSamples: {} | alpha:{}", i, numSamples, alpha);

            // array of samples and array of target function evaluations
            double[][] xs = new double[numSamples][M];
            double[] ys = new double[xs.length];

            // sample from surrogate and evaluate target at the samples
            Object[] fAndGamma = sampleNewGaussian(i, xs, ys, targetFn);
            MultivariateGaussianDistribution f = (MultivariateGaussianDistribution) fAndGamma[0];
            gamma = (double) fAndGamma[1];

            logTraceParameters(i, "before-computeMuHat");
            logger.trace("i: {} | alpha: {} | gamma: {}", i, alpha, gamma);
            //logger.info("i: {} | numSamples: {} | alpha:{} | gamma: {}, selfWeight: {}", i, numSamples, alpha, gamma, selfWeight);

            // ugly optimization, but since we don't need to draw any more
            // samples from f, we'll reuse its internal register as buffer
            // instead of making another deep copy of mu and sigma
            double[] mu_hat = f.mu; // effectively, mus[prevInd(i)]

            // compute mu_hat of current iteration i, eqn. (32)(top)
            computeMuHat(mu_hat, xs, ys, alpha, gamma, epsilon);
            updateMu(i, selfWeight, mu_hat);

            // broadcast mu_hat to neighbors
            publish(jedis, GSON.toJson(mu_hat), i, Message.PayloadType.MU);

            // compute mu, eqn. (32)(bottom)
            // wait for all neighbors' mu_hat for current iteration i
            muPhaser.awaitAdvance(i - 1); // phase is 0-based

            logTraceParameters(i, "before-computeSigmaHat");

            // compute sigma_hat of current iteration i, eqn. (33)(top)
            // remove "ugly" optimization since agents diffuse correlation matrix (instead covariance matrix) //double[][] sigma_hat = f.sigma;
            double[][] sigma_hat = new double[M][M];
            copy(sigmas[prevInd(i)], sigma_hat);

            // neither mu will receive updates at this point
            computeSigmaHat(sigma_hat, xs, ys, alpha, gamma, epsilon);
            updateSigma(i, selfWeight, sigma_hat);

            // at this point it's safe to clear mu_i-1 and sigma_i-1
            clearParameters(prevInd(i));

            // broadcast sigma_hat to neighbors
            publish(jedis, GSON.toJson(sigma_hat), i, Message.PayloadType.SIGMA);

            // compute sigma, eqn. (33)(bottom)
            // wait for all neighbors' sigma_hat for current iteration i
            sigmaPhaser.awaitAdvance(i - 1); // phase is 0-based

            synchronized (locks[currInd(i)]) {

                // Update covariance matrix from correlation matrix and mean
                //updateSigma(i);
                updateCovMat(i);

                // Check error
                double[] mu = mus[currInd(i)];
                rmse = rmse(mu, targetFn.getSoln());
                double outErr = abs(targetFn.getOptVal() - targetFn.f(mu));
                //csvLogger.info(String.format("%d, %.12f, %.12f, %s", i, outErr, rmse, GSON.toJson(mu)));
                csvLogger.info(String.format("%d, %.12f, %.12f", i, outErr, rmse));
                logger.info("completed iteration({}), outerr: {},  rmse: {}", i, outErr, rmse);
            }
            iteration_timer_context.stop();
        }
        // logger.info("final sigma: {\"sigma_{}_{}\": {{}}}", agentId, maxIter, ps.getSigmas()[maxIter % 2]);
        logger.info("final mu: {\"mu_{}_{}\": {{}}}", agentId, maxIter, mus[maxIter % 2]);

        // terminate all neighbor-listener threads
        neighPubSubs.forEach(jedisPubSub -> jedisPubSub.unsubscribe());

        // flush all buffered appenders
        LogManager.shutdown();
    }

    void logTraceParameters(int i, String msg) {
        if (logger.isTraceEnabled()) {
            synchronized (locks[currInd(i)]) {
                logger.trace("{} {\"mu_{}_{}\": {{}}}", msg, agentId, i, mus[currInd(i)]);
                // logger.trace("{} {\"sigma_{}_{}\": {{}}}", msg, agentId, i, ps.getSigmas()[ps.currInd(i)]);
            }
        }
    }

    void publish(Jedis jedis, String mu_hat, int i, Message.PayloadType type) {
        String out = GSON.toJson(new Message(i, agentId, mu_hat, type));
        jedis.publish(Integer.toString(agentId), out);
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

    JedisPubSub subscribeToBroadcast() {
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

    // TODO: 2 message classes, one for mu and one for sigma
    static class Message {
        enum PayloadType {MU, SIGMA}

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
        for (Map.Entry<Integer, Double> neighWeight : neighWeights.entrySet()) {

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

                    //logger.info("RX from agentId:{} | weight:{}", in.fromAgentId, neighWeight);

                    switch (in.type) {
                        case MU:
                            updateMu(in.i, neighWeight, in.payload);
                            muPhaser.arrive();
                            break;
                        case SIGMA:
                            updateSigma(in.i, neighWeight, in.payload);
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
