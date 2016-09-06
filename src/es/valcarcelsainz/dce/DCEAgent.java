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

    final Map<Integer, Double> neighWeights; // TODO: use immutable map
    final int agentId; // k
    final int maxIter;
    final double gammaQuantile;
    final double lowerBound;
    final double upperBound;
    final double epsilon;
    final double initSamples;
    final boolean regNoise;
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

    // synchronization locks for mu/sigma_i and mu/sigma_i+1
    Object[] locks = new Object[]{
            new Object(),
            new Object()
    };

    // constructor
    DCEAgent(Integer agentId, Map<Integer, Double> neighWeights, Integer maxIter, double gammaQuantile,
             double lowerBound, double upperBound, double epsilon, double initSamples, boolean regNoise, String redisHost, Integer redisPort,
             GlobalSolutionFunction targetFn, String resultsDirPath) {

        this.agentId = agentId;
        this.neighWeights = neighWeights;
        this.maxIter = maxIter;
        this.gammaQuantile = gammaQuantile;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.epsilon = epsilon;
        this.initSamples = initSamples;
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

        subscribeToBroadcast();
        subscribeToNeighbors();

        // TODO: jedisPubSub.unsubscribe();
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
                for (int i = 0; i < mus[j].length; i++) {
                    sigmas[j][i][i] = 1000;
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
        MultivariateGaussianDistribution f;
        try {
            f = new MultivariateGaussianDistribution(
                    // this constructor deep-copies mu[i] and sigma[i]
                    // later accessible via f.mean() and f.cov() respectively
                    mus[prevInd(i)], sigmas[prevInd(i)],
                    new myRandom(System.currentTimeMillis() + Thread.currentThread().getId()) // TODO: reuse Random instance
                    //new Random(System.currentTimeMillis() + Thread.currentThread().getId()) // TODO: reuse Random instance
            );
        } catch (IllegalArgumentException e) {
            //logger.info("{\"mu_{}_{}\": {{}}}", agentId, i - 1, mus[prevInd(i)]);
            //logger.info("{\"sigma_{}_{}\": {{}}}", agentId, i - 1, sigmas[prevInd(i)]);
            logger.info("i: {} | id: {}  singular matrix", i, agentId);
            throw e;
        }

        // TODO refer to eqn. (32)
        //  2) Que la primera distribución de muestreo sea una uniforme en la
        //  hyper-box. Es decir, sería algo como
        //  if (iteración == 1) {
        //      xs = Uniform( hyperbox );
        //  } else {
        //      xs = MultivariateGaussian ( mu, cov );
        //  }

        //  RandomDistribution h;
        //  if (1 == i) {
        //      UniformDistribution g = new UniformDistribution(lBound, uBound);
        //      h = g;
        //  } else {
        //      h = f;
        //  }
        //  double gamma = computeGamma2(xs, ys, () -> h.rand(), (double[] x) -> J.f(x), gammaQuantile);

        double gamma = computeGamma2(xs, ys, () -> f.rand(), (double[] x) -> J.f(x), gammaQuantile);
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

    static double computeGamma(double[][] xs, double[] ys, Supplier<double[]> f, Function<double[], Double> jfn, double gammaQuantile) {
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
    }

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
    static void computeSigmaHat(double[][] sigma_hat, double[] mu_hat, double[] mu_prev, double[] mu_curr,
                                double[][] xs, double[] ys, double alpha, double gamma, double epsilon) {

        boolean meth1 = false;
        boolean meth2 = false;
        boolean meth3 = true;

        int numSamples = ys.length;
        assert xs.length == ys.length;
        assert xs.length > 0;
        int M = xs[0].length;


        /*
        double[][] sigma_hat0 = new double[M][M];
        copy(sigma_hat, sigma_hat0);


        double[][] numer1 = new double[M][M];
        double denom1 = 0d;
        double[] mu_hat1 = new double[M];
        double[] mu_prev1 = new double[M];
        double[][] sigma_hat1 = new double[M][M];
        if (meth1) {
            double[] mu_prev0 = new double[M];
            copy(mu_prev, mu_prev0);
            double[] mu_hat0 = new double[M];
            copy(mu_hat, mu_hat0);

            copy(mu_hat0, mu_hat1);
            copy(mu_prev0, mu_prev1);
            copy(sigma_hat0, sigma_hat1);

            for (int xsInd = 0; xsInd < numSamples; xsInd++) {
                double[] x1 = new double[M];
                copy(xs[xsInd], x1);
                double y = ys[xsInd];
                double I = smoothIndicator(y, gamma, epsilon);
                // old version: (x-mu)*(x-mu)'
                minus(x1, mu_hat1);
                // scale(sqr(alpha * I / denom), x); // faster alternative? or numeric issues?
                double[][] A = new double[][]{x1}; // 1 x M
                plus(numer1, scaleA(I, atamm(A))); // M x M
                denom1 += I;
            }
            // old version: (mu_old - mu) * (mu_ol - mu)'  | v1: mu_curr; v2: mu_hat
            minus(mu_prev1, mu_hat1);
            double[][] B = new double[][]{mu_prev1}; // 1 x M
            plus(sigma_hat1, atamm(B)); // M x M
            scaleA(1d - alpha, sigma_hat1);
            plus(sigma_hat1, scaleA(alpha / denom1, numer1));
            logger.info("sigma_hat1: {{}}", sigma_hat1);
            //logger.info("numer1: {} | denom1: {}", numer1, denom1);
        }

        double[][] numer2 = new double[M][M];
        double denom2 = 0d;
        double[][] sigma_hat2 = new double[M][M];
        if (meth2) {

            copy(sigma_hat0, sigma_hat2);

            for (int xsInd = 0; xsInd < numSamples; xsInd++) {
                double[] x = xs[xsInd];
                double y = ys[xsInd];
                double I = smoothIndicator(y, gamma, epsilon);
                // scale(sqr(alpha * I / denom), x); // faster alternative? or numeric issues?
                double[][] A = new double[][]{x}; // 1 x M
                plus(numer2, scaleA(I, atamm(A))); // M x M
                denom2 += I;
            }
            double[][] B = new double[][]{mu_prev}; // 1 x M
            plus(sigma_hat2, atamm(B)); // M x M
            scaleA(1d - alpha, sigma_hat2);
            plus(sigma_hat2, scaleA(alpha / denom2, numer2));
            double[][] C = new double[][]{mu_hat}; // 1 x M
            minus(sigma_hat2, atamm(C)); // M x M
            logger.info("sigma_hat2: {{}}", sigma_hat2);
            //logger.info("numer2: {} | denom2: {}", numer2, denom2);
        }
        */


        if (meth3) {
            double[][] numer3 = new double[M][M];
            double denom3 = 0d;
            //double[][] sigma_hat3 = new double[M][M];
            //copy(sigma_hat0, sigma_hat3);
            for (int xsInd = 0; xsInd < numSamples; xsInd++) {
                double[] x = xs[xsInd];
                double y = ys[xsInd];
                double I = smoothIndicator(y, gamma, epsilon);
                // scale(sqr(alpha * I / denom), x); // faster alternative? or numeric issues?
                double[][] A = new double[][]{x}; // 1 x M
                plus(numer3, scaleA(I, atamm(A))); // M x M
                denom3 += I;
            }
            double[][] B = new double[][]{mu_prev}; // 1 x M
            //plus(sigma_hat3, atamm(B)); // M x M
            //scaleA(1d - alpha, sigma_hat3);
            //plus(sigma_hat3, scaleA(alpha / denom3, numer3));
            //copy(sigma_hat3, sigma_hat);


            plus(sigma_hat, atamm(B)); // M x M
            scaleA(1d - alpha, sigma_hat);
            plus(sigma_hat, scaleA(alpha / denom3, numer3));
            //logger.info("sigma_hat3: {{}}", sigma_hat3);
            //logger.info("numer3: {} | denom3: {}", numer3, denom3);

        }


    }

    void updateSigma(int i, double weight, double[][] sigma_hat) {
        synchronized (locks[currInd(i)]) {
            double[][] sigma = sigmas[currInd(i)];
            // using a+bc == b(a/b+c) to avoid allocating a new array
            //double inverse_weight = 1. / weight;
            for (int j = 0; j < sigma.length; j++) {
                for (int k = 0; k < sigma[0].length; k++) {
                    //sigma[j][k] *= inverse_weight;
                    //sigma[j][k] += sigma_hat[j][k];
                    //sigma[j][k] *= weight;
                    sigma[j][k] += weight*sigma_hat[j][k];
                }
            }
        }
    }

    void updateSigma(int i) {
        double[][] sigma = sigmas[currInd(i)];
        double[][] A = new double[][]{mus[currInd(i)]}; // 1 x M
        minus(sigma, atamm(A)); // M x M
    }


    void updateSigma(int i, double weight, String sigma_hat) {
        Type sigmaType = new TypeToken<double[][]>() {
        }.getType();
        double[][] sigmaHat = GSON.fromJson(sigma_hat, sigmaType);
        updateSigma(i, weight, sigmaHat);
    }

    // TODO: prevent multiple undesired calls to start()
    public void start() throws IOException {
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

        for (int i = 1; i <= maxIter; i++) {
            Timer.Context iteration_timer_context = iteration_timer.time();

            // weight given to our own estimates
            double selfWeight = neighWeights.get(agentId);

            int M = targetFn.getDim();
            numSamples = computeNumSamplesForIteration(i, initSamples);
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
            double[][] sigma_hat = f.sigma; // same ugly optimization

            // neither mu will receive updates at this point
            computeSigmaHat(sigma_hat, mu_hat, mus[prevInd(i)], mus[currInd(i)], xs, ys, alpha, gamma, epsilon);
            updateSigma(i, selfWeight, sigma_hat);

            // at this point it's safe to clear mu_i-1 and sigma_i-1
            clearParameters(prevInd(i));

            // broadcast sigma_hat to neighbors
            publish(jedis, GSON.toJson(sigma_hat), i, Message.PayloadType.SIGMA);

            // compute sigma, eqn. (33)(bottom)
            // wait for all neighbors' sigma_hat for current iteration i
            sigmaPhaser.awaitAdvance(i - 1); // phase is 0-based

            synchronized (locks[currInd(i)]) {

                // Convert correlation matrix into covariance matrix
                updateSigma(i);

                // Check error
                double[] mu = mus[currInd(i)];
                rmse = rmse(mu, targetFn.getSoln());
                csvLogger.info(String.format("%d, %f, %s", i, rmse, GSON.toJson(mu)));
                logger.info("completed iteration({}), rmse: {}", i, rmse);
            }
            iteration_timer_context.stop();
        }
        // logger.info("final sigma: {\"sigma_{}_{}\": {{}}}", agentId, maxIter, ps.getSigmas()[maxIter % 2]);
        logger.info("final mu: {\"mu_{}_{}\": {{}}}", agentId, maxIter, mus[maxIter % 2]);

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
