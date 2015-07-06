package es.valcarcelsainz.dce;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPubSub;

import java.lang.reflect.InvocationTargetException;
import java.util.Map;

/**
 * @author Marcos Sainz
 */
public class DCEAgent {

    private static final Logger logger =
            LoggerFactory.getLogger(DCEAgent.class);

    private final Integer agentId;
    private final Map<Integer, Double> neighWeights;
    private final Integer maxIter;

    public DCEAgent(Integer agentId, Map<Integer, Double> neighWeights, Integer maxIter) {
        this.agentId = agentId;
        this.neighWeights = neighWeights;
        this.maxIter = maxIter;
        subscribeToBroadcast();

        // TODO: jedisPubSub.unsubscribe();
    }

    public void start() {
        logger.info("start called!");
        try {
            Thread.sleep(10 * 1000); // get busy
        } catch (InterruptedException e) {
            logger.error(e.getMessage(), e);
        }
        logger.info("done with start.");
    }

    public void stop() {
        logger.info("stop called!");
    }

    public JedisPubSub subscribeToBroadcast() {
        final JedisPubSub jedisPubSub = new JedisPubSub() {
            @Override
            public void onMessage(String channel, String message) {
                logger.info("channel: {}, message: {}", channel, message);
                DCEAgent enclosingAgent = DCEAgent.this;
                java.lang.reflect.Method method;
                try {
                    method = enclosingAgent.getClass().getMethod(message);
                    method.invoke(enclosingAgent);
                } catch (NoSuchMethodException | InvocationTargetException | IllegalAccessException e) {
                    logger.error(e.getMessage(), e);
                }
            }
        };
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    logger.info("Connecting to redis");
                    Jedis jedis = new Jedis("localhost", 6379);
                    logger.info("subscribing to broadcast");
                    jedis.subscribe(jedisPubSub, "broadcast");
                    logger.info("subscribe returned, closing down");
                    jedis.quit();
                } catch (Exception e) {
                    logger.error(e.getMessage(), e);
                }
            }
        }, String.format("agentId: %s, mainThread", agentId)).start();
        return jedisPubSub;
    }

    public JedisPubSub[] subscribeToNeighbors() {

        // subscribe to the channel of each neighbor and the "broadcast" channel
        // loop through neighbors, create a pubsub object/thread for each channel

        final JedisPubSub jedisPubSub = new JedisPubSub() {
            @Override
            public void onMessage(String channel, String message) {
                // messageContainer.add(message);
                logger.info("channel: {}, message: {}", channel, message);
                // messageReceivedLatch.countDown();
            }
        };

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    logger.info("Connecting");
                    Jedis jedis = new Jedis("localhost", 6379);
                    logger.info("subscribing");
                    jedis.subscribe(jedisPubSub, "test");
                    logger.info("subscribe returned, closing down");
                    jedis.quit();
                } catch (Exception e) {
                    logger.error(e.getMessage());
                    logger.error(e.getStackTrace().toString());
                }
            }
        }, "subscriberThread").start();

        return null;
    }
}
