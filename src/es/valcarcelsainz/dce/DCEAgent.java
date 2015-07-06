package es.valcarcelsainz.dce;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPubSub;

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
    }

    private void run() throws InterruptedException {
        JedisPubSub jedisPubSub = subscribeToNeighbors()[0];
        // setupPublisher();

        // publish away!
        // publishLatch.countDown();

        // messageReceivedLatch.await();
        // logger.info("Got message: %s", messageContainer.iterator().next());

        jedisPubSub.unsubscribe();
    }

    public JedisPubSub subscribeToBroadcast() {
        final JedisPubSub jedisPubSub = new JedisPubSub() {
            @Override
            public void onMessage(String channel, String message) {
                // messageContainer.add(message);
                logger.info("channel: {}, message: {}", channel, message);
                // messageReceivedLatch.countDown();

                // call start() or stop() on the agent instance
                // can do dynamically with a bit of meta-programming
                // start or stop message then conveniently sent via redis-cli
                // once all agents in all threads and machines have subscribed
            }
        };
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    logger.info("Connecting");
                    Jedis jedis = new Jedis("localhost", 6379);
                    logger.info("subscribing");
                    jedis.subscribe(jedisPubSub, "broadcast");
                    logger.info("subscribe returned, closing down");
                    jedis.quit();
                } catch (Exception e) {
                    logger.error(e.getMessage());
                    logger.error(e.getStackTrace().toString());
                }
            }
        }, "broadcastSubscriberThread").start();
        return jedisPubSub;
    }

    public JedisPubSub [] subscribeToNeighbors() {

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
