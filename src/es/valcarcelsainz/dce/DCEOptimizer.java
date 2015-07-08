package es.valcarcelsainz.dce;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.lang3.ClassUtils;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.Level;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

/**
 * Main method, command-line parsing, logger setup, dce-agents init
 *
 * @author Marcos Sainz
 */
public class DCEOptimizer {

    // mvn exec:java -Dexec.mainClass="es.valcarcelsainz.dce.DCEOptimizer" -Dexec.args="-h"
    // MAVEN_OPTS="-ea" mvn clean install exec:java -Dexec.mainClass="es.valcarcelsainz.dce.DCEOptimizer" -Dexec.args="-w resources/hasting-weights/three-nodes.tsv -t Rosenbrock -i 3 -r localhost -l trace"
    public static void main(final String[] args) {
        final ArgumentParser parser = ArgumentParsers
                .newArgumentParser("dce-optimize")
                .description("Optimize a black-box multivariate function using (Distributed) Cross-Entropy");
        parser.addArgument("-w", "--weights-file")
                .nargs("?")
                .help("path to tab-delimited file containing Hasting weights");
        parser.addArgument("-t", "--target-function")
                .nargs(1)
                .required(true)
                .choices("Dejong", "Griewank", "Pinter", "Powell", "Rosenbrock", "Shekel", "Trigonometric")
                .help("target function to optimize (default: Pinter)");
        parser.addArgument("-o", "--agent-offset")
                .nargs("?")
                .type(Integer.class)
                .setDefault(0)
                .help("ignore agents prior to this offset in the weights file");
        parser.addArgument("-i", "--max-iterations")
                .nargs("?")
                .type(Integer.class)
                .setDefault(500)
                .help("maximum number of iterations to run");
        parser.addArgument("-l", "--log-level")
                .nargs("?")
                .choices("trace", "debug", "info")
                .setDefault("debug")
                .help("log level (default: debug)");
        parser.addArgument("-r", "--redis-host")
                .nargs("?")
                .help("redis host acting as message broker");
        parser.addArgument("-p", "--redis-port")
                .nargs("?")
                .type(Integer.class)
                .setDefault(6379)
                .help("redis port");
        try {
            Logger logger = LoggerFactory.getLogger(DCEOptimizer.class);
            Namespace parsedArgs = parser.parseArgs(args);
            logger.trace(parsedArgs.toString());

            setupLogger(parsedArgs.getString("log_level"));

            final GlobalSolutionFunction targetFn = getTargetFn(parsedArgs);
            logger.info("Target function: {} in M={} dimensions", targetFn.getClass().getName(), targetFn.getDim());
            logger.trace("Target function known global solution at: {}", Arrays.toString(targetFn.getSoln()));

            final int maxIter = parsedArgs.getInt("max_iterations");
            logger.info("Running {} max iterations", maxIter);

            final String redisHost = parsedArgs.getString("redis_host");
            final int redisPort = parsedArgs.getInt("redis_port");
            logger.info("Assuming redis server at {}:{}", redisHost, redisPort);

            Map<Integer, Map<Integer,Double>> agentToNeighborWeightsMap =
                    getAgentToNeighborWeightsMap(parsedArgs);
            for (Map.Entry<Integer,Map<Integer,Double>> entry : agentToNeighborWeightsMap .entrySet()) {
                Integer agentId = entry.getKey();
                Map<Integer, Double> neighWeights = entry.getValue();

                // instantiate dce-agent
                DCEAgent agent = new DCEAgent(agentId, neighWeights, maxIter, redisHost, redisPort, targetFn);
            }

        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        } catch (ClassNotFoundException | InstantiationException | IllegalAccessException | IOException e) {
            e.printStackTrace(System.err);
        }
    }

    public static GlobalSolutionFunction getTargetFn(Namespace parsedArgs) throws InstantiationException,
            IllegalAccessException, ClassNotFoundException {
        String targetFnClassName = String.format("%s.%s",
                DCEOptimizer.class.getPackage().getName(),
                parsedArgs.getList("target_function").get(0));
        final GlobalSolutionFunction targetFn =
                (GlobalSolutionFunction) ClassUtils.getClass(targetFnClassName).newInstance();
        return targetFn;
    }

    private static Map<Integer, Map<Integer,Double>> getAgentToNeighborWeightsMap(Namespace parsedArgs)
            throws IOException {
        Logger logger = LoggerFactory.getLogger(DCEOptimizer.class);
        String weights_path = parsedArgs.getString("weights_file");
        int agentOffset = parsedArgs.getInt("agent_offset");
        logger.info("Applying agent offset of {}", agentOffset);
        List<CSVRecord> agentWeightRecords = parseHastingWeights(weights_path);
        int numAgents = agentWeightRecords.size();
        final Map<Integer, Map<Integer,Double>> agentToNeighWeightsMap = new HashMap<>();
        int recordNumber = 0;
        for (CSVRecord record : agentWeightRecords) {
            if (recordNumber < agentOffset) {
                // skip record until we get to the agent offset
                // this is useful for distributed (multi-process) runs
                recordNumber++;
                continue;
            }
            Map<Integer,Double> neighWeights = new HashMap<>();
            double sumNeighWeights = 0.0; // just to check that neigh weights add to 1
            for (int neighIndex = 0; neighIndex < numAgents; neighIndex++) {
                double neighWeight = Double.parseDouble(record.get(neighIndex));
                sumNeighWeights += neighWeight;
                if (neighWeight > 0.0) {
                    // sparse storage of neighbor weights
                    neighWeights.put(neighIndex, neighWeight);
                }
            }
            assert Math.abs(sumNeighWeights - 1.0) < 1e-6;
            logger.trace(neighWeights.toString());
            agentToNeighWeightsMap.put(recordNumber++, neighWeights);
        }
        assert (numAgents - agentOffset) == agentToNeighWeightsMap.size();
        if (logger.isTraceEnabled()) {
            Integer[] agentIds = agentToNeighWeightsMap.keySet().toArray(new Integer[]{0});
            Arrays.sort(agentIds);
            logger.trace(Arrays.toString(agentIds));
        }
        return agentToNeighWeightsMap;
    }

    private static List<CSVRecord> parseHastingWeights(String weights_path) throws IOException {
        Logger logger = LoggerFactory.getLogger(DCEOptimizer.class);
        // http://commons.apache.org/proper/commons-csv/user-guide.html
        Reader in = new FileReader(weights_path);
        CSVParser csvParser = new CSVParser(in, CSVFormat.TDF);
        List<CSVRecord> weights = csvParser.getRecords(); // read csv into memory
        logger.info("Read {} and parsed Hasting weights for {} agents", weights_path, weights.size());
        return weights;
    }

    private static void setupLogger(final String level) {
        org.apache.log4j.Logger.getRootLogger().addAppender(new ConsoleAppender(
                new PatternLayout(PatternLayout.TTCC_CONVERSION_PATTERN), ConsoleAppender.SYSTEM_ERR));
        org.apache.log4j.Logger.getRootLogger().setLevel(Level.INFO);
        org.apache.log4j.Logger.getLogger("es.valcarcelsainz.dce").setLevel(Level.toLevel(level));
    }
    
}
