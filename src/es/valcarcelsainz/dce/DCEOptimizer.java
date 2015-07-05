package es.valcarcelsainz.dce;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.log4j.ConsoleAppender;
import org.apache.log4j.PatternLayout;
import org.apache.log4j.Level;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

/**
 * Main method, command-line parsing and logger setup.
 *
 * @author Marcos Sainz
 */
public class DCEOptimizer {

    // mvn exec:java -Dexec.mainClass="es.valcarcelsainz.dce.DCEOptimizer" -Dexec.args="-h"
    // MAVEN_OPTS="-ea" mvn clean install exec:java -Dexec.mainClass="es.valcarcelsainz.dce.DCEOptimizer" -Dexec.args="-w resources/hasting-weights/hundred-nodes-v1.tsv -r localhost"
    public static void main(final String[] args) {
        final ArgumentParser parser = ArgumentParsers.newArgumentParser("dce-optimize")
                .description("Optimize a black-box multivariate function using (Distributed) Cross-Entropy");
        parser.addArgument("-w", "--weights-file").nargs("?").help("path to tab-delimited file containing Hasting weights");
        parser.addArgument("-l", "--log-level").nargs("?").setDefault("DEBUG").help("log level (default: DEBUG)");
        parser.addArgument("-r", "--redis-host").nargs("?").help("redis host acting as message broker");
        parser.addArgument("-p", "--redis-port").nargs("?").setDefault(6379).help("redis port");

        try {
            Logger logger = LoggerFactory.getLogger(DCEOptimizer.class);
            Namespace parsedArgs = parser.parseArgs(args);
            logger.trace(parsedArgs.toString());
            String weights_path = parsedArgs.getString("weights_file");
            setupLogger(parsedArgs.getString("log_level"));

            // http://commons.apache.org/proper/commons-csv/user-guide.html
            Reader in = new FileReader(weights_path);
            CSVParser csvParser = new CSVParser(in, CSVFormat.TDF);
            List<CSVRecord> weights = csvParser.getRecords(); // read csv into memory
            int numAgents = weights.size();
            logger.info("Read {} and parsed Hasting weights for {} agents", weights_path, numAgents);
            for (CSVRecord record : weights) {
                Map<Integer,Double> neighWeights = new HashMap<>();
                double sumNeighWeights = 0.0;
                for (int neighIndex = 0; neighIndex < numAgents; neighIndex++) {
                    double neighWeight = Double.parseDouble(record.get(neighIndex));
                    sumNeighWeights += neighWeight;
                    if (neighWeight > 0.0) {
                        // sparse storage of neighbor weights
                        neighWeights.put(neighIndex, neighWeight);
                    }
                }
                assert Math.abs(sumNeighWeights - 1.0) < 1e-6;
                System.out.println(neighWeights);
            }

        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        } catch (IOException e) {
            e.printStackTrace(System.err);
        }
    }

    private static void setupLogger(final String level) {
        org.apache.log4j.Logger.getRootLogger().addAppender(new ConsoleAppender(
                new PatternLayout(PatternLayout.TTCC_CONVERSION_PATTERN), ConsoleAppender.SYSTEM_ERR));
        org.apache.log4j.Logger.getRootLogger().setLevel(Level.INFO);
        org.apache.log4j.Logger.getLogger("es.valcarcelsainz.dce").setLevel(Level.toLevel(level));
    }
    
}
