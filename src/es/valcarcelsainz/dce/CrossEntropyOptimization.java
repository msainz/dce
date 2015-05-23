package es.valcarcelsainz.dce;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.minimize.CostFunction;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

/**
 * Cross-Entropy Optimization algorithm to minimize a black-box cost function.
 *
 */
public final class CrossEntropyOptimization {

    private static final Logger LOG = LogManager
            .getLogger(CrossEntropyOptimization.class);

    private DoubleVector minimize(CostFunction f, DoubleVector pInput, int maxIterations, boolean verbose) {
        DoubleVector globalBestPosition = pInput;

        // loop as long as we haven't reached our max iterations
        for (int iteration = 0; iteration < maxIterations; iteration++) {

            double globalCost = f.evaluateCost(pInput).getCost();

            if (verbose) {
                LOG.info("Iteration " + iteration + " | Cost: " + globalCost);
            }
        }

        return globalBestPosition;
    }

    /**
     * Minimize a black-box function using cross-entropy optimization.
     *
     * @param f deterministic cost function to minimize. Note that the
     *          returned gradient will be ignored, as it is not needed in this algorithm.
     *          This function may be non-convex, non-differentiable.
     * @param pInput the initial starting point of the algorithm, i.e. the parameters
     *               that we are optimizing.
     * @param maxIterations how many iterations this algorithm should perform.
     * @param verbose if true prints progress to STDOUT.
     * @return optimized parameter set for the cost function.
     */
    public static DoubleVector minimizeFunction(CostFunction f, DoubleVector pInput,
                                                final int maxIterations, final boolean verbose) {
        return new CrossEntropyOptimization().minimize(f, pInput, maxIterations, verbose);
    }

    public static void main(String [] args) {
        System.out.println("Hello DCE!");
    }

}