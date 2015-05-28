package es.valcarcelsainz.dce;
import smile.math.MultivariateFunction;

/**
 * An interface representing a real function with a unique global solution.
 *
 * @author Marcos Sainz
 */
public abstract class GlobalSolutionFunction implements MultivariateFunction {

    /**
     * Return the name of the function.
     */
    public String getName() {
        return getClass().getSimpleName();
    }

    public abstract double [] getSoln();

    /**
     * Return the dimensionality of the function's support.
     */
    public int getDim() {
        return getSoln().length;
    }

    public void checkDim(double[] x) {
        if (x.length != getDim()) {
            throw new RuntimeException(
                    getName() + "only supports " + getDim() + " dimensions.");
        }
    }

    /**
     * Compute the value of the function at its solution.
     */
    public double getMax() {
        return f(getSoln());
    }

}
