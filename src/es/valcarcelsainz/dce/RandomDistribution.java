package es.valcarcelsainz.dce;

/**
 * Created by love on 23/12/16.
 */
public interface RandomDistribution {
    double[] rand(final long seed);
    double[] rand();
}
