package es.valcarcelsainz.dce;

import java.util.Arrays;
import java.util.Iterator;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;


public class DCEAgentTest {

    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void testComputeGamma() throws Exception {
        double gamma, gammaQuantile;

        /*
         * example 1
         */
        gammaQuantile = 0.;
        Double[][] example1 = new Double[][] { {0.}, {1.} };
        Iterator<Double[]> iterator1 = Arrays.asList(example1).iterator();
        gamma = DCEAgent.computeGamma(new double[2][1], new double[2],
                () -> ArrayUtils.toPrimitive(iterator1.next()),
                (double[] x) -> x[0],
                gammaQuantile);

        assertEquals(0., gamma, 1e-14);

        /*
         * example 2
         */
        gammaQuantile = 1.;
        Double[][] example2 = new Double[][] { {0.}, {1.} };
        Iterator<Double[]> iterator2 = Arrays.asList(example2).iterator();
        gamma = DCEAgent.computeGamma(new double[2][1], new double[2],
                () -> ArrayUtils.toPrimitive(iterator2.next()),
                (double[] x) -> x[0],
                gammaQuantile);

        assertEquals(1., gamma, 1e-14);

        /*
         * example 3
         */
        gammaQuantile = 0.9;
        Double[][] example3 = new Double[][] { {0.}, {1.}, {2.}, {3.}, {4.}, {5.}, {6.}, {7.}, {8.}, {9.} };
        Iterator<Double[]> iterator3 = Arrays.asList(example3).iterator();
        gamma = DCEAgent.computeGamma(new double[10][1], new double[10],
                () -> ArrayUtils.toPrimitive(iterator3.next()),
                (double[] x) -> x[0],
                gammaQuantile);

        assertEquals(9., gamma, 1e-14);
    }
}
