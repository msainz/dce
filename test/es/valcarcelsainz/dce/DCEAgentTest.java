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


       /*
         * example 4
         */
        gammaQuantile = 0.5;
        Double[][] example4 = new Double[][] { {0.0}, {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9} };
        Iterator<Double[]> iterator4 = Arrays.asList(example4).iterator();
        gamma = DCEAgent.computeGamma(new double[10][1], new double[10],
                () -> ArrayUtils.toPrimitive(iterator4.next()),
                (double[] x) -> x[0],
                gammaQuantile);
        //assertEquals(0.5, gamma, 1e-14);


        /*
         * example 5
         */
        gammaQuantile = 0.2;
        Double[][] example5= new Double[][] { {0.0}, {0.01}, {0.02}, {0.03}, {0.04} };
        Iterator<Double[]> iterator5 = Arrays.asList(example5).iterator();
        gamma = DCEAgent.computeGamma(new double[5][1], new double[5],
                () -> ArrayUtils.toPrimitive(iterator5.next()),
                (double[] x) -> x[0],
                gammaQuantile);
        //assertEquals(0.01, gamma, 1e-14);


        /*
         * example 6
         */
        gammaQuantile = 0.7;
        Double[][] example6= new Double[][] { {0.0}, {0.01}, {0.02}, {0.03}, {0.04}, {0.05}, {0.06}, {0.07}, {0.08}, {0.09}  };
        Iterator<Double[]> iterator6 = Arrays.asList(example6).iterator();
        gamma = DCEAgent.computeGamma(new double[10][1], new double[10],
                () -> ArrayUtils.toPrimitive(iterator6.next()),
                (double[] x) -> x[0],
                gammaQuantile);
        //assertEquals(0.07, gamma, 1e-14);


        /*
         * example 7
         */
        gammaQuantile = 0.98;
        Double[][] example7= new Double[][] { {0.0}, {0.01}, {0.012}, {0.013}, {0.014}, {0.015}, {0.016}, {0.01801}, {0.01805}, {0.0181},
                {0.1812}, {0.1813}, {0.1814}, {0.1815}, {0.1816}, {0.1817}, {0.1818}, {0.18183}, {0.19}, {0.191}};
        Iterator<Double[]> iterator7 = Arrays.asList(example7).iterator();
        gamma = DCEAgent.computeGamma(new double[10][1], new double[10],
                () -> ArrayUtils.toPrimitive(iterator7.next()),
                (double[] x) -> x[0],
                gammaQuantile);
        assertEquals(0.0191, gamma, 1e-14);
    }
}
