/*
 *  Java Information Dynamics Toolkit (JIDT)
 *  Copyright (C) 2012, Joseph T. Lizier
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package infodynamics.measures.continuous.kraskov;

import infodynamics.measures.continuous.MultiVariateInfoMeasureCalculatorCommon;
import infodynamics.utils.EuclideanUtils;
import infodynamics.utils.NeighbourNodeData;
import infodynamics.utils.KdTree;
import infodynamics.utils.UnivariateNearestNeighbourSearcher;
import infodynamics.utils.MathsUtils;
import infodynamics.utils.MatrixUtils;

import java.util.PriorityQueue;
import java.util.Calendar;
import java.util.Random;

/**
 * <p>Base class with common functionality for child class implementations of
 * multivariate information measures on a given multivariate
 * <code>double[][]</code> set of
 * observations (extending {@link MultiVariateInfoMeasureCalculatorCommon}),
 * using Kraskov-Stoegbauer-Grassberger (KSG) estimation
 * (see Kraskov et al., below).</p>
 *    
 * <p>Usage is as per the paradigm outlined for {@link MultiVariateInfoMeasureCalculatorCommon},
 * with:
 * <ul>
 *  <li>For constructors see the child classes.</li>
 *  <li>Further properties are defined in {@link #setProperty(String, String)}.</li>
 *  <li>Computed values are in <b>nats</b>, not bits!</li>
 *  </ul>
 * </p>
 * 
 * <p>Finally, note that {@link Cloneable} is implemented allowing clone()
 *  to produce only an automatic shallow copy, which is fine
 *  for the statistical significance calculation it is intended for
 *  (none of the array data will be changed there).
 * </p>
 * 
 * <p><b>References:</b><br/>
 * <ul>
 *  <li>Rosas, F., Mediano, P., Gastpar, M, Jensen, H.,
 *   <a href="http://dx.doi.org/10.1103/PhysRevE.100.032305">"Quantifying high-order
 *   interdependencies via multivariate extensions of the mutual information"</a>,
 *   Physical Review E 100, (2019) 032305.</li>
 *
 *  <li>Kraskov, A., Stoegbauer, H., Grassberger, P., 
 *   <a href="http://dx.doi.org/10.1103/PhysRevE.69.066138">"Estimating mutual information"</a>,
 *   Physical Review E 69, (2004) 066138.</li>
 * </ul>
 * @author Pedro A.M. Mediano (<a href="pmediano at pm.me">email</a>,
 * <a href="http://www.doc.ic.ac.uk/~pam213">www</a>)
 */
public abstract class MultiVariateInfoMeasureCalculatorKraskov
  extends MultiVariateInfoMeasureCalculatorCommon
  implements Cloneable { // See comments on clonability above

  /**
   * we compute distances to the kth nearest neighbour
   */
  protected int k = 4;
    
  /**
   * The norm type in use (see {@link #PROP_NORM_TYPE})
   */
  protected int normType = EuclideanUtils.NORM_MAX_NORM;
    
  /**
   * Property name for the number of K nearest neighbours used in
   * the KSG algorithm (default 4).
   */
  public final static String PROP_K = "k";
  /**
   * Property name for what type of norm to use between data points
   *  for each marginal variable -- Options are defined by 
   *  {@link KdTree#setNormType(String)} and the
   *  default is {@link EuclideanUtils#NORM_MAX_NORM}.
   */
  public final static String PROP_NORM_TYPE = "NORM_TYPE";
  /**
   * Property name for an amount of random Gaussian noise to be
   *  added to the data (default 1e-8 to match the noise order in MILCA toolkit.).
   */
  public static final String PROP_ADD_NOISE = "NOISE_LEVEL_TO_ADD";
  /**
   * Property name for a dynamics exclusion time window 
   * otherwise known as Theiler window (see Kantz and Schreiber).
   * Default is 0 which means no dynamic exclusion window.
   */
  public static final String PROP_DYN_CORR_EXCL_TIME = "DYN_CORR_EXCL"; 
  /**
   * Property name for the number of parallel threads to use in the
   *  computation (default is to use all available)
   */
  public static final String PROP_NUM_THREADS = "NUM_THREADS";
  /**
   * Valid property value for {@link #PROP_NUM_THREADS} to indicate
   *  that all available processors should be used. 
   */
  public static final String USE_ALL_THREADS = "USE_ALL";

  /**
   * Whether to add an amount of random noise to the incoming data
   */
  protected boolean addNoise = true;
  /**
   * Amount of random Gaussian noise to add to the incoming data
   */
  protected double noiseLevel = (double) 1e-8;
  /**
   * Whether we use dynamic correlation exclusion
   */
  protected boolean dynCorrExcl = false;
  /**
   * Size of dynamic correlation exclusion window.
   */
  protected int dynCorrExclTime = 0;
  /**
   * Number of parallel threads to use in the computation;
   *  defaults to use all available.
   */
  protected int numThreads = Runtime.getRuntime().availableProcessors();
  /**
   * Protected k-d tree data structure (for fast nearest neighbour searches)
   *  representing the joint space
   */
  protected KdTree kdTreeJoint;
  // /**
  //  * protected data structures (for fast nearest neighbour searches)
  //  *  representing the marginal spaces
  //  */
  // protected KdTree[] rangeSearchersInMarginals;
  /**
   * Protected data structures (for fast nearest neighbour searches)
   *  representing the marginal spaces of each individual variable
   */
  protected UnivariateNearestNeighbourSearcher[] rangeSearchersInSmallMarginals;
  /**
   * Protected data structures (for fast nearest neighbour searches)
   *  representing the marginal spaces of each set of (D-1) variables
   */
  protected KdTree[] rangeSearchersInBigMarginals;
  /**
   * Constant for digamma(k), with k the number of nearest neighbours selected
   */
  protected double digammaK;
  /**
   * Constant for digamma(N), with N the number of samples.
   */
  protected double digammaN;


  public void initialise(int dimensions) {
    this.dimensions = dimensions;
    lastAverage = 0.0;
    totalObservations = 0;
    isComputed = false;
    observations = null;
    kdTreeJoint = null;
    rangeSearchersInSmallMarginals = null;
    rangeSearchersInBigMarginals = null;
  }

  /**
   * Sets properties for the KSG multivariate measure calculator.
   *  New property values are not guaranteed to take effect until the next call
   *  to an initialise method. 
   *  
   * <p>Valid property names, and what their
   * values should represent, include:</p>
   * <ul>
   *  <li>{@link #PROP_K} -- number of k nearest neighbours to use in joint kernel space
   *      in the KSG algorithm (default is 4).</li>
   *  <li>{@link #PROP_NORM_TYPE} -- normalization type to apply to 
   *      working out the norms between the points in each marginal space.
   *      Options are defined by {@link KdTree#setNormType(String)} -
   *      default is {@link EuclideanUtils#NORM_MAX_NORM}.</li>
   *  <li>{@link #PROP_DYN_CORR_EXCL_TIME} -- a dynamics exclusion time window,
   *      also known as Theiler window (see Kantz and Schreiber);
   *      default is 0 which means no dynamic exclusion window.</li>
   *  <li>{@link #PROP_ADD_NOISE} -- a standard deviation for an amount of
   *    random Gaussian noise to add to
   *      each variable, to avoid having neighbourhoods with artificially
   *      large counts. (We also accept "false" to indicate "0".)
   *      The amount is added in after any normalisation,
   *      so can be considered as a number of standard deviations of the data.
   *      (Recommended by Kraskov. MILCA uses 1e-8; but adds in
   *      a random amount of noise in [0,noiseLevel) ).
   *      Default 1e-8 to match the noise order in MILCA toolkit..</li>
   * </ul>
   * 
   * <p>Unknown property values are ignored.</p>
   * 
   * @param propertyName name of the property
   * @param propertyValue value of the property
   * @throws Exception for invalid property values
   */
  public void setProperty(String propertyName, String propertyValue) throws Exception {
    boolean propertySet = true;
    if (propertyName.equalsIgnoreCase(PROP_K)) {
      k = Integer.parseInt(propertyValue);
    } else if (propertyName.equalsIgnoreCase(PROP_NORM_TYPE)) {
      normType = KdTree.validateNormType(propertyValue);
    } else if (propertyName.equalsIgnoreCase(PROP_DYN_CORR_EXCL_TIME)) {
      dynCorrExclTime = Integer.parseInt(propertyValue);
      dynCorrExcl = (dynCorrExclTime > 0);
    } else if (propertyName.equalsIgnoreCase(PROP_ADD_NOISE)) {
      if (propertyValue.equals("0") ||
          propertyValue.equalsIgnoreCase("false")) {
        addNoise = false;
        noiseLevel = 0;
      } else {
        addNoise = true;
        noiseLevel = Double.parseDouble(propertyValue);
      }
    } else if (propertyName.equalsIgnoreCase(PROP_NUM_THREADS)) {
      if (propertyValue.equalsIgnoreCase(USE_ALL_THREADS)) {
        numThreads = Runtime.getRuntime().availableProcessors();
      } else { // otherwise the user has passed in an integer:
        numThreads = Integer.parseInt(propertyValue);
      }
    } else {
      // No property was set here
      propertySet = false;
      // try the superclass:
      super.setProperty(propertyName, propertyValue);
    }
    if (debug && propertySet) {
      System.out.println(this.getClass().getSimpleName() + ": Set property " + propertyName +
          " to " + propertyValue);
    }
  }

  @Override
  public void finaliseAddObservations() throws Exception {
    super.finaliseAddObservations();

    if ((observations == null) || (observations[0].length == 0)) {
      throw new Exception("Computing measure with a null set of data");
    }
    if (observations.length <= k + 2*dynCorrExclTime) {
      throw new Exception("There are less observations provided (" +
          observations.length +
          ") than required for the number of nearest neighbours parameter (" +
          k + ") and any dynamic correlation exclusion (" + dynCorrExclTime + ")");
    }
    
    // Normalise the data if required
    if (normalise) {
      // We can overwrite these since they're already
      //  a copy of the users' data.
      MatrixUtils.normalise(observations);
    }
    
    // Add small random noise if required
    if (addNoise) {
      Random random = new Random();
      // Add Gaussian noise of std dev noiseLevel to the data
      for (int r = 0; r < observations.length; r++) {
        for (int c = 0; c < dimensions; c++) {
          observations[r][c] += random.nextGaussian()*noiseLevel;
        }
      }
    }

    // Set the constants:
    digammaK = MathsUtils.digamma(k);
    digammaN = MathsUtils.digamma(totalObservations);
  }

  /**
   * Internal method to ensure that the Kd-tree data structures to represent the
   * observational data have been constructed (should be called prior to attempting
   * to use these data structures)
   */
  protected void ensureKdTreesConstructed() throws Exception {

    // We need to construct the k-d trees for use by the child
    //  classes. We check each tree for existence separately
    //  since source can be used across original and surrogate data
    // TODO can parallelise these -- best done within the kdTree --
    //  though it's unclear if there's much point given that
    //  the tree construction itself afterwards can't really be well parallelised.
    if (kdTreeJoint == null) {
      kdTreeJoint = new KdTree(observations);
      kdTreeJoint.setNormType(normType);
    }
    if (rangeSearchersInSmallMarginals == null) {
      rangeSearchersInSmallMarginals = new UnivariateNearestNeighbourSearcher[dimensions];
      for (int d = 0; d < dimensions; d++) {
        rangeSearchersInSmallMarginals[d] = new UnivariateNearestNeighbourSearcher(
            MatrixUtils.selectColumn(observations, d));
        rangeSearchersInSmallMarginals[d].setNormType(normType);
      }
    }
    if (rangeSearchersInBigMarginals == null) {
      rangeSearchersInBigMarginals = new KdTree[dimensions];
      for (int d = 0; d < dimensions; d++) {
        rangeSearchersInBigMarginals[d] = new KdTree(
            MatrixUtils.selectColumns(observations, allExcept(d, dimensions)));
        rangeSearchersInBigMarginals[d].setNormType(normType);
      }
    }

  }

  /**
   * {@inheritDoc} 
   * 
   * @return the average measure in nats (not bits!)
   */
  public double computeAverageLocalOfObservations() throws Exception {
    // Compute the measure
    double startTime = Calendar.getInstance().getTimeInMillis();
    lastAverage = computeFromObservations(false)[0];
    isComputed = true;
    if (debug) {
      Calendar rightNow2 = Calendar.getInstance();
      long endTime = rightNow2.getTimeInMillis();
      System.out.println("Calculation time: " + ((endTime - startTime)/1000.0) + " sec" );
    }
    return lastAverage;
  }

  /**
   * {@inheritDoc}
   * 
   * @return the "time-series" of local measure values in nats (not bits!)
   * @throws Exception
   */
  public double[] computeLocalOfPreviousObservations() throws Exception {
    double[] localValues = computeFromObservations(true);
    lastAverage = MatrixUtils.mean(localValues);
    isComputed = true;
    return localValues;
  }

  /**
   * This method, specified in {@link MultiVariateInfoMeasureCalculatorCommon}
   * is not implemented yet here.
   */
  public double[] computeLocalUsingPreviousObservations(double[][] states) throws Exception {
    // TODO If this is implemented, will need to normalise the incoming
    //  observations the same way that previously supplied ones were
    //  normalised (if they were normalised, that is)
    throw new Exception("Local method not implemented yet");
  }

  /**
   * This protected method handles the multiple threads which
   *  computes either the average or local measure (over parts of the total
   *  observations), computing the
   *  distances between all tuples in time.
   * 
   * <p>The method returns:<ol>
   *  <li>for (returnLocals == false), an array of size 1,
   *      containing the average measure </li>
   *  <li>for (returnLocals == true), the array of local
   *      measure values</li>
   *  </ol>
   * 
   * @param returnLocals whether to return an array or local values, or else
   *  sums of these values
   * @return either the average measure, or array of local measure value,
   *  in nats not bits
   * @throws Exception
   */
  protected double[] computeFromObservations(boolean returnLocals) throws Exception {
    
    double[] returnValues = null;

    ensureKdTreesConstructed();
    
    if (numThreads == 1) {
      // Single-threaded implementation:
      returnValues = partialComputeFromObservations(0, totalObservations, returnLocals);
      
    } else {
      // We're going multithreaded:
      if (returnLocals) {
        // We're computing locals
        returnValues = new double[totalObservations];
      } else {
        // We're computing average
        returnValues = new double[1];
      }
      
      // Distribute the observations to the threads for the parallel processing
      int lTimesteps = totalObservations / numThreads; // each thread gets the same amount of data
      int res = totalObservations % numThreads; // the first thread gets the residual data
      if (debug) {
        System.out.printf("Computing Kraskov Multi-Info with %d threads (%d timesteps each, plus %d residual)%n",
            numThreads, lTimesteps, res);
      }
      Thread[] tCalculators = new Thread[numThreads];
      KraskovThreadRunner[] runners = new KraskovThreadRunner[numThreads];
      for (int t = 0; t < numThreads; t++) {
        int startTime = (t == 0) ? 0 : lTimesteps * t + res;
        int numTimesteps = (t == 0) ? lTimesteps + res : lTimesteps;
        if (debug) {
          System.out.println(t + ".Thread: from " + startTime +
              " to " + (startTime + numTimesteps)); // Trace Message
        }
        runners[t] = new KraskovThreadRunner(this, startTime, numTimesteps, returnLocals);
        tCalculators[t] = new Thread(runners[t]);
        tCalculators[t].start();
      }
      
      // Here, we should wait for the termination of the all threads
      //  and collect their results
      for (int t = 0; t < numThreads; t++) {
        if (tCalculators[t] != null) { // TODO Ipek: can you comment on why we're checking for null here?
          tCalculators[t].join(); 
        }
        // Now we add in the data from this completed thread:
        if (returnLocals) {
          // We're computing local measure; copy these local values
          //  into the full array of locals
          System.arraycopy(runners[t].getReturnValues(), 0, 
              returnValues, runners[t].myStartTimePoint, runners[t].numberOfTimePoints);
        } else {
          // We're computing the average measure, keep the running sums of digammas and counts
          MatrixUtils.addInPlace(returnValues, runners[t].getReturnValues());
        }
      }
    }
    
    return returnValues;

  }
  
  /**
   * Protected method to be used internally for threaded implementations.  This
   * method implements the guts of each Kraskov algorithm, computing the number
   * of nearest neighbours in each dimension for a sub-set of the data points.
   * It is intended to be called by one thread to work on that specific sub-set
   * of the data.
   *
   * <p>Child classes should implement the computation of any specific measure
   * using this method.</p>
   * 
   * <p>The method returns:<ol>
   *  <li>for average measures (returnLocals == false), the relevant sums of
   *    digamma(n_x+1) in each marginal
   *     for a partial set of the observations</li>
   *  <li>for local measures (returnLocals == true), the array of local values</li>
   *  </ol>
   * 
   * @param startTimePoint start time for the partial set we examine
   * @param numTimePoints number of time points (including startTimePoint to examine)
   * @param returnLocals whether to return an array or local values, or else
   *  sums of these values
   * @return an array of sum of digamma(n_x+1) for each marginal x, then
   *  sum of n_x for each marginal x (these latter ones are for debugging purposes).
   * @throws Exception
   */
  protected abstract double[] partialComputeFromObservations(
      int startTimePoint, int numTimePoints, boolean returnLocals) throws Exception;

  /**
   * Private class to handle multi-threading of the Kraskov algorithms.
   * Each instance calls partialComputeFromObservations()
   * to compute nearest neighbours for a part of the data.
   * 
   * 
   * @author Joseph Lizier (<a href="joseph.lizier at gmail.com">email</a>,
   * <a href="http://lizier.me/joseph/">www</a>)
   * @author Ipek Özdemir
   */
  private class KraskovThreadRunner implements Runnable {
    protected MultiVariateInfoMeasureCalculatorKraskov calc;
    protected int myStartTimePoint;
    protected int numberOfTimePoints;
    protected boolean computeLocals;
    
    protected double[] returnValues = null;
    protected Exception problem = null;
    
    public static final int INDEX_SUM_DIGAMMAS = 0;

    public KraskovThreadRunner(
        MultiVariateInfoMeasureCalculatorKraskov calc,
        int myStartTimePoint, int numberOfTimePoints,
        boolean computeLocals) {
      this.calc = calc;
      this.myStartTimePoint = myStartTimePoint;
      this.numberOfTimePoints = numberOfTimePoints;
      this.computeLocals = computeLocals;
    }
    
    /**
     * Return the values from this part of the data,
     *  or throw any exception that was encountered by the 
     *  thread.
     * 
     * @return an exception previously encountered by this thread.
     * @throws Exception
     */
    public double[] getReturnValues() throws Exception {
      if (problem != null) {
        throw problem;
      }
      return returnValues;
    }
    
    /**
     * Start the thread for the given parameters
     */
    public void run() {
      try {
        returnValues = calc.partialComputeFromObservations(
            myStartTimePoint, numberOfTimePoints, computeLocals);
      } catch (Exception e) {
        // Store the exception for later retrieval
        problem = e;
        return;
      }
    }
  }
  // end class KraskovThreadRunner

}
