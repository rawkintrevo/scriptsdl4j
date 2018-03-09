package org.rawkintrevo.scenegen

import java.io.{File, IOException}
import java.net.URL
import java.nio.charset.Charset
import java.util.Random

import org.apache.commons.io.FileUtils
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.{RmsProp, Sgd}
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction


import scala.collection.mutable

// For UI Server
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage


/** GravesLSTM Character modelling example
  *
  * @author Alex Black
  *         *
  *         Example: Train a LSTM RNN to generates text, one character at a time.
  *         This example is somewhat inspired by Andrej Karpathy's blog post,
  *         "The Unreasonable Effectiveness of Recurrent Neural Networks"
  *         http://karpathy.github.io/2015/05/21/rnn-effectiveness/
  *         *
  *         This example is set up to train on the Complete Works of William Shakespeare, downloaded
  *         from Project Gutenberg. Training on other text sources should be relatively easy to implement.
  *         *
  *         For more details on RNNs in DL4J, see the following:
  *         http://deeplearning4j.org/usingrnns
  *         http://deeplearning4j.org/lstm
  *         http://deeplearning4j.org/recurrentnetwork
  *
  *         See updated version here https://github.com/deeplearning4j/dl4j-examples/pull/589/files
  */
object App {
  @throws[Exception]
  def main(args: Array[String]) {
    val lstmLayerSize = 1024                    //Number of units in each LSTM layer
    val miniBatchSize = 128 // 256, but keep increments of 32, maybe 64(2056) next  //Size of mini batch to use when  training
    val exampleLength = 1022 //65536/8  // ~8k  // and episode ~ 30k e.g. devide by 2   //Length of each training example sequence to use. This could certainly be increased
    val tbpttLength = 128                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    val numEpochs = 1000                           //Total number of training epochs
    val generateSamplesEveryNMinibatches = 1000   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
    val nSamplesToGenerate = 1                  //Number of samples to generate after each training epoch
    val nCharactersToSample = 1024               //Length of each sample to generate
    val generationActors = Array("LAFORGE", "DATA", "TROI", "RIKER")
    val generationInitialization = "Stardate"         //Optional character initialization; a random character is used if null
    val learningRate = 0.1
    // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
    // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
    val rng = new Random(12345)

    val uiServer = UIServer.getInstance

    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
    val statsStorage = new InMemoryStatsStorage //Alternative: new FileStatsStorage(File), for saving and loading later

    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage)

    //Then add the StatsListener to collect this information from the network, as it trains
    // replaced by WebUI

    //Get a DataSetIterator that handles vectorization of text into something we can use to train
    // our GravesLSTM network.
    val iter = getIterator(miniBatchSize, exampleLength)
    val nOut = iter.totalOutcomes
    println("loaded " + nOut + " items, building network")

    //Set up network configuration:
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      //      .updater(new RmsProp.Builder().learningRate(0.1).build())
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(learningRate)
      .seed(12345)
      .regularization(true)
      .l2(0.001)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.RMSPROP)
      .list()
      .layer(0, new LSTM.Builder().nIn(iter.inputColumns).nOut(lstmLayerSize)  // GravesLSTM doesn't support CuDNN - for gpus should use just lstm
        .activation(Activation.TANH).build())                                   // see https://deeplearning4j.org/quickref
      .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
        .activation(Activation.TANH).build())
//      .layer(2, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
//        .activation(Activation.TANH).build())
      .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)  //MCXENT + softmax for classification
        .nIn(lstmLayerSize).nOut(nOut).build())
      .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
      .pretrain(false).backprop(true)
      .build()

    println("configuration set...")
    val net = new MultiLayerNetwork(conf)
    println("network created, intializing")
    net.init()
    println("network initialized")
    net.setListeners(new StatsListener(statsStorage))
    // surf to http://localhost:9000/train

    println("StatsListener initialized")

    val layers = net.getLayers
    var totalNumParams = 0
    println("summary statistics on inputs:")
    println("input layer size: " + iter.inputColumns)
    for (i <- layers.indices) {
      val nParams = layers(i).numParams
      println("Number of parameters in layer " + i + ": " + nParams)
      totalNumParams += nParams
    }
    println("Total number of network parameters: " + totalNumParams)

    //Do training, and then generate and print samples from network
    var miniBatchNumber = 0
    var i = 0
    for (i <- 0 until numEpochs) {
      while (iter.hasNext) {
        val ds = iter.next
        net.fit(ds)
      }
        println("--------------------")
        println("Completed Epoch" + i + " samples of size " + iter.totalExamples + "x" + exampleLength + " characters")
        println("Sampling characters from network given initialization \"" +
          (if (Option(generationInitialization).isEmpty) "" else generationInitialization) + "\"")
        val samples = sampleCharactersFromNetwork(generationInitialization,
                                                  generationActors,
                                                  net, iter, rng, nCharactersToSample, nSamplesToGenerate)
        for (j <- samples.indices) {
          println("----- Sample " + j + " -----")
          println(samples(j))
          println()
        }
//      }
      iter.reset() //Reset iterator for another epoch
    }
    println("\n\nExample complete")
  }

  /** Downloads Shakespeare training data and stores it locally (temp directory). Then set up and return a simple
    * DataSetIterator that does vectorization based on the text.
    *
    * @param miniBatchSize  Number of text segments in each training mini-batch
    * @param sequenceLength Number of characters in each text segment.
    */
  @throws[Exception]
  def getIterator(miniBatchSize: Int, sequenceLength: Int): CharFileIterator = {

    val fileLocation = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scenes/star-trek-tng"
    val f = new File(fileLocation)
    if (!f.exists) throw new IOException("File does not exist: " + fileLocation) //Download problem?
    val validCharacters: Array[Char] = CharFileIterator.getDefaultCharacterSet //Which characters are allowed? Others will be removed
    new CharFileIterator(fileLocation, Charset.forName("UTF-8"), miniBatchSize, sequenceLength, validCharacters, new Random(12345))
  }

  /** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
    * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
    * Note that the initalization is used for all samples
    *
    * @param _initialization     String, may be null. If null, select a random character as initialization for all samples
    * @param charactersToSample Number of characters to sample from network (excluding initialization)
    * @param net                MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
    * @param iter               CharacterIterator. Used for going from indexes back to characters
    */
  private def sampleCharactersFromNetwork(_initialization: String,
                                          _initActors: Array[String],
                                          net: MultiLayerNetwork,
                                          iter: CharFileIterator,
                                          rng: Random,
                                          charactersToSample: Int,
                                          numSamples: Int): Array[String] = {
    //Set up initialization. If no initialization: use a random character
    val initialization = if (_initialization == null) {
      String.valueOf(iter.getRandomCharacter)
    } else _initialization
    //Create input for initialization
    val initializationInput = Nd4j.zeros(numSamples, iter.inputColumns, initialization.length)
    val init = initialization.toCharArray
    for (i <- init.indices) {
      val idx = iter.convertCharacterToIndex(init(i))
      for (j <- 0 until numSamples) {
        initializationInput.putScalar(Array[Int](j, idx, i), 1.0f)
        // Add Actors
        for (a <- _initActors) {
          initializationInput.putScalar(Array[Int](j, iter.validCharacters.length + iter.actorToIdxMap(a), i), 1.0f)
        }
      }
    }
    val sb = mutable.ArrayBuffer.empty[StringBuilder]
    (0 until numSamples).foreach { _ =>
      sb += new StringBuilder(initialization)
    }

    //Sample from network (and feed samples back into input) one character at a time (for all samples)
    //Sampling is done in parallel here
    net.rnnClearPreviousState()
    var output = net.rnnTimeStep(initializationInput)
    output = output.tensorAlongDimension(output.size(2) - 1, 1, 0) //Gets the last time step output

    for (i <- 0 until charactersToSample) {
      //Set up next input (single time step) by sampling from previous output
      val nextInput = Nd4j.zeros(numSamples, iter.inputColumns)
      //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
      for (s <- 0 until numSamples) {
        val outputProbDistribution = new Array[Double](iter.totalOutcomes)
        for (j <- outputProbDistribution.indices) {
          outputProbDistribution(j) = output.getDouble(s, j)
        }
        val sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng)
        //Prepare next time step input
        nextInput.putScalar(Array[Int](s, sampledCharacterIdx), 1.0f)
        // Add Actors
        for (a <- _initActors) {
          nextInput.putScalar(Array[Int](s, iter.validCharacters.length + iter.actorToIdxMap(a)), 1.0f)
        }
        sb(s).append(iter.convertIndexToCharacter(sampledCharacterIdx)) //Add sampled character to StringBuilder (human readable output)
      }
      output = net.rnnTimeStep(nextInput) //Do one time step of forward pass
    }
    val out = new Array[String](numSamples)
    for (i <- 0 until numSamples) {
      out(i) = sb(i).toString
    }
    out
  }

  /**
    * Given a probability distribution over discrete classes, sample from the distribution
    * and return the generated class index.
    *
    * @param distribution Probability distribution over classes. Must sum to 1.0
    */
  def sampleFromDistribution(distribution: Array[Double], rng: Random): Int = {
    val d = rng.nextDouble
    val i = distribution
      .toIterator
      .scanLeft(0.0)({ case (acc, p) => acc + p })
      .drop(1)
      .indexWhere(_ >= d)
    if (i >= 0) {
      i
    } else {
      //Should never happen if distribution is a valid probability distribution
      throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + distribution.sum)
    }
  }

}