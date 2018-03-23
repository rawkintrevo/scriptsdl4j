package org.rawkintrevo.word2vec

import java.io.{File, IOException}
import java.net.URL
import java.nio.charset.Charset
import java.nio.file.Files
import java.util
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
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import scala.collection.JavaConverters._

import scala.collection.mutable

// For UI Server
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import scala.collection.JavaConverters._


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

  val rng = new Random(12345)


  @throws[Exception]
  def main(args: Array[String]) {
    val lstmLayerSize = 2048    //Number of units in each LSTM layer
    val miniBatchSize = 32
    val exampleLength = 12       // minimum length of a "sentance"
    val tbpttLength = 4         //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    val numEpochs = 1000        //Total number of training epochs
    val generateSamplesEveryNMinibatches = 5   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
    val nSamplesToGenerate = 1                  //Number of samples to generate after each training epoch
    val nWordsToSample = 8               //Length of each sample to generate
//    val generationActors = Array("LAFORGE", "DATA", "TROI", "RIKER")
//    val generationTopics: Array[Float] = new Array[Float](60)
//    generationTopics(19) = 0.51.toFloat // Romulans
//    generationTopics(23) = 0.49.toFloat // Battle
    val generationInitialization = "captain"         //Optional character initialization; a random word is used if null
    val learningRate = 0.001
    // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.



    val uiServer = UIServer.getInstance

    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
    val statsStorage = new InMemoryStatsStorage //Alternative: new FileStatsStorage(File), for saving and loading later

    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage)

    //Get a DataSetIterator that handles vectorization of text into something we can use to train
    // our GravesLSTM network.
    println("loading iterator")
    val iter = getIterator(miniBatchSize, exampleLength)
    val nOut = iter.totalOutcomes
    println("loaded " + nOut + " items, building network")
    println(iter.word2idxMap.size)
    println(iter.idx2wordMap.size)
    val vecTest = iter.word2vec("Starfleet")
    println(s"vecTest shape: ${vecTest.shape().mkString(",")}")
    val vecTest2 = iter.vec2word(vecTest)
    println(s"found word: $vecTest2")

    //Set up network configuration:
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      //      .updater(new RmsProp.Builder().learningRate(0.1).build())
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(learningRate)
      .seed(12345)
      .regularization(true)
      .l2(0.01)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.ADAM)
      .list()
      .layer(0, new LSTM.Builder().nIn(iter.inputColumns).nOut(lstmLayerSize)  // GravesLSTM doesn't support CuDNN - for gpus should use just lstm
        .activation(Activation.TANH).build())                                   // see https://deeplearning4j.org/quickref
      .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
        .activation(Activation.TANH).build())
////      .layer(2, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
//        .activation(Activation.TANH).build())
      .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)
        .nIn(lstmLayerSize).nOut(nOut).build())
      .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
      .pretrain(false).backprop(true)
      .build()


    val net = new MultiLayerNetwork(conf)
    net.setListeners(new ScoreIterationListener(1))

    net.init()

    net.setListeners(new StatsListener(statsStorage))
    // surf to http://localhost:9000/train



    val layers = net.getLayers
    var totalNumParams = 0
    for (i <- layers.indices) {
      val nParams = layers(i).numParams
      println("Number of parameters in layer " + i + ": " + nParams)
      totalNumParams += nParams
    }
    println("Total number of network parameters: " + totalNumParams)

    //Do training, and then generate and print samples from network


    for (i <- 0 until numEpochs) {
      var miniBatchNumber = 0
      while (iter.hasNext) {
        val ds = iter.next

        val features = ds.getFeatures
        val labels = ds.getLabels
        //        println(s"features.shape: ${features.shape().mkString(",")}")
        //        println(s"labels.shape: ${labels.shape().mkString(",")}")
        //        //        for (i <- 0 until features.shape()(0)){
        //        //          for (j <- 0 until features.shape()(1)) {
        //        //            val featureVec = features.get(NDArrayIndex.point(i), NDArrayIndex.all, NDArrayIndex.point(j))
        //        //            val labelVec = features.get(NDArrayIndex.point(i), NDArrayIndex.all, NDArrayIndex.point(j))
        //        //            println(s"vec2word feature $j : ${iter.vec2word(featureVec)}")
        //        //            println(s"vec2word label   $j : ${iter.vec2word(labelVec)}")
        //        //          }
        //        //        }
        net.fit(ds)

        if (miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
          println(s"miniBatchNumber : $miniBatchNumber")
          println("--------------------")
          println("Sampling characters from network given initialization \"" +
            (if (Option(generationInitialization).isEmpty) "" else generationInitialization) + "\"")
          val samples: Array[String] = sampleWordsFromNetwork(generationInitialization,
            net, iter, rng, nWordsToSample, nSamplesToGenerate)
          for (j <- samples.indices) {
            println("----- Sample " + j + " -----")
            println(samples(j))
            println()
          }
          println("Completed mini-batch " + miniBatchNumber + " samples of size " + iter.totalExamples + "x" + exampleLength + " words")
        }
        miniBatchNumber += 1
      }
      println(s"Completed Epoch ${i}")
      iter.reset() //Reset iterator for another epoch
    }
    println("\n\nExample complete")
  }

  val fileLocation = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/corpus.txt"
  /** Downloads Shakespeare training data and stores it locally (temp directory). Then set up and return a simple
    * DataSetIterator that does vectorization based on the text.
    *
    * @param miniBatchSize  Number of text segments in each training mini-batch
    * @param sequenceLength Number of characters in each text segment.
    */
  @throws[Exception]
  def getIterator(miniBatchSize: Int, sequenceLength: Int): WordIterator2 = {

    val f = new File(fileLocation)
    if (!f.exists) throw new IOException("File does not exist: " + fileLocation) //Download problem?
//    val validCharacters: Array[Char] = CharFileIterator.getDefaultCharacterSet //Which characters are allowed? Others will be removed
    new WordIterator2(fileLocation, miniBatchSize, sequenceLength, rng)
  }


//  private def sampleWordsFromNetwork(net: MultiLayerNetwork,
//                                     rng: Random
//                                     ) = {
//
//    net.
//
//  }
  /** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
    * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
    * Note that the initalization is used for all samples
    *
    * @param _initialization     String, may be null. If null, select a random character as initialization for all samples
    * @param wordToSample Number of characters to sample from network (excluding initialization)
    * @param net                MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
    * @param iter               CharacterIterator. Used for going from indexes back to characters
    */
  private def sampleWordsFromNetwork(_initialization: String,
//                                          _initActors: Array[String],
//                                          _initTopics: Array[Float],
                                          net: MultiLayerNetwork,
                                          iter: WordIterator2,
                                          rng: Random,
                                          wordToSample: Int,
                                          numSamples: Int): Array[String] = {

    //Set up initialization. If no initialization: use a random character
    val initialization: String = if (_initialization == null) {
      String.valueOf(iter.getRandomWord)
    } else _initialization
    //Create input for initialization
    val tokens = iter.t.create(_initialization).getTokens.asScala
    val initializationInput = Nd4j.create(Array[Int](1, iter.vectorSize, tokens.length), 'f')
    val vectors = Nd4j.zeros(tokens.length, iter.inputColumns)
    for (j <- tokens.indices) {

      vectors.putRow(1, iter.word2vec(tokens(j)))
    }


    // Put wordvectors into features array at the following indices:
    // 1) Document (i)
    // 2) All vector elements which is equal to NDArrayIndex.interval(0, vectorSize)
    // 3) All elements between 0 and the length of the current sequence
    initializationInput.put(Array[INDArrayIndex](NDArrayIndex.point(0), NDArrayIndex.all, NDArrayIndex.interval(0, tokens.length)), vectors)

    println(s"recovered word: ${iter.vec2word(initializationInput.getRow(0))}")

//
//    //Sample from network (and feed samples back into input) one character at a time (for all samples)
//    //Sampling is done in parallel here
    net.rnnClearPreviousState()
    var output: INDArray = net.rnnTimeStep(iter.word2vec(_initialization))//initializationInput)

    //output = output.tensorAlongDimension(output.size(2) - 1, 1, 0) //Gets the last time step output

    val outString = new Array[String](wordToSample)

    for (i <- 0 until wordToSample) {
      val outputProbDistribution = new Array[Double](iter.totalOutcomes)

      for (j <- outputProbDistribution.indices) {
        outputProbDistribution(j) = output.getDouble(0, j)
      }
      val sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng)
      outString(i) = iter.idx2wordMap(sampledCharacterIdx) //iter.vec2word(output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(0)))
      output = iter.word2vec(iter.idx2wordMap(sampledCharacterIdx))
      output = net.rnnTimeStep(output) //Do one time step of forward pass
    }

    Array(outString.mkString(" "))
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