package org.rawkintrevo.word2vec

import org.apache.commons.io.FileUtils
import org.apache.commons.io.FilenameUtils
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import java.io.File
import java.io.IOException
import java.nio.charset.Charset
import java.nio.file.Files
import java.util
import java.util.{NoSuchElementException, Random}
import scala.collection.JavaConversions._
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer

import scala.collection.JavaConversions._

/** This is a DataSetIterator that is specialized for the IMDB review dataset used in the Word2VecSentimentRNN example
  * It takes either the train or test set data from this data set, plus a WordVectors object (typically the Google News
  * 300 pretrained vectors from https://code.google.com/p/word2vec/) and generates training data sets.<br>
  * Inputs/features: variable-length time series, where each word (with unknown words removed) is represented by
  * its Word2Vec vector representation.<br>
  * Labels/target: a single class (negative or positive), predicted at the final time step (word) of each review
  *
  * @author Alex Black
  */


/**
  * @param textFilePath  the directory of the IMDB review data set

  * @param batchSize      Size of each minibatch for training
  * @param truncateLength If reviews exceed

  */
@throws[IOException]
class WordIterator2(val textFilePath: String,
                    val batchSize: Int,
                    val truncateLength: Int,
                    val rng: Random) extends DataSetIterator {

    var vectorSize: Int = _
//  val p = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (if (train) "train"
//  else "test") + "/pos/") + "/")
//  val n = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (if (train) "train"
//  else "test") + "/neg/") + "/")
//  positiveFiles = p.listFiles
//  negativeFiles = n.listFiles
  var cursor: Int = 0
  val t: TokenizerFactory = new DefaultTokenizerFactory
  t.setTokenPreProcessor(new CommonPreprocessor)

  val miniBatchOffsets: util.LinkedList[Integer] = new util.LinkedList[Integer]

  val wordVectors: WordVectors = {
    println("loading word2vec model")
    val gModel = new File("/home/rawkintrevo/gits/scriptsdl4j/data/GoogleNews-vectors-negative300.bin.gz")
//    val w2v = WordVectorSerializer.loadStaticModel(gModel)
    val w2v = WordVectorSerializer.readWord2VecModel(gModel)
    println("word2vec model loaded")

    vectorSize = w2v.getWordVector(w2v.vocab.wordAtIndex(0)).length
    w2v
  }

  val word2vecModel = wordVectors
  override def next(num: Int): DataSet = {
//    if (cursor >= positiveFiles.length + negativeFiles.length) throw new NoSuchElementException
    try
      nextDataSet(num)
    catch {
      case e: IOException =>
        throw new RuntimeException(e)
    }
  }
  var maxLength = 0

  @throws[IOException]
  private def nextDataSet(num: Int) = {
    val currMinibatchSize: Int = Math.min(num, miniBatchOffsets.size)
    val textFileEncoding = Charset.forName("UTF-8")
    //First: load reviews to String. Alternate positive and negative reviews
    println("reading lines")
    val lines: util.List[String] = Files.readAllLines(new File(textFilePath).toPath, textFileEncoding)
    //Second: tokenize reviews and filter out unknown words
    val allTokens = new util.ArrayList[util.List[String]](lines.size)

    println("lines read, creating tokens")

    for (s <- lines) { // this is just to prove that small batches work (then rewire to create random small batches)
      val tokens = t.create(s).getTokens
      val tokensFiltered = new util.ArrayList[String]

      for (t <- tokens) {
        if (wordVectors.hasWord(t)) tokensFiltered.add(t)
      }
      allTokens.add(tokensFiltered)
      maxLength = Math.max(maxLength, tokensFiltered.size)
    }
    //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
    maxLength = math.min(maxLength, truncateLength)
    println(s"tokens created")
    //Create data for training
    //Here: we have reviews.size() examples of varying lengths
    val features = Nd4j.create(Array[Int](lines.size, vectorSize, maxLength), 'f')
    val labels = Nd4j.create(Array[Int](lines.size, vectorSize, maxLength), 'f')
    //Two labels: positive or negative
    //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
    //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
    val featuresMask = Nd4j.zeros(lines.size, maxLength)
    val labelsMask = Nd4j.zeros(lines.size, maxLength)
    var i = 0
    println("starting while loop")
    while ( { i < 2}) {//lines.size }) {
      println(i)

      val tokens = allTokens.get(i)
      if (tokens.length > 1) {
        // Get the truncated sequence length of document (i)
        val seqLength = Math.min(tokens.size, maxLength)
        // Get all wordvectors for the current document and transpose them to fit the 2nd and 3rd feature shape
        val vectors: INDArray = wordVectors.getWordVectors(tokens.subList(0, seqLength)).transpose
        // Put wordvectors into features array at the following indices:
        // 1) Document (i)
        // 2) All vector elements which is equal to NDArrayIndex.interval(0, vectorSize)
        // 3) All elements between 0 and the length of the current sequence
        features.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.all, NDArrayIndex.interval(0, seqLength)), vectors)

        // Assign "1" to each position where a feature is present, that is, in the interval of [0, seqLength)
        featuresMask.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.interval(0, seqLength)), 1)

        val labs = wordVectors.getWordVectors(tokens.subList(1, seqLength)).transpose
        labels.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.all, NDArrayIndex.interval(0, seqLength - 1)), labs)

        labelsMask.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.interval(0, seqLength - 1)), 1)
      }

      i += 1
    }
    new DataSet(features, labels, featuresMask, labelsMask)
  }

  override def totalExamples: Int ={
    val textFileEncoding = Charset.forName("UTF-8")
    //First: load reviews to String. Alternate positive and negative reviews
    val reviews: util.List[String] = Files.readAllLines(new File(textFilePath).toPath, textFileEncoding)
    reviews.size()
  }

  override def inputColumns: Int = vectorSize

  override def totalOutcomes: Int = vectorSize

  override def reset(): Unit = {
    cursor = 0
  }

  override def resetSupported = true

  override def asyncSupported = true

  override def batch: Int = batchSize


  override def numExamples: Int = totalExamples

  override def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = {
    throw new UnsupportedOperationException
  }

  override def getLabels: util.List[String] = util.Arrays.asList("positive", "negative")

  override def hasNext: Boolean = miniBatchOffsets.size > 0

  override def next: DataSet = next(batchSize)

  override def remove(): Unit = {
  }

  override def getPreProcessor = throw new UnsupportedOperationException("Not implemented")

  /** Convenience method for loading review to String */
//  @throws[IOException]
//  def loadReviewToString(index: Int): String = {
//    "foo"
//  }

  /**
    * Used post training to convert a String to a features INDArray that can be passed to the network output method
    *
    * @param reviewContents Contents of the review to vectorize
    * @param maxLength      Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not nruncate
    * @return Features array for the given input String
    */
  def loadFeaturesFromString(reviewContents: String, maxLength: Int): INDArray = {
    val tokens = t.create(reviewContents).getTokens
    val tokensFiltered = new util.ArrayList[String]
    import scala.collection.JavaConversions._
    for (t <- tokens) {
      if (wordVectors.hasWord(t)) tokensFiltered.add(t)
    }
    val outputLength = Math.max(maxLength, tokensFiltered.size)
    val features = Nd4j.create(1, vectorSize, outputLength)
    var j = 0
    while ( {
      j < tokens.size && j < maxLength
    }) {
      val token = tokens.get(j)
      val vector = wordVectors.getWordVectorMatrix(token)
      features.put(Array[INDArrayIndex](NDArrayIndex.point(0), NDArrayIndex.all, NDArrayIndex.point(j)), vector)

      {
        j += 1; j - 1
      }
    }
    features
  }

  def convertINDArrayToWord(v: INDArray): String =
    word2vecModel.wordsNearest(v, 1).iterator().next()

  def getRandomWord: String = {
    val vocab =word2vecModel.vocab()
    vocab.wordAtIndex(rng.nextInt(vocab.numWords()))
  }
}
