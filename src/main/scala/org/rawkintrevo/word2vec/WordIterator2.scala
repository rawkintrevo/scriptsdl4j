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
import java.util.{NoSuchElementException, Random, Collections}
import scala.collection.JavaConversions._
import scala.util.parsing.json._
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
  * @param rng            Random Number Generator

  */
@throws[IOException]
class WordIterator2(val textFilePath: String,
                    val batchSize: Int,
                    val truncateLength: Int,
                    val rng: Random) extends DataSetIterator {

  var vectorSize: Int = 300
  var cursor: Int = 0
  val t: TokenizerFactory = new DefaultTokenizerFactory
  t.setTokenPreProcessor(new CommonPreprocessor)

  val miniBatchOffsets: util.LinkedList[Integer] = new util.LinkedList[Integer]

  initOffsets()

  override def next(num: Int): DataSet = {
    try
      nextDataSet(num)
    catch {
      case e: IOException =>
        throw new RuntimeException(e)
    }
  }

  var maxLength = 0

  var word2idxMap : Map[String, Int] = _
  var idx2wordMap : Map[Int, String] = _

  def loadOutputVocab(): Unit = {
    val textFileEncoding = Charset.forName("UTF-8")
    var lines: util.List[String] = Files.readAllLines(new File(textFilePath).toPath, textFileEncoding)
    val allTokens = scala.collection.mutable.ArrayBuffer.empty[String]
    for (s <- lines) { // this is just to prove that small batches work (then rewire to create random small batches)
      val tokens = t.create(s).getTokens
      for (t <- tokens) allTokens += t
    }
    val allUniqueTokens = allTokens.distinct.toArray
    word2idxMap = allUniqueTokens.indices.map(i => (allUniqueTokens(i), i)).toMap[String, Int]
    idx2wordMap = allUniqueTokens.indices.map(i => (i, allUniqueTokens(i))).toMap[Int, String]
  }

  loadOutputVocab()

  @throws[IOException]
  private def nextDataSet(num: Int) = {

    val textFileEncoding = Charset.forName("UTF-8")
    val startIdx = miniBatchOffsets.removeFirst()
    var lines: util.List[String] = Files.readAllLines(new File(textFilePath).toPath, textFileEncoding)

    val currMinibatchSize: Int = Math.min(batchSize, lines.length - startIdx)
    lines = lines.slice(startIdx, startIdx + currMinibatchSize)
    //Second: tokenize reviews and filter out unknown words
    val allTokens = new util.ArrayList[util.List[String]](lines.size)
    for (s <- lines) { // this is just to prove that small batches work (then rewire to create random small batches)
      val tokens = t.create(s).getTokens
      val tokensFiltered = new util.ArrayList[String]
      for (t <- tokens) tokensFiltered.add(t)
      allTokens.add(tokensFiltered)
      maxLength = Math.max(maxLength, tokensFiltered.size)
    }
    //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
    maxLength = math.min(maxLength, truncateLength)

    //Create data for training

    var i = 0
    val features = Nd4j.create(Array[Int](lines.size, vectorSize, maxLength), 'f')
    val labels = Nd4j.create(Array[Int](lines.size, word2idxMap.size, maxLength), 'f')


    val featuresMask = Nd4j.zeros(lines.size, maxLength)
    val labelsMask = Nd4j.zeros(lines.size, maxLength)

    while ( { i < lines.size }) {

      val tokens = allTokens.get(i)
      if (tokens.length > 1) {
        // Get the truncated sequence length of document (i)
        val seqLength = Math.min(tokens.size, maxLength)
        // Get all wordvectors for the current document and transpose them to fit the 2nd and 3rd feature shape
        val vectors = Nd4j.zeros(seqLength, inputColumns)
        for (j <- 0 until seqLength) {
//          println(s"token: ${tokens(j)}")
          vectors.put(Array[INDArrayIndex](NDArrayIndex.point(j)), word2vec(tokens(j)))
//          println(s"vec2word: ${vec2word(vectors.get(NDArrayIndex.point(j)))}")
        }


//        for (j <- 0 until seqLength -1 ){
//          println(s"vec2word label $j : ${vec2word(labs.get(NDArrayIndex.point(j)))}")
//
//        }

        // Put wordvectors into features array at the following indices:
        // 1) Document (i)
        // 2) All vector elements which is equal to NDArrayIndex.interval(0, vectorSize)
        // 3) All elements between 0 and the length of the current sequence
//        println(s"vectors.shape: ${vectors.shape.mkString(",")}")
//        println(s"features.shape: ${features.shape.mkString(",")}")
        features.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.all, NDArrayIndex.interval(0, seqLength)), vectors.transpose())
        // Assign "1" to each position where a feature is present, that is, in the interval of [0, seqLength)
        featuresMask.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.interval(0, seqLength)), 1)
//        println(s"featureMask: ${featuresMask.get(NDArrayIndex.point(i))}")

        for (t <- 0 until seqLength){

          val wordIdx = word2idxMap(tokens(t))
          labels.putScalar(Array[Int](i, wordIdx, t), 1.0)
        }
//        labels.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.all, NDArrayIndex.interval(0, seqLength - 1)), labs.transpose())
//        for (j <- 0 until seqLength -1 ){
//          val featureVec = features.get(NDArrayIndex.point(i), NDArrayIndex.all, NDArrayIndex.point(j))
//          println(s"feature shape: ${featureVec.shape().mkString(",")}")
//          println(s"vec2word feature2 $j : ${vec2word(featureVec)}")
//          println(s"featureVec input: ${labs.get(NDArrayIndex.point(j)).toString}")
//          println(s"featureVec recovered: ${featureVec.toString}")
//        }
        labelsMask.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.interval(0, seqLength - 1)), 1)
      }

      i += 1
    }
    new DataSet(features, labels, featuresMask, labelsMask)
  }

  override def totalExamples: Int ={
    val textFileEncoding = Charset.forName("UTF-8")
    val lines: util.List[String] = Files.readAllLines(new File(textFilePath).toPath, textFileEncoding)
    lines.size()
  }

  override def inputColumns: Int = vectorSize

  override def totalOutcomes: Int = idx2wordMap.size

  override def reset(): Unit = {
    initOffsets()
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


  /*********************************************************************************************************************
    * Custom Shit Starts here
    * **************************************************************************************************************/

  def initOffsets(): Unit = {
    for (i <- 0 until totalExamples by batchSize) {
      miniBatchOffsets.add(i)
    }
    Collections.shuffle(miniBatchOffsets, rng)
  }

  def word2vec(word: String): INDArray = {
    if (word.trim() == ""){
      return Nd4j.zeros(300)
    }
    val url = s"http://localhost:9001/word2vec/${word.trim()}"
    val result = scala.io.Source.fromURL(url).mkString
    val parsed : Option[Any] = JSON.parseFull(result)
    parsed match {
      case Some(m: Map[String, List[Double]]) => {
        if (m("foundResult")(0) == 1) return Nd4j.create(m("vec").toArray)
        else {
//          println(s"Word '$word' doesn't exist")
          return Nd4j.zeros(300)
        }
      }
      case _ => {
        println("this is wierd...")
        Nd4j.zeros(300)
      }
    }
  }

  def vec2word(vec: INDArray): String = {
    var vec2: INDArray = Nd4j.zeros(300)
    if (vec.shape()(0) == 300) {
      vec2 = vec
    } else {
      vec2 = vec.transpose()
    }
    val myArray = new Array[Double](300)
    for (i <- 0 until 300) {
      myArray(i) = vec2.getDouble(i,0)
    }

    val params = (0 until 300).map(i => s"a$i=${myArray(i)}").mkString("&")
    val url = s"http://localhost:9001/vec2word?$params"
    val result = scala.io.Source.fromURL(url).mkString
    val parsed : Option[Any] = JSON.parseFull(result)
    parsed match {
      case Some(m: Map[String, String]) => m("word")
      case _ =>  ""
    }
  }
  /*

    * @return
    */
  def getRandomWord: String = {
    val randArray = Seq.fill(300)(rng.nextFloat)
    val params = (0 until 300).map(i => s"a$i=${randArray(i)}").mkString("&")
    val url = s"http://localhost:9001/vec2word?$params"
    val result = scala.io.Source.fromURL(url).mkString
    val parsed : Option[Any] = JSON.parseFull(result)
    parsed match {
      case Some(m: Map[String, String]) => m("word")
      case None =>  ""
    }
  }
}
