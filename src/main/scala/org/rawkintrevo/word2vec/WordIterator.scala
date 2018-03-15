package org.rawkintrevo.word2vec


import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import java.io.{File, IOException}
import java.nio.charset.Charset
import java.nio.file.Files
import java.util
import java.util.{Collections, Random}

import org.deeplearning4j.models.word2vec.VocabWord
import org.deeplearning4j.models.word2vec.wordstore.VocabCache
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
import org.deeplearning4j.text.sentenceiterator.SentenceIterator
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.nd4j.linalg.api.ndarray.INDArray

import scala.collection.JavaConverters._

/** A simple DataSetIterator for use in the GravesLSTMCharModellingExample.
  * Given a text file and a few options, generate feature vectors and labels for training,
  * where we want to predict the next character in the sequence.<br>
  * This is done by randomly choosing a position in the text file, at offsets of 0, exampleLength, 2*exampleLength, etc
  * to start each sequence. Then we convert each character to an index, i.e., a one-hot vector.
  * Then the character 'a' becomes [1,0,0,0,...], 'b' becomes [0,1,0,0,...], etc
  *
  * Feature vectors and labels are both one-hot vectors of same length
  *
  * The key difference between this and the original, is that this creates one dataset for each of the files in a directory
  *
  * @author Trevor Grant's hack of Alex Black's work
  *
  */
object WordIterator {
  /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
  def getMinimalCharacterSet: Array[Char] = (
    ('a' to 'z') ++ ('A' to 'Z') ++ ('0' until '9') ++
      Array('!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t')
    ).toArray

  /** As per getMinimalCharacterSet(), but with a few extra characters */
  def getDefaultCharacterSet: Array[Char] = {
    getMinimalCharacterSet ++
      Array('@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_', '\\', '|', '<', '>')
  }


}


/**
  * @param textFilePath     Path to text file to use for generating samples
  * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
  * @param miniBatchSize    Number of examples per mini-batch
  * @param exampleLength    Number of characters in each input/output vector
  * @param rng              Random number generator, for repeatability if required
  * @throws IOException If text file cannot  be loaded
  */
@throws[IOException]
class WordIterator(
                        val textFilePath: String,
                        val textFileEncoding: Charset,
                        var miniBatchSize: Int, //Size of each minibatch (number of examples)
                        var exampleLength: Int, //Length of each example/minibatch (number of characters)
//                        var validCharacters: Array[Char], //Valid characters
                        var rng: Random
                      ) extends DataSetIterator {

  if (!new File(textFilePath).exists)
    throw new IOException("Could not access file (does not exist): " + textFilePath)

  if (miniBatchSize <= 0)
    throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)")

  import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
  import org.deeplearning4j.models.word2vec.Word2Vec

  val word2vecModel: Word2Vec = {
    println("loading word2vec model")
    val gModel = new File("/home/rawkintrevo/gits/scriptsdl4j/data/GoogleNews-vectors-negative300.bin.gz")
    val w2v = WordVectorSerializer.readWord2VecModel(gModel)
    println("word2vec model loaded")
    w2v
  }



  def getRandomWord(): String = {
    val vocab =word2vecModel.getVocab
    vocab.wordAtIndex(rng.nextInt(vocab.numWords()))
  }

  val t: TokenizerFactory = new DefaultTokenizerFactory
  //Offsets for which script in allChars to be used in minibatch
  private val exampleStartOffsets: util.LinkedList[Integer] = new util.LinkedList[Integer]

  //  initializeOffsets()

  def loadWordVecsFromFile(filename: String): Array[Array[Array[Double]]] = {
    // Line, Word, Vector
    val lineIter = new BasicLineIterator(filename)
    println(s"attempting to load wordVecSeqs")
    t.setTokenPreProcessor(new CommonPreprocessor())
    val textFilePath = filename
    val lines: util.List[String] = Files.readAllLines(new File(textFilePath).toPath, textFileEncoding)
    val maxSize: Int  = lines.size + lines.asScala.map(_.length - exampleLength).sum //add lines.size() to account for newline characters at end of each line
    val wordVectorSeqs = Array.ofDim[Array[Double]](maxSize, exampleLength)

    var lineIdx: Int = 0

    while (lineIter.hasNext) {
      val thisLine = lineIter.nextSentence()
      val theseTokens = t.create(thisLine)
      var wordIdx = 0
      val allTokenArray: Array[String] = theseTokens.getTokens.toArray().map(_.toString)
      val tokenArray = allTokenArray.map(w => if (word2vecModel.hasWord(w)) w )
      if (tokenArray.length > exampleLength) {
        for (i <- 0 until math.max(tokenArray.length - exampleLength, 1)) {
          for (thisToken <- tokenArray.slice(i, i + exampleLength)){
            val thisVector: Array[Double] = word2vecModel.getWordVector(thisToken.toString)
            wordVectorSeqs(lineIdx)(wordIdx) = thisVector
            wordIdx += 1
          }
          lineIdx += 1
        }
      }
    }
    println(s"loaded ${wordVectorSeqs.length} wordVecSeqs")
    wordVectorSeqs
  }

  val wordVecSeqs: Array[Array[Array[Double]]] = {
    println("attempting to load wordvecs")
    loadWordVecsFromFile(textFilePath)
  }

//
//  def loadAllFileChars(directory: String): Array[Array[Char]] = {
//    def getListOfFiles(dir: String):List[String] = {
//      val d = new File(dir)
//      if (d.exists && d.isDirectory) {
//        d.listFiles.filter(_.isFile).toList.map(_.toString)
//      } else {
//        List[String]()
//      }
//    }
//
//    val allFiles = getListOfFiles(directory)
//    val allChars = new Array[Array[Char]](allFiles.length)
//    for (i <- 0 until allFiles.length) {
//      allChars(i) = loadCharsFromFile(allFiles(i))
//    }
//    allChars
//  }
//
//  private val allChars: Array[Array[Char]] = loadAllFileChars(textFilePath)

  initializeOffsets()

//  def convertIndexToCharacter(idx: Int): Char =
//    validCharacters(idx)
//
    def convertINDArrayToWord(v: INDArray): String =
      word2vecModel.wordsNearest(v, 1).iterator().next()

//  def convertCharacterToIndex(c: Char): Int =
//    charToIdxMap(c)
//
//  def getRandomCharacter: Char =
//    validCharacters((rng.nextDouble * validCharacters.length).toInt)

  def hasNext: Boolean =
    exampleStartOffsets.size > 0
  //    cursorPos < allChars.length

  def next: DataSet =
    next(miniBatchSize)

//  val allActorsTextFilePath = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/tng-all-chars.txt"
//  val validActors: Array[String] = Files.readAllLines(new File(allActorsTextFilePath).toPath, textFileEncoding)
//    .toArray()
//    .map(_.toString)
//
//
//  val actorToIdxMap: Map[String, Int] =
//    validActors.indices.map({ i => (validActors(i), i)}).toMap
//
//  val nTopics: Int = Files.readAllLines(new File("/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/topics.txt").toPath, textFileEncoding).size()


  def next(num: Int): DataSet = {
    if (exampleStartOffsets.size == 0) throw new NoSuchElementException
    val currMinibatchSize: Int = Math.min(num, exampleStartOffsets.size)
    //    print("Current Minibatch Size: " + currMinibatchSize + "\n")
    //Allocate space:
    //Note the order here:
    // dimension 0 = number of examples in minibatch
    // dimension 1 = size of each vector (i.e., number of characters)
    // dimension 2 = length of each time series/example
    //Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
    val dim1 = wordVecSeqs(0)(0).length
    val input = Nd4j.create(Array[Int](currMinibatchSize, dim1, exampleLength), 'f')
    val labels = Nd4j.create(Array[Int](currMinibatchSize, dim1, exampleLength), 'f')

//    val input = Nd4j.create(Array[Int](currMinibatchSize, validCharacters.length + validActors.length + nTopics, exampleLength), 'f')
//    val labels = Nd4j.create(Array[Int](currMinibatchSize, validCharacters.length, exampleLength), 'f')

    for (i <- 0 until currMinibatchSize) {
      val startIdx = exampleStartOffsets.removeFirst()

      // Get the actors for this scene (artifact from charIterator Days)
//      val sceneLines = allChars(startIdx).mkString("").split("\n")
//      val actors = sceneLines(0).split(" ")
//      val topics = sceneLines(1).split(" ").map(_.toFloat)
//      val headerLines = 2
//      val scriptData: Array[Char] = sceneLines.slice(headerLines, sceneLines.length).mkString("\n").toCharArray
//      var currCharIdx = charToIdxMap(scriptData(0))

      var currWordVec = Nd4j.create(wordVecSeqs(startIdx)(0))
      //Current input
//      var c: Int = 0
      for (j <- 1 until exampleLength) {
//        val nextCharIdx = charToIdxMap(scriptData(j)) //Next character to predict
        val nextWordVec = Nd4j.create(wordVecSeqs(startIdx)(j))
//        input.putScalar(Array[Int](i, currCharIdx, c), 1.0)

        input.put(Array[Int](i, j), currWordVec)
//        for (a <- actors) {
//          if (actorToIdxMap.contains(a)) {
//            input.putScalar(Array[Int](i, validCharacters.length + actorToIdxMap(a), c), 1.0)
//          } else {input.putScalar(Array[Int](i, validCharacters.length + actorToIdxMap("GUEST0"), c), 1.0) } // a lazy hack bc occasionally there is a "GUEST" or "P" or "T" and I can't figure out where this is coming from
//        }
//        for (t_i <- topics.indices) {
//          input.putScalar(Array[Int](i, validCharacters.length + validActors.length + t_i, c), topics(t_i))
//        }
//        labels.putScalar(Array[Int](i, nextCharIdx, c), 1.0)
        labels.put(Array[Int](i,j), nextWordVec)
//        currCharIdx = nextCharIdx
        currWordVec = nextWordVec
//        c += 1
      }
    }
    new DataSet(input, labels)
  }

  def totalExamples: Int =
//    allChars.length
    wordVecSeqs.length

  def inputColumns: Int =
//    validCharacters.length + validActors.length + nTopics
    wordVecSeqs(0)(0).length

  def totalOutcomes: Int =
//    validCharacters.length
    inputColumns

  def reset() {
    exampleStartOffsets.clear()
    initializeOffsets()
  }

  private def initializeOffsets(): Unit = {
    // Randomly Select Indexes from wordVecSeqs
    // This uses ALL examples for each epoch (which is def of an epoch)
    for (i <- 0 until wordVecSeqs.length){
      exampleStartOffsets.add(i)
    }
    Collections.shuffle(exampleStartOffsets, rng)
  }

  def resetSupported: Boolean =
    true

  def asyncSupported: Boolean =
    true

  def batch: Int =
    miniBatchSize

  def cursor: Int =
  //    cursorPos
    totalExamples - exampleStartOffsets.size

  def numExamples: Int =
    totalExamples

  def setPreProcessor(preProcessor: DataSetPreProcessor) {
    throw new UnsupportedOperationException("Not implemented")
  }

  def getPreProcessor: DataSetPreProcessor = {
    throw new UnsupportedOperationException("Not implemented")
  }

  def getLabels: util.List[String] = {
    throw new UnsupportedOperationException("Not implemented")
  }

  override def remove() {
    throw new UnsupportedOperationException
  }

}
