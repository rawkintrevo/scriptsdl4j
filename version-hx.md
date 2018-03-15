

## V1

- [Install CUDA](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
-  Hack to get CUDA 9.1 (on ubuntu 17.04) to work with CUDA 8.0 (per DL4J)
```bash
 cd /usr/local/cuda-9.1/lib64
 sudo ln -s libcudart.so libcudart.so.8.0
 sudo ln -s libcublas.so libcublas.so.8.0
 sudo ln -s libcusparse.so libcusparse.so.8.0
 sudo ln -s libcusolver.so libcusolver.so.8.0
```
- Set Env Variables
```bash
export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64\
	${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

```

- `mvn clean package`
- `cd target`
- `java -jar scripts*with-dep*`

```bash
16:06:55.978 [main] INFO  o.d.o.l.ScoreIterationListener - Score at iteration 2619 is 3.6143330293180833
--------------------
Completed 0 minibatches of size 32x1000 characters
Sampling characters from network given initialization ""
----- Sample 0 -----
CARD: These will be your own words, your Honour. What do you expect Ferester be honour's statement the prisoner will not be harmed? Make here? 
ZORN OC: This is Zorn, Captain. 
PICARD: I see. A Captain's rank means nothing to you. 
RIKER: We have alled can we have a long mission aboard, repeat no sta

----- Sample 1 -----
CARD: But even when we wore costumes like that we'de that is alvaid. 
DATA: Inadvisable as you perimeser with the Captain? 
PICARD: I see your elcomet to Farpoint Station? Our now appears if the Enterprise, in the under
to Bandiccity

t's to plated in a shuttlecraft. 
Q: But you must still prove that

----- Sample 2 -----
CARD: But even when we wore costumes like commant it it the mosting the
Bridge) 
PICARD: You are an and was the mask is now. 
CRUSHER: (looking at that is places hilling choses on that vessel. I'm sonding
our away, sir. The fathed have a family. 
WESLEY: Just a look, at the Bridge. I'll stay in the p

----- Sample 3 -----
CARD: But even when we wore costumes like that we'd already started
to make rapid progress. 
Q: Oh ye, Captain? 
PICARD: I don't unders, your honourawally. Goly, sir. No I trugal tures our semoss it probaciin. 
Q: Yas the feeling I've to beam over? 
DATA: I spaner) 
DATA: We're right there, this is a
```

**Thoughts** It's starting to memorize the script. 

We need to 
1. Get full scripts in for training
2. Lengthen up Sample size
3. Get the outputs to be less
4. Run over night.

Code is in scriptgen_v1

# V2

1. Based on gitter chat- decided to stitch all episodes into one big file, with "TITLE: " breaking episodes up.
2. Update params
3. remove noisy output
4. push to text file


```
val lstmLayerSize = 200                     //Number of units in each GravesLSTM layer
val miniBatchSize = 64                      //Size of mini batch to use when  training
val exampleLength = 50000                    //Length of each training example sequence to use. This could certainly be increased
val tbpttLength = 50                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
val numEpochs = 100                           //Total number of training epochs
val generateSamplesEveryNMinibatches = 1000   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
val nSamplesToGenerate = 4                  //Number of samples to generate after each training epoch
val nCharactersToSample = 1000               //Length of each sample to generate
val generationInitialization = "TITLE:\nrawkintrevo's new Ford"
```
2333 MB of GPU Memory used- you've got some room to grow :D

#v3

```
val lstmLayerSize = 256                     //Number of units in each GravesLSTM layer
    val miniBatchSize = 64                      //Size of mini batch to use when  training
    val exampleLength = 65536                    //Length of each training example sequence to use. This could certainly be increased
    val tbpttLength = 64                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    val numEpochs = 100                           //Total number of training epochs
    val generateSamplesEveryNMinibatches = 1000   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
    val nSamplesToGenerate = 4                  //Number of samples to generate after each training epoch
    val nCharactersToSample = 1000               //Length of each sample to generate
    val generationInitialization = "TITLE:\nrawkintrevo's new Ford\n"
```

3503 MB of GPU Mem used (4200 later in training)

1. Added visualizer and using shaded jar now

# v4

1. on rec from Alex, decreased learning rate from .1 to .01 (want `Update:Parameter Ratios (Mean Magnitudes): log10` to be in range of -3)
2. increased iteratations from 1 to 100 (bc copying from/to GPU is costly)

```
val lstmLayerSize = 256/2                     //Number of units in each GravesLSTM layer
val miniBatchSize = 128                      //Size of mini batch to use when  training
val exampleLength = 65536/4                    //Length of each training example sequence to use. This could certainly be increased
val tbpttLength = 64                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
val numEpochs = 100                           //Total number of training epochs
val generateSamplesEveryNMinibatches = 500   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
val nSamplesToGenerate = 3                  //Number of samples to generate after each training epoch
val nCharactersToSample = 1000               //Length of each sample to generate
val generationInitialization = "TITLE:\nReturn of the Jedi\n"
```

2289 MB GPU Mem

Notes: thing I wanted at -3 is there. But no output (probably to high of iterations)
Learning is _very_ slow, (but speeding up)wish I could see output.

# v5

Turned iterations down to 10
added `+=1` to `miniBatchNumber`

#v6

- 4 layers

```
val lstmLayerSize = 256                    //Number of units in each GravesLSTM layer
val miniBatchSize = 32                      //Size of mini batch to use when  training
val exampleLength = 65536/16                    //Length of each training example sequence to use. This could certainly be increased
val tbpttLength = 64                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
val numEpochs = 100                           //Total number of training epochs
val generateSamplesEveryNMinibatches = 500   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
val nSamplesToGenerate = 3                  //Number of samples to generate after each training epoch
val nCharactersToSample = 1000               //Length of each sample to generate
val generationInitialization = "TITLE:\nReturn of the Jedi\n" 
```
 
1300MB Gpu

#v7

- Increased Backprop tt
- Iteratoins = 1

```
val lstmLayerSize = 256                    //Number of units in each GravesLSTM layer
val miniBatchSize = 32                      //Size of mini batch to use when  training
val exampleLength = 65536/16                    //Length of each training example sequence to use. This could certainly be increased
val tbpttLength = 256                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
val numEpochs = 100                           //Total number of training epochs
val generateSamplesEveryNMinibatches = 500   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
val nSamplesToGenerate = 1                  //Number of samples to generate after each training epoch
val nCharactersToSample = 3000               //Length of each sample to generate
val generationInitialization = "TITLE:\nReturn of the Jedi\n" 
```

1400 MB GPU

#v8

- Introduced learning rate schedule (0.01, until 5000 then 0.001)
- greatly increased mini batchsize

```
val lstmLayerSize = 256                    //Number of units in each GravesLSTM layer
val miniBatchSize = 32 * 32 // 1024, but keep increments of 32, maybe 64(2056) next  //Size of mini batch to use when  training
val exampleLength = 65536/2                    //Length of each training example sequence to use. This could certainly be increased
val tbpttLength = 256                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
val numEpochs = 100                           //Total number of training epochs
val generateSamplesEveryNMinibatches = 500   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
val nSamplesToGenerate = 1                  //Number of samples to generate after each training epoch
val nCharactersToSample = 1000               //Length of each sample to generate
val generationInitialization = "TITLE:\nReturn of the Jedi\n" 
```
5119 MB GPU Mem
Got warnings: `Out of [DEVICE] memory, host memory will be used instead: deviceId: [0]`

Still OOM after redux in param sizes- reducing miniBatchSize

after a few more iterations:

```
val lstmLayerSize = 128                    //Number of units in each GravesLSTM layer
val miniBatchSize = 32 * 8 // 256, but keep increments of 32, maybe 64(2056) next  //Size of mini batch to use when  training
val exampleLength = 65536/4  // ~16k  // and episode ~ 30k e.g. devide by 2   //Length of each training example sequence to use. This could certainly be increased
val tbpttLength = 256                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
val numEpochs = 100                           //Total number of training epochs
val generateSamplesEveryNMinibatches = 500   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
val nSamplesToGenerate = 1                  //Number of samples to generate after each training epoch
val nCharactersToSample = 1000               //Length of each sample to generate
val generationInitialization = "TITLE:\nReturn of the Jedi\n"
```

Not getting output and learning slower


#v9

- lower minibatch size back to 32
- keep example length long
- dropped back to original config

```
val lstmLayerSize = 200                    //Number of units in each GravesLSTM layer
val miniBatchSize = 32 * 1 // 256, but keep increments of 32, maybe 64(2056) next  //Size of mini batch to use when  training
val exampleLength = 65536/8  // ~8k  // and episode ~ 30k e.g. devide by 2   //Length of each training example sequence to use. This could certainly be increased
val tbpttLength = 128                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
val numEpochs = 100                           //Total number of training epochs
val generateSamplesEveryNMinibatches = 500   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
val nSamplesToGenerate = 1                  //Number of samples to generate after each training epoch
val nCharactersToSample = 1000               //Length of each sample to generate
val generationInitialization = "TITLE:\nReturn of the Jedi\n"
```

Strange spikes in error/score.

possible will have to go back and create dataset iterator such that each file is a dataset. 
on each epoch iterate through all datasets.

Or a scene interator that trains on: Characters, keywords, area.

#v10

- Updated char iterator to read file by file (next time do scene by scene- currently on full scripts)
- Seeing HUGE error score now.


#v11
- Scene based 
- Input `NAME NAME2 NAME3\nkeyword1 keyword2 ... keyword9\n`
- expanded from minimal charset to default charset (includes > and [,])

#v12

- updated backprop to 512
- only output examples at every 5k (from 500- stopped early bc output file was way too big)
- was still looking at scripts (not scenes)

NOTE: CharIterator wasn't working irght, only pulled a few scenes and memorized them- updated charIterator

#v13 

Fixed CharFileIterator to randomly select scenes
- only generate sample at end of EPOCH (as opposed to after each miniBatch)


#v14

- Add layer
- Epochs from 100-> 1000

Adding a layer decreases parameter update speed by ~ 20%

```
val lstmLayerSize = 200                    //Number of units in each GravesLSTM layer
val miniBatchSize = 128 // 256, but keep increments of 32, maybe 64(2056) next  //Size of mini batch to use when  training
val exampleLength = 1024 //65536/8  // ~8k  // and episode ~ 30k e.g. devide by 2   //Length of each training example sequence to use. This could certainly be increased
val tbpttLength = 256                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
val numEpochs = 1000                           //Total number of training epochs
val generateSamplesEveryNMinibatches = 5000   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
val nSamplesToGenerate = 1                  //Number of samples to generate after each training epoch
val nCharactersToSample = 1024               //Length of each sample to generate
val generationInitialization = "PICARD DATA TREVO RIKER\ndream power puppy adult happy blue friends apache starfleet\n"
```

#v15

- lr drops at 15000 (instead of 5k)
- REASON: We seem to stop getting better at the point where it switches- (or takes ALOT longer anyway)

NOTES:
- Started outperforming v14 (at 31k iters) around iter 7100 (in terms of score)
- Early termination on iter 7180 

# v16

- installed(?) cuDNN (should give performance boost)
- todo start saving models at end of run (save every N iters- so if you stop early you still have a saved model N ~= 25)
- removed lr step down (to 0.01)

```
val lstmLayerSize = 256                    //Number of units in each LSTM layer
val miniBatchSize = 256 // 256, but keep increments of 32, maybe 64(2056) next  //Size of mini batch to use when  training
val exampleLength = 1024 //65536/8  // ~8k  // and episode ~ 30k e.g. devide by 2   //Length of each training example sequence to use. This could certainly be increased
val tbpttLength = 256                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
val numEpochs = 1000                           //Total number of training epochs
val generateSamplesEveryNMinibatches = 5000   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
val nSamplesToGenerate = 1                  //Number of samples to generate after each training epoch
val nCharactersToSample = 1024               //Length of each sample to generate
val generationInitialization = "LAFORGE DATA TROI RIKER\ndream power puppy adult happy blue friends apache starfleet\n"
```    

#v17

cuDNN is working- seeing 2.5x speed up.

3500MB on GPU

Best (and "longest")run yet- around score 171 (iter 25k) conversations start becoming coherent- see epoch 298

Need to start saving output.

Around epoch 500, less coherent conversations, more spelling errors, etc.

At 1000 epochs, still babbly.  Introduce word2vec. 

Results: 
- unable to determine characters, though it does seem that it was taking sugguestions.
- coherent thought, still not really happening. (Even with 256 tbptt)

#v18

- Redoing character detection (removing `[on viewscreen]` etc. and `TROI + PULASKI` )
- removing topics (will come back later, want to focus on character sugguestions first)
- OOM Errors on initial setup:

```scala
val lstmLayerSize = 1024                    //Number of units in each LSTM layer
val miniBatchSize = 256 // 256, but keep increments of 32, maybe 64(2056) next  //Size of mini batch to use when  training
val exampleLength = 1024 //65536/8  // ~8k  // and episode ~ 30k e.g. devide by 2   //Length of each training example sequence to use. This could certainly be increased
val tbpttLength = 128                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
val numEpochs = 1000                           //Total number of training epochs
val generateSamplesEveryNMinibatches = 5000   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
val nSamplesToGenerate = 1                  //Number of samples to generate after each training epoch
val nCharactersToSample = 1024               //Length of each sample to generate
val generationActors = Array("LAFORGE", "DATA", "TROI", "RIKER")
val generationInitialization = "Stardate"    
```

- reduced minibatch to 128, dropped 3rd layer

```scala
val lstmLayerSize = 1024                    //Number of units in each LSTM layer
val miniBatchSize = 128 // 256, but keep increments of 32, maybe 64(2056) next  //Size of mini batch to use when  training
val exampleLength = 1024 //65536/8  // ~8k  // and episode ~ 30k e.g. devide by 2   //Length of each training example sequence to use. This could certainly be increased
val tbpttLength = 128                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
val numEpochs = 1000                           //Total number of training epochs
val generateSamplesEveryNMinibatches = 5000   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
val nSamplesToGenerate = 1                  //Number of samples to generate after each training epoch
val nCharactersToSample = 1024               //Length of each sample to generate
val generationActors = Array("LAFORGE", "DATA", "TROI", "RIKER")
val generationInitialization = "Stardate"  
```

- notes:  doing awesome at character selection.
- apparently stopped writing output somewhere along the way...

# v19

- Added layer back in
GPU MEM - 3950MB

- 4 layers- having a hard time learning names/words, score is ~8 but having issues with spelling, actor selection

#v20

- dropped back to 3 layers- lower lr
- hit minimum (score 10) around iter 10k, then flat lined, and eventaully got worse

# v21

- add topic one-hots

```bash
val lstmLayerSize = 1024                    //Number of units in each LSTM layer
val miniBatchSize = 128 // 256, but keep increments of 32, maybe 64(2056) next  //Size of mini batch to use when  training
val exampleLength = 996 //1022 //65536/8
val tbpttLength = 128                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
val numEpochs = 1000                           //Total number of training epochs
val generateSamplesEveryNMinibatches = 1000   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
val nSamplesToGenerate = 1                  //Number of samples to generate after each training epoch
val nCharactersToSample = 1024               //Length of each sample to generate
val generationActors = Array("LAFORGE", "DATA", "TROI", "RIKER")
val generationTopics: Array[Float] = new Array[Float](60)
generationTopics(19) = 0.11.toFloat // Romulans
generationTopics(23) = 0.09.toFloat // Battle
val generationInitialization = "Stardate"         //Optional character initialization; a random character is used if null
val learningRate = 0.1
// Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
// Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
val rng = new Random(12345)
```

# v22 

- lower lr

```bash
    val lstmLayerSize = 1024                    //Number of units in each LSTM layer
    val miniBatchSize = 128 // 256, but keep increments of 32, maybe 64(2056) next  //Size of mini batch to use when  training
    val exampleLength = 996 //1022 //65536/8
    val tbpttLength = 128                        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    val numEpochs = 1000                           //Total number of training epochs
    val generateSamplesEveryNMinibatches = 1000   //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
    val nSamplesToGenerate = 1                  //Number of samples to generate after each training epoch
    val nCharactersToSample = 1024               //Length of each sample to generate
    val generationActors = Array("LAFORGE", "DATA", "TROI", "RIKER")
    val generationTopics: Array[Float] = new Array[Float](60)
    generationTopics(19) = 0.11.toFloat // Romulans
    generationTopics(23) = 0.09.toFloat // Battle
    val generationInitialization = "Stardate"         //Optional character initialization; a random character is used if null
    val learningRate = 0.01
    // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
    // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
    val rng = new Random(12345)
```

# v22a

- updated topics to .51 and .49 respectively
- seeing more "shields" "phasers" "Romulan" etc- but at the cost of worse output wrt grammer/spelling. 

# v22b 
- wasn't including topics in output...

# v23

- word2vec: [some basic understanding of Star Trek](http://bionlp-www.utu.fi/wv_demo/) Set to google news and try typing in `Romulan` `Klingon` `Captain_Picard`
- we'll use google 300 for word2vec (will try training our own soon or possible just skip to though-vectors)
