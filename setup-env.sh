#!/usr/bin/env bash

# Select CUDA 8 / CUDA 9
# sudo update-alternatives --config cuda

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\
	${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


# Hack to get CUDA 9.1 (on ubuntu 17.04) to work with CUDA 8.0 (per DL4J)
# cd /usr/local/cuda-9.1/lib64
# sudo ln -s libcudart.so libcudart.so.8.0
# sudo ln -s libcublas.so libcublas.so.8.0
# sudo ln -s libcusparse.so libcusparse.so.8.0
# sudo ln -s libcusolver.so libcusolver.so.8.0

# for cuDNN


# Run Jar
cd target
java -classpath .:scripts-dl4j-0.1-SNAPSHOT-shaded.jar org.rawkintrevo.scenegen.App

# or build and run jar
mvn clean package -DskipTests && cd target && java -classpath .:scripts-dl4j-0.1-SNAPSHOT-shaded.jar org.rawkintrevo.scenegen.App >> ../output/v16-output.txt