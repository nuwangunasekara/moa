# MOA (Massive Online Analysis)
[![Build Status](https://travis-ci.org/Waikato/moa.svg?branch=master)](https://travis-ci.org/Waikato/moa)
[![Maven Central](https://img.shields.io/maven-central/v/nz.ac.waikato.cms.moa/moa-pom.svg)](https://mvnrepository.com/artifact/nz.ac.waikato.cms)
[![DockerHub](https://img.shields.io/badge/docker-available-blue.svg?logo=docker)](https://hub.docker.com/r/waikato/moa)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

![MOA][logo]

[logo]: http://moa.cms.waikato.ac.nz/wp-content/uploads/2014/11/LogoMOA.jpg "Logo MOA"

MOA is the most popular open source framework for data stream mining, with a very active growing community ([blog](http://moa.cms.waikato.ac.nz/blog/)). It includes a collection of machine learning algorithms (classification, regression, clustering, outlier detection, concept drift detection and recommender systems) and tools for evaluation. Related to the WEKA project, MOA is also written in Java, while scaling to more demanding problems.

http://moa.cms.waikato.ac.nz/

## Using MOA

* [Getting Started](http://moa.cms.waikato.ac.nz/getting-started/)
* [Documentation](http://moa.cms.waikato.ac.nz/documentation/)
* [About MOA](http://moa.cms.waikato.ac.nz/details/)

MOA performs BIG DATA stream mining in real time, and large scale machine learning. MOA can be extended with new mining algorithms, and new stream generators or evaluation measures. The goal is to provide a benchmark suite for the stream mining community. 

## Mailing lists
* MOA users: http://groups.google.com/group/moa-users
* MOA developers: http://groups.google.com/group/moa-development 

## Citing MOA
If you want to refer to MOA in a publication, please cite the following JMLR paper: 

> Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer (2010);
> MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604


# Run MOA experiments with [DJL](https://djl.ai)
## Requirements
* [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
## Datasets
Datasets could be downloaded from:
[https://www.dropbox.com/s/y2d9v0kayorm23b/datasets.tar.gz?dl=0](https://www.dropbox.com/s/y2d9v0kayorm23b/datasets.tar.gz?dl=0)

## How to set up environment
###From source root run:
> bash ./moa/src/main/scripts/reinit_conda.sh ~/Desktop/conda/ moa/src/main/scripts/conda.yml
## How to build MOA for
###From source root run:
> bash ./moa/src/main/scripts/build_moa.sh ~/Desktop/m2/ ~/Desktop/conda/
## Run GUI
> bash ./moa/src/main/scripts/build_moa.sh ~/Desktop/m2/ ~/Desktop/conda/  ~/Desktop/djl.ai
## Run experiments
###From < results dir > run:
> bash < moa source root >/moa/src/main/scripts/run_moa.sh < dataset dir > < results dir > < DJL cache dir > < directory for separate maven repository > < directory for conda environment >

e.g
> bash ~/Desktop/moa_fork/moa/src/main/scripts/run_moa.sh ~/Desktop/datasets/ ~/Desktop/results/Exp1/ ~/Desktop/djl.ai/ ~/Desktop/m2/ ~/Desktop/conda/


