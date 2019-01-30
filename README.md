# LArCV: Liquid Argon Computer Vision

Data format and image processing tools, routines, and framework for LAr TPC-derived images. Developing as bridge between LArSoft and Caffe/Tensorflow.

This is the UBDL_DEV branch, which is the version to be used when
deploying the (new) UB DL reco. chain on FNAL (or Tufts).

## Installation

### (direct) Dependencies

* Cmake (3.10) REQUIRED
* ROOT  (>6)   REQUIRED
* Python 2 or 3 (optional)
* OpenCV 3 (optional)
* LArLite (optional)
* LArSoft (optional)

### Setup

0. Dependencies to build with are determined through the presence of environment variables or executables in your PATH:

  * ROOT: determined through the ability to run rootcling (ROOT6) or rootcint (ROOT5)
  * OPENCV: the presence of OPENCV_INCDIR and OPENCV_LIBDIR
  * LArLite: the presence of LARLITE_BASEDIR (created when one runs configure/setup.sh in the larlite root folder)
  * LArSoft: if MRB_TOP defined and the uboonecode source directories found, will build LArSoft API
  

1. clone the repository

      git clone https://github.com/LArbys/LArCV.git

2. go into the LArCV directory
3. run the build configuration script

      source configure.sh
      
4. build

      make
      

That's it.


## Wiki

Checkout the [Wiki](https://github.com/LArbys/LArCV/wiki) for notes on using this code.
