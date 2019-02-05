# LArCV: Liquid Argon Computer Vision

Data format and image processing tools, routines, and framework for LAr TPC-derived images. Developing as bridge between LArSoft and Caffe/Tensorflow.

This is the UBDL_DEV branch, which is the version to be used when
deploying the (new) UB DL reco. chain on FNAL (or Tufts).

Planned features

* CMake support, for easier exporting of package into other projects (done)
* python2 and python3 support 
* ROOT utilities for quick, cheap visualization from command line or python interpretor
* ROOT Eve Viewer by Ralitsa
* Row indices are in positive time order (same as LArCV2).
  But we provide mechanism to read old larcv1 data, then reverse convention.
  This way data is not obsolete, but any code that assumes reverse time order is!
  We do this by specifying in IOManager that Image2D loaded from branches are reversetimeorder.
  For those image2d's, we reshape the array. [done]
* Propagate changes of tick-forward order to rest of code.
    * imagemeta methods [done]
    * image2d methods
    * pyrgb viewer
    * ... more
* Canvas support for Supera API. No need for lar.
* Remove UB-stuff from app (now in its own [repo](https://github.com/LArbys/ublarcvapp).
* Proper documentation off all classes
* Convenient multi-process data-loader into DL frameworks (still kind of clunky right now).
  examples and unittests.
* Image2D -> BJSON -> Image2D support for message passing to GPU server [done]
* maintain Vic's pyqtgraph viewer, add 3d and pmt views (from pylard) [done]
* pyutil doesn't need transpose (row,col) means same in image2d as in numpy array
* export to eigen representations (maps, copied dense and sparse matrices)

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
