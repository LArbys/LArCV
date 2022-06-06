# LArCV: Liquid Argon Computer Vision

Data format and image processing tools, routines, and framework for LAr TPC-derived images.
Developed as bridge between LArSoft and deep learning frameworks, e.g. PyTorch, Caffe, Tensorflow.

We originally developed for MicroBooNE, but are now using this library across experiments.
MicroBooNE specific code has been moved into a new repository, [ublarcvapp](https://github.com/larbys/ublarcvapp)
dependent on this library and `larlite`.

One recent big change is that the assumed row order is now in postive time order (same as LArCV2).
We, however, are attempting to maintain the ability to read old "tickbackward" files created for MicroBooNE.
When reading tickbackward images, the data is flipped along the row axis in order to be treated as "tickforward" data.

Planned features
* ROOT utilities for quick, cheap visualization from command line or python interpretor
* Proper documentation off all classes
* Convenient multi-process data-loader into DL frameworks (still kind of clunky right now).
  examples and unittests.
* Image2D -> BJSON -> Image2D support for message passing to GPU server [in progress]
* maintain Vic's pyqtgraph viewer, add 3d and pmt views (from pylard) [done]
* pyutil doesn't need transpose (row,col) means same in image2d as in numpy array

Recently completed features
* CMake support, for easier exporting of package into other projects (done)
* python2 and python3 support 
* Row indices are in positive time order (same as LArCV2).
  But we provide mechanism to read old larcv1 data, then reverse convention.
  This way data is not obsolete, but any code that assumes reverse time order is!
  We do this by specifying in IOManager that Image2D loaded from branches are reversetimeorder.
  For those image2d's, we reshape the array. [done]
* Remove UB-stuff from app (now in its own repo, [ublarcvapp](https://github.com/LArbys/ublarcvapp)).  
* Propagate changes of tick-forward order to rest of code.
    * imagemeta methods [done]
    * image2d methods
    * pyrgb viewer
    * ... more


## Installation

### (direct) Dependencies

* Cmake (3.10) REQUIRED
* ROOT  (>6)   REQUIRED
* Python 2 or 3 (optional)
* OpenCV 4 (optional)

### Setup

0. Dependencies to build with are determined through the presence of environment variables or executables in your PATH:

  * ROOT: usually can setup the environment variables we need by runningn the `thisroot.sh` script.
  * OPENCV: need the presence of environment variables OPENCV_INCDIR and OPENCV_LIBDIR that point to the include and library directories, respectively

1. clone the repository

        git clone https://github.com/LArbys/LArCV.git

2. go into the LArCV directory
3. run the build configuration script

        source configure.sh
      
4. make a build folder somewhere, e.g. in the same folder as this README.

        mkdir build

5. go into the folder and run cmake

        cd build
        cmake -DUSE_PYTHON3=ON ../

   (If you require `python2`, probably deprecated in the future, you can use the cmake flag `-DUSE_PYTHON2=ON` instead.)

6. then make and install the code

        make install


The output of `make install` will be libaries and headers in `build/installed/`.
A cmake config file is provided in `build/installed/lib/larcv` in case you want to incorporate the library into other projects.


## Wiki

Checkout the [Wiki](https://github.com/LArbys/LArCV/wiki) for notes on using this code.
