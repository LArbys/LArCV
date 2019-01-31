#!/bin/bash

# Use this to not build supera (can cause conflicts when building supera in larsoft)
#export LARCV_NOSUPERA=1

# Use this to not build w/ opencv
#export LARCV_NOOPENCV=1

# clean up previously set env
if [[ -z $FORCE_LARCV_BASEDIR ]]; then
    where="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    export LARCV_BASEDIR=${where}
else
    export LARCV_BASEDIR=$FORCE_LARCV_BASEDIR
fi

# Abort if ROOT not installed. Let's check rootcint for this.
if [ `command -v rootcling` ]; then
    export LARCV_ROOT6=1
else 
    if [[ -z `command -v rootcint` ]]; then
	echo
	echo Looks like you do not have ROOT installed.
	echo You cannot use LArCV w/o ROOT!
	echo Aborting.
	echo
	return 1;
    fi
fi

# check for either clang or gcc
export LARCV_CXX=clang++
if [ -z `command -v $LARCV_CXX` ]; then
    export LARCV_CXX=g++
    if [ -z `command -v $LARCV_CXX` ]; then
        echo
        echo Looks like you do not have neither clang or g++!
        echo You need one of those to compile LArCaffe... Abort config...
        echo
        return 1;
    fi
fi

# set the build dir
unset LARCV_BUILDDIR
if [[ -z $LARCV_BUILDDIR ]]; then
    export LARCV_BUILDDIR=$LARCV_BASEDIR/build
fi

# set important directories
export LARCV_COREDIR=$LARCV_BASEDIR/larcv/core
export LARCV_APPDIR=$LARCV_BASEDIR/larcv/app
export LARCV_LIBDIR=$LARCV_BUILDDIR/installed/lib
export LARCV_INCDIR=$LARCV_BUILDDIR/installed/include
export LARCV_BINDIR=$LARCV_BUILDDIR/installed/bin

# set the version
export LARCV_VERSION=1

# Check OpenCV
export LARCV_OPENCV=1
if [[ -z $OPENCV_INCDIR ]]; then
    export LARCV_OPENCV=0
fi
if [[ -z $OPENCV_LIBDIR ]]; then
    export LARCV_OPENCV=0
fi

# Check Numpy
export LARCV_NUMPY=`$LARCV_BASEDIR/bin/check_numpy`

# Check for libtorch
export LARCV_LIBTORCH=1
if [[ -z $LIBTORCH_INCDIR ]]; then
    export LARCV_LIBTORCH=0
fi
if [[ -z $LIBTORCH_LIBDIR ]]; then
    export LARCV_LIBTORCH=0
fi

# Set ANN directories
if [[ -z $LARCV_ANN ]]; then
    export LARCV_ANN=1
fi

if [ $LARCV_ANN -eq 1 ]; then
    export ANN_INCDIR=$LARCV_BASEDIR/app/ann_1.1.2/include
    export ANN_LIBDIR=$LARCV_BASEDIR/app/ann_1.1.2/lib
    printf "\033[93mANN: approximate nearest neighboor\033[00m\n"
    echo "    Found ANN package"
fi

# warning for missing support
missing=""
if [ $LARCV_OPENCV -eq 0 ]; then
    missing+=" OpenCV"
fi
if [ $LARCV_NUMPY -eq 0 ]; then
    missing+=" Numpy"
fi
if [ $LARCV_ANN -eq 0 ]; then
    missing+=" ANN"
fi
if [ $LARCV_LIBTORCH -eq 0 ]; then
    missing+=" libTorch"
else
    printf "\033[93mlibTorch\033[00m Building with libtorch (PyTorch C++ API) support \n"
fi
if [[ $missing ]]; then
    printf "\033[93mWarning\033[00m ... missing$missing support. Build without them.\n";
fi

echo
printf "\033[93mLArCV\033[00m FYI shell env. may useful for external packages:\n"
printf "    \033[95mLARCV_INCDIR\033[00m   = $LARCV_INCDIR\n"
printf "    \033[95mLARCV_LIBDIR\033[00m   = $LARCV_LIBDIR\n"
printf "    \033[95mLARCV_BUILDDIR\033[00m = $LARCV_BUILDDIR\n"

# modify paths
[[ ":$PATH:" != *":${LARCV_BASEDIR}/bin:"* ]] && export PATH="${LARCV_BASEDIR}/bin:${PATH}"
[[ ":$LD_LIBRARY_PATH:" != *":${LARCV_LIBDIR}/bin:"* ]] && export LD_LIBRARY_PATH="${LARCV_LIBDIR}:${LD_LIBRARY_PATH}"
[[ ":$DYLD_LIBRARY_PATH:" != *":${LARCV_LIBDIR}/bin:"* ]] && export DYLD_LIBRARY_PATH="${LARCV_LIBDIR}:${DYLD_LIBRARY_PATH}"

# paths if using OPENCV (NO_OPENCV to explicitly disallow)
if [[ -z $LARCV_NOOPENCV ]]; then
    [[ ":$LD_LIBRARY_PATH:" != *":${OPENCV_LIBDIR}/bin:"* ]] && export LD_LIBRARY_PATH="${OPENCV_LIBDIR}:${LD_LIBRARY_PATH}"
    [[ ":$DYLD_LIBRARY_PATH:" != *":${OPENCV_LIBDIR}/bin:"* ]] && export DYLD_LIBRARY_PATH="${OPENCV_LIBDIR}:${DYLD_LIBRARY_PATH}"    
fi

mkdir -p $LARCV_BUILDDIR;
mkdir -p $LARCV_LIBDIR;
mkdir -p $LARCV_BINDIR;

# add to python path
[[ ":$PYTHONPATH:" != *":${LARCV_BASEDIR}/python:"* ]] && export PYTHONPATH="${LARCV_BASEDIR}/python:${PATH}"

echo
echo "Finish configuration. To build, type:"
echo "> cd \$LARCV_BUILDDIR"
echo "> make "
echo
