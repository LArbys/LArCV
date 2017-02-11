#!/bin/bash

# clean up previously set env
if [[ -z $FORCE_LARCV_BASEDIR ]]; then
    where="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    export LARCV_BASEDIR=${where}
else
    export LARCV_BASEDIR=$FORCE_LARCV_BASEDIR
fi

# set the build dir
unset LARCV_BUILDDIR
if [[ -z $LARCV_BUILDDIR ]]; then
    export LARCV_BUILDDIR=$LARCV_BASEDIR/build
fi

export LARCV_COREDIR=$LARCV_BASEDIR/core
export LARCV_APPDIR=$LARCV_BASEDIR/app
export LARCV_LIBDIR=$LARCV_BUILDDIR/lib
export LARCV_INCDIR=$LARCV_BUILDDIR/include
export LARCV_BINDIR=$LARCV_BUILDDIR/bin

# Abort if ROOT not installed. Let's check rootcint for this.
if [ `command -v rootcling` ]; then
    export LARCV_ROOT6=1
else 
    if [[ -z `command -v rootcint` ]]; then
	echo
	echo Looks like you do not have ROOT installed.
	echo You cannot use LArLite w/o ROOT!
	echo Aborting.
	echo
	return 1;
    fi
fi

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

# Check ann
export LARCV_ANN=1
export ANN_INCDIR=$LARCV_BASEDIR/app/ann_1.1.2/include
export ANN_LIBDIR=$LARCV_BASEDIR/app/ann_1.1.2/lib
if [[ -z $ANN_INCDIR ]]; then
    export LARCV_ANN=0
fi
if [[ -z $ANN_LIBDIR ]]; then
    export LARCV_ANN=0
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
if [[ $missing ]]; then
    printf "\033[93mWarning\033[00m ... missing$missing support. Build without them.\n";
fi

echo
printf "\033[93mLArCV\033[00m FYI shell env. may useful for external packages:\n"
printf "    \033[95mLARCV_INCDIR\033[00m   = $LARCV_INCDIR\n"
printf "    \033[95mLARCV_LIBDIR\033[00m   = $LARCV_LIBDIR\n"
printf "    \033[95mLARCV_BUILDDIR\033[00m = $LARCV_BUILDDIR\n"

export PATH=$LARCV_BASEDIR/bin:$LARCV_BINDIR:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LARCV_LIBDIR
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$LARCV_LIBDIR

if [[ -z $NO_OPENCV ]]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENCV_LIBDIR
    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$OPENCV_LIBDIR
fi

mkdir -p $LARCV_BUILDDIR;
mkdir -p $LARCV_LIBDIR;
mkdir -p $LARCV_BINDIR;

export LD_LIBRARY_PATH=$LARCV_LIBDIR:$LD_LIBRARY_PATH
export PYTHONPATH=$LARCV_BASEDIR/python:$PYTHONPATH

if [[ $LARLITE_BASEDIR ]]; then
    printf "\033[93mLArLite\033[00m\n"
    echo "    Found larlite set up @ \$LARLITE_BASEDIR=${LARLITE_BASEDIR}"
    echo "    Preparing APILArLite package for build (making sym links)"
    target=$LARCV_APPDIR/Supera/larfmwk_shared/*
    for f in $target
    do
	ln -sf $f $LARCV_APPDIR/Supera/APILArLite/
    done
fi

if [[ $ANN_LIBDIR ]]; then
    printf "\033[93mANN: approximate nearest neighboor\033[00m\n"
    echo "    Found ANN package"
fi

if [[ -d $MRB_TOP/srcs/uboonecode/uboone ]]; then
    printf "\033[93mLArSoft\033[00m\n"
    echo "    Found local larsoft @ \$MRB_TOP=${MRB_TOP}"
    echo "    Preparing APILArSoft package for build (making sym links)"
    target=$LARCV_APPDIR/Supera/larfmwk_shared/*
    for f in $target
    do
	ln -sf $f $LARCV_APPDIR/Supera/APILArSoft/
    done
    if [ ! -d $MRB_TOP/srcs/uboonecode/uboone/Supera ]; then
	ln -s $LARCV_APPDIR/Supera/APILArSoft $MRB_TOP/srcs/uboonecode/uboone/Supera
    fi
fi

if [[ -d $MRB_TOP/srcs/argoneutcode/ ]]; then
    printf "\033[93mLArSoft\033[00m\n"
    echo "    Found local larsoft @ \$MRB_TOP=${MRB_TOP}"
    echo "    Preparing APILArSoft package for build (making sym links)"
    target=$LARCV_APPDIR/Supera/larfmwk_shared/*
    for f in $target
    do
    ln -sf $f $LARCV_APPDIR/Supera/APILArSoft/
    done
    if [ ! -d $MRB_TOP/srcs/argoneutcode/Supera ]; then
    ln -s $LARCV_APPDIR/Supera/APILArSoft $MRB_TOP/srcs/argoneutcode/Supera
    fi
fi


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

echo
echo "Finish configuration. To build, type:"
echo "> cd \$LARCV_BUILDDIR"
echo "> make "
echo
