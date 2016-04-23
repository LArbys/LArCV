#
# This is an example GNUmakefile for my packages
#
PACKAGE_NAME = HiRes

# specific names for this package
SOURCES = $(wildcard *.cxx)
FMWK_HEADERS = LinkDef.h
HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))
#HEADERS += IOManager.inl

# include options for this package
INCFLAGS  = -I.                       #Include itself
INCFLAGS += $(shell larcv-config --includes)

# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

include $(LARCV_BASEDIR)/Makefile/Makefile.${OSNAME}

CVLIBDIR = /usr/local/opt/opencv3/lib

ifeq ($(LARCV_OPENCV),1)
	CXXFLAGS += -DHAS_OPENCV
	LDFLAGS += $(shell larcv-config --libs) -lLArCVPMTWeights -L$(CVLIBDIR) -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
endif

# call the common GNUmakefile
include $(LARCV_BASEDIR)/Makefile/GNUmakefile.CORE

pkg_build:
pkg_clean:
