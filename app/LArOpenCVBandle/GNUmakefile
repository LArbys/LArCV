#
# This is an example GNUmakefile for my packages
#
PACKAGE_NAME = LArOpenCVBandle

# specific names for this package
SOURCES = $(wildcard *.cxx)
FMWK_HEADERS = LinkDef.h
HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))
#HEADERS += IOManager.inl

# include options for this package
INCFLAGS  = -I.                       #Include itself
INCFLAGS += $(shell basictool-config --includes)
INCFLAGS += $(shell larcv-config --includes)
INCFLAGS += -I$(LARLITE_COREDIR)
INCFLAGS += -I$(LAROPENCV_BASEDIR)
INCFLAGS += -I$(GEO2D_BASEDIR)
INCFLAGS += -I$(LARLITE_USERDEVDIR)/RecoTool/
INCFLAGS += -I$(LAROPENCV_IMAGECLUSTERDIR)
INCFLAGS += -I$(LARCV_APPDIR)/NuFilter

LDFLAGS += -L$(LARLITE_LIBDIR)
LDFLAGS += -lLArOpenCV_ImageClusterBase
LDFLAGS += -lLArOpenCV_ImageClusterAlgoData
LDFLAGS += -lLArOpenCV_ImageClusterCluster
LDFLAGS += -lLArOpenCV_ImageClusterDirection
LDFLAGS += -lLArOpenCV_ImageClusterUtil
LDFLAGS += -lLArOpenCV_ImageClusterMatch
LDFLAGS += -lLArOpenCV_ImageClusterMerge
LDFLAGS += -lLArOpenCV_ImageClusterFilter
#LDFLAGS += -lLArOpenCV_ImageClusterReCluster
LDFLAGS += -lLArOpenCV_ImageClusterStartPoint
LDFLAGS += -lLArOpenCV_ImageClusterDebug
LDFLAGS += -lLArOpenCV_Utils -lLArOpenCV_Core -lRecoTool_ClusterRecoUtil
LDFLAGS += -lBasicTool_FhiclLite

# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

include $(LARCV_BASEDIR)/Makefile/Makefile.${OSNAME}

LDFLAGS += $(shell larcv-config --libs) 
# call the common GNUmakefile
include $(LARCV_BASEDIR)/Makefile/GNUmakefile.CORE

pkg_build:
pkg_clean:
