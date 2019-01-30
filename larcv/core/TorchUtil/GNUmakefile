#
# Package with Torch utilities -- for PyTorch C++ API
#
PACKAGE_NAME = TorchUtil

# specific names for this package
SOURCES = $(wildcard *.cxx)
FMWK_HEADERS = LinkDef.h
#HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))
HEADERS = 
HEADERS_DEST = larcv/core/$(PACKAGE_NAME)

# include options for this package
INCFLAGS  = -I.                       #Include itself
INCFLAGS += $(shell larcv-config --includes)
INCFLAGS += -I$(LARCV_INCDIR)
INCFLAGS += -I$(LIBTORCH_INCDIR) -I$(LIBTORCH_INCDIR)/torch/csrc/api/include
INCFLAGS += -Wno-unused-variable -Wno-unused-parameter -fpermissive

# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

include $(LARCV_BASEDIR)/Makefile/Makefile.${OSNAME}

LDFLAGS += -L$(LIBTORCH_LIBDIR) -ltorch -lcaffe2

# call the common GNUmakefile
include $(LARCV_BASEDIR)/Makefile/GNUmakefile.CORE

pkg_build: # nothing extra for this package
pkg_clean: # nothing extra for this package
