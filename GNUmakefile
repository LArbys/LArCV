
ifndef LARCV_BASEDIR
ERROR_MESSAGE := $(error LARCV_BASEDIR is not set... run configure.sh!)
endif

OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

include $(LARCV_BASEDIR)/Makefile/Makefile.${OSNAME}

CORE_SUBDIRS := Base DataFormat Processor CPPUtil
ifeq ($(LARCV_NUMPY),1)
CORE_SUBDIRS += PyUtil
endif
ifeq ($(LARCV_OPENCV),1)
  CORE_SUBDIRS += CVUtil
endif

APP_SUBDIRS := ImageAna ImageMod Filter PMTWeights HiResDivider Merger APICaffe
ifdef LARLITE_BASEDIR
APP_SUBDIRS += Supera/APILArLite VertexImg
  ifdef LAROPENCV_BASEDIR
  APP_SUBDIRS += LArOpenCVBandle
  endif
endif

#ifeq ($(LARCV_LLBANDLE),1)
#  APP_SUBDIRS += LLBandle/LArCVFlashMatch
#endif

.phony: all clean

all: obj lib

clean: clean_app clean_core
	@rm -f $(LARCV_LIBDIR)/liblarcv.so
clean_core:
	@for i in $(CORE_SUBDIRS); do ( echo "" && echo "Cleaning $$i..." && cd $(LARCV_COREDIR)/$$i && rm -rf $(LARCV_BUILDDIR)/$$i && rm -rf $(LARCV_BUILDDIR)/lib/*$ii.* ) || exit $$?; done
clean_app:
	@for i in $(APP_SUBDIRS); do ( echo "" && echo "Cleaning $$i..." && cd $(LARCV_APPDIR)/$$i && rm -rf $(LARCV_BUILDDIR)/$$i && rm -rf $(LARCV_BUILDDIR)/lib/*$ii.* ) || exit $$?; done

obj:
	@echo
	@echo Building core...
	@echo
	@for i in $(CORE_SUBDIRS); do ( echo "" && echo "Compiling $$i..." && cd $(LARCV_COREDIR)/$$i && $(MAKE) ) || exit $$?; done
	@echo Building app...
	@for i in $(APP_SUBDIRS); do ( echo "" && echo "Compiling $$i..." && cd $(LARCV_APPDIR)/$$i && $(MAKE) ) || exit $$?; done

lib: obj
	@ echo
	@ if [ `python ${LARCV_BASEDIR}/bin/libarg.py build` ]; then \
	    echo Linking library...; \
	    $(SOMAKER) $(SOFLAGS) $(shell python $(LARCV_BASEDIR)/bin/libarg.py); \
	  else \
	   echo Nothing to be done for lib...; \
	fi
	@echo 
