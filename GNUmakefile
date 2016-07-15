
ifndef LARCV_BASEDIR
ERROR_MESSAGE := $(error LARCV_BASEDIR is not set... run configure.sh!)
endif

OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

include $(LARCV_BASEDIR)/Makefile/Makefile.${OSNAME}

all:
	@echo
	@echo Building core...
	@echo
	@$(MAKE) $(ARGS) --directory=$(LARCV_COREDIR)
	@echo
	@echo Building app...
	@echo
	@$(MAKE) $(ARGS) --directory=$(LARCV_APPDIR)
	@echo 
	@echo Linking libs...
	@$(SOMAKER) $(SOFLAGS) -o liblarcv.so $(shell python $(LARCV_BASEDIR)/bin/libarg.py)
	@echo 

clean:
	@echo
	@echo Cleaning core...
	@echo
	@$(MAKE) clean --directory=$(LARCV_COREDIR)
	@echo
	@echo Cleaning app...
	@echo
	@$(MAKE) clean --directory=$(LARCV_APPDIR)
	@echo
	@echo Cleaning lib...
	@echo
	@rm -f $(LARCV_LIBDIR)/liblarcv.so
	@echo
