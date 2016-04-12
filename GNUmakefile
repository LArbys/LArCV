
ifndef LARCV_BASEDIR
ERROR_MESSAGE := $(error LARCV_BASEDIR is not set... run configure.sh!)
endif

all:
	@make --directory=$(LARCV_COREDIR)
ifdef LARLITE_BASEDIR
	@make --directory=$(LARCV_SUPERADIR)
endif

clean:
	@make clean --directory=$(LARCV_COREDIR)
ifdef LARLITE_BASEDIR
	@make clean --directory=$(LARCV_SUPERADIR)
endif

