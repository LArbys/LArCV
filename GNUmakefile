
ifndef LARCV_BASEDIR
ERROR_MESSAGE := $(error LARCV_BASEDIR is not set... run configure.sh!)
endif

all:
	@echo
	@echo Building core...
	@echo
	@make --directory=$(LARCV_COREDIR)
	@echo
	@echo Building app...
	@echo
	@make --directory=$(LARCV_APPDIR)

clean:
	@echo
	@echo Cleaning core...
	@echo
	@make clean --directory=$(LARCV_COREDIR)
	@echo
	@echo Cleaning app...
	@echo
	@make clean --directory=$(LARCV_APPDIR)

