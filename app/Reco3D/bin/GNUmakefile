# Include your header file location                                                                                                                                   
CXXFLAGS += -I. $(shell root-config --cflags) -g
CXXFLAGS += $(shell larlite-config --includes)
CXXFLAGS += $(shell larlite-config --includes)/../UserDev
CXXFLAGS += $(shell larcv-config --includes)
CXXFLAGS += -I$(LARCV_APPDIR)/Reco3D

# Include your shared object lib location                                                                                                                             
LDFLAGS += $(shell larlite-config --libs)
LDFLAGS += $(shell larcv-config --libs)
LDFLAGS += $(shell root-config --libs) -lPhysics -lMatrix

# platform-specific options                                                                                                                                           
OSNAME = $(shell uname -s)
include $(LARCV_BASEDIR)/Makefile/Makefile.${OSNAME}

# Add your program below with a space after the previous one.                                                                                                         
# This makefile compiles all binaries specified below.                                                                                                                
PROGRAMS = run_reco

all:            $(PROGRAMS)

$(PROGRAMS): $(PROGRAMS).cxx
	@echo '<<compiling' $@'>>'
	@$(CXX) $@.cxx -o $@ $(CXXFLAGS) -std=c++1y $(LDFLAGS)
	@rm -rf *.dSYM
	@echo '<<installing' $@'>>'
	@cp run_reco $(LARCV_BUILDDIR)/bin/
	@chmod 755 $(LARCV_BUILDDIR)/bin/run_reco
clean:
	rm -f $(PROGRAMS)
	rm $(LARCV_BUILDDIR)/bin/run_reco