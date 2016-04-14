CXXFLAGS += $(shell larcv-config --includes)

LDFLAGS += $(shell larcv-config --libs)

OSNAME = $(shell uname -s)
include $(LARCV_BASEDIR)/Makefile/Makefile.${OSNAME}

PROGRAMS = calculate_mean

TARGETS = $(addsuffix .cc, $(PROGRAMS))

all: $(PROGRAMS)

$(PROGRAMS): 
	@echo '<<compiling' $@'>>'
	@$(CXX) $@.cc -o $(LARCV_BINDIR)/$@ $(CXXFLAGS) $(LDFLAGS) 
	@rm -rf *.dSYM

clean:	
	@for i in $(PROGRAMS); do ( echo "" && echo "Cleaning $$i..." && rm -f $(LARCV_BINDIR)/$$i || exit $$?; done
