all: test_json run_test

INCLUDES =`root-config --cflags`
INCLUDES +=`larcv-config --includes`
INCLUDES +=-Iinclude
INCLUDES += $(shell laropencv-config --includes)
INCLUDES += $(shell geo2d-config --includes)
INCLUDES += $(shell larlite-config --includes)
INCLUDES += -I$(LARLITE_BASEDIR)/UserDev/


LIBS = `larcv-config --libs`
LIBS += `laropencv-config --libs`
LIBS += $(shell geo2d-config --libs)
LUBS += $(shell larlite-config --libs) -lBasicTool_FhiclLite
LIBS +=`root-config --ldflags --libs`

test_json: test_json.cxx
	g++ -fPIC -g $(INCLUDES) -o test_json test_json.cxx $(LIBS)

clean:
	@rm test_json

run_test: run_test.cxx
	g++ -fPIC -g $(INCLUDES) run_test.cxx -o run_test $(LIBS)


clean:
	@rm test_json run_test
