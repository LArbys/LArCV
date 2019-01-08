all: test_json

LIBS=`root-config --ldflags`
LIBS += `larcv-config --libs`

INCLUDES =`root-config --cflags`
INCLUDES +=`larcv-config --includes`
INCLUDES +=-Iinclude

test_json: test_json.cxx
	g++ -fPIC -g $(INCLUDES) -o test_json test_json.cxx $(LIBS)


clean:
	@rm test_json
