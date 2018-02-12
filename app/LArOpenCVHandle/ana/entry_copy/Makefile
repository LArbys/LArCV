CC=g++
CFLAGS=-g
INCFLAGS =-I.
INCFLAGS+=$(shell root-config --cflags)

LDFLAGS=-L.
LDFLAGS+=$(shell root-config --libs)

SOURCES=entry_copy.cxx
OBJECTS=$(SOURCES:.cxx=.o)
EXECUTABLE=entry_copy

all: $(OBJECTS) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $(OBJECTS) $(LDFLAGS)

$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $(INCFLAGS) -c $< -o $@

clean:
	rm *.o entry_copy
