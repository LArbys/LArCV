CXX=g++

CXXFLAGS = -g -fPIC
INCLUDES =`root-config --cflags`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlite-config --includes`
#INCLUDES += -std=c++17
# INCLUDES += -I$(UBLARCVAPP_INCDIR)

LIBS = `root-config --libs`
LIBS += `larlite-config --libs`
LIBS += `larcv-config --libs`
# LIBS += -L$(UBLARCVAPP_LIBDIR)

all:  thrumu_maker thrumu_maker_tickforward


thrumu_maker: thrumu_maker.cxx
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o thrumu_maker $(LIBS)

thrumu_maker_tickforward: thrumu_maker_tickforward.cxx
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o thrumu_maker_tickforward $(LIBS)

clean:
	rm thrumu_maker thrumu_maker_tickforward
