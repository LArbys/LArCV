#ifndef __VERTEX_CXX__
#define __VERTEX_CXX__

#include "Vertex.h"

namespace larcv {
    
  // Interaction ID default constructor
  Vertex::Vertex()
    : _x(0)
    , _y(0)
    , _z(0)
    , _t(0)
  {Approx();}
  
  Vertex::Vertex(double x, double y, double z, double t)
    : _x(x), _y(y), _z(z), _t(t)
  {}
  
  void Vertex::Reset(){
    _x = _y = _z = _t = 0;
  }
  
  void Vertex::Approx()
  {
    _x = (double)( ((signed long long)(_x * 1.e12)) * 1.e-12 );
    _y = (double)( ((signed long long)(_y * 1.e12)) * 1.e-12 );
    _z = (double)( ((signed long long)(_z * 1.e12)) * 1.e-12 );
    _t = (double)( ((signed long long)(_t * 1.e12)) * 1.e-12 );
  }
}

#endif  
