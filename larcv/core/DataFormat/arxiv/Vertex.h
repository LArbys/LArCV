#ifndef __VERTEX_H__
#define __VERTEX_H__

namespace larcv {

  class Vertex {
    friend class Vertex;
    
  public:
    /// Particle ID default constructor
    Vertex();
    Vertex(double x, double y, double z, double t);
    
    /// Reset function for x, y, z, t
    void Reset();
    
    double X() const { return _x; }
    double Y() const { return _y; }
    double Z() const { return _z; }
    double T() const { return _t; }
    
    /// Default destructor
    virtual ~Vertex(){};
    
    inline bool operator== (const Vertex& rhs) const
    {
      return ( _x == rhs._x && _y == rhs._y && _z == rhs._z && _t == rhs._t ); 
    }
    
    inline bool operator!= (const Vertex& rhs) const
    {
      return !((*this) == rhs);
    }
    
    inline bool operator< (const Vertex& rhs) const
    {
      if( _x     < rhs._x ) return true;
      if( rhs._x < _x     ) return false;
      if( _y     < rhs._y ) return true;
      if( rhs._y < _y     ) return false;
      if( _z     < rhs._z ) return true;
      if( rhs._z < _z     ) return false;
      if( _t     < rhs._t ) return true;
      if( rhs._t < _t     ) return false;
      
      return false;
    }
  private:
    
    double _x, _y, _z, _t;
    
    void Approx();
  };
}

#endif
