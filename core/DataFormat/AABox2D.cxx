#ifndef AABOX2D_CXX
#define AABOX2D_CXX

#include "AABox2D.h"
#include "Base/larbys.h"
namespace larcv {

  AABox2D::AABox2D()
    : _min()
    , _max()
  {}

  AABox2D::AABox2D(const double x_min, const double y_min,
		   const double x_max, const double y_max)
    : _min ( x_min, y_min )
    , _max ( x_max, y_max )
  {
    if(_min.X() >= _max.X()) throw larbys("AABox x_min >= x_max!");
    if(_min.Y() >= _max.Y()) throw larbys("AABox y_min >= y_max!");
  }

  AABox2D::AABox2D(const Vector2D& min, const Vector2D& max)
    : _min ( min )
    , _max ( max )
  {
    if(_min.X() >= _max.X()) throw larbys("AABox x_min >= x_max!");
    if(_min.Y() >= _max.Y()) throw larbys("AABox y_min >= y_max!");
  }

  const Vector2D& AABox2D::Min() const { return _min; }
  
  const Vector2D& AABox2D::Max() const { return _max; }

  void AABox2D::Min(const double x, const double y)
  {
    if(x >= _max.X()) throw larbys("AABox x_min >= x_max!");
    if(y >= _max.Y()) throw larbys("AABox y_min >= y_max!");
  }
  void AABox2D::Max(const double x, const double y)
  {
    if(x <= _min.X()) throw larbys("AABox x_min >= x_max!");
    if(y <= _min.Y()) throw larbys("AABox y_min >= y_max!");
  }

  bool AABox2D::Contain(const Vector2D &pt) const {
    return !( (pt.X() < _min.X() || _max.X() < pt.X()) // point is outside X boundaries OR
	      || 
	      (pt.Y() < _min.Y() || _max.Y() < pt.Y()) // point is outside Y boundaries 
	      );
  }
  

}
#endif
