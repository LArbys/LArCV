/**
 * \file LineSegment2D.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class LineSegment2D
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef LINESEGMENT2D_H
#define LINESEGMENT2D_H

#include <iostream>
#include "DataFormatTypes.h"
#include "Vector2D.h"
#include "Base/larbys.h"

namespace larcv {
  /**
     \class LineSegment2D
     User defined class LineSegment2D ... these comments are used to generate
     doxygen documentation!
  */
  class LineSegment2D {

  private:
    /// Default constructor
    LineSegment2D() : _start() , _end()
    {}
    
  public:
    /// Alternative constructor 1
    LineSegment2D(double x1, double y1, double x2, double y2)
      : _start(x1,y1) , _end(x2,y2) {}
    /// Alternative constructor 2
    LineSegment2D(const Vector2D& p1, const Vector2D& p2)
      : _start(p1) , _end(p2) {}
    /// Alternative constructor 3
    LineSegment2D(Vector2D&& p1, Vector2D&& p2)
      : _start(std::move(p1)) , _end(std::move(p2)) {}
    /// Default destructor
    ~LineSegment2D(){}

    /// Start point getter
    const Vector2D& Start() const { return _start; }
    /// End point getter
    const Vector2D& End() const { return _end; }
    /// Slope getter
    double Slope() const { return (_end.Y() - _start.Y()) / (_end.X() - _start.X()); }
    /// Direction vector
    Vector2D Dir() const
    { auto res = _end; res -= _start; res /= res.Length(); return res; }

  private:
    Vector2D _start; ///< start of 2D line
    Vector2D _end;   ///< end of 2D line
  };
}

#endif
/** @} */ // end of doxygen group 

