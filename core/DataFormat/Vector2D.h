/**
 * \file Vector2D.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class Vector2D
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef VECTOR2D_H
#define VECTOR2D_H

#include <iostream>
#include "DataFormatTypes.h"
#include "Pixel2D.h"
#include <cmath>

namespace larcv {
  /**
     \class Vector2D
     User defined class Vector2D ... these comments are used to generate
     doxygen documentation!
  */
  class Vector2D {
    
  public:
    
    /// Default constructor
    Vector2D(double x=0, double y=0) : _x(x) , _y(y) {}
    /// Alternative constructor
    Vector2D(const Vector2D& rhs) : _x(rhs.X() + 0.5) , _y(rhs.Y() + 0.5) {}
    /// Default destructor
    ~Vector2D(){}

    /// Pixel getter
    Pixel2D Pixel(double pixel_width=1.0, double pixel_height=1.0) const
    { return Pixel2D((size_t)(_x / pixel_width) , (size_t)(_y / pixel_height)); }
    
    /// X getter
    double X() const { return _x; }
    /// Y getter
    double Y() const { return _y; }
    /// Magnitude
    double Length() const
    { return sqrt(LengthSquared()); }
    /// Magnitude
    double LengthSquared() const
    { return pow(_x,2) + pow(_y,2); }
    
    /// X setter
    void X(double x) { _x = x; }
    /// Y setter
    void Y(double y) { _y = y; }
    /// X Y setter
    void Set(double x, double y) { _x = x; _y = y; }
    /// Normalize
    void Normalize()
    { (*this) /= Length(); }

    //
    // uniry operators
    //
    inline Vector2D& operator += ( const Vector2D& rhs )
    { _x += rhs._x; _y += rhs._y; return (*this); }
    inline Vector2D& operator -= ( const Vector2D& rhs )
    { _x -= rhs._x; _y -= rhs._y; return (*this); }
    inline Vector2D& operator /= ( const double&   rhs )
    { _x /= rhs; _y /= rhs; return (*this); }
    inline Vector2D& operator *= ( const double&   rhs )
    { _x *= rhs; _y *= rhs; return (*this); }
    inline Vector2D& operator  = ( const Vector2D& rhs )
    { _x  = rhs._x; _y  = rhs._y; return (*this); }

    //
    // binary operators
    //
    inline Vector2D  operator +  ( const Vector2D& rhs )
    { auto res = (*this); res += rhs; return res; }
    inline Vector2D  operator -  ( const Vector2D& rhs )
    { auto res = (*this); res -= rhs; return res; }
    inline Vector2D  operator /  ( const double&   rhs )
    { auto res = (*this); res /= rhs; return res; }
    inline Vector2D  operator *  ( const double&   rhs )
    { auto res = (*this); res *= rhs; return res; }
    inline double    operator *  ( const Vector2D& rhs )
    { return _x * rhs._x + _y + rhs._y; }
    
  private:
    double _x; ///< X position
    double _y; ///< Y position
    
  };

}

#endif
/** @} */ // end of doxygen group 

