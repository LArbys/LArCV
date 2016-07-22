/**
 * \file AABox2D.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class AABox2D
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/

#ifndef AABOX2D_H
#define AABOX2D_H

#include "Vector2D.h"

namespace larcv {

  /**                                                                                                                              
     \class AABox2D                                                                                                                  
     @brief Representation of a 2D rectangular box which sides are aligned w/ coordinate axis.  
     A representation of an Axis-Aligned-Boundary-Box, a simple & popular representation of   \n
     3D boundary box for collision detection. The concept was taken from the reference,       \n
     Real-Time-Collision-Detection (RTCD), and in particular Ch. 4.2 (page 77):               \n
                                                                                                                                   
     Ref: http://realtimecollisiondetection.net                                                 
                                                                                                
     This class uses one of the simplest representation for AABox: "min-max" representation.  \n
     Though this method requires storing 6 floating point values, class attributes (i.e.      \n
     "min" and "max" points) store intuitive values for most UB analyzers. Also it simplifies \n
     utility function implementations.                                                        \n
     
     This class is borrowed from LArLite: UserDev/BasicTool/GeoAlgo package and simplified into 2D.\n
  */
  class AABox2D {

  public:

    /// Default constructor
    AABox2D();

    /// Default destructor
    virtual ~AABox2D(){};

    /// Alternative ctor (0)
    AABox2D(const double x_min, const double y_min,
	    const double x_max, const double y_max);

    /// Altenartive ctor (1)
    AABox2D(const Vector2D& min, const Vector2D& max);

    //
    // Attribute accessor
    //
    const Vector2D& Min() const; ///< Minimum point getter
    const Vector2D& Max() const; ///< Maximum point getter
    void Min(const double x, const double y); ///< Minimum point setter
    void Max(const double x, const double y); ///< Maximum point setter
    bool Contain(const Vector2D &pt) const; ///< Test if a point is contained within the box

  protected:

    Vector2D _min; ///< Minimum point
    Vector2D _max; ///< Maximum point                               

  };
}

#endif
