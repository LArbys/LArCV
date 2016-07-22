/**
 * \file Pixel2DCluster.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class Pixel2DCluster
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef PIXEL2DCLUSTER_H
#define PIXEL2DCLUSTER_H

#include <iostream>
#include "DataFormatTypes.h"
#include "Pixel2D.h"

namespace larcv {

  class EventPixelCollection;

  /**
     \class Pixel2DCluster
     User defined class Pixel2DCluster ... these comments are used to generate
     doxygen documentation!
  */
  class Pixel2DCluster : public std::vector<larcv::Pixel2D> {

    friend class EventPixel2D;

  public:
    
    /// Default constructor
    Pixel2DCluster()
      : std::vector<larcv::Pixel2D>() {}
    /// Alternative constructor 1
    Pixel2DCluster(std::vector<larcv::Pixel2D>&& rhs)
      : std::vector<larcv::Pixel2D>(std::move(rhs)) {}
    /// Alternative constructor 2
    Pixel2DCluster(const std::vector<larcv::Pixel2D>& rhs)
      : std::vector<larcv::Pixel2D>(rhs) {}
    /// Default destructor
    ~Pixel2DCluster(){}

    // Pixel Intensity Sum
    double Intensity() const
    { double res = 0.;
      for(auto const& px : (*this)) { res += px.Intensity(); }
      return res;
    }

    // Cluster ID
    size_t ID() const
    { return _id; }

    // Pool among duplicate pixels
    void Pool(const PoolType_t type);

    //
    // uniry operator
    //
    inline Pixel2DCluster& operator += (const Pixel2D& rhs)
    { this->push_back(rhs); return (*this); }
    inline Pixel2DCluster& operator += (const Pixel2DCluster& rhs)
    { for(auto const& px : rhs) { (*this) += px; }; return (*this); }

  private:

    size_t _id;
    
  };
}

#endif
/** @} */ // end of doxygen group 

