/**
 * \file EventPixel2D.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class EventPixel2D
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef EVENTPIXEL2D_H
#define EVENTPIXEL2D_H

#include <iostream>
#include <map>
#include "EventBase.h"
#include "Pixel2D.h"
#include "Pixel2DCluster.h"
#include "ImageMeta.h"
#include "DataProductFactory.h"
namespace larcv {
  
  /**
    \class EventPixel2D
    Event-wise class to store a collection of larcv::Pixel2D and larcv::Pixel2DCluster
  */
  class EventPixel2D : public EventBase {
    
  public:
    
    /// Default constructor
    EventPixel2D() {
      clear();
    }
    
    /// Default destructor
    virtual ~EventPixel2D(){}

    /// Clears an array of larcv::Pixel2D
    void clear();

    /// Retrieve Pixel2D for a plane
    const std::vector<larcv::Pixel2D>& Pixel2DArray(const ::larcv::PlaneID_t plane);
    /// Retrieve Pixel2DCluster for a plane
    const std::vector<larcv::Pixel2DCluster>& Pixel2DClusterArray(const ::larcv::PlaneID_t plane);
    /// Retrieve ImageMeta for a plane (for simple Pixel2D collection)
    const ImageMeta& Meta(const ::larcv::PlaneID_t plane) const;
    /// Retrieve ImageMeta for a plane (for Pixel2DCluster)
    const std::vector<larcv::ImageMeta>& ClusterMetaArray(const ::larcv::PlaneID_t plane) const;
    /// Retrieve ImageMeta for a plane (for Pixel2DCluster)
    const ImageMeta& ClusterMeta(const ::larcv::PlaneID_t plane, const size_t) const;

    const std::map< ::larcv::PlaneID_t, std::vector<larcv::Pixel2D> >& Pixel2DArray() const
    { return _pixel_m; }
    
    const std::map< ::larcv::PlaneID_t, std::vector<larcv::Pixel2DCluster> >& Pixel2DClusterArray() const
    { return _cluster_m; }

    const std::map< ::larcv::PlaneID_t, larcv::ImageMeta>& MetaArray() const
    { return _meta_m; }

    const std::map< ::larcv::PlaneID_t, std::vector< ::larcv::ImageMeta> >& ClusterMetaArray() const
    { return _cluster_meta_m; }

    /// Set ImageMeta
    void SetMeta(const larcv::PlaneID_t plane, const ImageMeta& meta)
    { _meta_m[plane] = meta;}

    /// Insert larcv::Pixel2D into a collection
    void Append(const larcv::PlaneID_t plane, const Pixel2D& pixel);
    /// Insert larcv::Pixel2DCluster into a collection
    void Append(const larcv::PlaneID_t plane, const Pixel2DCluster& cluster);
    /// Insert larcv::Pixel2DCluster into a collection
    void Append(const larcv::PlaneID_t plane, const Pixel2DCluster& cluster, const ImageMeta&);
    
#ifndef __CINT__
    /// Emplace larcv::Pixel2D into a collection
    void Emplace(const larcv::PlaneID_t plane, Pixel2D&& pixel);
    /// Emplace larcv::Pixel2DCluster into a collection
    void Emplace(const larcv::PlaneID_t plane, Pixel2DCluster&& cluster, const ImageMeta&);
#endif

    void reverseTickOrder(); ///< convert tick-backward pixel order to tick-forward (hack during dllee_unified->ubdl transition)

  private:

    std::map< ::larcv::PlaneID_t, std::vector< ::larcv::Pixel2D > >        _pixel_m;
    std::map< ::larcv::PlaneID_t, std::vector< ::larcv::Pixel2DCluster > > _cluster_m;
    std::map< ::larcv::PlaneID_t, ::larcv::ImageMeta > _meta_m;
    std::map< ::larcv::PlaneID_t, std::vector< ::larcv::ImageMeta > > _cluster_meta_m;
  };

  /**
     \class larcv::EventPixel2D
     \brief A concrete factory class for larcv::EventPixel2D
  */
  class EventPixel2DFactory : public DataProductFactoryBase {
  public:
    /// ctor
    EventPixel2DFactory() { DataProductFactory::get().add_factory(kProductPixel2D,this); }
    /// dtor
    ~EventPixel2DFactory() {}
    /// create method
    EventBase* create() { return new EventPixel2D; }
  };

}

#endif
/** @} */ // end of doxygen group 

