/**
 * \file EventClusterMask.h
 *
 * \ingroup DataFormat
 *
 * \brief Class def header for a class EventClusterMask2D
 * \built to house _clustermask_vv a vector of vectors of ClusterMasks
 * \supposed to house collections of clustermasks by plane in a single event
 *
 * @author Joshua Mills
 */

/** \addtogroup DataFormat

    @{*/
#ifndef EVENTCLUSTERMASK_H
#define EVENTCLUSTERMASK_H

#include <iostream>
#include "EventBase.h"
#include "ClusterMask.h"
#include "DataProductFactory.h"
namespace larcv {

  /**
    \class EventClusterMask
    Event-wise class to store a collection of larcv::ClusterMask
  */
  class EventClusterMask : public EventBase {

  public:

    /// Default constructor
    EventClusterMask(){}

    /// Default destructor
    virtual ~EventClusterMask(){}

    /// Clears an array of larcv::ClusterMask
    void clear();

    /// Const reference getter to a vector of vector<larcv::ClusterMask>
    const std::vector<std::vector<larcv::ClusterMask>>& as_vector() const { return _clustermask_vv; }

    /// Deprecated (use as_vector): const reference getter to a vector of vector<larcv::ClusterMask>
    const std::vector<std::vector<larcv::ClusterMask>>& clustermask_array() const { return _clustermask_vv; }

    /// std::vector<larcv::ClusterMask> const reference getter for a specified index number
    const std::vector<ClusterMask>& at(size_t id) const;

    /// Inserter into larcv::ClusterMask array
    void append(const std::vector<ClusterMask>& mask_v);
#ifndef __CINT__
    /// Emplace into std::vector<std::vector<larcv::ClusterMask>> array
    void emplace(std::vector<ClusterMask>&& mask_v);
    /// Emplace into std::vector<std::vector<larcv::ClusterMask>> array
    void emplace(std::vector<std::vector<larcv::ClusterMask>>&& mask_vv);
    /// std::move to retrieve content std::vector<std::vector<larcv::ClusterMask>> array
    void move(std::vector<std::vector<larcv::ClusterMask>>& mask_vv)
    { mask_vv = std::move(_clustermask_vv); }
#endif


    /// Mutable reference to one of the planes in _clustermask_vv
    std::vector<ClusterMask>& mod_mask_v_at(int id);
    /// call reserver for _image_vv
    void reserve( size_t s);
    /// Mutable reference to _clustermask_vv
    std::vector<std::vector<larcv::ClusterMask>>& as_mod_vector() { return _clustermask_vv; }


  private:

    std::vector<std::vector<larcv::ClusterMask>> _clustermask_vv;

  };

  /**
     \class larcv::EventClusterMask
     \brief A concrete factory class for larcv::EventClusterMask2D
  */
  class EventClusterMaskFactory : public DataProductFactoryBase {
  public:
    /// ctor
    EventClusterMaskFactory() { DataProductFactory::get().add_factory(kProductClusterMask,this); }
    /// dtor
    ~EventClusterMaskFactory() {}
    /// create method
    EventBase* create() { return new EventClusterMask; }
  };

}

#endif
/** @} */ // end of doxygen group
