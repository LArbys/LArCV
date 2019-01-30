/**
 * \file EventVoxel3D.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class EventVoxel3D
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef EVENTVOXEL3D_H
#define EVENTVOXEL3D_H

#include <iostream>
#include "EventBase.h"
#include "Voxel3D.h"
#include "DataProductFactory.h"
namespace larcv {
  /**
    \class EventVoxel3D
    Event-wise class to store a collection of larcv::Voxel3D
  */
  class EventVoxel3D : public EventBase,
		       public Voxel3DSet {
    
  public:
    
    /// Default constructor
    EventVoxel3D(){}
    
    /// Default destructor
    ~EventVoxel3D(){}

    /// larcv::Voxel3D array clearer
    void clear();

  };

  /**
     \class larcv::EventVoxel3D
     \brief A concrete factory class for larcv::EventVoxel3D
  */
  class EventVoxel3DFactory : public DataProductFactoryBase {
  public:
    /// ctor
    EventVoxel3DFactory() { DataProductFactory::get().add_factory(kProductVoxel3D,this); }
    /// dtor
    ~EventVoxel3DFactory() {}
    /// create method
    EventBase* create() { return new EventVoxel3D; }
  };

  
}
#endif
/** @} */ // end of doxygen group 

