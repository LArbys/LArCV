#ifndef __LARCV_CORE_DATAFORMAT_EVENT_CLUSTER_VOXEL3D_H__
#define __LARCV_CORE_DATAFORMAT_EVENT_CLUSTER_VOXEL3D_H__

#include "larcv/core/DataFormat/ClusterVoxel3D.h"
#include "larcv/core/DataFormat/EventBase.h"
#include "larcv/core/DataFormat/DataProductFactory.h"

namespace larcv {

  
  /**
    \class EventClusterVoxel3D
    \brief Event-wise class to store a collection of VoxelSet (cluster) collection
  */
  class EventClusterVoxel3D : public EventBase,
			      public ClusterVoxel3D {

  public:

    /// Default constructor
    EventClusterVoxel3D() {}

    /// Default destructor
    virtual ~EventClusterVoxel3D() {}

    /// EventBase::clear() override
    inline void clear() {EventBase::clear(); ClusterVoxel3D::clear_data();}
    
  };

  // Template instantiation for IO
  // template<> inline std::string product_unique_name<larcv::EventClusterVoxel3D>() { return "cluster3d"; }
  // template EventClusterVoxel3D& IOManager::get_data<larcv::EventClusterVoxel3D>(const std::string&);
  // template EventClusterVoxel3D& IOManager::get_data<larcv::EventClusterVoxel3D>(const ProducerID_t);

  /**
     \class larcv::EventClusterVoxel3D
     \brief A concrete factory class for larcv::EventClusterVoxel3D
  */
  class EventClusterVoxel3DFactory : public DataProductFactoryBase {
  public:
    /// ctor
    EventClusterVoxel3DFactory()
    {
      //DataProductFactory::get().add_factory(product_unique_name<larcv::EventClusterVoxel3D>(), this);
      DataProductFactory::get().add_factory(kProductClusterVoxel3D, this);      
    }
    /// dtor
    ~EventClusterVoxel3DFactory() {}
    /// create method
    EventBase* create() { return new EventClusterVoxel3D; }
  };
  
}

#endif

