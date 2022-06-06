#ifndef __LARCV_CORE_DATAFORMAT_EVENTSPARSETENSOR3D_H__
#define __LARCV_CORE_DATAFORMAT_EVENTSPARSETENSOR3D_H__

#include "DataProductFactory.h"
#include "DataFormatTypes.h"
#include "SparseTensor3D.h"

namespace larcv {

  /**
    \class EventSparseTensor3D
    \brief Event-wise class to store a collection of VoxelSet 
  */
  class EventSparseTensor3D : public EventBase,
			      public SparseTensor3D{

  public:

    /// Default constructor
    EventSparseTensor3D() {}

    /// Default destructor
    virtual ~EventSparseTensor3D() {}

    /// EventBase::clear() override
    inline void clear() {EventBase::clear(); SparseTensor3D::clear_data();}
    
  };
  
  /**
     \class larcv::EventSparseTensor3D
     \brief A concrete factory class for larcv::EventSparseTensor3D
  */
  class EventSparseTensor3DFactory : public DataProductFactoryBase {
  public:
    /// ctor
    EventSparseTensor3DFactory()
    {
      DataProductFactory::get().add_factory(kProductSparseTensor3D, this);
    }
    /// dtor
    ~EventSparseTensor3DFactory() {}
    /// create method
    EventBase* create() { return new EventSparseTensor3D; }
  };
  
  // Template instantiation for IO
  // template<> inline std::string product_unique_name<larcv::EventSparseTensor3D>() { return "sparse3d"; }
  // template EventSparseTensor3D& IOManager::get_data<larcv::EventSparseTensor3D>(const std::string&);
  // template EventSparseTensor3D& IOManager::get_data<larcv::EventSparseTensor3D>(const ProducerID_t);
  
}

#endif
