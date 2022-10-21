#ifndef __LARCV_CORE_DATAFORMAT_EVENTSPARSETENSOR2D_H__
#define __LARCV_CORE_DATAFORMAT_EVENTSPARSETENSOR2D_H__

#include "EventBase.h"
#include "DataProductFactory.h"
#include "DataFormatTypes.h"
#include "SparseTensor2D.h"

namespace larcv {

  /**
     \class EventSparseTensor2D
     \brief Event-wise class to store a collection of VoxelSet (cluster) per projection id
  */
  class EventSparseTensor2D : public larcv::EventBase {
    
  public:

    /// Default constructor
    EventSparseTensor2D() {}

    /// Default destructor
    virtual ~EventSparseTensor2D() {}

    //
    // Read-access
    //
    /// Access to all stores larcv::SparseTensor2D
    inline const std::vector<larcv::SparseTensor2D>& as_vector() const { return _tensor_v; }
    /// Access SparseTensor2D of a specific projection ID
    const larcv::SparseTensor2D& sparse_tensor_2d(const ProjectionID_t id) const;
    /// Number of valid projection id
    inline size_t size() const { return _tensor_v.size(); }

    //
    // Write-access
    //
    /// EventBase::clear() override
    inline void clear() {EventBase::clear(); _tensor_v.clear();}
    /// Emplace data
    void emplace(larcv::SparseTensor2D&& clusters);
    /// Set data
    void set(const larcv::SparseTensor2D& clusters);
    /// Emplace a new element
    void emplace(larcv::VoxelSet&& cluster, larcv::ImageMeta&& meta);
    /// Set a new element
    void set(const larcv::VoxelSet& cluster, const larcv::ImageMeta& meta);

  private:

    std::vector<larcv::SparseTensor2D> _tensor_v;
    
  };

  /**
     \class larcv::EventSparseTensor2D
     \brief A concrete factory class for larcv::EventSparseTensor2D
  */
  class EventSparseTensor2DFactory : public DataProductFactoryBase {
  public:
    /// ctor
    // EventSparseTensor2DFactory()
    //   { DataProductFactory::get().add_factory(product_unique_name<larcv::EventSparseTensor2D>(), this); }
    EventSparseTensor2DFactory()
    { DataProductFactory::get().add_factory(larcv::kProductSparseTensor2D, this); }
    /// dtor
    ~EventSparseTensor2DFactory() {}
    /// create method
    EventBase* create() { return new EventSparseTensor2D; }
  };
  
  
}


#endif
