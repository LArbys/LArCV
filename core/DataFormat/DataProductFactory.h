/**
 * \file DataProductFactory.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class DataProductFactory
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef __DATAPRODUCTFACTORY_H__
#define __DATAPRODUCTFACTORY_H__

#include <iostream>
#include <map>
#include "Base/larcv_base.h"
#include "Base/larbys.h"
#include "ProductMap.h"
#include "EventBase.h"
#include "DataFormatTypes.h"
#include "UtilFunc.h"
#include <sstream>
namespace larcv {

  class EventBase;
  /**
     \class DataProductFactoryBase
     \brief Abstract base class for factory (to be implemented per data product)
  */
  class DataProductFactoryBase {
  public:
    /// Default ctor
    DataProductFactoryBase(){}
    /// Default dtor (virtual)
    virtual ~DataProductFactoryBase(){}
    /// Abstract constructor method
    virtual EventBase* create() = 0;
  };

  /**
     \class ClusterAlgoFactory
     \brief Factory class for instantiating event product instance by larcv::IOManager
     This factory class can instantiate a specified product instance w/ provided producer name. \n
     The actual factory core (to which each product class must register creation factory instance) is \n
     a static std::map. Use static method to get a static instance (larcv::DataProductFactory::get) \n
     to access a factory.
  */
  class DataProductFactory : public larcv_base  {

  public:
    /// Default ctor, shouldn't be used
    DataProductFactory() : larcv_base("DataProductFactory")
    {}
    /// Default dtor
    ~DataProductFactory() {_factory_map.clear();}
    /// Static sharable instance getter
    static DataProductFactory& get()
    { if(!_me) _me = new DataProductFactory; return *_me; }
    /// Factory registration method (should be called by global factory instance in algorithm header)
    void add_factory(const ProductType_t type, larcv::DataProductFactoryBase* factory)
    {
      LARCV_INFO() << "Registering a factory " << factory << " of type " << ProductName(type) << std::endl;
      
      auto iter = _factory_map.find(type);

      if(iter != _factory_map.end()) {
	LARCV_CRITICAL() << "Attempted a duplicate registration of Data product "
			 << ProductName(type) << " to a factory!" << std::endl;
	throw larbys();
      }

      _factory_map[type] = factory;
    }
    /// Factory creation method (should be called by clients, possibly you!)
    EventBase* create(const ProductType_t type, const std::string producer) {
      auto iter = _factory_map.find(type);
      if(iter == _factory_map.end() || !((*iter).second)) {
	LARCV_ERROR() << "Found no registered class " << ProductName(type) << std::endl;
	return nullptr;
      }
      auto ptr = (*iter).second->create();
      ptr->_producer = producer;
      return ptr;
    }

    /// List registered products
    void list() const {
      std::stringstream ss;
      ss << "    Listing registered products:" << std::endl;
      for(auto const& type_factory : _factory_map) {
	ss << "    Type: " << type_factory.first
	   << " ... Name: " << ProductName(type_factory.first)
	   << " ... Factory @ " << type_factory.second
	   << std::endl;
      }
      ss << std::endl;
      LARCV_NORMAL() << ss.str() << std::endl;
    }

    std::string ROIType2String(const ROIType_t type)
    { return ROIType2String(type); }

    ROIType_t String2ROIType(const std::string& name)
    { return String2ROIType(name); }

  private:
    /// Static factory container
    std::map<larcv::ProductType_t,larcv::DataProductFactoryBase*> _factory_map;
    /// Static self
    static DataProductFactory* _me;
  };
}
#endif
/** @} */ // end of doxygen group 

