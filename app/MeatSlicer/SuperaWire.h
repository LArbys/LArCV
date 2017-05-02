/**
 * \file SuperaWire.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class SuperaWire
 *
 * @author kazuhiro
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __SUPERAWIRE_H__
#define __SUPERAWIRE_H__

#include "SuperaBase.h"
#include "FMWKInterface.h"
#include "DataFormat/Image2D.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class SuperaWire ... these comments are used to generate
     doxygen documentation!
  */
  class SuperaWire : public SuperaBase {

  public:
    
    /// Default constructor
    SuperaWire(const std::string name="SuperaWire");
    
    /// Default destructor
    ~SuperaWire(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  };

  /**
     \class larcv::SuperaWireFactory
     \brief A concrete factory class for larcv::SuperaWire
  */
  class SuperaWireProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    SuperaWireProcessFactory() { ProcessFactory::get().add_factory("SuperaWire",this); }
    /// dtor
    ~SuperaWireProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new SuperaWire(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

