/**
 * \file ChStatusConverter.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class ChStatusConverter
 *
 * @author kterao
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __CHSTATUSCONVERTER_H__
#define __CHSTATUSCONVERTER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "DataFormat/storage_manager.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class ChStatusConverter ... these comments are used to generate
     doxygen documentation!
  */
  class ChStatusConverter : public ProcessBase {

  public:
    
    /// Default constructor
    ChStatusConverter(const std::string name="ChStatusConverter");
    
    /// Default destructor
    ~ChStatusConverter(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    larlite::storage_manager _io;
    std::string _in_producer;
    std::string _out_producer;
  };

  /**
     \class larcv::ChStatusConverterFactory
     \brief A concrete factory class for larcv::ChStatusConverter
  */
  class ChStatusConverterProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    ChStatusConverterProcessFactory() { ProcessFactory::get().add_factory("ChStatusConverter",this); }
    /// dtor
    ~ChStatusConverterProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new ChStatusConverter(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

