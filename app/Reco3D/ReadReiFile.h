/**
 * \file ReadReiFile.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class ReadReiFile
 *
 * @author vgenty
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __READREIFILE_H__
#define __READREIFILE_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class ReadReiFile ... these comments are used to generate
     doxygen documentation!
  */
  class ReadReiFile : public ProcessBase {

  public:
    
    /// Default constructor
    ReadReiFile(const std::string name="ReadReiFile");
    
    /// Default destructor
    ~ReadReiFile(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  };

  /**
     \class larcv::ReadReiFileFactory
     \brief A concrete factory class for larcv::ReadReiFile
  */
  class ReadReiFileProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    ReadReiFileProcessFactory() { ProcessFactory::get().add_factory("ReadReiFile",this); }
    /// dtor
    ~ReadReiFileProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new ReadReiFile(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

