/**
 * \file MCinfoRetriever.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class MCinfoRetriever
 *
 * @author Rui
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __MCINFORETRIEVER_H__
#define __MCINFORETRIEVER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class MCinfoRetriever ... these comments are used to generate
     doxygen documentation!
  */
  class MCinfoRetriever : public ProcessBase {

  public:
    
    /// Default constructor
    MCinfoRetriever(const std::string name="MCinfoRetriever");
    
    /// Default destructor
    ~MCinfoRetriever(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  };

  /**
     \class larcv::MCinfoRetrieverFactory
     \brief A concrete factory class for larcv::MCinfoRetriever
  */
  class MCinfoRetrieverProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    MCinfoRetrieverProcessFactory() { ProcessFactory::get().add_factory("MCinfoRetriever",this); }
    /// dtor
    ~MCinfoRetrieverProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new MCinfoRetriever(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

