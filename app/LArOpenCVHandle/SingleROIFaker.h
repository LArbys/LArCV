/**
 * \file SingleROIFaker.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class SingleROIFaker
 *
 * @author vgenty
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __SINGLEROIFAKER_H__
#define __SINGLEROIFAKER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class SingleROIFaker ... these comments are used to generate
     doxygen documentation!
  */
  class SingleROIFaker : public ProcessBase {

  public:
    
    /// Default constructor
    SingleROIFaker(const std::string name="SingleROIFaker");
    
    /// Default destructor
    ~SingleROIFaker(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    std::string _img_producer;
    std::string _roi_producer;

  };

  /**
     \class larcv::SingleROIFakerFactory
     \brief A concrete factory class for larcv::SingleROIFaker
  */
  class SingleROIFakerProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    SingleROIFakerProcessFactory() { ProcessFactory::get().add_factory("SingleROIFaker",this); }
    /// dtor
    ~SingleROIFakerProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new SingleROIFaker(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

