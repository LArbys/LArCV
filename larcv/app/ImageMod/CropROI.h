/**
 * \file CropROI.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class CropROI
 *
 * @author kazuhiro
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __CROPROI_H__
#define __CROPROI_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class CropROI ... these comments are used to generate
     doxygen documentation!
  */
  class CropROI : public ProcessBase {

  public:
    
    /// Default constructor
    CropROI(const std::string name="CropROI");
    
    /// Default destructor
    ~CropROI(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  private:

    size_t _roi_idx;
    std::string _roi_producer;
    std::string _input_producer;
    std::string _output_producer;
    std::vector<size_t> _image_idx;
  };

  /**
     \class larcv::CropROIFactory
     \brief A concrete factory class for larcv::CropROI
  */
  class CropROIProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    CropROIProcessFactory() { ProcessFactory::get().add_factory("CropROI",this); }
    /// dtor
    ~CropROIProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new CropROI(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

