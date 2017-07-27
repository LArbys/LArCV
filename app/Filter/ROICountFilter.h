/**
 * \file ROICountFilter.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class ROICountFilter
 *
 * @author drinkingkazu
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __ROICOUNTFILTER_H__
#define __ROICOUNTFILTER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class ROICountFilter ... these comments are used to generate
     doxygen documentation!
  */
  class ROICountFilter : public ProcessBase {

  public:
    
    /// Default constructor
    ROICountFilter(const std::string name="ROICountFilter");
    
    /// Default destructor
    ~ROICountFilter(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  private:

    std::string _roi_producer;
    size_t _max_roi_count;
    std::vector<size_t> _roi_count_v;
  };

  /**
     \class larcv::ROICountFilterFactory
     \brief A concrete factory class for larcv::ROICountFilter
  */
  class ROICountFilterProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    ROICountFilterProcessFactory() { ProcessFactory::get().add_factory("ROICountFilter",this); }
    /// dtor
    ~ROICountFilterProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new ROICountFilter(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

