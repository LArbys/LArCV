/**
 * \file RunFilter.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class RunFilter
 *
 * @author drinkingkazu
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __RUNFILTER_H__
#define __RUNFILTER_H__

#include "ProcessBase.h"
#include "ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class RunFilter ... these comments are used to generate
     doxygen documentation!
  */
  class RunFilter : public ProcessBase {

  public:
    
    /// Default constructor
    RunFilter(const std::string name="RunFilter");
    
    /// Default destructor
    ~RunFilter(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  private:
    std::set<size_t> _run_s; ///< a set of runs to be filtered out
    std::string _producer;   ///< a producer name
    ProductType_t _type;     ///< a product type 
  };

  /**
     \class larcv::RunFilterFactory
     \brief A concrete factory class for larcv::RunFilter
  */
  class RunFilterProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    RunFilterProcessFactory() { ProcessFactory::get().add_factory("RunFilter",this); }
    /// dtor
    ~RunFilterProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new RunFilter(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

