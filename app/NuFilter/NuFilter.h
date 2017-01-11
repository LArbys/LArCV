/**
 * \file Back2Back.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class Back2Back
 *
 * @author vgenty
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __NUFILTER_H__
#define __NUFILTER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "DataFormat/EventROI.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class NuFilter ... these comments are used to generate
     doxygen documentation!
  */
  class NuFilter : public ProcessBase {

  public:
    
    /// Default constructor
    NuFilter(const std::string name="NuFilter");
    
    /// Default destructor
    ~NuFilter(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    const bool selected() const { return _selected; }
    
  private:
    
    bool MCSelect(const EventROI* ev_roi);
    
    std::string _roi_producer_name;
    
    uint _nu_pdg;
    
    double _min_nu_init_e;
    double _max_nu_init_e;
    double _dep_sum_lepton;
    double _dep_sum_proton;

    bool _selected;
    bool _select_signal;
    bool _select_background;

    struct this_proton{
      int trackid;
      int parenttrackid;
      float depeng;
    };

    
  };

  /**
     \class larcv::NuFilterFactory
     \brief A concrete factory class for larcv::NuFilter
  */
  class NuFilterProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    NuFilterProcessFactory() { ProcessFactory::get().add_factory("NuFilter",this); }
    /// dtor
    ~NuFilterProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new NuFilter(instance_name); }
    
  };

}

#endif
/** @} */ // end of doxygen group 

