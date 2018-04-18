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

  private:

    bool _mc_available;
    
    bool MCSelect(const EventROI* ev_roi);
    
    std::string _true_roi_producer_name;
    std::string _reco_roi_producer_name;
    std::string _rse_producer;

    int _nu_pdg;
    int _interaction_mode;
    
    double _min_nu_init_e;
    double _max_nu_init_e;
    double _dep_sum_lepton;
    double _dep_sum_proton;

    bool _select_signal;
    bool _select_background;
    
    struct aparticle{
      int pdg;
      int trackid;
      int ptrackid;
      bool primary;
      float depeng;
    };

    int _n_fail_nupdg;
    int _n_fail_ccqe;
    int _n_fail_nuE;
    int _n_fail_lepton_dep;
    int _n_fail_proton_dep;
    int _n_pass;
    int _n_calls;
    int _n_fail_unknowns;
    int _n_fail_inter;
    
    TTree* _event_tree;

    int _run;
    int _subrun;
    int _event; 
    int _entry;
    int _number_croi;
    int _selected;
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

