/**
 * \file RSEFilter.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class RSEFilter
 *
 * @author kazuhiro
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __RSEFILTER_H__
#define __RSEFILTER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include <map>
namespace larcv {

  class RSEID {
  public:
    RSEID(size_t run_val=0, size_t subrun_val=0, size_t event_val=0)
      : run(run_val)
      , subrun(subrun_val)
      , event(event_val)
    {}
    ~RSEID(){}

    inline bool operator < (const RSEID& rhs) const
    { if(run < rhs.run) return true;
      if(run > rhs.run) return false;
      if(subrun < rhs.subrun) return true;
      if(subrun > rhs.subrun) return false;
      if(event < rhs.event) return true;
      if(event > rhs.event) return false;
      return false;
    }

    inline bool operator == (const RSEID& rhs) const
    { return (run == rhs.run && subrun == rhs.subrun && event == rhs.event); }

    inline bool operator != (const RSEID& rhs) const
    { return !( (*this) == rhs ); }

    inline bool operator > (const RSEID& rhs) const
    { return ( (*this) != rhs && !((*this) < rhs) ); }

    size_t run, subrun, event;
  };

  /**
     \class ProcessBase
     User defined class RSEFilter ... these comments are used to generate
     doxygen documentation!
  */
  class RSEFilter : public ProcessBase {

  public:
    
    /// Default constructor
    RSEFilter(const std::string name="RSEFilter");
    
    /// Default destructor
    ~RSEFilter(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  private:

    TTree* _tree;

    std::map<larcv::RSEID,bool> _rse_m;
    std::string _ref_producer;
    size_t _ref_type;

    std::string _fname;
    int _run;
    int _subrun;
    int _event;

  };

  /**
     \class larcv::RSEFilterFactory
     \brief A concrete factory class for larcv::RSEFilter
  */
  class RSEFilterProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    RSEFilterProcessFactory() { ProcessFactory::get().add_factory("RSEFilter",this); }
    /// dtor
    ~RSEFilterProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new RSEFilter(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

