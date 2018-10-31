/**
 * \file SSNetChecker.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class SSNetChecker
 *
 * @author vgenty
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __SSNETCHECKER_H__
#define __SSNETCHECKER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

#include "TTree.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class SSNetChecker ... these comments are used to generate
     doxygen documentation!
  */
  class SSNetChecker : public ProcessBase {

  public:
    
    /// Default constructor
    SSNetChecker(const std::string name="SSNetChecker");
    
    /// Default destructor
    ~SSNetChecker(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    std::string _rse_producer;
    std::string _adc_producer;
    std::string _ssnet_producer;
    std::string _roi_producer;

    TTree* _outtree;

    int _run;
    int _subrun;
    int _event;
    int _entry;
    int _broken;
    int _valid_roi;
    std::string _fname;

    void SetFileName(const std::string& s) { _fname = s; }
  };

  /**
     \class larcv::SSNetCheckerFactory
     \brief A concrete factory class for larcv::SSNetChecker
  */
  class SSNetCheckerProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    SSNetCheckerProcessFactory() { ProcessFactory::get().add_factory("SSNetChecker",this); }
    /// dtor
    ~SSNetCheckerProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new SSNetChecker(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

