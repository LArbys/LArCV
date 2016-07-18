/**
 * \file ProcessBase.h
 *
 * \ingroup Processor
 * 
 * \brief Class def header for a class ProcessBase
 *
 * @author drinkingkazu
 */

/** \addtogroup Processor

    @{*/
#ifndef PROCESSBASE_H
#define PROCESSBASE_H

#include "Base/Watch.h"
#include "DataFormat/IOManager.h"
#include "ProcessorTypes.h"
namespace larcv {

  class ProcessDriver;
  class ProcessFactory;
  /**
     \class ProcessBase
     User defined class ProcessBase ... these comments are used to generate
     doxygen documentation!
  */
  class ProcessBase : public larcv_base {
    friend class ProcessDriver;
    friend class ProcessFactory;

  public:
    
    /// Default constructor
    ProcessBase(const std::string name="ProcessBase");
    
    /// Default destructor
    virtual ~ProcessBase(){}

    virtual void configure(const PSet&) = 0;

    virtual void initialize() = 0;

    virtual bool process(IOManager& mgr) = 0;

    virtual void finalize() = 0;

    virtual bool is(const std::string question) const;

    bool event_creator() const
    { return _event_creator; }

    bool has_ana_file() const
    { return _fout != nullptr; }

    TFile& ana_file()
    { if(!_fout) throw larbys("ana file does not exist"); return *_fout; }

  private:

    void _configure_(const PSet&);

    bool _process_(IOManager& mgr);

    bool _event_creator;    ///< special flag to mark this algorithm an event creator
#ifndef __CINT__
    larcv::Watch _watch;    ///< algorithm profile stopwatch
#endif
    double _proc_time;      ///< algorithm execution time record (cumulative)
    size_t _proc_count;     ///< algorithm execution counter (cumulative)
    
    larcv::ProcessID_t _id; ///< unique algorithm identifier
    bool _profile;          ///< measure process time if profile flag is on
    std::string _typename;  ///< process type from factory
    TFile* _fout;           ///< output analysis file
  };
}

#endif
/** @} */ // end of doxygen group 

