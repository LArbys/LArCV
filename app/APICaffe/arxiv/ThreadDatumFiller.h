/**
 * \file ThreadDatumFiller.h
 *
 * \ingroup Processor
 * 
 * \brief Class def header for a class ThreadDatumFiller
 *
 * @author drinkingkazu
 */

/** \addtogroup Processor

    @{*/
#ifndef THREADDATUMFILLER_H
#define THREADDATUMFILLER_H

#include <vector>
#include "DataFormat/IOManager.h"
#include "DatumFillerBase.h"

namespace larcv {
  /**
     \class ThreadDatumFiller
     User defined class ThreadDatumFiller ... these comments are used to generate
     doxygen documentation!
  */
  class ThreadDatumFiller : public larcv_base {
    
  public:
    
    /// Default constructor
    ThreadDatumFiller(std::string name);
    
    /// Default destructor
     ThreadDatumFiller(){}

    void reset();

    void configure(const std::string config_file);

    void configure(const PSet& cfg);

    void initialize();

    void override_input_file(const std::vector<std::string>& flist);

    void batch_process(size_t nentries);

    void finalize();

    ProcessID_t process_id(std::string name) const;

    std::vector<std::string> process_names() const;

    const std::map<std::string,size_t>& process_map() const;

    const ProcessBase* process_ptr(ProcessID_t id) const;

    const IOManager& io() const { return _io; }

  private:

    bool _process_entry_();
#ifndef __CINT__
    DatumFillerBase* _filler;
    std::vector<size_t> _batch_entries;
    bool _enable_filter;
    bool _random_access;
    IOManager _io;
    std::map<std::string,larcv::ProcessID_t> _proc_m;
    std::vector<larcv::ProcessBase*> _proc_v;
    bool _processing;
    bool _batch_processing;
    TFile* _fout;    
    std::string _fout_name;
#endif
  };
}

#endif
/** @} */ // end of doxygen group 

