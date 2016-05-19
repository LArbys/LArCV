/**
 * \file ProcessDriver.h
 *
 * \ingroup Processor
 * 
 * \brief Class def header for a class ProcessDriver
 *
 * @author drinkingkazu
 */

/** \addtogroup Processor

    @{*/
#ifndef PROCESSDRIVER_H
#define PROCESSDRIVER_H

#include <vector>
#include "DataFormat/IOManager.h"
#include "ProcessBase.h"

namespace larcv {
  /**
     \class ProcessDriver
     User defined class ProcessDriver ... these comments are used to generate
     doxygen documentation!
  */
  class ProcessDriver : public larcv_base {
    
  public:
    
    /// Default constructor
    ProcessDriver(std::string name);
    
    /// Default destructor
    ~ProcessDriver(){}

    void reset();

    void configure(const std::string config_file);

    void configure(const PSet& cfg);

    void initialize();

    void override_input_file(const std::vector<std::string>& flist);

    void override_output_file(const std::string fname);

    bool process_entry();
    
    bool process_entry(size_t entry,bool force_reload=false);

    void batch_process(size_t start_entry=0, size_t num_entries=0);

    void finalize();

    ProcessID_t process_id(std::string name) const;

    std::vector<std::string> process_names() const;

    const std::map<std::string,size_t>& process_map() const;

    const ProcessBase* process_ptr(ProcessID_t id) const;

    const IOManager& io() const { return _io; }

  private:

    bool _process_entry_();
#ifndef __CINT__
    size_t _current_entry;
    bool _enable_filter;
    bool _random_access;
    std::vector<size_t> _access_entry_v;
    IOManager _io;
    std::map<std::string,larcv::ProcessID_t> _proc_m;
    std::vector<larcv::ProcessBase*> _proc_v;
    bool _processing;
    TFile* _fout;    
    std::string _fout_name;
    bool _has_event_creator;
#endif
  };
}

#endif
/** @} */ // end of doxygen group 

