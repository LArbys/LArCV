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
#include "larcv/core/DataFormat/IOManager.h"
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
    virtual ~ProcessDriver();

    void reset();

    void set_iomanager( IOManager* io ) { _io = io; _external_io=true; };

    void configure(const std::string config_file);

    void configure(const PSet& cfg);

    void initialize();

    void override_input_file(const std::vector<std::string>& flist);

    void override_output_file(const std::string fname);

    void override_ana_file(const std::string fname);

    bool process_entry(bool autosave_entry=true);
    
    bool process_entry(size_t entry,bool force_reload=false, bool autosave_entry=true);

    const EventBase& event_id() const { return _io->last_event_id(); }

    void batch_process(size_t start_entry=0, size_t num_entries=0);

    void finalize();

    void random_access(bool doit) { _random_access = doit; }

    ProcessID_t process_id(std::string name) const;

    std::vector<std::string> process_names() const;

    const std::map<std::string,size_t>& process_map() const;

    const ProcessBase* process_ptr(ProcessID_t id) const;

    const IOManager& io() const { return *_io; }

    IOManager& io_mutable() { return *_io; }

    void set_id(size_t run, size_t subrun, size_t event)
    { _io->set_id(run,subrun,event); }

    size_t get_tree_index( size_t entry ) const;

    ProcessBase* process_ptr_mutable( ProcessID_t id );

  protected:

    bool _process_entry_( bool autosave_entry=true );
    bool _run_process_( ProcessBase* p ) { return p->process( *_io ); };
    
#ifndef __CINT__
    size_t _current_entry;
    bool _enable_filter;
    bool _random_access;
    bool _process_good_status;
    bool _process_cleared; 
    IOManager* _io;   
    std::vector<size_t> _access_entry_v;
    std::map<std::string,larcv::ProcessID_t> _proc_m;
    std::vector<larcv::ProcessBase*> _proc_v;
    bool _processing;
    TFile* _fout;    
    std::string _fout_name;

    size_t _batch_start_entry;
    size_t _batch_num_entry;

    bool _external_io; ///< true if using an externally-provided IOManager
    bool _has_event_creator;
#endif
  };
}

#endif
/** @} */ // end of doxygen group 

