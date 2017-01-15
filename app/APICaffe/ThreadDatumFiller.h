/**
 * \file ThreadDatumFiller.h
 *
 * \ingroup APICaffe
 *
 * \brief Class def header for a class ThreadDatumFiller
 *
 * @author kazuhiro
 */

/** \addtogroup APICaffe

    @{*/
#ifndef THREADDATUMFILLER_H
#define THREADDATUMFILLER_H

#include "Processor/ProcessDriver.h"
#include "DatumFillerBase.h"
#include "APICaffeTypes.h"
#ifndef __CINT__
#include <thread>
#endif

namespace larcv {
  /**
     \class ThreadDatumFiller
     User defined class ThreadDatumFiller ... these comments are used to generate
     doxygen documentation!
  */
  class ThreadDatumFiller : public larcv_base {

  public:

    /// Default constructor
    ThreadDatumFiller(std::string name = "ThreadDatumFiller");

    /// Default destructor
    ~ThreadDatumFiller();

    /// copy ctor
    //ThreadDatumFiller(const ThreadDatumFiller& rhs) = delete;

    void reset();

    void configure(const std::string config_file);

    void configure(const PSet& cfg);

    bool batch_process(size_t nentries=0);

    void set_next_index(size_t index);

    void set_next_batch(const std::vector<size_t>& index_v);

    bool thread_config() const { return _use_threading; }

    bool thread_running() const { return (_thread_state != kThreadStateIdle); }

    size_t process_ctr() const { return _num_processed; }

    size_t get_n_entries() const { return _driver.io().get_n_entries(); }

    const std::vector<size_t>& processed_entries() const { return _batch_entries; }

    const std::vector<larcv::EventBase>& processed_events() const { return _batch_events; }

    const std::vector<int> dim(bool image=true) const;

    const std::vector<float>& data() const;

    const std::vector<float>& labels() const;

    const std::vector<float>& weights() const;

    const std::vector<std::vector<larcv::ImageMeta> >& meta() const;
    
    const ProcessDriver* pd() { return &_driver; }
    
  private:

    bool _batch_process_(size_t nentries);
    bool _processing;
    ThreadFillerState_t _thread_state;
    bool _use_threading;
    bool _random_access;
    bool _configured;
    bool _enable_filter;
    size_t _num_processed;
    std::vector<size_t> _batch_entries;
    std::vector<larcv::EventBase> _batch_events;
    std::vector<int> _dim_v;
    ProcessDriver _driver;

    DatumFillerBase* _filler;
    #ifndef __CINT__
    std::thread _th;
    #endif
    std::vector<std::string> _input_fname_v;
    size_t _optional_next_index;
    std::vector<size_t> _optional_next_index_v;
  };
}

#endif
/** @} */ // end of doxygen group

