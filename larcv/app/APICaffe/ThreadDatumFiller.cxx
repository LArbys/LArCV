#ifndef THREADDATUMFILLER_CXX
#define THREADDATUMFILLER_CXX

#include "ThreadDatumFiller.h"
#include "Base/LArCVBaseUtilFunc.h"
#include <random>
#include <sstream>
#include <unistd.h>

namespace larcv {
  ThreadDatumFiller::ThreadDatumFiller(std::string name)
    : larcv_base(name)
    , _processing(false)
    , _thread_state(kThreadStateIdle)
    , _random_access(false)
    , _configured(false)
    , _enable_filter(false)
    , _num_processed(0)
    , _dim_v(4, 0)
    , _driver(name + "ProcessDriver")
    , _filler(nullptr)
    , _th()
    , _optional_next_index(kINVALID_SIZE)
  {}

  ThreadDatumFiller::~ThreadDatumFiller()
  {
    if (_th.joinable()) _th.join();
    if (_processing) _driver.finalize();
  }

  void ThreadDatumFiller::set_next_index(size_t index)
  {
    if (thread_running()) {
      LARCV_CRITICAL() << "Cannot set next index while thread is running!" << std::endl;
      throw larbys();
    }
    if( _optional_next_index_v.size() ) {
      LARCV_CRITICAL() << "Next batch indecies already set! Cannot call this function..." << std::endl;
      throw larbys();
    }
    _optional_next_index = index;
  }

  void ThreadDatumFiller::set_next_batch(const std::vector<size_t>& index_v)
  {
    if (thread_running()) {
      LARCV_CRITICAL() << "Cannot set next index while thread is running!" << std::endl;
      throw larbys();
    }
    if( _optional_next_index != kINVALID_SIZE ) {
      LARCV_CRITICAL() << "Next batch indecies already set! Cannot call this function..." << std::endl;
      throw larbys();
    }
    _optional_next_index_v = index_v;
  }

  void ThreadDatumFiller::reset()
  {
    if (_processing) {
      LARCV_NORMAL() << "Finalizing..." << std::endl;
      _driver.finalize();
    }
    _filler = nullptr;
    _driver.reset();
    _configured = false;
    _processing = false;
    _num_processed = 0;
    _optional_next_index = kINVALID_SIZE;
  }

  void ThreadDatumFiller::configure(const std::string config_file)
  {
    LARCV_DEBUG() << "Called" << std::endl;
    // check state
    if (_processing) {
      LARCV_CRITICAL() << "Must call finalize() before calling initialize() after starting to process..." << std::endl;
      throw larbys();
    }
    // check cfg file
    if (config_file.empty()) {
      LARCV_CRITICAL() << "Config file not set!" << std::endl;
      throw larbys();
    }

    // check cfg content top level
    auto main_cfg = CreatePSetFromFile(config_file);
    if (!main_cfg.contains_pset(name())) {
      LARCV_CRITICAL() << "ThreadDatumFiller configuration (" << name() << ") not found in the config file (dump below)" << std::endl
                       << main_cfg.dump()
                       << std::endl;
      throw larbys();
    }
    auto const cfg = main_cfg.get<larcv::PSet>(name());
    configure(cfg);
  }

  void ThreadDatumFiller::configure(const PSet& orig_cfg)
  {

    reset();
    PSet cfg(_driver.name());
    for (auto const& value_key : orig_cfg.value_keys())
      cfg.add_value(value_key, orig_cfg.get<std::string>(value_key));
    std::cout<<"\033[93m setting verbosity \033[00m" << cfg.get<unsigned short>("Verbosity", 2) << std::endl;
    set_verbosity( (msg::Level_t)(cfg.get<unsigned short>("Verbosity", 2)) );
    _enable_filter = cfg.get<bool>("EnableFilter");
    _random_access = cfg.get<bool>("RandomAccess");
    _use_threading = cfg.get<bool>("UseThread", true);
    _input_fname_v = cfg.get<std::vector<std::string> >("InputFiles");

    // Brew read-only configuration
    PSet io_cfg("IOManager");
    std::stringstream ss;
    ss << logger().level();
    io_cfg.add_value("Verbosity", ss.str());
    io_cfg.add_value("Name", name() + "IOManager");
    io_cfg.add_value("IOMode", "0");
    io_cfg.add_value("OutFileName", "");
    io_cfg.add_value("StoreOnlyType", "[]");
    io_cfg.add_value("StoreOnlyName", "[]");

    for (auto const& pset_key : orig_cfg.pset_keys()) {
      if (pset_key == "IOManager") {
	auto const& orig_io_cfg = orig_cfg.get_pset(pset_key);
	if(orig_io_cfg.contains_value("ReadOnlyName"))
	  io_cfg.add_value("ReadOnlyName", orig_io_cfg.get<std::string>("ReadOnlyName"));
	if(orig_io_cfg.contains_value("ReadOnlyType"))
	  io_cfg.add_value("ReadOnlyType", orig_io_cfg.get<std::string>("ReadOnlyType"));
        LARCV_NORMAL() << "IOManager configuration will be ignored..." << std::endl;
      } else { cfg.add_pset(orig_cfg.get_pset(pset_key)); }
    }
    cfg.add_pset(io_cfg);

    // Configure the driver
    _driver.configure(cfg);

    // override input file
    _driver.override_input_file(_input_fname_v);

    // override random access to be false always
    _driver.random_access(false);

    // Make sure event_creator does not exist
    ProcessID_t last_process_id = 0;
    ProcessID_t datum_filler_id = kINVALID_SIZE;
    for (auto const& process_name : _driver.process_names()) {

      ProcessID_t id = _driver.process_id(process_name);

      if (id > last_process_id) last_process_id = id;

      auto ptr = _driver.process_ptr(id);

      LARCV_INFO() << "Process " << process_name << " = DatumFiller: " << ptr->is("DatumFiller") << std::endl;

      if (ptr->is("DatumFiller")) {
        if (datum_filler_id != kINVALID_SIZE) {
          LARCV_CRITICAL() << "Duplicate DatumFillers: id=" << datum_filler_id
                           << " vs. id=" << id << std::endl;
          throw larbys();
        }
        datum_filler_id = id;
      }
    }

    if (datum_filler_id == kINVALID_SIZE) {
      LARCV_CRITICAL() << "DatumFiller not found in process list..." << std::endl;
      throw larbys();
    }

    if (datum_filler_id != last_process_id) {
      LARCV_CRITICAL() << "DatumFiller not the last process..." << std::endl;
      throw larbys();
    }
    // Retrieve the filler ptr
    _filler = (DatumFillerBase*)(_driver.process_ptr(datum_filler_id));
    _configured = true;
  }

  const std::vector<int> ThreadDatumFiller::dim(bool image) const
  {
    if (!_processing) {
      LARCV_CRITICAL() << "Dimension is not known before start processing!" << std::endl;
      throw larbys();
    }
    if (thread_running()) {
      LARCV_CRITICAL() << "Thread is currently running (cannot retrieve data)" << std::endl;
      throw larbys();
    }
    return _filler->dim(image);
  }

  const std::string& ThreadDatumFiller::producer(DatumFillerBase::FillerDataType_t dtype) const
  {
    if (!_processing) {
      LARCV_CRITICAL() << "Data is not available before start processing!" << std::endl;
      throw larbys();
    }
    if (thread_running()) {
      LARCV_CRITICAL() << "Thread is currently running (cannot retrieve data)" << std::endl;
      throw larbys();
    }
    return _filler->producer(dtype);
  }

  const std::vector<float>& ThreadDatumFiller::data() const
  {
    if (!_processing) {
      LARCV_CRITICAL() << "Data is not available before start processing!" << std::endl;
      throw larbys();
    }
    if (thread_running()) {
      LARCV_CRITICAL() << "Thread is currently running (cannot retrieve data)" << std::endl;
      throw larbys();
    }
    return _filler->data(DatumFillerBase::kFillerImageData);
  }

  const std::vector<float>& ThreadDatumFiller::labels() const
  {
    if (!_processing) {
      LARCV_CRITICAL() << "Label is not available before start processing!" << std::endl;
      throw larbys();
    }
    if (thread_running()) {
      LARCV_CRITICAL() << "Thread is currently running (cannot retrieve data)" << std::endl;
      throw larbys();
    }
    return _filler->data(DatumFillerBase::kFillerLabelData);
  }


  const std::vector<float>& ThreadDatumFiller::multiplicities() const
  {
    if (!_processing) {
      LARCV_CRITICAL() << "Multiplicities is not available before start processing!" << std::endl;
      throw larbys();
    }
    if (thread_running()) {
      LARCV_CRITICAL() << "Thread is currently running (cannot retrieve data)" << std::endl;
      throw larbys();
    }
    return _filler->data(DatumFillerBase::kFillerMultiplicityData);
  }
  
  const std::vector<float>& ThreadDatumFiller::weights() const
  {
    if (!_processing) {
      LARCV_CRITICAL() << "Weight is not available before start processing!" << std::endl;
      throw larbys();
    }
    if (thread_running()) {
      LARCV_CRITICAL() << "Thread is currently running (cannot retrieve data)" << std::endl;
      throw larbys();
    }
    return _filler->data(DatumFillerBase::kFillerWeightData);
  }
  
  const std::vector<std::vector<larcv::ImageMeta> >& ThreadDatumFiller::meta() const
  {
    if (!_processing) {
      LARCV_CRITICAL() << "Weight is not available before start processing!" << std::endl;
      throw larbys();
    }
    if (thread_running()) {
      LARCV_CRITICAL() << "Thread is currently running (cannot retrieve data)" << std::endl;
      throw larbys();
    }
    return _filler->meta();
  }
  

  bool ThreadDatumFiller::batch_process(size_t nentries)
  {
    LARCV_DEBUG() << " start" << std::endl;
    if(nentries && _optional_next_index_v.size() && nentries != _optional_next_index_v.size()) {
      LARCV_CRITICAL() << "# entries specified != size of specified next-batch indicies!" << std::endl;
      throw larbys();
    }
    if(!nentries) nentries = _optional_next_index_v.size();
    
    if (thread_running()) return false;
    if (_th.joinable()) {
      LARCV_INFO() << "Thread has finished running but not joined. "
                   << "You might want to retrieve data?" << std::endl;
      _th.join();
    }
    if (!_processing && !_configured) {
      LARCV_CRITICAL() << "Must call configure() before run process!" << std::endl;
      throw larbys();
    }
    if ( _use_threading ) {
      LARCV_INFO() << "Instantiating thread..." << std::endl;
      _thread_state = kThreadStateStarting;
      std::thread t(&ThreadDatumFiller::_batch_process_, this, nentries);
      _th = std::move(t);
      usleep(1000);
      while(!_processing) usleep(500);
    }
    else {
      LARCV_INFO() << "No Thread..." << std::endl;
      _batch_process_( nentries );
    }
    return true;
  }

  bool ThreadDatumFiller::_batch_process_(size_t nentries)
  {
    LARCV_DEBUG() << " start" << std::endl;
    _thread_state = kThreadStateRunning;
    _filler->_nentries = nentries;
    if (!_processing) {
      LARCV_INFO() << "Initializing for 1st time processing" << std::endl;
      _driver.initialize();
      _processing = true;
    }

    size_t last_entry = kINVALID_SIZE - 1;
    if (_batch_entries.size()) last_entry = _batch_entries.back();
    _batch_entries.resize(nentries, 0);
    _batch_events.clear();
    _batch_events.reserve(nentries);

    _filler->_nentries = nentries;
    _filler->batch_begin();

    size_t valid_ctr = 0;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, _driver.io().get_n_entries() - 1);
    if (_random_access)
      LARCV_INFO() << "Generating random numbers from 0 to " << _driver.io().get_n_entries() << std::endl;

    // counter for _optional_next_index_v
    size_t next_batch_ctr = 0;
    
    LARCV_INFO() << "Entering process loop" << std::endl;
    while (valid_ctr < nentries) {
      size_t entry = last_entry + 1;
      if(_optional_next_index_v.size()) {
	if(next_batch_ctr >= _optional_next_index_v.size())
	  break;
	entry = _optional_next_index_v[next_batch_ctr];
	++next_batch_ctr;
      }else{
	if (_optional_next_index != kINVALID_SIZE) {
	  entry = _optional_next_index;
	  _optional_next_index = kINVALID_SIZE;
	}
	if (entry == kINVALID_SIZE) entry = 0;

	if (_random_access) {
	  entry = dis(gen);
	  while (entry == last_entry) entry = dis(gen);
	}
	else if (entry >= _driver.io().get_n_entries()) entry -= _driver.io().get_n_entries();

      }

      LARCV_INFO() << "Processing entry: " << entry
		   << " (tree index=" << _driver.get_tree_index( entry ) << ")" << std::endl;

      last_entry = entry;
      bool good_status = _driver.process_entry(entry, true);
      if (_enable_filter && !good_status) {
        LARCV_INFO() << "Filter enabled: bad event found" << std::endl;
        continue;
      }
      LARCV_INFO() << "Finished processing event id: " << _driver.event_id().event_key() << std::endl;

      _batch_entries[valid_ctr] = _driver.get_tree_index( entry );
      _batch_events.push_back(_driver.event_id());
      ++valid_ctr;
      LARCV_INFO() << "Processed good event: valid entry counter = " << valid_ctr << " : " << _batch_events.size() << std::endl;
    }
    _num_processed += valid_ctr;
    _filler->batch_end();
    _thread_state = kThreadStateIdle;
    _optional_next_index = kINVALID_SIZE;
    _optional_next_index_v.clear();
    LARCV_DEBUG() << " end" << std::endl;
    return true;
  }
}

#endif
