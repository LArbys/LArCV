#ifndef THREADDATUMFILLER_CXX
#define THREADDATUMFILLER_CXX
#include <sstream>
#include <iomanip>
#include <random>
#include "ThreadDatumFiller.h"
#include "Processor/ProcessFactory.h"
#include "Base/LArCVBaseUtilFunc.h"
namespace larcv {

  ThreadDatumFiller::ThreadDatumFiller(std::string name)
    : larcv_base(name)
    , _filler(nullptr)
    , _enable_filter(false)
    , _random_access(false)
    , _io(IOManager::kREAD)
    , _proc_v()
    , _processing(false)
    , _fout(nullptr)
    , _fout_name("")
  {}

  void ThreadDatumFiller::reset()
  {
    LARCV_DEBUG() << "Called" << std::endl;
    _io.reset();
    _enable_filter = _random_access = false;
    for(size_t i=0; i<_proc_v.size(); ++i) { delete _proc_v[i]; _proc_v[i]=nullptr; }
    _proc_v.clear();
    _processing = false;
    _fout = nullptr;
    _fout_name = "";
    if(_filler) delete _filler;
  }

  void ThreadDatumFiller::override_input_file(const std::vector<std::string>& flist)
  {
    LARCV_DEBUG() << "Called" << std::endl;
    _io.clear_in_file();
    for(auto const& f : flist) _io.add_in_file(f);
  }

  ProcessID_t ThreadDatumFiller::process_id(std::string name) const
  {
    LARCV_DEBUG() << "Called" << std::endl;
    auto iter = _proc_m.find(name);
    if(iter == _proc_m.end()) {
      LARCV_CRITICAL() << "Process w/ name " << name << " not found..." << std::endl;
      throw larbys();
    }
    return (*iter).second;
  }

  std::vector<std::string> ThreadDatumFiller::process_names() const
  {
    LARCV_DEBUG() << "Called" << std::endl;
    std::vector<std::string> res;
    res.reserve(_proc_m.size());
    for(auto const& name_id : _proc_m) res.push_back(name_id.first);
    return res;
  }

  const std::map<std::string,larcv::ProcessID_t>& ThreadDatumFiller::process_map() const
  { LARCV_DEBUG() << "Called" << std::endl; return _proc_m; }

  const ProcessBase* ThreadDatumFiller::process_ptr(size_t id) const
  {
    LARCV_DEBUG() << "Called" << std::endl;
    if(id >= _proc_v.size()) {
      LARCV_CRITICAL() << "Invalid ID requested: " << id << std::endl;
      throw larbys();
    }
    return _proc_v[id];
  }

  void ThreadDatumFiller::configure(const std::string config_file)
  {
    LARCV_DEBUG() << "Called" << std::endl;
    // check state
    if(_processing) {
      LARCV_CRITICAL() << "Must call finalize() before calling initialize() after starting to process..." << std::endl;
      throw larbys();
    }
    // check cfg file
    if(config_file.empty()) {
      LARCV_CRITICAL() << "Config file not set!" << std::endl;
      throw larbys();
    }

    // check cfg content top level
    auto main_cfg = CreatePSetFromFile(config_file);
    if(!main_cfg.contains_pset(name())) {
      LARCV_CRITICAL() << "ThreadDatumFiller configuration (" << name() << ") not found in the config file (dump below)" << std::endl
		       << main_cfg.dump()
		       << std::endl;
      throw larbys();
    }
    auto const cfg = main_cfg.get<larcv::PSet>(name());
    configure(cfg);
  }

  void ThreadDatumFiller::configure(const PSet& cfg) 
  {
    // check process config exists
    if(!cfg.contains_pset("ProcessList")) {
      LARCV_CRITICAL() << "ProcessList config not found!" << std::endl
		       << cfg.dump() << std::endl;
      throw larbys();
    }
    // check if filler config exists
    auto filler_type = cfg.get<std::string>("FillerType","");
    if(filler_type.empty()) {
      LARCV_CRITICAL() << "FillerType not set!" << std::endl
            << cfg.dump() << std::endl;
      throw larbys();
    }
    if(!cfg.contains_pset(filler_type)) {
      LARCV_CRITICAL() << "Configuration for a filler " 
            << filler_type << " not found!" << std::endl
            << cfg.dump() << std::endl;
      throw larbys();
    }

    reset();
    LARCV_INFO() << "Retrieving IO config" << std::endl;
    auto const io_config = cfg.get<larcv::PSet>("IOManager");
    LARCV_INFO() << "Retrieving ProcessList" << std::endl;
    auto const proc_config = cfg.get<larcv::PSet>("ProcessList");

    // Prepare IO manager
    //LARCV_INFO() << "Configuring IO" << std::endl;

    // Set ThreadDatumFiller
    LARCV_INFO() << "Retrieving self (ThreadDatumFiller) config" << std::endl;
    set_verbosity((msg::Level_t)(cfg.get<unsigned short>("Verbosity",logger().level())));
    _enable_filter = cfg.get<bool>("EnableFilter");
    _random_access = cfg.get<bool>("RandomAccess");
    _fout_name = cfg.get<std::string>("AnaFile","");

    // Process list
    auto process_instance_type_v = cfg.get<std::vector<std::string> >("ProcessType");
    auto process_instance_name_v = cfg.get<std::vector<std::string> >("ProcessName");

    if(process_instance_type_v.size() != process_instance_name_v.size()) {
      LARCV_CRITICAL() << "ProcessType and ProcessName config parameters have different length! "
		       << "(" << process_instance_type_v.size() << " vs. " << process_instance_name_v.size() << ")" << std::endl;
      throw larbys();
    }

    LARCV_INFO() << "Start looping process list to instantiate processes" << std::endl;
    for(auto& p : _proc_v) if(p) { delete p; }
    _proc_v.clear();
    _proc_m.clear();

    for(size_t i=0; i<process_instance_type_v.size(); ++i) {
      auto const& name = process_instance_name_v[i];
      auto const& type = process_instance_type_v[i];
      if(_proc_m.find(name) != _proc_m.end()) {
	       LARCV_CRITICAL() << "Duplicate Process name found: " << name << std::endl;
	       throw larbys("Duplicate algorithm name found!");
      }
      size_t id = _proc_v.size();
      LARCV_NORMAL() << "Instantiating Process ID=" << id << " Type: " << type << " w/ Name: " << name << std::endl;

      auto ptr = ProcessFactory::get().create(type,name);
      //ptr->_id = id;
      ptr->_configure_(proc_config.get_pset(name));
      if(ptr->_event_creator) {
         LARCV_CRITICAL() << "Event creator is not allowed!" << std::endl;
         throw larbys();
       }
      _proc_m[name] = id;
      _proc_v.push_back(ptr);
    }

    LARCV_INFO() << "Configuring the filler: " << filler_type << std::endl;
    _filler = (DatumFillerBase*)(ProcessFactory::get().create(filer_type,filler_type));
    _filler->_configure_(cfg.get(filer_type));

    LARCV_DEBUG() << " done" << std::endl;
  }

  void ThreadDatumFiller::initialize()
  {
    LARCV_DEBUG() << "Called" << std::endl;
    // check state
    if(_processing) {
      LARCV_CRITICAL() << "Must call finalize() before calling initialize() after starting to process..." << std::endl;
      throw larbys();
    }

    // Initialize process
    for(auto& p : _proc_v) {
      LARCV_INFO() << "Initializing: " << p->name() << std::endl;
      p->initialize();
    }
    // Initialize filler
    LARCV_INFO() << "Initializing: " << _filler->name() << std::endl;
    _filler->initialize();

    // Initialize IO
    LARCV_INFO() << "Initializing IO " << std::endl;
    _io.initialize();

    // Handle invalid cases
    auto const nentries = _io.get_n_entries();

    // Prepare analysis output file if needed
    if(!_fout_name.empty()) {
      LARCV_NORMAL() << "Opening analysis output file " << _fout_name << std::endl;
      _fout = TFile::Open(_fout_name.c_str(),"RECREATE");
    }

    // Change state from to-be-initialized to to-process
    _processing = true;
    _batch_processing = false;
  }

  void ThreadDatumFiller::batch_process(size_t nentries)
  {
    LARCV_DEBUG() << " start" << std::endl;

    if(!_processing) {
      LARCV_CRITICAL() << "Must call initialize() before start processing!" << std::endl;
      throw larbys();
    }

    _batch_processing = true;
    size_t last_entry = 0;
    if(_batch_entries.size()) last_entry = _batch_entries.back();
    _batch_entries.resize(nentries,0);

    _filler->_nentries = nentries;
    _filler->batch_begin();

    size_t valid_ctr = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0,_io.get_n_entries());

    while(valid_ctr < nentries) {
      size_t entry = ++last_entry;
      if(_random_access) entry = dis(gen);
      else if(entry >= _io.get_n_entries()) entry -= _io.get_n_entries();

      _io.read_entry(entry);
      last_entry = entry;
      bool good_status=true;

      for(auto& p : _proc_v) {
        good_status = good_status && p->_process_(_io);
        if(!good_status && _enable_filter) break;
      }
      if(!good_status && _enable_filter) continue;

      good_status = _filler->process(_io);
      if(!good_status) continue;

      _batch_entries[valid_ctr] = entry;
      ++valid_ctr;
    }

    _filler->batch_end();
    _batch_processing = false;

    LARCV_DEBUG() << " end" << std::endl;
  }

  void ThreadDatumFiller::finalize()
  {
    LARCV_DEBUG() << "called" << std::endl;

    for(auto& p : _proc_v) {
      LARCV_INFO() << "Finalizing: " << p->name() << std::endl;
      p->finalize(_fout);
    }
    LARCV_INFO() << "Finalizing: " << _filler->name() << std::endl;
    _filler->finalize(_fout);

    // Profile repor
    LARCV_INFO() << "Compiling time profile..." << std::endl;
    std::stringstream ss;
    for(auto& p : _proc_v) {
      if(!p->_profile) continue;
      ss << "  \033[93m" << std::setw(20) << std::setfill(' ') << p->name() << "\033[00m"
	 << " ... # call " << std::setw(5) << std::setfill(' ') << p->_proc_count
	 << " ... total time " << p->_proc_time << " [s]"
	 << " ... average time " << p->_proc_time / p->_proc_count << " [s/process]"
	 << std::endl;
    }

    std::string msg(ss.str());
    if(!msg.empty()) 
      LARCV_NORMAL() << "Simple time profiling requested and run..." << std::endl
		     << "  ================== " << name() << " Profile Report ==================" << std::endl
		     << msg
		     << std::endl;

    if(_fout) {
      LARCV_NORMAL() << "Closing analysis output file..." << std::endl;
      _fout->Close();
    }

    LARCV_INFO() << "Finalizing IO..." << std::endl;
    _io.finalize();
    LARCV_INFO() << "Resetting..." << std::endl;
    reset();
  }

}

#endif
