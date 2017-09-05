#ifndef __RSEFILTER_CXX__
#define __RSEFILTER_CXX__

#include "RSEFilter.h"
#include "CPPUtil/CSVReader.h"

namespace larcv {

  static RSEFilterProcessFactory __global_RSEFilterProcessFactory__;

  RSEFilter::RSEFilter(const std::string name)
    : ProcessBase(name), _tree(nullptr)
  {}
    
  void RSEFilter::configure(const PSet& cfg)
  {
    _ref_producer=cfg.get<std::string>("RefProducer");
    _ref_type=cfg.get<size_t>("RefType");
    
    auto file_path=cfg.get<std::string>("CSVFilePath");
    auto format=cfg.get<std::string>("Format","II");
    auto data = larcv::read_csv(file_path,format);
    auto const& run_v = data.get<int>("run");
    auto const& subrun_v = data.get<int>("subrun");
    auto const& event_v = data.get<int>("event");
    _rse_m.clear();
    RSEID rse_id;
    for(size_t i=0; i<run_v.size(); ++i) {
      rse_id.run = run_v[i];
      rse_id.subrun = subrun_v[i];
      rse_id.event = event_v[i];
      _rse_m[rse_id] = false;
    }
    LARCV_INFO() << "Registered: " << _rse_m.size() << " unique events to be kept..." << std::endl;
  }

  void RSEFilter::initialize()
  {
    
    _tree = new TTree("RSEFilter","");
    _tree->Branch("fname"  , &_fname);
    _tree->Branch("run"    , &_run   , "run/I");
    _tree->Branch("subrun" , &_subrun, "subrun/I");
    _tree->Branch("event"  , &_event , "event/I");
  }

  bool RSEFilter::process(IOManager& mgr)
  {
    auto ptr = mgr.get_data((larcv::ProductType_t)_ref_type,_ref_producer);

    auto const rse = RSEID(ptr->run(),ptr->subrun(),ptr->event());
    auto itr = _rse_m.find(rse);

    bool keepit = (itr != _rse_m.end());
    LARCV_INFO() << "Event key: " << ptr->event_key() << " ... keep it? " << keepit << std::endl;

    if(keepit) {
      if((*itr).second) LARCV_WARNING() << "Run " << rse.run << " Event " << rse.event << " is duplicated!!!" << std::endl;
      (*itr).second = true;
    }

    _fname  = (std::string) mgr.file_list().front();
    _run    = (int) ptr->run();
    _subrun = (int) ptr->subrun();
    _event  = (int) ptr->event();
    _tree->Fill();

    return keepit;
  }

  void RSEFilter::finalize()
  {
    for(auto const& rse_used : _rse_m) {
      if(rse_used.second) continue;
      LARCV_WARNING() << "Event ID not found in data file (unused): Run " << rse_used.first.run
		      << " Event " << rse_used.first.event << std::endl;
    }
    _tree->Write();
  }

}
#endif
