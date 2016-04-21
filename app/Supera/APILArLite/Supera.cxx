#ifndef LARLITE_SUPERA_CXX
#define LARLITE_SUPERA_CXX

#include "Supera.h"
#include "Base/LArCVBaseUtilFunc.h"
#include "DataFormat/chstatus.h"

namespace larlite {

  Supera::Supera() 
  {_name = "Supera"; _fout=nullptr;}

  bool Supera::initialize() {

    auto main_cfg = ::larcv::CreatePSetFromFile(_config_file).get<larcv::PSet>(_name);
    //std::cout<<main_cfg.dump()<<std::endl;

    _core.configure(main_cfg);

    _core.initialize();

    return true;
  }
  
  bool Supera::analyze(storage_manager* storage) {

    _core.clear_data();
    _core.set_id(storage->run_id(),storage->subrun_id(),storage->event_id());

    auto wire_h = storage->get_data<event_wire>(_core.producer_wire());

    if(!wire_h) { throw DataFormatException("Could not load wire data!"); }

    auto opdigit_h = storage->get_data<event_opdetwaveform>(_core.producer_opdigit());

    if(!opdigit_h) { throw DataFormatException("Could not load opdetwaveform data!"); }

    if(_core.store_chstatus()) {

      auto chstatus_h = storage->get_data<event_chstatus>(_core.producer_chstatus());

      if(!chstatus_h || chstatus_h->empty()) 

	throw ::larcv::larbys("ChStatus not found or empty!");

      for(auto const& chs : *chstatus_h) {

	auto const& pid = chs.plane();
	auto const& status_v = chs.status();
	for(size_t wire=0; wire<status_v.size(); ++wire)
	  _core.set_chstatus(pid.Plane,wire,status_v[wire]);
      }

    }

    bool status=true;
    if(_core.use_mc()) {

      auto mctruth_h  = storage->get_data<event_mctruth>(_core.producer_generator());
      auto mctrack_h  = storage->get_data<event_mctrack>(_core.producer_mcreco());
      auto mcshower_h = storage->get_data<event_mcshower>(_core.producer_mcreco());

      if(!mctruth_h || !mctrack_h || !mcshower_h) throw DataFormatException("Necessary MC info missing...");

      if(_core.producer_simch().empty()) {

	std::vector<larlite::simch> empty_simch;
	status = _core.process_event(*opdigit_h, *wire_h, *mctruth_h, *mctrack_h, *mcshower_h, empty_simch);
	
      }else{

	auto simch_h = storage->get_data<event_simch>(_core.producer_simch());
	if(!simch_h) throw DataFormatException("SimChannel requested but not available");
	status = _core.process_event(*opdigit_h, *wire_h, *mctruth_h, *mctrack_h, *mcshower_h, *simch_h);
	
      }
    }else{
      std::vector<larlite::mctruth>  empty_mctruth;
      std::vector<larlite::mctrack>  empty_mctrack;
      std::vector<larlite::mcshower> empty_mcshower;
      std::vector<larlite::simch>    empty_simch;
      status = _core.process_event(*opdigit_h, *wire_h,empty_mctruth,empty_mctrack,empty_mcshower,empty_simch);
    }
    return status;
  }

  bool Supera::finalize() {
    _core.finalize();
    return true;
  }

}
#endif
