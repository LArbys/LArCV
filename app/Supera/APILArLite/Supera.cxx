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

    event_wire* wire_h = nullptr;
    event_hit*  hit_h  = nullptr;

    if(!_core.producer_wire().empty()) {

      wire_h = storage->get_data<event_wire>(_core.producer_wire());

      if(!wire_h) { throw DataFormatException("Could not load wire data!"); }

    }

    if(!_core.producer_hit().empty()) {

      hit_h = storage->get_data<event_hit>(_core.producer_hit());

      if(!hit_h) { throw DataFormatException("Could not load hit data!"); }

    }

    if(!hit_h && !wire_h)
      throw DataFormatException("No wire nor hit producer specified...");
    if(hit_h && wire_h) 
      throw DataFormatException("Both hit and wire producer specified (one has to be empty)");

    auto opdigit_h = storage->get_data<event_opdetwaveform>(_core.producer_opdigit());

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
        std::vector<larlite::wire> empty_wire;
        std::vector<larlite::hit> empty_hit;
        if(opdigit_h) {
          if(wire_h)
            status = _core.process_event(*opdigit_h, *wire_h, empty_hit, 
              *mctruth_h, *mctrack_h, *mcshower_h, empty_simch);
          else
            status = _core.process_event(*opdigit_h, empty_wire, *hit_h, 
              *mctruth_h, *mctrack_h, *mcshower_h, empty_simch);
        }
        else{
          std::vector<larlite::opdetwaveform> empty_opdigit;
          if(wire_h)
            status = _core.process_event(empty_opdigit, *wire_h, empty_hit, 
              *mctruth_h, *mctrack_h, *mcshower_h, empty_simch);
          else
            status = _core.process_event(empty_opdigit, empty_wire, *hit_h, 
              *mctruth_h, *mctrack_h, *mcshower_h, empty_simch);
        }	
      }else{

        auto simch_h = storage->get_data<event_simch>(_core.producer_simch());
        if(!simch_h) throw DataFormatException("SimChannel requested but not available");

        std::vector<larlite::wire> empty_wire;
        std::vector<larlite::hit> empty_hit;
        if(opdigit_h) {
          if(wire_h)
            status = _core.process_event(*opdigit_h, *wire_h, empty_hit, 
              *mctruth_h, *mctrack_h, *mcshower_h, *simch_h);
          else
            status = _core.process_event(*opdigit_h, empty_wire, *hit_h, 
              *mctruth_h, *mctrack_h, *mcshower_h, *simch_h);
        }
        else{
          std::vector<larlite::opdetwaveform> empty_opdigit;
          if(wire_h)
            status = _core.process_event(empty_opdigit, *wire_h, empty_hit, 
              *mctruth_h, *mctrack_h, *mcshower_h, *simch_h);
          else
            status = _core.process_event(empty_opdigit, empty_wire, *hit_h, 
              *mctruth_h, *mctrack_h, *mcshower_h, *simch_h);
        } 
      }
    }else{
      std::vector<larlite::mctruth>  empty_mctruth;
      std::vector<larlite::mctrack>  empty_mctrack;
      std::vector<larlite::mcshower> empty_mcshower;
      std::vector<larlite::simch>    empty_simch;
      std::vector<larlite::wire> empty_wire;
      std::vector<larlite::hit> empty_hit;
      if(opdigit_h) {
        if(wire_h)
          status = _core.process_event(*opdigit_h, *wire_h, empty_hit, 
            empty_mctruth, empty_mctrack, empty_mcshower, empty_simch);
        else
          status = _core.process_event(*opdigit_h, empty_wire, *hit_h, 
            empty_mctruth, empty_mctrack, empty_mcshower, empty_simch);
      }
      else{
        std::vector<larlite::opdetwaveform> empty_opdigit;
        if(wire_h)
          status = _core.process_event(empty_opdigit, *wire_h, empty_hit, 
            empty_mctruth, empty_mctrack, empty_mcshower, empty_simch);
        else
          status = _core.process_event(empty_opdigit, empty_wire, *hit_h, 
            empty_mctruth, empty_mctrack, empty_mcshower, empty_simch);
      } 
    }
    return status;
  }

  bool Supera::finalize() {
    _core.finalize();
    return true;
  }

}
#endif
