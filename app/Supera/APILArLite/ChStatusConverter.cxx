#ifndef __CHSTATUSCONVERTER_CXX__
#define __CHSTATUSCONVERTER_CXX__

#include "ChStatusConverter.h"
#include "DataFormat/EventChStatus.h"
#include "DataFormat/chstatus.h"
namespace larcv {

  static ChStatusConverterProcessFactory __global_ChStatusConverterProcessFactory__;

  ChStatusConverter::ChStatusConverter(const std::string name)
    : ProcessBase(name)
    , _io(::larlite::storage_manager::kWRITE)
  {}
    
  void ChStatusConverter::configure(const PSet& cfg)
  {
    _io.set_out_filename("out.root");
    _in_producer=cfg.get<std::string>("InputProducer");
    _out_producer=cfg.get<std::string>("OutputProducer");
  }

  void ChStatusConverter::initialize()
  {
    _io.open();
  }

  bool ChStatusConverter::process(IOManager& mgr)
  {

    static size_t entry=0;

    auto in_chstatus = (::larcv::EventChStatus*)(mgr.get_data(::larcv::kProductChStatus,_in_producer));

    auto out_chstatus = _io.get_data<larlite::event_chstatus>(_out_producer);

    _io.set_id(in_chstatus->run(),in_chstatus->subrun(),in_chstatus->event());

    for(auto const& plane_status : in_chstatus->ChStatusMap()) {

      ::larlite::chstatus status;
      ::larlite::geo::PlaneID pid(0,0,plane_status.first);
      status.set_status(pid,plane_status.second.as_vector());

      out_chstatus->emplace_back(std::move(status));
    }

    entry++;

    _io.next_event();
    return true;
  }

  void ChStatusConverter::finalize()
  {
    _io.close();
  }

}
#endif
