#ifndef LARLITE_CHSTATUSFINDER_CXX
#define LARLITE_CHSTATUSFINDER_CXX

#include "ChStatusFinder.h"
#include "DataFormat/chstatus.h"
#include "DataFormat/EventChStatus.h"

namespace larlite {

  bool ChStatusFinder::initialize() {

    _io.initialize();

    _num_entries = _io.get_n_entries();

    if(_out_producer.empty()) throw DataFormatException("Output producer forgotten...");
    if(_in_producer.empty()) throw DataFormatException("Input producer forgotten...");

    return true;
  }
  
  bool ChStatusFinder::analyze(storage_manager* storage) {

    static size_t entry=0;

    _io.read_entry(entry);
  
    auto in_chstatus = (::larcv::EventChStatus*)(_io.get_data(::larcv::kProductChStatus,_in_producer));

    auto out_chstatus = storage->get_data<event_chstatus>(_out_producer);

    storage->set_id(in_chstatus->run(),in_chstatus->subrun(),in_chstatus->event());

    for(auto const& plane_status : in_chstatus->ChStatusMap()) {

      ::larlite::chstatus status;
      ::larlite::geo::PlaneID pid(0,0,plane_status.first);
      status.set_status(pid,plane_status.second.as_vector());
      
      out_chstatus->emplace_back(std::move(status));
    }

    entry++;

    if(entry == _num_entries) {
      _io.finalize();
      throw DataFormatException("Finished");
    }
    
    return true;
  }

  bool ChStatusFinder::finalize() {

    // This function is called at the end of event loop.
    // Do all variable finalization you wish to do here.
    // If you need, you can store your ROOT class instance in the output
    // file. You have an access to the output file through "_fout" pointer.
    //
    // Say you made a histogram pointer h1 to store. You can do this:
    //
    // if(_fout) { _fout->cd(); h1->Write(); }
    //
    // else 
    //   print(MSG::ERROR,__FUNCTION__,"Did not find an output file pointer!!! File not opened?");
    //
  
    return true;
  }

}
#endif
