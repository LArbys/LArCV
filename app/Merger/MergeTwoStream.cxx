#ifndef MERGETWOSTREAM_CXX
#define MERGETWOSTREAM_CXX

#include "MergeTwoStream.h"
#include "Base/UtilFunc.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventChStatus.h"

namespace larcv {

  MergeTwoStream::MergeTwoStream() : larcv_base("MergeTwoStream")
				   , _driver1("Stream1")
				   , _driver2("Stream2")
				   , _prepared(false)
				   , _num_proc1(0)
				   , _num_proc2(0)
  {}

  void MergeTwoStream::override_input_file(const std::vector<std::string> driver1,
					   const std::vector<std::string> driver2)
  {
    if(_prepared) {
      LARCV_CRITICAL() << "Cannot re-configure after initialized..." << std::endl;
      throw larbys();
    }
    if(driver1.size())
      _driver1.override_input_file(driver1);
    if(driver2.size())
      _driver2.override_input_file(driver2);
  }

  void MergeTwoStream::override_output_file(std::string out_fname)
  {
    if(_prepared) {
      LARCV_CRITICAL() << "Cannot re-configure after initialized..." << std::endl;
      throw larbys();
    }
    _io.set_out_file(out_fname);
  }

  void MergeTwoStream::configure(std::string cfg_file)
  {
    if(_prepared) {
      LARCV_CRITICAL() << "Cannot re-configure after initialized..." << std::endl;
      throw larbys();
    }
    auto main_cfg = CreatePSetFromFile(cfg_file);
    auto const& cfg = main_cfg.get_pset(name());
    if(!cfg.contains_pset("Stream1")) {
      LARCV_CRITICAL() << "Stream1 parameter set not found..." << std::endl;
      throw larbys();
    }
    if(!cfg.contains_pset("Stream2")) {
      LARCV_CRITICAL() << "Stream2 parameter set not found..." << std::endl;
      throw larbys();
    }
    if(!cfg.contains_pset("IOManager")) {
      LARCV_CRITICAL() << "Main IOManager parameter set not found..." << std::endl;
      throw larbys();
    }
    _proc1_name = cfg.get<std::string>("ImageHolder1");
    if(_proc1_name.empty()) {
      LARCV_CRITICAL() << "ImageHolder1 name is empty" << std::endl;
      throw larbys();
    }
    _proc2_name = cfg.get<std::string>("ImageHolder2");
    if(_proc2_name.empty()) {
      LARCV_CRITICAL() << "ImageHolder2 name is empty" << std::endl;
      throw larbys();
    }

    LARCV_WARNING() << "Note Stream1 should contain image to-be-overlayed with ROI (Stream2 is the base image)"<<std::endl;
    LARCV_NORMAL() << "Registered Image+ROI ImageHolder: Stream1::" << _proc1_name << std::endl;
    LARCV_NORMAL() << "Registered Image     ImageHolder: Stream2::" << _proc2_name << std::endl;

    set_verbosity((msg::Level_t)(cfg.get<unsigned short>("Verbosity",logger().level())));

    _io = IOManager(cfg.get_pset("IOManager"));
    _driver1.configure(cfg.get_pset("Stream1"));
    _driver2.configure(cfg.get_pset("Stream2"));
    _min_chstatus=cfg.get<short>("MinChannelStatus");
  }
  
  void MergeTwoStream::initialize()
  {
    _io.initialize();
    _driver1.initialize();
    _driver2.initialize();
    // retrieve image holder 1 & 2
    auto const id1 = _driver1.process_id(_proc1_name);
    _proc1 = (ImageHolder*)(_driver1.process_ptr(id1));
    auto const id2 = _driver2.process_id(_proc2_name);
    _proc2 = (ImageHolder*)(_driver2.process_ptr(id2));
    _prepared=true;
    _num_proc1 = 0;
    _num_proc2 = 0;
    _num_proc_max = std::min(_driver1.io().get_n_entries(),_driver2.io().get_n_entries());
    _num_proc_frac = _num_proc_max/10 + 1;
    LARCV_NORMAL() << "Processing max " << _num_proc_max << " entries..." << std::endl;
  }
  
  bool MergeTwoStream::process()
  {
    if(!_prepared) {
      LARCV_CRITICAL() << "Must call initialize() beore process()" << std::endl;
      throw larbys();
    }

    while(_num_proc1 < _num_proc_max) {
      ++_num_proc1;
      if(_driver1.process_entry()) break;
    }
    while(_num_proc2 < _num_proc_max) {
      ++_num_proc2;
      if(_driver2.process_entry()) break;
    }

    if(_num_proc1 >= _num_proc_max) return false;
    if(_num_proc2 >= _num_proc_max) return false;

    LARCV_INFO() << "Processing stream1 entry " << _num_proc1 
		 << " ... stream2 entry " << _num_proc2 << std::endl;

    size_t num_proc = std::max(_num_proc1,_num_proc2);
    if(!_num_proc_frac) {
      LARCV_NORMAL() << "Processing entry " << num_proc << "/" << _num_proc_max << std::endl;
    }else if(num_proc%_num_proc_frac==0){
      LARCV_NORMAL() << "Processing " << num_proc/_num_proc_frac * 10 << "%" << std::endl;
    }

    std::map<larcv::PlaneID_t,larcv::ChStatus> chstatus_m;
    std::vector<larcv::Image2D> image1_v;
    std::vector<larcv::Image2D> image2_v;

    auto const& roi1 = _proc1->roi();
    _proc1->move(image1_v);

    _proc2->move(image2_v);
    _proc2->move(chstatus_m);

    // Check size
    if(image1_v.size() != image2_v.size()) {
      LARCV_ERROR() << "# of stream1 image do not match w/ stream2! Skipping this entry..." << std::endl;
      return true;
    }
    if(roi1.BB().empty()) {
      LARCV_ERROR() << "No ROI found. skipping..." << std::endl;
      return true;
    }
    if(roi1.BB().size() != image1_v.size()) {
      LARCV_ERROR() << "# of image do not match w/ # of ROI! Skipping this entry..." << std::endl;
      return true;
    }

    // Check PlaneID
    for(size_t i=0; i<image1_v.size(); ++i) {
      auto const& image1 = image1_v[i];
      auto const& image2 = image2_v[i];
      if(image1.meta().plane() != image2.meta().plane()) {
	LARCV_ERROR() << "Plane ID mismatch! skipping..." << std::endl;
	return true;
      }
      if(chstatus_m.find(image1.meta().plane()) == chstatus_m.end()) {
	LARCV_ERROR() << "Plane ID " << image1.meta().plane() << " not found for ch status!" << std::endl;
	return true;
      }
    }

    // All check done
    auto event_image = (EventImage2D*)(_io.get_data(kProductImage2D,"merged"));
    for(size_t i=0; i<image1_v.size(); ++i) {

      // Overlay image
      auto& image1 = image1_v[i];
      auto& image2 = image2_v[i];
      auto const& bb = roi1.BB(image1.meta().plane());
      auto cropped = image1.crop(bb);
      image2.overlay(cropped);

      auto const& meta = image2.meta();
      std::vector<float> null_col(meta.rows(),0);
      // Impose ChStatus
      auto const& stat_v = chstatus_m[image1.meta().plane()].as_vector();
      for(size_t wire_num=0; wire_num < stat_v.size(); ++wire_num) {
	auto const& stat = stat_v[wire_num];
	if(stat < _min_chstatus) {
	  auto col = meta.col((double)wire_num);
	  image2.copy(0,col,null_col);
	}
      }
      
      event_image->Emplace(std::move(image2));
    }

    auto event_roi = (EventROI*)(_io.get_data(kProductROI,"merged"));
    event_roi->Append(roi1);

    _io.set_id(_proc2->run(),_proc2->subrun(),_proc2->event());
    _io.save_entry();
    
    return true;
  }

  void MergeTwoStream::batch_process()
  {
    while(1) if(!process()) break;
    LARCV_NORMAL() << "Finished 100%" << std::endl;
  }
  
  void MergeTwoStream::finalize()
  {
    if(!_prepared) {
      LARCV_CRITICAL() << "Must call initialize() beore process()" << std::endl;
      throw larbys();
    }

    _driver2.finalize();
    _driver1.finalize();
    _io.finalize();

  }

}
#endif
