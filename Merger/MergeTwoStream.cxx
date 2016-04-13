#ifndef MERGETWOSTREAM_CXX
#define MERGETWOSTREAM_CXX

#include "MergeTwoStream.h"
#include "Base/UtilFunc.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/ROI.h"
namespace larcv {

  MergeTwoStream::MergeTwoStream() : larcv_base("MergeTwoStream")
				   , _driver1("Stream1")
				   , _driver2("Stream2")
				   , _prepared(false)
  {}

  void MergeTwoStream::configure(std::string cfg_file)
  {
    if(_prepared) {
      LARCV_CRITICAL() << "Cannot re-configure after initialized..." << std::endl;
      throw larbys();
    }
    auto cfg = CreatePSetFromFile(cfg_file);
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

    _io = IOManager(cfg.get_pset("IOManager"));
    _driver1.configure(cfg.get_pset("Stream1"));
    _driver2.configure(cfg.get_pset("Stream2"));

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
  }
  
  void MergeTwoStream::process()
  {
    if(!_prepared) {
      LARCV_CRITICAL() << "Must call initialize() beore process()" << std::endl;
      throw larbys();
    }

    while(1) if(_driver1.process_entry()) break;
    while(1) if(_driver2.process_entry()) break;

    std::vector<larcv::Image2D> image1_v;
    std::vector<larcv::Image2D> image2_v;
    std::vector<larcv::Image2D> roi1_v;

    // Check size
    if(image1_v.size() != image2_v.size()) {
      LARCV_ERROR() << "# of stream1 image do not match w/ stream2! Skipping this entry..." << std::endl;
      return;
    }
    if(roi1_v.empty()) {
      LARCV_ERROR() << "No ROI found. skipping..." << std::endl;
      return;
    }

    // Check PlaneID
    for(size_t i=0; i<image1_v.size(); ++i) {
      auto const& image1 = image1_v[i];
      auto const& image2 = image2_v[i];
      if(image1.meta().plane() != image2.meta().plane()) {
	LARCV_ERROR() << "Plane ID mismatch! skipping..." << std::endl;
	return;
      }
    }

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
