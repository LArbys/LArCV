#ifndef LARLITE_SUPERA_CXX
#define LARLITE_SUPERA_CXX

#include "Supera.h"
#include "FhiclLite/ConfigManager.h"
#include "LArUtil/Geometry.h"
#include "DataFormat/wire.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/mctruth.h"
#include "DataFormat/simch.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include "Base/larbys.h"
#include "SuperaUtils.h"
#include "DataFormat/ProductMap.h"
#include "MCParticleTree.h"

namespace larlite {

  Supera::Supera() : _logger("Supera")		   
		   , _larcv_io(::larcv::IOManager::kWRITE)
  {_name = "Supera"; _fout=nullptr;}

  bool Supera::initialize() {

    ::fcllite::ConfigManager cfg_mgr(_name);

    auto geom = ::larutil::Geometry::GetME();

    cfg_mgr.AddCfgFile(_config_file);

    auto const& main_cfg = cfg_mgr.Config().get_pset(_name);

    std::cout<<main_cfg.dump()<<std::endl;

    _larcv_io.set_out_file(main_cfg.get<std::string>("OutFileName"));

    _producer_wire   = main_cfg.get<std::string>("WireProducer");
    _producer_gen    = main_cfg.get<std::string>("GenProducer");
    _producer_mcreco = main_cfg.get<std::string>("MCRecoProducer");

    _min_time = main_cfg.get<double>("MinTime");
    _min_wire = main_cfg.get<double>("MinWire");
    
    _event_image_rows = main_cfg.get<std::vector<size_t> >("EventImageRows");
    _event_image_cols = main_cfg.get<std::vector<size_t> >("EventImageCols");
    _event_comp_rows  = main_cfg.get<std::vector<size_t> >("EventCompRows");
    _event_comp_cols  = main_cfg.get<std::vector<size_t> >("EventCompCols");

    // Check/Enforce conditions
    _logger.set((::larcv::msg::Level_t)(main_cfg.get<unsigned short>("Verbosity")));
    _mctree_verbosity = (::larcv::msg::Level_t)(main_cfg.get<unsigned short>("MCTreeVerbosity"));
    _cropper_verbosity = (::larcv::msg::Level_t)(main_cfg.get<unsigned short>("CropperVerbosity"));

    if(geom->Nplanes() != _event_image_rows.size()) throw larcv::larbys("EventImageRows size != # planes!");
    if(geom->Nplanes() != _event_image_cols.size()) throw larcv::larbys("EventImageCols size != # planes!");
    if(geom->Nplanes() != _event_comp_rows.size())  throw larcv::larbys("EventCompRows size != # planes!");
    if(geom->Nplanes() != _event_comp_cols.size())  throw larcv::larbys("EventCompCols size != # planes!");
    
    for(auto const& v : _event_image_rows){ if(!v) throw larcv::larbys("Event-Image row size is 0!"); }
    for(auto const& v : _event_image_cols){ if(!v) throw larcv::larbys("Event-Image col size is 0!"); }
    for(auto const& v : _event_comp_rows){ if(!v) throw larcv::larbys("Event-Image row comp factor is 0!"); }
    for(auto const& v : _event_comp_cols){ if(!v) throw larcv::larbys("Event-Image col comp factor is 0!"); }

    _larcv_io.initialize();
    return true;
  }
  
  bool Supera::analyze(storage_manager* storage) {

    auto ev_wire = storage->get_data<event_wire>(_producer_wire);

    if(!ev_wire) { throw DataFormatException("Could not load wire data!"); }

    auto geom = ::larutil::Geometry::GetME();

    //auto& image_v = _larcv_io.get_data<larcv::EventImage2D>("event_image");
    auto image_v = (::larcv::EventImage2D*)(_larcv_io.get_data(::larcv::kProductImage2D,"event_image"));

    std::map<larcv::PlaneID_t,larcv::ImageMeta> image_meta_m;
    for(size_t p=0; p<geom->Nplanes(); ++p) {

      size_t cols = _event_image_cols[p] * _event_comp_cols[p];
      size_t rows = _event_image_rows[p] * _event_comp_rows[p];

      auto meta = ::larcv::ImageMeta(cols,rows,rows,cols,_min_wire,_min_time+rows-1,p);
      image_meta_m.insert(std::make_pair(p,meta));
      image_v->Emplace(::larcv::supera::Extract<larlite::wire>(meta,*ev_wire));

    }

    //
    // ROIs
    //
    ::larcv::supera::MCParticleTree<larlite::mctruth,larlite::mctrack,larlite::mcshower> mctp;
    mctp.set_verbosity(_mctree_verbosity);
    mctp.GetCropper().set_verbosity(_cropper_verbosity);
    mctp.DefinePrimary(*(storage->get_data<event_mctruth>(_producer_gen)));
    mctp.RegisterSecondary(*(storage->get_data<event_mctrack>(_producer_mcreco)));
    mctp.RegisterSecondary(*(storage->get_data<event_mcshower>(_producer_mcreco)));
    mctp.UpdatePrimaryROI();
    auto int_roi_v = mctp.GetPrimaryROI();

    auto roi_v = (::larcv::EventROI*)(_larcv_io.get_data(::larcv::kProductROI,"event_roi"));

    for(auto& int_roi : int_roi_v) {

      //
      // Primary: store overlapped ROI
      //
      std::vector<larcv::ImageMeta> pri_bb_v;
      
      for(auto const& bb : int_roi.first.BB()) {
	auto iter = image_meta_m.find(bb.plane());
	if(iter == image_meta_m.end()) continue;
	try{
	  pri_bb_v.push_back(bb.overlap((*iter).second));
	}catch(const ::larcv::larbys& err){
	  break;
	}
      }
      
      if(pri_bb_v.size() != int_roi.first.BB().size()) {
	LARCV_NORMAL() << "Requested to register Interaction..." << std::endl
		     << int_roi.first.dump() << std::endl;
	LARCV_NORMAL() << "No overlap found in image region and Interaction ROI. Skipping..." << std::endl;
	continue;
      }
      int_roi.first.SetBB(pri_bb_v);
      LARCV_INFO() << "Registering Interaction..." << std::endl
		   << int_roi.first.dump() << std::endl;
      roi_v->Append(int_roi.first);

      //
      // Secondaries
      //
      for(auto& roi : int_roi.second) {

	std::vector<larcv::ImageMeta> sec_bb_v;
      
	for(auto const& bb : roi.BB()) {
	  auto iter = image_meta_m.find(bb.plane());
	  if(iter == image_meta_m.end()) continue;
	  try{
	    sec_bb_v.push_back(bb.overlap((*iter).second));
	  }catch(const ::larcv::larbys& err) {
	    break;
	  }
	}
	if(sec_bb_v.size() != roi.BB().size()) {
	  LARCV_INFO() << "Requested to register Secondary..." << std::endl
		       << roi.dump() << std::endl;
	  LARCV_INFO() << "No overlap found in image region and Particle ROI. Skipping..." << std::endl;
	  continue;
	}
	roi.SetBB(sec_bb_v);
	LARCV_INFO() << "Registering Secondary..." << std::endl
		     << roi.dump() << std::endl;
	roi_v->Append(roi);
      }
    }

    _larcv_io.save_entry();
    
    return true;
  }

  bool Supera::finalize() {

    _larcv_io.finalize();
    return true;
  }

}
#endif
