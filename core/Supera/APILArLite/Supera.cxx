#ifndef LARLITE_SUPERA_CXX
#define LARLITE_SUPERA_CXX

#include "Supera.h"
#include "FhiclLite/ConfigManager.h"
#include "LArUtil/Geometry.h"
#include "DataFormat/wire.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include "Base/larbys.h"
#include "SuperaUtils.h"
#include "DataFormat/ProductMap.h"

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

    _producer_simch  = main_cfg.get<std::string>("SimChProducer");
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
    _mctp.configure(main_cfg.get<larcv::supera::Config_t>("MCParticleTree"));
    //_mctree_verbosity = (::larcv::msg::Level_t)(main_cfg.get<unsigned short>("MCTreeVerbosity"));
    //_cropper_verbosity = (::larcv::msg::Level_t)(main_cfg.get<unsigned short>("CropperVerbosity"));

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

    _larcv_io.set_id(storage->run_id(), storage->subrun_id(), storage->event_id());

    auto wire_h = storage->get_data<event_wire>(_producer_wire);                                                                                        
    //art::Handle<std::vector<recob::Wire> > wire_h; e.getByLabel(_producer_wire,wire_h);

    if(!wire_h) { throw DataFormatException("Could not load wire data!"); }
    //if(!wire_h.isValid()) { throw larcv::larbys("Could not load wire data!"); }

    //auto const* geom = ::lar::providerFrom<geo::Geometry>();
    auto geom = ::larutil::Geometry::GetME();

    auto event_image_v = (::larcv::EventImage2D*)(_larcv_io.get_data(::larcv::kProductImage2D,"event_image"));

    //
    // 0) Construct Event-image ROI
    //
    std::map<larcv::PlaneID_t,larcv::ImageMeta> image_meta_m;
    for(size_t p=0; p<geom->Nplanes(); ++p) {

      size_t cols = _event_image_cols[p] * _event_comp_cols[p];
      size_t rows = _event_image_rows[p] * _event_comp_rows[p];

      auto meta = ::larcv::ImageMeta(cols,rows,
				     rows,cols,
				     _min_wire,_min_time+rows,
				     p);
      image_meta_m.insert(std::make_pair(p,meta));

      LARCV_INFO() << "Creating Event image frame for plane " << p << " " << meta.dump();
    }

    //
    // 1) Construct Interaction/Particle ROIs
    //
    //art::Handle<std::vector<simb::MCTruth> > mctruth_h;  e.getByLabel( _producer_gen,    mctruth_h  );
    //art::Handle<std::vector<sim::MCTrack > > mctrack_h;  e.getByLabel( _producer_mcreco, mctrack_h  );
    //art::Handle<std::vector<sim::MCShower> > mcshower_h; e.getByLabel( _producer_mcreco, mcshower_h );
    auto mctruth_h  = storage->get_data<event_mctruth>(_producer_gen);
    auto mctrack_h  = storage->get_data<event_mctrack>(_producer_mcreco);
    auto mcshower_h = storage->get_data<event_mcshower>(_producer_mcreco);
    _mctp.clear();
    _mctp.DefinePrimary(*mctruth_h);
    _mctp.RegisterSecondary(*(mctrack_h));
    if(_producer_simch.empty()) {
      _mctp.RegisterSecondary(*(mcshower_h));
    }else{
      //art::Handle<std::vector<sim::SimChannel> > simch_h; e.getByLabel( _producer_simch, simch_h );
      auto simch_h = storage->get_data<event_simch>(_producer_simch);
      _mctp.RegisterSecondary(*(mcshower_h),*(simch_h));
    }
    _mctp.UpdatePrimaryROI();
    auto int_roi_v = _mctp.GetPrimaryROI();

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
	  auto trimmed = (*iter).second.overlap(bb);
	  pri_bb_v.push_back(trimmed);
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
	    auto trimmed = (*iter).second.overlap(bb);
	    sec_bb_v.push_back(trimmed);
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

    //
    // If no ROI, skip this event
    //
    if(roi_v->ROIArray().empty()) {
      _larcv_io.save_entry();
      return true;
      //return;
    }
    //
    // If no Interaction ImageMeta (Interaction ROI object w/ no real ROI), skip this event
    //
    bool skip = true;
    for(auto const& roi : roi_v->ROIArray()) {
      if(roi.MCSTIndex() != ::larcv::kINVALID_INDEX) continue;
      if(roi.BB().size() == geom->Nplanes()) {
	skip=false;
	break;
      }
    }
    if(skip) {
      _larcv_io.save_entry();
      return true;
      //return;
    }
    
    //
    // Extract image if there's any ROI
    //
    for(size_t p=0; p<geom->Nplanes(); ++p) {

      auto const& full_meta = (*(image_meta_m.find(p))).second;

      // Create full resolution image
      _full_image.reset(full_meta);
      ::larcv::supera::Fill<larlite::wire>(_full_image,*wire_h);
      _full_image.index(event_image_v->Image2DArray().size());

      // Now extract each high-resolution interaction image
      for(auto const& roi : roi_v->ROIArray()) {
	// Only care about interaction
	if(roi.MCSTIndex() != ::larcv::kINVALID_INDEX) continue;
	auto const& roi_meta = roi.BB(p);
	// Retrieve cropped full resolution image
	auto int_img_v = (::larcv::EventImage2D*)(_larcv_io.get_data(::larcv::kProductImage2D,Form("mcint%02d",roi.MCTIndex())));
	auto hires_img = _full_image.crop(roi_meta);
	int_img_v->Emplace(std::move(hires_img));
      }

      // Finally compress and store as event image
      auto comp_meta = ::larcv::ImageMeta(_full_image.meta());
      comp_meta.update(_event_image_rows[p],_event_image_cols[p]);
      ::larcv::Image2D img(std::move(comp_meta),
			   std::move(_full_image.copy_compress(_event_image_rows[p],_event_image_cols[p])));
      event_image_v->Emplace(std::move(img));
    }
    
    _larcv_io.save_entry();
    return true;
    //return;
  }

  bool Supera::finalize() {

    _larcv_io.finalize();
    return true;
  }

}
#endif
