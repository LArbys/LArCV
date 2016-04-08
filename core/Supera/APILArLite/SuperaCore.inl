#ifndef __SUPERACORE_INL__
#define __SUPERACORE_INL__

#include "Base/larbys.h"
#include "SuperaUtils.h"
#include "DataFormat/ProductMap.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventImage2D.h"
namespace larcv {
  namespace supera {

    template <class S, class T, class U, class V, class W>
    SuperaCore<S,T,U,V,W>::SuperaCore() : _logger("Supera")
			     , _larcv_io(::larcv::IOManager::kWRITE)
    { _configured = false; }

    template <class S, class T, class U, class V, class W>
    void SuperaCore<S,T,U,V,W>::initialize() {
      _larcv_io.initialize();
    }
    
    template <class S, class T, class U, class V, class W>
    void SuperaCore<S,T,U,V,W>::configure(const Config_t& main_cfg) {

      _use_mc = main_cfg.get<bool>("UseMC");

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
      
      _skip_empty_image = main_cfg.get<bool>("SkipEmptyImage");
      
      // Check/Enforce conditions
      _logger.set((::larcv::msg::Level_t)(main_cfg.get<unsigned short>("Verbosity")));
      _mctp.configure(main_cfg.get<larcv::supera::Config_t>("MCParticleTree"));
      
      if(::larcv::supera::Nplanes() != _event_image_rows.size()) throw larcv::larbys("EventImageRows size != # planes!");
      if(::larcv::supera::Nplanes() != _event_image_cols.size()) throw larcv::larbys("EventImageCols size != # planes!");
      if(::larcv::supera::Nplanes() != _event_comp_rows.size())  throw larcv::larbys("EventCompRows size != # planes!");
      if(::larcv::supera::Nplanes() != _event_comp_cols.size())  throw larcv::larbys("EventCompCols size != # planes!");
      
      for(auto const& v : _event_image_rows){ if(!v) throw larcv::larbys("Event-Image row size is 0!"); }
      for(auto const& v : _event_image_cols){ if(!v) throw larcv::larbys("Event-Image col size is 0!"); }
      for(auto const& v : _event_comp_rows){ if(!v) throw larcv::larbys("Event-Image row comp factor is 0!"); }
      for(auto const& v : _event_comp_cols){ if(!v) throw larcv::larbys("Event-Image col comp factor is 0!"); }
      
      _configured = true;
    }

    template <class S, class T, class U, class V, class W>    
    bool SuperaCore<S,T,U,V,W>::process_event(const std::vector<S>& wire_v,
					      const std::vector<T>& mctruth_v,
					      const std::vector<U>& mctrack_v,
					      const std::vector<V>& mcshower_v,
					      const std::vector<W>& simch_v)
    {
      if(!_configured) throw larbys("Call configure() first!");

      _larcv_io.clear_entry();
      
      auto event_image_v = (::larcv::EventImage2D*)(_larcv_io.get_data(::larcv::kProductImage2D,"event_image"));
      
      //
      // 0) Construct Event-image ROI
      //
      std::map<larcv::PlaneID_t,larcv::ImageMeta> image_meta_m;
      for(size_t p=0; p< ::larcv::supera::Nplanes(); ++p) {
	
	size_t cols = _event_image_cols[p] * _event_comp_cols[p];
	size_t rows = _event_image_rows[p] * _event_comp_rows[p];
	
	auto meta = ::larcv::ImageMeta(cols,rows,
				       rows,cols,
				       _min_wire,_min_time+rows,
				       p);
	image_meta_m.insert(std::make_pair(p,meta));
	
	LARCV_INFO() << "Creating Event image frame for plane " << p << " " << meta.dump();
      }

      if(!_use_mc) {
	// No MC: take an event picture and done
	
	for(size_t p=0; p < ::larcv::supera::Nplanes(); ++p) {
	  
	  auto const& full_meta = (*(image_meta_m.find(p))).second;
	  
	  // Create full resolution image
	  _full_image.reset(full_meta);
	  ::larcv::supera::Fill<S>(_full_image,wire_v);
	  _full_image.index(event_image_v->Image2DArray().size());
	
	  // Finally compress and store as event image
	  auto comp_meta = ::larcv::ImageMeta(_full_image.meta());
	  comp_meta.update(_event_image_rows[p],_event_image_cols[p]);
	  ::larcv::Image2D img(std::move(comp_meta),
			       std::move(_full_image.copy_compress(_event_image_rows[p],_event_image_cols[p])));
	  event_image_v->Emplace(std::move(img));
	}
	
	_larcv_io.save_entry();
	return true;
      }
      
      //
      // 1) Construct Interaction/Particle ROIs
      //
      _mctp.clear();
      _mctp.DefinePrimary(mctruth_v);
      _mctp.RegisterSecondary(mctrack_v);
      if(_producer_simch.empty())
	_mctp.RegisterSecondary(mcshower_v);
      else
	_mctp.RegisterSecondary(mcshower_v,simch_v);
      
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
	if(!_skip_empty_image) _larcv_io.save_entry();
	return true;
      }
      //
      // If no Interaction ImageMeta (Interaction ROI object w/ no real ROI), skip this event
      //
      bool skip = true;
      for(auto const& roi : roi_v->ROIArray()) {
	if(roi.MCSTIndex() != ::larcv::kINVALID_INDEX) continue;
	if(roi.BB().size() == ::larcv::supera::Nplanes()) {
	  skip=false;
	  break;
	}
      }
      if(skip) {
	if(!_skip_empty_image) _larcv_io.save_entry();
	return true;
      }
      
      //
      // Extract image if there's any ROI
      //
      for(size_t p=0; p < ::larcv::supera::Nplanes(); ++p) {
	
	auto const& full_meta = (*(image_meta_m.find(p))).second;
	
	// Create full resolution image
	_full_image.reset(full_meta);
	::larcv::supera::Fill<S>(_full_image,wire_v);
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
    }

    template<class S, class T, class U, class V, class W>
    void SuperaCore<S,T,U,V,W>::finalize() {
      if(!_configured) throw larbys("Call configure() first!");
      _larcv_io.finalize();
      _larcv_io.reset();
    }
  }
}
#endif
