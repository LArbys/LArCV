#ifndef __SUPERACORE_INL__
#define __SUPERACORE_INL__

#include "Base/larbys.h"
#include "DataFormat/ProductMap.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventChStatus.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/UtilFunc.h"


namespace larcv {
  namespace supera {

    template <class R, class S, class T, class U, class V, class W>
    SuperaCore<R,S,T,U,V,W>::SuperaCore() : _logger("Supera")
			     , _larcv_io(::larcv::IOManager::kWRITE)
    {
      _configured = false;
      _use_mc = false;
      _store_chstatus = false;
      //_larcv_io.set_verbosity(::larcv::msg::kDEBUG);
    }
       

    template <class R, class S, class T, class U, class V, class W>
    void SuperaCore<R,S,T,U,V,W>::initialize() {
      _larcv_io.initialize();
    }
    
    template <class R, class S, class T, class U, class V, class W>
    void SuperaCore<R,S,T,U,V,W>::configure(const Config_t& main_cfg) {

      _use_mc = main_cfg.get<bool>("UseMC");
      _store_chstatus = main_cfg.get<bool>("StoreChStatus");
      _larcv_io.set_out_file(main_cfg.get<std::string>("OutFileName"));
      
      _producer_key     = main_cfg.get<std::string>("ProducerKey");
      _producer_digit   = main_cfg.get<std::string>("DigitProducer");
      _producer_simch   = main_cfg.get<std::string>("SimChProducer");
      _producer_wire    = main_cfg.get<std::string>("WireProducer");
      _producer_gen     = main_cfg.get<std::string>("GenProducer");
      _producer_mcreco  = main_cfg.get<std::string>("MCRecoProducer");
      _producer_opdigit = main_cfg.get<std::string>("OpDigitProducer");
      
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

      if(_store_chstatus) {
	_channel_to_plane_wire.clear();
	_channel_to_plane_wire.resize(::larcv::supera::Nchannels());
	for(size_t i=0; i < ::larcv::supera::Nchannels(); ++i) {
	  auto& plane_wire = _channel_to_plane_wire[i];
	  auto const wid = ::larcv::supera::ChannelToWireID(i);
	  plane_wire.first  = wid.Plane;
	  plane_wire.second = wid.Wire;
	}

	for(size_t i=0; i < ::larcv::supera::Nplanes(); ++i) {
	  ::larcv::ChStatus status;
	  status.Plane(i);
	  status.Initialize(::larcv::supera::Nwires(i),::larcv::chstatus::kUNKNOWN);
	  _status_m.emplace(status.Plane(),status);
	}
      }
      _configured = true;
    }

    template <class R, class S, class T, class U, class V, class W>
    void SuperaCore<R,S,T,U,V,W>::set_chstatus(unsigned int ch, short status)
    {
      if(ch >= _channel_to_plane_wire.size()) throw ::larcv::larbys("Invalid channel to store status!");
      auto const& plane_wire = _channel_to_plane_wire[ch];
      _status_m[plane_wire.first].Status(plane_wire.second,status);
    }

    template <class R, class S, class T, class U, class V, class W>    
    bool SuperaCore<R,S,T,U,V,W>::process_event(const std::vector<R>& opdigit_v,
						const std::vector<S>& wire_v,
						const std::vector<T>& mctruth_v,
						const std::vector<U>& mctrack_v,
						const std::vector<V>& mcshower_v,
						const std::vector<W>& simch_v)
    {
      if(!_configured) throw larbys("Call configure() first!");

      //
      // 0) Store channel status if requested
      //
      if(_store_chstatus) {

	auto event_chstatus = (::larcv::EventChStatus*)(_larcv_io.get_data(::larcv::kProductChStatus,_producer_key));
	for(auto const& id_status : _status_m)
	  event_chstatus->Insert(id_status.second);

	// Reset status
	for(auto& plane_status : _status_m) plane_status.second.Reset(::larcv::chstatus::kUNKNOWN);
	
      }
      
      auto event_image_v = (::larcv::EventImage2D*)(_larcv_io.get_data(::larcv::kProductImage2D,_producer_key));
      
      //
      // 0) Construct Event-image ROI
      //
      std::map<larcv::PlaneID_t,larcv::ImageMeta> image_meta_m;
      for(size_t p=0; p< ::larcv::supera::Nplanes(); ++p) {
	
	size_t cols = _event_image_cols[p] * _event_comp_cols[p];
	size_t rows = _event_image_rows[p] * _event_comp_rows[p];
	
	auto meta = ::larcv::ImageMeta(cols,rows,rows,cols,
				       _min_wire,_min_time+rows,p);
	image_meta_m.insert(std::make_pair(p,meta));
	
	LARCV_INFO() << "Creating Event image frame:" << meta.dump();
      }

      if(!_use_mc) {
	// No MC: take an event picture and done
	
	for(size_t p=0; p < ::larcv::supera::Nplanes(); ++p) {
	  
	  auto const& full_meta = (*(image_meta_m.find(p))).second;
	  
	  // Create full resolution image
	  _full_image.reset(full_meta);
	  fill(_full_image,wire_v);
	  _full_image.index(event_image_v->Image2DArray().size());
	
	  // Finally compress and store as event image
	  auto comp_meta = ::larcv::ImageMeta(_full_image.meta());
	  comp_meta.update(_event_image_rows[p],_event_image_cols[p]);
	  ::larcv::Image2D img(std::move(comp_meta),
			       std::move(_full_image.copy_compress(_event_image_rows[p],_event_image_cols[p])));
	  event_image_v->Emplace(std::move(img));
	}

	// OpDigit
	std::string op_producer = "op_" + _producer_key;
	auto opdigit_image_v = (::larcv::EventImage2D*)(_larcv_io.get_data(::larcv::kProductImage2D,op_producer));
	::larcv::ImageMeta op_meta(32,1500,1500,32,0,1499);
	::larcv::Image2D op_img(op_meta);
	fill(op_img,opdigit_v);
	opdigit_image_v->Emplace(std::move(op_img));
	
	_larcv_io.save_entry();
	return true;
      }
      
      //
      // 1) Construct Interaction/Particle ROIs
      //
      _mctp.clear();
      _mctp.DefinePrimary(mctruth_v);
      if(_producer_simch.empty()) {
	_mctp.RegisterSecondary(mctrack_v);
	_mctp.RegisterSecondary(mcshower_v);
      }else{
	_mctp.RegisterSecondary(mctrack_v,simch_v);
	_mctp.RegisterSecondary(mcshower_v,simch_v);
      }
      
      _mctp.UpdatePrimaryROI();
      auto int_roi_v = _mctp.GetPrimaryROI();
      
      auto roi_v = (::larcv::EventROI*)(_larcv_io.get_data(::larcv::kProductROI,_producer_key));
      
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
	auto& sec_roi_v = int_roi.second;
	std::vector<bool> sec_used_v(sec_roi_v.size(),false);
	std::vector<larcv::ROI> sec_roi_out_v;
	for(size_t sec_index=0; sec_index<sec_roi_v.size(); ++sec_index) {

	  if(sec_used_v[sec_index]) continue;

	  // Copy ROI (don't use reference)
	  auto sec_roi = sec_roi_v[sec_index];

	  // If gamma and parent is Pi0, search for sibling & merge two
	  if(sec_roi.PdgCode() == 22 && sec_roi.ParentPdgCode() == 111) {
	    size_t sibling_index = sec_index;
	    for(size_t i=sec_index+1; i<sec_roi_v.size(); ++i) {
	      if(sec_roi_v[i].ParentTrackID() == sec_roi.ParentTrackID()) {
		sibling_index = i;
		break;
	      }
	    }
	    if(sibling_index!=sec_index) {
	      // Merge
	      auto sibling_roi = sec_roi_v[sibling_index];
	      std::vector<larcv::ImageMeta> merged_bb_v = sec_roi.BB();
	      for(auto& bb : merged_bb_v) {
		auto const& sibling_bb = sibling_roi.BB(bb.plane());
		bb = bb.inclusive(sibling_bb);
	      }
	      sec_roi.SetBB(merged_bb_v);
	      sec_roi.Position(sec_roi.ParentPosition());
	      sec_roi.Momentum(sec_roi.ParentPx(),sec_roi.ParentPy(),sec_roi.ParentPx());
	      sec_roi.ParentPosition(0,0,0,0);
	      sec_roi.ParentMomentum(0.,0.,0.);
	      sec_roi.PdgCode(111);
	      sec_roi.EnergyDeposit(sec_roi.EnergyDeposit() + sibling_roi.EnergyDeposit());
	      sec_roi.EnergyInit(sec_roi.EnergyInit() + sibling_roi.EnergyInit());
	      sec_roi.TrackID(sec_roi.ParentTrackID());
	      sec_roi.ParentPdgCode(0);
	      sec_roi.ParentTrackID(::larcv::kINVALID_UINT);
	      sec_used_v[sibling_index] = true;
	    }
	  }
	  sec_used_v[sec_index] = true;
	  std::vector<larcv::ImageMeta> sec_bb_v;
	  
	  for(auto const& bb : sec_roi.BB()) {
	    auto iter = image_meta_m.find(bb.plane());
	    if(iter == image_meta_m.end()) continue;
	    try{
	      auto trimmed = (*iter).second.overlap(bb);
	      sec_bb_v.push_back(trimmed);
	    }catch(const ::larcv::larbys& err) {
	      break;
	    }
	  }
	  if(sec_bb_v.size() != sec_roi.BB().size()) {
	    LARCV_INFO() << "Requested to register Secondary..." << std::endl
			 << sec_roi.dump() << std::endl;
	    LARCV_INFO() << "No overlap found in image region and Particle ROI. Skipping..." << std::endl;
	    continue;
	  }
	  sec_roi.SetBB(sec_bb_v);
	  LARCV_INFO() << "Registering Secondary..." << std::endl
		       << sec_roi.dump() << std::endl;
	  roi_v->Append(sec_roi);
	}
	
	//
	// Secondaries
	//
	/*
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
	*/
      }


	
      //
      // If no ROI, skip this event
      //
      if(roi_v->ROIArray().empty()) {
	if(!_skip_empty_image) _larcv_io.save_entry();
	return (!_skip_empty_image);
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
	return (!_skip_empty_image);
      }
      
      // OpDigit
      std::string op_producer = "op_" + _producer_key;
      auto opdigit_image_v = (::larcv::EventImage2D*)(_larcv_io.get_data(::larcv::kProductImage2D,op_producer));
      ::larcv::ImageMeta op_meta(32,1500,1500,32,0,1499);
      ::larcv::Image2D op_img(op_meta);
      fill(op_img,opdigit_v);
      opdigit_image_v->Emplace(std::move(op_img));      
      
      //
      // Extract image if there's any ROI
      //
      for(size_t p=0; p < ::larcv::supera::Nplanes(); ++p) {
	
	auto const& full_meta = (*(image_meta_m.find(p))).second;
	
	// Create full resolution image
	_full_image.reset(full_meta);
	fill(_full_image,wire_v);
	_full_image.index(event_image_v->Image2DArray().size());
	
	// Now extract each high-resolution interaction image
	for(auto const& roi : roi_v->ROIArray()) {
	  // Only care about interaction
	  if(roi.MCSTIndex() != ::larcv::kINVALID_INDEX) continue;
	  auto const& roi_meta = roi.BB(p);
	  // Retrieve cropped full resolution image
	  auto int_img_v = (::larcv::EventImage2D*)(_larcv_io.get_data(::larcv::kProductImage2D,Form("%s_int%02d",_producer_key.c_str(),roi.MCTIndex())));
	  LARCV_INFO() << "Cropping an interaction image (high resolution) @ plane " << p << std::endl
		       << roi_meta.dump() << std::endl;
	  auto hires_img = _full_image.crop(roi_meta);
	  int_img_v->Emplace(std::move(hires_img));
	}
	
	// Finally compress and store as event image
	auto comp_meta = ::larcv::ImageMeta(_full_image.meta());
	comp_meta.update(_event_image_rows[p],_event_image_cols[p]);

	LARCV_INFO() << "Compressing an event image! " << std::endl
		     << "From: " << _full_image.meta().dump() << std::endl
		     << "To: " << comp_meta.dump() << std::endl;

	::larcv::Image2D img(std::move(comp_meta),
			     std::move(_full_image.copy_compress(_event_image_rows[p],_event_image_cols[p])));
	event_image_v->Emplace(std::move(img));
      }

      //
      // Semantic Segmentation
      //
      std::string sem_producer = "segment_" + _producer_key;
      auto event_semimage_v = (::larcv::EventImage2D*)(_larcv_io.get_data(::larcv::kProductImage2D,sem_producer));
      std::vector<larcv::Image2D> sem_images;
      /*// For full plane image + full resolution
      for(auto const& plane_img : image_meta_m) 
	sem_images.emplace_back(::larcv::Image2D(plane_img.second));
      */
      // For interaction ROI + full resolution
      for(auto const& roi : roi_v->ROIArray()) {
	if(roi.MCSTIndex() != ::larcv::kINVALID_INDEX) continue;
	if(roi.BB().empty()) continue;
	for(auto const& bb : roi.BB()) {
	  auto sem_image = ::larcv::Image2D(bb);
	  sem_image.paint(::larcv::kROIUnknown);
	  sem_images.emplace_back(sem_image);
	}
	break;
      }
      if(!sem_images.empty())
	fill(sem_images,mctrack_v,mcshower_v,simch_v);
      event_semimage_v->Emplace(std::move(sem_images));
      
      _larcv_io.save_entry();
      return true;      
    }

    template<class R, class S, class T, class U, class V, class W>
    void SuperaCore<R,S,T,U,V,W>::finalize() {
      if(!_configured) throw larbys("Call configure() first!");
      _larcv_io.finalize();
      _larcv_io.reset();
    }

    template <class R, class S, class T, class U, class V, class W>
    void SuperaCore<R,S,T,U,V,W>::fill(Image2D& img, const std::vector<R>& opdigit_v, int time_offset)
    {
      auto const& meta = img.meta();
      for(auto const& opdigit : opdigit_v) {
	if(opdigit.size()<1000) continue;
	auto const col = opdigit.ChannelNumber();
	if(meta.min_x() > col) continue;
	if(col >= meta.max_x()) continue;
	//
	// HACK: right way is to use TimeService + trigger.
	//       for now I just record PMT beamgate tick=0 as edge of an image (w/ offset)
	//
	size_t nskip = 0;
	if(time_offset < 0) nskip = (-1 * time_offset);
	if(nskip >= opdigit.size()) continue;
	size_t num_pixel = std::min(meta.rows(),opdigit.size() - nskip);
	img.copy(0,col,&((std::vector<short>)opdigit)[nskip],num_pixel);
      }
    }

    template <class R, class S, class T, class U, class V, class W>
    void SuperaCore<R,S,T,U,V,W>::fill(std::vector<Image2D>& img_v,
				     const std::vector<U>& mct_v,
				     const std::vector<V>& mcs_v,
				     const std::vector<W>& sch_v,
				     const int time_offset)
    {
      LARCV_INFO() << "Filling semantic-segmentation ground truth image..." << std::endl;
      for(auto const& img : img_v) 

	LARCV_INFO() << img.meta().dump();
      
      static std::vector<larcv::ROIType_t> track2type_v(1e4,::larcv::kROIUnknown);
      for(auto& v : track2type_v) v = ::larcv::kROIUnknown;
      for(auto const& mct : mct_v) {
	if(mct.TrackID() >= track2type_v.size())
	  track2type_v.resize(mct.TrackID()+1,::larcv::kROIUnknown);
	track2type_v[mct.TrackID()] = ::larcv::PDG2ROIType(mct.PdgCode());
      }
      for(auto const& mcs : mcs_v) {
	if(mcs.TrackID() >= track2type_v.size())
	  track2type_v.resize(mcs.TrackID()+1,::larcv::kROIUnknown);
	auto const roi_type = ::larcv::PDG2ROIType(mcs.PdgCode());
	track2type_v[mcs.TrackID()] = roi_type;
	for(auto const& id : mcs.DaughterTrackID()) {
	  if(id >= track2type_v.size())
	    track2type_v.resize(id+1,::larcv::kROIUnknown);
	  track2type_v[id] = roi_type;
	}
      }

      static std::vector<float> column;
      for(auto const& img : img_v) {
	if(img.meta().rows() >= column.size())
	  column.resize(img.meta().rows()+1,(float)(::larcv::kROIUnknown));
      }

      for(auto const& sch : sch_v) {

	auto ch = sch.Channel();
	auto const& wid = ::larcv::supera::ChannelToWireID(ch);
	auto const& plane = wid.Plane;
	auto& img = img_v.at(plane);
	auto const& meta = img.meta();

	size_t col = wid.Wire;
	if(col < meta.min_x()) continue;
	if(meta.max_x() <= col) continue;
	if(plane != img.meta().plane()) continue;

	col -= (size_t)(meta.min_x());
	
	// Initialize column vector
	for(auto& v : column) v = (float)(::larcv::kROIUnknown);

	for(auto const tick_ides : sch.TDCIDEMap()) {
	  int tick = (TPCTDC2Tick((double)(tick_ides.first)) + time_offset);
	  if(tick < meta.min_y()) continue;
	  if(tick >= meta.max_y()) continue;
	  // Where is this tick in column vector?
	  size_t index = (size_t)(meta.max_y() - tick);
	  // Pick type
	  double energy=0;
	  ::larcv::ROIType_t roi_type=::larcv::kROIUnknown;
	  for(auto const& edep : tick_ides.second) {
	    if(edep.energy < energy) continue;
	    if(edep.trackID >= track2type_v.size()) continue;
	    auto temp_roi_type = track2type_v[edep.trackID];
	    if(temp_roi_type==::larcv::kROIUnknown) continue;
	    energy = edep.energy;
	    roi_type = temp_roi_type;
	  }
	  column[index]=roi_type;
	}
	// mem-copy column vector
	img.copy(0,col,column,img.meta().rows());
      }
    }
    
    template <class R, class S, class T, class U, class V, class W>
    void SuperaCore<R,S,T,U,V,W>::fill(Image2D& img, const std::vector<S>& wires, const int time_offset)
    {
      //int nticks = meta.rows();
      //int nwires = meta.cols();
      auto const& meta = img.meta();
      size_t row_comp_factor = (size_t)(meta.pixel_height());
      const int ymax = meta.max_y() - 1; // Need in terms of row coordinate
      const int ymin = (meta.min_y() >= 0 ? meta.min_y() : 0);
      img.paint(0.);

      LARCV_INFO() << "Filling an image..." << std::endl
		   << meta.dump()
		   << "(ymin,ymax) = (" << ymin << "," << ymax << ")" << std::endl;
      
      for(auto const& wire : wires) {

	auto const& wire_id = ChannelToWireID(wire.Channel());
	
	if((int)(wire_id.Plane) != meta.plane()) continue;

	size_t col=0;
	try{
	  col = meta.col(wire_id.Wire);
	}catch(const larbys&){
	  continue;
	}

	for(auto const& range : wire.SignalROI().get_ranges()) {
	  
	  auto const& adcs = range.data();
	  //double sumq = 0;
	  //for(auto const& v : adcs) sumq += v;
	  //sumq /= (double)(adcs.size());
	  //if(sumq<3) continue;
	  
	  int start_index = range.begin_index() + time_offset;
	  int end_index   = start_index + adcs.size() - 1;
	  if(start_index > ymax || end_index < ymin) continue;

	  if(row_comp_factor>1) {

	    for(size_t index=0; index<adcs.size(); ++index) {
	      if((int)index + start_index < ymin) continue;
	      if((int)index + start_index > ymax) break;
	      auto row = meta.row((double)(start_index+index));
	      img.set_pixel(row,col,adcs[index]);
	    }
	  }else{
	    // Fill matrix from start_index => end_index of matrix row
	    // By default use index 0=>length-1 index of source vector
	    int nskip=0;
	    int nsample=adcs.size();
	    if(end_index   > ymax) {
	      LARCV_DEBUG() << "End index (" << end_index << ") exceeding image bound (" << ymax << ")" << std::endl;
	      nsample   = adcs.size() - (end_index - ymax);
	      end_index = ymax;
	      LARCV_DEBUG() << "Corrected End index = " << end_index << std::endl;
	    }
	    if(start_index < ymin) {
	      LARCV_DEBUG() << "Start index (" << start_index << ") exceeding image bound (" << ymin << ")" << std::endl;
	      nskip = ymin - start_index;
	      nsample -= nskip;
	      start_index = ymin;
	      LARCV_DEBUG() << "Corrected Start index = " << start_index << std::endl;
	    }
	    LARCV_DEBUG() << "Calling a reverse_copy..." << std::endl
			  << "      source wf : start index = " << range.begin_index() << " length = " << adcs.size() << std::endl
			  << "      (row,col) : (" << (ymax - end_index) << "," << col << ")" << std::endl
			  << "      nskip     : "  << nskip << std::endl
			  << "      nsample   : "  << nsample << std::endl;
	    try{
	      img.reverse_copy(ymax - end_index,
			       col,
			       adcs,
			       nskip,
			       nsample);
	    }catch(const ::larcv::larbys& err) {
	      LARCV_CRITICAL() << "Attempted to fill an image..." << std::endl
			       << meta.dump()
			       << "(ymin,ymax) = (" << ymin << "," << ymax << ")" << std::endl
			       << "Called a reverse_copy..." << std::endl
			       << "      source wf : start index = " << range.begin_index() << " length = " << adcs.size() << std::endl
			       << "      (row,col) : (" << (ymax - end_index) << "," << col << ")" << std::endl
			       << "      nskip     : "  << nskip << std::endl
			       << "Re-throwing an error:" << std::endl;
	      throw err;
	    }
	  }
	}
      }
    }

  }
}
#endif
