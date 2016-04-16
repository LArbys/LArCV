

#ifndef __SUPERA_CROPPER_INL__
#define __SUPERA_CROPPER_INL__

#include "Base/larbys.h"
#include "FMWKInterface.h"
#include <TLorentzVector.h> // ROOT
#include <set>
namespace larcv {

  namespace supera {

    template<class T, class U, class V>
    void Cropper<T,U,V>::configure(const larcv::supera::Config_t& cfg)
    {
      LARCV_DEBUG() << "start" << std::endl;
      set_verbosity((::larcv::msg::Level_t)(cfg.get<unsigned short>("Verbosity")));
      _time_padding  = cfg.get<unsigned int>("TimePadding");
      _wire_padding  = cfg.get<unsigned int>("WirePadding");
      _target_width  = cfg.get<unsigned int>("TargetWidth");
      _target_height = cfg.get<unsigned int>("TargetHeight");
      _compression_factor = cfg.get<unsigned int>("CompressionFactor");
      _min_width     = cfg.get<unsigned int>("MinWidth");
      _min_height    = cfg.get<unsigned int>("MinHeight");

      LARCV_NORMAL() << "Configuration called..." << std::endl
		     << " Padding   (wire,time) = (" << _wire_padding << "," << _time_padding << ")" << std::endl
		     << " Min. Size (wire,time) = (" << _min_width << "," << _min_height << ")" << std::endl;
    }

    template<class T, class U, class V>
    WTRangeArray_t Cropper<T,U,V>::WireTimeBoundary(const T& mct, const std::vector<V>& sch_v) const
    {
      LARCV_DEBUG() << "start" << std::endl;
      // result is N planes' wire boundary + time boundary (N+1 elements)
      WTRangeArray_t result(::larcv::supera::Nplanes()+1);

      for(auto const& sch : sch_v) {

	auto const wire = ChannelToWireID(sch.Channel());
	auto& wrange = result[wire.Plane];
	auto& trange = result.back();

	for(auto const& tdc_ide_v : sch.TDCIDEMap()) {
	  bool store=false;
	  double tick = TPCTDC2Tick((double)(tdc_ide_v.first));
	  if(tick<0 || tick >= NumberTimeSamples()) continue;
	  tick += 0.5;
	  for(auto const& ide : tdc_ide_v.second){
	    unsigned int trackid = (ide.trackID < 0 ? (-1 * ide.trackID) : ide.trackID);
	    if(trackid != mct.TrackID()) continue;

	    store=true;
	    LARCV_INFO() << "IDE wire = " << (unsigned int)wire.Wire
			 << " ... tick = " << (unsigned int)tick << std::endl
			 << " (tdc=" << tdc_ide_v.first << ", x=" << ide.x << ")" << std::endl;
	    break;
	  }
	  if(!store) continue;
	  if(!wrange.valid()) wrange.Set((unsigned int)wire.Wire,(unsigned int)(wire.Wire));
	  else wrange += (unsigned int)(wire.Wire);  
	  if(!trange.valid()) trange.Set((unsigned int)(tick),(unsigned int)(tick));
	  else trange += (unsigned int)(tick);
	}
      }
      
      for(auto& r : result) if(!r.valid()) r.Set(0,0);

      for(size_t plane=0; plane <= larcv::supera::Nplanes(); ++plane)
	  
	LARCV_INFO() << "Single MCShower ... Plane " << plane 
		     << " bound " << result[plane].Start() << " => " << result[plane].End() << std::endl;
      
      return result;
    }
    
    template<class T, class U, class V>
    WTRangeArray_t Cropper<T,U,V>::WireTimeBoundary(const T& mct) const
    {
      LARCV_DEBUG() << "start" << std::endl;
      const double drift_velocity = ::larcv::supera::DriftVelocity()*1.0e-3; // make it cm/ns
      const int tick_max = ::larcv::supera::NumberTimeSamples();
      const double wireplaneoffset_cm = 0.0; //cm (made up)
      TVector3 xyz; xyz[0] = xyz[1] = xyz[2] = 0.;

      // result is N planes' wire boundary + time boundary (N+1 elements)
      WTRangeArray_t result(::larcv::supera::Nplanes()+1); 
      
      for(auto& step : mct) {
	
	// Figure out time
	int tick = (unsigned int)( ::larcv::supera::TPCG4Time2Tick(step.T() + ((step.X()+wireplaneoffset_cm) / drift_velocity))) + 1;

	if(tick < 0 || tick >= tick_max) continue;
	
	auto& trange = result.back();
	if(!trange.valid()) trange.Set((unsigned int)tick,(unsigned int)tick); // 1st time: "set" it
	else trange += (unsigned int)tick; // >1st time: "add (include)" it
	
	// Figure out wire per plane
	xyz[0] = step.X();
	xyz[1] = step.Y();
	xyz[2] = step.Z();
	LARCV_INFO() << "(x,t,v) = (" << xyz[0] << "," << step.T() << "," << drift_velocity << ") ... tick = " << tick << std::endl;
	for(size_t plane=0; plane < larcv::supera::Nplanes(); ++plane) {

	  auto wire_id = ::larcv::supera::NearestWire(xyz,plane);
	  auto& wrange = result[plane];
	  LARCV_INFO() << "(y,z) = (" << xyz[1] << "," << xyz[2] << ") ... @ plane " << plane << " wire = " << wire_id
		       << (wrange.valid() ? " updating wire-range" : " setting wire-range") << std::endl;
	  if(!wrange.valid()) wrange.Set((unsigned int)wire_id,(unsigned int)wire_id);
	  else wrange += (unsigned int)wire_id;

	}
      }
      
      for(auto& r : result) if(!r.valid()) r.Set(0,0);
	//if(!r.valid() || (r.End() - r.Start()) < 2) r.Set(0,0);

      for(size_t plane=0; plane <= larcv::supera::Nplanes(); ++plane)
	  
	LARCV_INFO() << "Single MCTrack ... Plane " << plane 
		     << " bound " << result[plane].Start() << " => " << result[plane].End() << std::endl;
      
      return result;
    }

    template<class T, class U, class V>
    WTRangeArray_t Cropper<T,U,V>::WireTimeBoundary(const U& mcs) const
    {
      LARCV_DEBUG() << "start" << std::endl;
      const double drift_velocity = ::larcv::supera::DriftVelocity()*1.0e-3; // make it cm/ns
      const int tick_max = ::larcv::supera::NumberTimeSamples();
      double xyz[3] = {0.};

      // result is N planes' wire boundary + time boundary (N+1 elements)
      WTRangeArray_t result(::larcv::supera::Nplanes()+1); 
      
      auto const& detprofile = mcs.DetProfile();
      double energy = detprofile.E();
      
      double showerlength = 13.8874 + 0.121734*energy - (3.75571e-05)*energy*energy;
      showerlength *= 2.5;
      //double showerlength = 100.0;
      double detprofnorm = sqrt( detprofile.Px()*detprofile.Px() + detprofile.Py()*detprofile.Py() + detprofile.Pz()*detprofile.Pz() );
      TLorentzVector showerend;
      const double wireplaneoffset_cm = 0.0; //cm (made up)
      showerend[0] = detprofile.X()+showerlength*(detprofile.Px()/detprofnorm); 
      showerend[1] = detprofile.Y()+showerlength*(detprofile.Py()/detprofnorm); 
      showerend[2] = detprofile.Z()+showerlength*(detprofile.Pz()/detprofnorm); 
      showerend[3] = detprofile.T();
      //std::cout << "showerlength: " << showerlength << " norm=" << detprofnorm << std::endl;
      
      std::vector< TLorentzVector > step_v;
      step_v.push_back( detprofile.Position() );
      step_v.push_back( showerend );
      
      for(auto& step : step_v) {
	
	// Figure out time
	int tick = (unsigned int)( ::larcv::supera::TPCG4Time2Tick(step.T() + ( (step.X()+wireplaneoffset_cm) / drift_velocity))) + 1;

	if(tick < 0 || tick >= tick_max) continue;
	
	auto& trange = result.back();
	if(!trange.valid()) trange.Set((unsigned int)tick,(unsigned int)tick);
	else trange += (unsigned int)tick;
	
	// Figure out wire per plane
	xyz[0] = step.X();
	xyz[1] = step.Y();
	xyz[2] = step.Z();
	for(size_t plane=0; plane < larcv::supera::Nplanes(); ++plane) {

	  auto wire_id = ::larcv::supera::NearestWire(xyz, plane);

	  auto& wrange = result[plane];
	  if(!wrange.valid()) wrange.Set((unsigned int)wire_id,(unsigned int)wire_id);
	  else wrange += (unsigned int)wire_id;
	}

	LARCV_INFO() << "(x,t,v) = (" << xyz[0] << "," << step.T() << "," << drift_velocity << ") ... tick = " << tick << std::endl;
      }
      
      for(auto& r : result) if(!r.valid()) r.Set(0,0);

      for(size_t plane=0; plane <= larcv::supera::Nplanes(); ++plane)
	  
	LARCV_INFO() << "Single MCShower ... Plane " << plane 
		     << " bound " << result[plane].Start() << " => " << result[plane].End() << std::endl;
      
      return result;
    }

    template<class T, class U, class V>
    WTRangeArray_t Cropper<T,U,V>::WireTimeBoundary(const U& mcs, const std::vector<V>& sch_v) const
    {
      LARCV_DEBUG() << "start" << std::endl;
      // result is N planes' wire boundary + time boundary (N+1 elements)
      WTRangeArray_t result(::larcv::supera::Nplanes()+1);

      std::set<unsigned int> daughters;
      for(auto const& trackid : mcs.DaughterTrackID()) daughters.insert(trackid);
      daughters.insert(mcs.TrackID());
      
      for(auto const& sch : sch_v) {

	auto const wire = ChannelToWireID(sch.Channel());
	auto& wrange = result[wire.Plane];
	auto& trange = result.back();

	for(auto const& tdc_ide_v : sch.TDCIDEMap()) {
	  bool store=false;
	  double tick = TPCTDC2Tick((double)(tdc_ide_v.first));
	  if(tick<0 || tick >= NumberTimeSamples()) continue;
	  tick += 0.5;
	  for(auto const& ide : tdc_ide_v.second){
	    unsigned int trackid = (ide.trackID < 0 ? (-1 * ide.trackID) : ide.trackID);
	    if(daughters.find(trackid) != daughters.end()) {
	      store=true;
	      LARCV_INFO() << "IDE wire = " << (unsigned int)wire.Wire
			   << " ... tick = " << (unsigned int)tick << std::endl
			   << " (tdc=" << tdc_ide_v.first << ", x=" << ide.x << ")" << std::endl;
	      break;
	    }
	  }
	  if(!store) continue;
	  if(!wrange.valid()) wrange.Set((unsigned int)wire.Wire,(unsigned int)(wire.Wire));
	  else wrange += (unsigned int)(wire.Wire);  
	  if(!trange.valid()) trange.Set((unsigned int)(tick),(unsigned int)(tick));
	  else trange += (unsigned int)(tick);
	}
      }
      
      for(auto& r : result) if(!r.valid()) r.Set(0,0);

      for(size_t plane=0; plane <= larcv::supera::Nplanes(); ++plane)
	  
	LARCV_INFO() << "Single MCShower ... Plane " << plane 
		     << " bound " << result[plane].Start() << " => " << result[plane].End() << std::endl;
      
      return result;
    }

    template<class T, class U, class V>
    std::vector<larcv::ImageMeta> Cropper<T,U,V>::WTRange2BB(const WTRangeArray_t& wtrange_v) const
    {
      LARCV_DEBUG() << "start" << std::endl;
      std::vector<larcv::ImageMeta> bb_v;
      if(wtrange_v.empty()) return bb_v;

      bb_v.reserve(wtrange_v.size()-1);
      for(size_t i=0; i<wtrange_v.size()-1; ++i) {
	auto const& wrange = wtrange_v[i];
	auto const& trange = wtrange_v.back();
	double width,height;
	double origin_x, origin_y;
	size_t rows,cols;
	width=wrange.End()-wrange.Start();
	height=trange.End()-trange.Start();
	origin_x=wrange.Start();
	origin_y=trange.End();
	rows = trange.End() - trange.Start();
	cols = wrange.End() - wrange.Start();
	//if(wrange.valid() && (wrange.End() - wrange.Start())) {
	if(wrange.valid() && wrange.Start()) {
	  width = wrange.End() - wrange.Start() + 1 + 2 * _wire_padding;
	  cols = wrange.End() - wrange.Start() + 1 + 2 * _wire_padding;
	  origin_x = wrange.Start() - _wire_padding;
	}
	//if(trange.valid() && (trange.End() - trange.Start())) {
	if(trange.valid() && trange.Start()) {
	  height = trange.End() - trange.Start() + 1 + 2 * _time_padding;
	  rows = trange.End() - trange.Start() + 1 + 2 * _time_padding;
	  origin_y = trange.End() + _time_padding;
	}

	if( (width - 2 * _wire_padding) < _min_width || (height - 2 * _time_padding) < _min_height ) {
	  LARCV_INFO() << "Making an empty ImageMeta (too small) based on WTRange_t for Plane "<< i << std::endl
		       << "      W: " << wrange.Start() << " => " << wrange.End() << (wrange.valid() ? " good" : " bad")
		       << " ... T: " << trange.Start() << " => " << trange.End() << (trange.valid() ? " good" : " bad") << std::endl;
	}else{
	  LARCV_INFO() << "Constructing ImageMeta from WTRange_t for Plane "<< i
		       << " w/ padding (w,t) = (" << _wire_padding << "," << _time_padding << ")" << std::endl
		       << "      W: " << wrange.Start() << " => " << wrange.End() << (wrange.valid() ? " good" : " bad")
		       << " ... T: " << trange.Start() << " => " << trange.End() << (trange.valid() ? " good" : " bad") << std::endl
		       << "      Origin: (" << origin_x << "," << origin_y << ")" << std::endl
		       << "      Rows = " << rows << " Cols = " << cols << " ... Height = " << height << " Width = " << width << std::endl;
	  bb_v.emplace_back(width,height,rows,cols,origin_x,origin_y,i);
	}

      }
      return bb_v;
    }

    template<class T, class U, class V>
    ::larcv::ROI Cropper<T,U,V>::ParticleROI( const T& mct ) const
    {
      LARCV_DEBUG() << "start" << std::endl;
      LARCV_INFO() << "Assessing MCTrack G4Track ID = " << mct.TrackID() << " PdgCode " << mct.PdgCode() << std::endl;
      auto wtrange_v = WireTimeBoundary(mct);
      auto bb_v = WTRange2BB(wtrange_v);
      ::larcv::ROI res;
      res.SetBB(bb_v);
      res.Shape(::larcv::kShapeTrack);
      res.Type(::larcv::PdgCode2ROIType(mct.PdgCode()));
      if(mct.size())
	res.EnergyDeposit(mct.front().E() - mct.back().E());
      else
	res.EnergyDeposit(0);
      res.EnergyInit(mct.Start().E());
      res.Position(mct.Start().X(),mct.Start().Y(),mct.Start().Z(),mct.Start().T());
      res.Momentum(mct.Start().Px(),mct.Start().Py(),mct.Start().Pz());
      res.PdgCode(mct.PdgCode());
      res.ParentPdgCode(mct.MotherPdgCode());
      res.TrackID(mct.TrackID());
      res.ParentTrackID(mct.MotherTrackID());
      res.ParentPosition(mct.MotherStart().X(),
			 mct.MotherStart().Y(),
			 mct.MotherStart().Z(),
			 mct.MotherStart().T());
      res.ParentMomentum(mct.MotherStart().Px(),
			 mct.MotherStart().Py(),
			 mct.MotherStart().Pz());
			 
      return res;
    }

    template<class T, class U, class V>
    ::larcv::ROI Cropper<T,U,V>::ParticleROI( const T& mct, const std::vector<V>& sch_v ) const
    {
      LARCV_DEBUG() << "start" << std::endl;
      LARCV_INFO() << "Assessing MCTrack G4Track ID = " << mct.TrackID() << " PdgCode " << mct.PdgCode() << std::endl;
      auto wtrange_v = WireTimeBoundary(mct,sch_v);
      auto bb_v = WTRange2BB(wtrange_v);
      ::larcv::ROI res;
      res.SetBB(bb_v);
      res.Shape(::larcv::kShapeTrack);
      res.Type(::larcv::PdgCode2ROIType(mct.PdgCode()));
      if(mct.size())
	res.EnergyDeposit(mct.front().E() - mct.back().E());
      else
	res.EnergyDeposit(0);
      res.EnergyInit(mct.Start().E());
      res.Position(mct.Start().X(),mct.Start().Y(),mct.Start().Z(),mct.Start().T());
      res.Momentum(mct.Start().Px(),mct.Start().Py(),mct.Start().Pz());
      res.PdgCode(mct.PdgCode());
      res.ParentPdgCode(mct.MotherPdgCode());
      res.TrackID(mct.TrackID());
      res.ParentTrackID(mct.MotherTrackID());
      res.ParentPosition(mct.MotherStart().X(),
			 mct.MotherStart().Y(),
			 mct.MotherStart().Z(),
			 mct.MotherStart().T());
      res.ParentMomentum(mct.MotherStart().Px(),
			 mct.MotherStart().Py(),
			 mct.MotherStart().Pz());
      return res;
    }

    template <class T, class U, class V>
    ::larcv::ROI Cropper<T,U,V>::ParticleROI( const U& mcs ) const
    {
      LARCV_DEBUG() << "start" << std::endl;
      LARCV_INFO() << "Assessing MCShower G4Track ID = " << mcs.TrackID() << " PdgCode " << mcs.PdgCode() << std::endl;
      auto wtrange_v = WireTimeBoundary(mcs);
      auto bb_v = WTRange2BB(wtrange_v);
      ::larcv::ROI res;
      res.SetBB(bb_v);
      res.Shape(::larcv::kShapeShower);
      res.Type(::larcv::PdgCode2ROIType(mcs.PdgCode()));
      res.EnergyDeposit(mcs.DetProfile().E());
      res.EnergyDeposit(0);
      res.EnergyInit(mcs.Start().E());
      res.Position(mcs.Start().X(),mcs.Start().Y(),mcs.Start().Z(),mcs.Start().T());
      res.Momentum(mcs.Start().Px(),mcs.Start().Py(),mcs.Start().Pz());
      res.PdgCode(mcs.PdgCode());
      res.ParentPdgCode(mcs.MotherPdgCode());
      res.TrackID(mcs.TrackID());
      res.ParentTrackID(mcs.MotherTrackID());
      res.ParentPosition(mcs.MotherStart().X(),
			 mcs.MotherStart().Y(),
			 mcs.MotherStart().Z(),
			 mcs.MotherStart().T());
      res.ParentMomentum(mcs.MotherStart().Px(),
			 mcs.MotherStart().Py(),
			 mcs.MotherStart().Pz());
      return res;
    }

    template <class T, class U, class V>
    ::larcv::ROI Cropper<T,U,V>::ParticleROI( const U& mcs, const std::vector<V>& sch_v ) const
    {
      LARCV_DEBUG() << "start" << std::endl;
      LARCV_INFO() << "Assessing MCShower G4Track ID = " << mcs.TrackID() << " PdgCode " << mcs.PdgCode() << std::endl;
      auto wtrange_v = WireTimeBoundary(mcs,sch_v);
      auto bb_v = WTRange2BB(wtrange_v);
      ::larcv::ROI res;
      res.SetBB(bb_v);
      res.Shape(::larcv::kShapeShower);
      res.Type(::larcv::PdgCode2ROIType(mcs.PdgCode()));
      res.EnergyDeposit(mcs.DetProfile().E());
      res.EnergyDeposit(0);
      res.EnergyInit(mcs.Start().E());
      res.Position(mcs.Start().X(),mcs.Start().Y(),mcs.Start().Z(),mcs.Start().T());
      res.Momentum(mcs.Start().Px(),mcs.Start().Py(),mcs.Start().Pz());
      res.PdgCode(mcs.PdgCode());
      res.ParentPdgCode(mcs.MotherPdgCode());
      res.TrackID(mcs.TrackID());
      res.ParentTrackID(mcs.MotherTrackID());
      res.ParentPosition(mcs.MotherStart().X(),
			 mcs.MotherStart().Y(),
			 mcs.MotherStart().Z(),
			 mcs.MotherStart().T());
      res.ParentMomentum(mcs.MotherStart().Px(),
			 mcs.MotherStart().Py(),
			 mcs.MotherStart().Pz());
      return res;
    }

    /*
    Range_t Cropper::Format(const Range_t& range, unsigned short plane_id) const
    {
      // we check the quality of Range_t
      // if empty we return empty range
      // if inconsistent we throw

      // never filled
      if ( larcaffe::RangeEmpty(range)  ) {
	//std::cout << "Empty Range" << std::endl;
	Range_t emptyrange(0,0);
	return emptyrange;
      }
      
      if ( !larcaffe::RangeValidity(range) ) {
	//std::cout << "Invalid Range" << std::endl;
	Range_t emptyrange(0,0);
	return emptyrange;
      }

      if(range.End() < range.Start()) {
	logger().LOG(msg::kERROR,__FUNCTION__,__LINE__)
	  << "Inconsistent boundary given: max (" << range.End() << ") < min (" << range.Start() << ") !" << std::endl;
	throw larbys();
      }
      if(plane_id > larcv::supera::Nplanes()) {
	logger().LOG(msg::kERROR,__FUNCTION__,__LINE__)
	  << "Invalid plane ID (" << plane_id << ") ... must be <= " << larcv::supera::Nplanes() << std::endl;
	throw larbys();
      }
      
      const bool wire = plane_id < larcv::supera::Nplanes();
      const int target_width = ( wire ? _target_width : _target_height );
      const int padding = ( wire ? _wire_padding : _time_padding );
      const int max = ( wire ? (::larcv::supera::Nwires(plane_id)) : (::larcv::supera::NumberTimeSamples()));

      if(target_width > max) {
        logger().LOG(msg::kERROR,__FUNCTION__,__LINE__)
	  << "Image size (" << target_width << ") too large for plane " << plane_id << " (only goes 0=>" << max << ")!" << std::endl;
	throw larbys();
      }
      if(_compression_factor && (int)(target_width * _compression_factor) > max) {
        logger().LOG(msg::kERROR,__FUNCTION__,__LINE__)
	  << "Image size (" << target_width * _compression_factor << ") too large for plane " << plane_id << " (only goes 0=>" << max << ")!" << std::endl;
	throw larbys();
      }
      
      Range_t result;

      if((int)(range.Start()) > max) {
        if(logger().info())
	  logger().LOG(msg::kINFO,__FUNCTION__,__LINE__)
	    << "Image lower bound (" << range.Start() << ") too large for plane " << plane_id << " (only goes 0=>" << max << ")" << std::endl;
	return result;
      }
      
      // checks are over
      result.setFilled();

      if(logger().debug())
	logger().LOG(msg::kINFO,__FUNCTION__,__LINE__)
          << "Set bounds: target " << target_width  << std::endl;

      const int center = ( range.Start()  + range.End() ) / 2;

      int upper_bound, lower_bound;
      if(!_compression_factor) {
	upper_bound = center + (((int)(range.End()) - center + padding) / target_width) * target_width - 1;
	lower_bound = center - ((center - (int)(range.Start()) - padding) / target_width) * target_width ;
	if(upper_bound < (int)(range.End())) upper_bound += target_width;
	if(lower_bound > (int)(range.Start()) ) lower_bound -= target_width;
      }else{
	upper_bound = center + _compression_factor * target_width / 2 - 1;
	lower_bound = center - _compression_factor * target_width / 2;
      }

      if(logger().debug())
	logger().LOG(msg::kINFO,__FUNCTION__,__LINE__)
	  << "Preliminary bounds: " << lower_bound << " => " << upper_bound << std::endl;

      // Case 1: extension do not cross hard-limit boundaries
      if(lower_bound >= 0 && upper_bound < max) {
	result.Start()  = lower_bound;
	result.End() = upper_bound;
	if(logger().info())
	  logger().LOG(msg::kINFO,__FUNCTION__,__LINE__) 
	    << "Range [a] @ plane " << plane_id
	    << " ... Before " << range.Start()  << " => " << range.End()
	    << " ... After " << result.Start() << " => " << result.End()
	    << std::endl;
	return result;
      }
      
      // Case 2: touching only the max bound
      if(lower_bound >=0 && upper_bound >= max) {
	// just need to cover range min from the max-edge
	result.End() = max - 1;
	if(!_compression_factor) {
	  result.Start()  = max - target_width * ((max - (int)(range.Start()) - padding) / target_width);
	  if(result.Start() > range.Start() && (int)(result.Start()) > target_width) result.Start() -= target_width;
	}
	else result.Start() = max - (target_width * _compression_factor);

	if(logger().info())
	  logger().LOG(msg::kINFO,__FUNCTION__,__LINE__) 
	    << "Range [b] @ plane " << plane_id
	    << " ... Before " << range.Start()  << " => " << range.End()
	    << " ... After " << result.Start() << " => " << result.End()
	    << std::endl;
	return result;
      }
      
      // Case3: touching only the min bound
      if(upper_bound < max && lower_bound < 0) {
	result.Start()  = 0; // set to lower bound
	if(!_compression_factor) {
	  result.End() = target_width * (((int)(range.End()) + padding) / target_width);
	  if((result.End() < range.End()) || range.End()==0 ) result.End() += target_width;
	  if((int)(result.End()) >= max) result.End() -= target_width;
	  result.End() -= 1;
	}
	else result.End() = target_width * _compression_factor - 1;
	
	if(logger().info())
	  logger().LOG(msg::kINFO,__FUNCTION__,__LINE__) 
	    << "Range [c] @ plane " << plane_id
	    << " ... Before " << range.Start()  << " => " << range.End()
	    << " ... After " << result.Start() << " => " << result.End()
	    << std::endl;
	return result;
      }
      
      // Case 4: touching both bounds
      if(!_compression_factor) {
	while(upper_bound >= max) upper_bound -= target_width;
	while(lower_bound <  0  ) lower_bound += target_width;
      }else{
	logger().LOG(msg::kCRITICAL,__FUNCTION__,__LINE__)
	  << "Logic error: for a fixed scale factor this error should not be raised..." << std::endl;
	throw larbys();
      }

      result.Start()  = lower_bound;
      result.End() = upper_bound;
      
      if(logger().info())
	logger().LOG(msg::kINFO,__FUNCTION__,__LINE__) 
	  << "Range [d] @ plane " << plane_id
	  << " ... Before " << range.Start()  << " => " << range.End()
	  << " ... After " << result.Start() << " => " << result.End()
	  << std::endl;
      return result;
    }
    
    RangeArray_t Cropper::Format(const RangeArray_t& boundary) const
    {
      if(boundary.size() != larcv::supera::Nplanes()+1) {
	logger().LOG(msg::kERROR,__FUNCTION__,__LINE__)
	  << "Size of handed boundary is not # planes (wires) + 1 (time)!" << std::endl;
	throw larbys();
      }
      
      RangeArray_t result(boundary.size());   
      
      for(size_t plane=0; plane <= larcv::supera::Nplanes(); ++plane)
	
	result[plane] = Format(boundary[plane], plane);
      
      return result;
    }
    */  
  }
}
#endif

