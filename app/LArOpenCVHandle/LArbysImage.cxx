#ifndef __LARBYSIMAGE_CXX__
#define __LARBYSIMAGE_CXX__

#include "LArbysImage.h"
#include "Base/ConfigManager.h"

namespace larcv {

  static LArbysImageProcessFactory __global_LArbysImageProcessFactory__;

  LArbysImage::LArbysImage(const std::string name)
    : ProcessBase(name),
      _PreProcessor(),
      _TrackShowerAna(),
      _LArbysImageMaker()
  {}
      
  void LArbysImage::configure(const PSet& cfg)
  {
    _adc_producer    = cfg.get<std::string>("ADCImageProducer");
    _track_producer  = cfg.get<std::string>("TrackImageProducer","");
    _shower_producer = cfg.get<std::string>("ShowerImageProducer","");
    _output_producer = cfg.get<std::string>("OutputImageProducer","");

    _LArbysImageMaker.Configure(cfg.get<larcv::PSet>("LArbysImageMaker"));
    
    _preprocess = cfg.get<bool>("PreProcess",true);
    if (_preprocess) {
      LARCV_INFO() << "Preprocessing image" << std::endl;
      _PreProcessor.Configure(cfg.get<larcv::PSet>("PreProcessor"));
    }
    
    _tsanalyze = cfg.get<bool>("TSAnalyzeOnly",false);
    if (_tsanalyze) {
      LARCV_INFO() << "Analyzing Tracks and Showers only" << std::endl;
      _TrackShowerAna.Configure(cfg.get<larcv::PSet>("TrackShowerAnalysis"));
    }
    
    _process_count = 0;
    _process_time_image_extraction = 0;
    _process_time_analyze = 0;
    _process_time_cluster_storage = 0;
    
    _plane_weights = cfg.get<std::vector<float> >("MatchPlaneWeights");

    ::fcllite::PSet copy_cfg(_alg_mgr.Name(),cfg.get_pset(_alg_mgr.Name()).data_string());
    _alg_mgr.Configure(copy_cfg.get_pset(_alg_mgr.Name()));
    _alg_mgr.MatchPlaneWeights() = _plane_weights;

    auto const output_cluster_alg_name = cfg.get<std::string>("OutputClusterAlgName","");
    _output_cluster_alg_id = ::larocv::kINVALID_ALGO_ID;
    if(!output_cluster_alg_name.empty())
      _output_cluster_alg_id = _alg_mgr.GetClusterAlgID(output_cluster_alg_name);
  }
  
  void LArbysImage::initialize()
  { }
  
  bool LArbysImage::process(IOManager& mgr)
  {
    _adc_img_mgr.clear();
    _track_img_mgr.clear();
    _shower_img_mgr.clear();
    _alg_mgr.ClearData();

    ::larocv::Watch watch_all, watch_one;
    watch_all.Start();
    watch_one.Start();

    auto img_data_v = _LArbysImageMaker.ExtractImage(mgr,_adc_producer);
    for(auto& img_data : img_data_v)
      _adc_img_mgr.emplace_back(std::move(std::get<0>(img_data)),std::move(std::get<1>(img_data)));
    
    img_data_v = _LArbysImageMaker.ExtractImage(mgr,_track_producer);
    for(auto& img_data : img_data_v)
      _track_img_mgr.emplace_back(std::move(std::get<0>(img_data)),std::move(std::get<1>(img_data)));
    
    img_data_v = _LArbysImageMaker.ExtractImage(mgr,_shower_producer);
    for(auto& img_data : img_data_v)
      _shower_img_mgr.emplace_back(std::move(std::get<0>(img_data)),std::move(std::get<1>(img_data)));
    
    _process_time_image_extraction += watch_one.WallTime();

    for (size_t plane = 0; plane < _adc_img_mgr.size(); ++plane) {

      auto const& img  = _adc_img_mgr.img_at(plane);
      auto      & meta = _adc_img_mgr.meta_at(plane);
      auto const& roi  = _adc_img_mgr.roi_at(plane);

      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
      _alg_mgr.Add(img, meta, roi, 0);
    }

    for (size_t plane = 0; plane < _track_img_mgr.size(); ++plane) {

      auto const& img  = _track_img_mgr.img_at(plane);
      auto      & meta = _track_img_mgr.meta_at(plane);
      auto const& roi  = _track_img_mgr.roi_at(plane);

      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;

      _alg_mgr.Add(img, meta, roi, 1);

    }

    for (size_t plane = 0; plane < _shower_img_mgr.size(); ++plane) {

      auto const& img  = _shower_img_mgr.img_at(plane);
      auto      & meta = _shower_img_mgr.meta_at(plane);
      auto const& roi  = _shower_img_mgr.roi_at(plane);

      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;

      _alg_mgr.Add(img, meta, roi, 2);

    }
    LARCV_DEBUG() << "Process index " << mgr.current_entry() << std::endl;

    if (_preprocess) {
      //give a single plane @ a time to pre processor
      auto& adc_img_v= _alg_mgr.InputImagesRW(0,true);
      auto& trk_img_v= _alg_mgr.InputImagesRW(1,true);
      auto& shr_img_v= _alg_mgr.InputImagesRW(2,true);
      auto nplanes = adc_img_v.size();
      for(size_t plane_id=0;plane_id<nplanes;++plane_id) {
	LARCV_DEBUG() << "Preprocess image set @ "<< " plane " << plane_id << std::endl;
	if (!_PreProcessor.PreProcess(adc_img_v[plane_id],trk_img_v[plane_id],shr_img_v[plane_id])) {
	  LARCV_CRITICAL() << "... could not be preprocessed, abort!" << std::endl;
	  throw larbys();
	}
      }
    }
    
    if(_tsanalyze) {
      //given a single plane @ a time run track shower analyzis
      auto& adc_img_v= _alg_mgr.InputImages(0);
      auto& trk_img_v= _alg_mgr.InputImages(1);
      auto& shr_img_v= _alg_mgr.InputImages(2);
      auto nplanes = adc_img_v.size();
      for(size_t plane_id=0;plane_id<nplanes;++plane_id) {
    	LARCV_DEBUG() << "TrackShowerAnalyze image set @ "<< " plane " << plane_id << std::endl;
    	if (!_TrackShowerAna.Analyze(adc_img_v[plane_id],trk_img_v[plane_id],shr_img_v[plane_id])) {
    	  LARCV_CRITICAL() << "... could not be preprocessed, abort!" << std::endl;
    	  throw larbys();
    	}
      }      
      return true;
    }
    
    _alg_mgr.Process();

    watch_one.Start();
     
    return true;

    if(!_output_producer.empty())
      this->store_clusters(mgr);

    _process_time_cluster_storage += watch_one.WallTime();

    _process_time_analyze += watch_all.WallTime();

    ++_process_count;

    _tree->Fill();
    
    return true;
  }

  void LArbysImage::store_clusters(IOManager& mgr) {

    if(_output_producer.empty()) {
      LARCV_CRITICAL() << "Cannot store clusters when no output producer name is specified!" << std::endl;
      throw larbys();
    }

    auto ev_adc_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_adc_producer));
    if(!ev_adc_image) {
      LARCV_CRITICAL() << "Image by adc producer " << _adc_producer << " not found..." << std::endl;
      throw larbys();
    }
    
    EventImage2D* ev_track_image = nullptr;
    
    if(!_track_producer.empty()) {
      ev_track_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_track_producer));
      if(!ev_track_image) {
	LARCV_CRITICAL() << "Image by track producer " << _track_producer << " not found..." << std::endl;
	throw larbys();
      }
    }
    
    EventImage2D* ev_shower_image = nullptr;

    if(!_shower_producer.empty()) {
      ev_shower_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_shower_producer));
      if(!ev_shower_image) {
	LARCV_CRITICAL() << "Image by shower producer " << _shower_producer << " not found..." << std::endl;
	throw larbys();
      }
    }

    auto ev_image_out = (EventImage2D*)(mgr.get_data(kProductImage2D,_output_producer));
    if(!ev_image_out){
      LARCV_CRITICAL() << "Could not create an output image container by name " << _output_producer << std::endl;
      throw larbys();
    }

    // FIXME not supporting shower image output yet... just too busy for now...
    auto const& track_image_v = ev_track_image->Image2DArray();

    for(size_t i=0; i<track_image_v.size(); ++i) {
      
      auto const& cvmeta = track_image_v[i].meta();
      
      Image2D out_img(cvmeta);
      
      for(size_t row=0; row<cvmeta.rows(); ++row) {
	for(size_t col=0; col<cvmeta.cols(); ++col) {    
	  auto x = cvmeta.pos_x(col);
	  auto y = cvmeta.pos_y(row);
	  
	  auto cid = _alg_mgr.ClusterID(x,y,i,_output_cluster_alg_id);
	  if(cid == ::larocv::kINVALID_CLUSTER_ID) continue;
	  out_img.set_pixel(row,col,(float)cid+1.);
	}
      }
      ev_image_out->Emplace(std::move(out_img));
    }
  }
  
  
  void LArbysImage::finalize()
  {
    if ( has_ana_file() ) 
      _alg_mgr.Finalize(&(ana_file()));

    if ( _tsanalyze ) 
      _TrackShowerAna.Finalize(&(ana_file()));
    
  }

}
#endif
