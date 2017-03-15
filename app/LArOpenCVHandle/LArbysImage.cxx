#ifndef __LARBYSIMAGE_CXX__
#define __LARBYSIMAGE_CXX__

#include "LArbysImage.h"
#include "Base/ConfigManager.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include "LArbysImageOut.h"
#include "LArbysImageResult.h"
namespace larcv {

  static LArbysImageProcessFactory __global_LArbysImageProcessFactory__;

  LArbysImage::LArbysImage(const std::string name)
    : ProcessBase(name),
      _PreProcessor(),
      _TrackShowerAna(),
      _LArbysImageMaker(),
      _LArbysImageAnaBase_ptr(nullptr)
  {}
      
  void LArbysImage::configure(const PSet& cfg)
  {
    _adc_producer    = cfg.get<std::string>("ADCImageProducer");
    _track_producer  = cfg.get<std::string>("TrackImageProducer","");
    _shower_producer = cfg.get<std::string>("ShowerImageProducer","");
    _adc_producer    = cfg.get<std::string>("ROIProducer","");
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

    auto const ana_class_name = cfg.get<std::string>("LArbysImageAnaClass","");
    if(ana_class_name.empty()) return;

    if      (ana_class_name == "LArbysImageOut"    ) _LArbysImageAnaBase_ptr = new LArbysImageOut("LArbysImageOut");
    else if (ana_class_name == "LArbysImageResult" ) _LArbysImageAnaBase_ptr = new LArbysImageResult("LArbysImageResult");
    else {
      LARCV_CRITICAL() << "LArbysImageAna class name " << ana_class_name << " not recognized..." << std::endl;
      throw larbys();
    }
    
    if(_LArbysImageAnaBase_ptr)
      _LArbysImageAnaBase_ptr->Configure(cfg.get<larcv::PSet>("LArbysImageAnaConfig"));
  }
  
  void LArbysImage::initialize()
  {
    if( _LArbysImageAnaBase_ptr ) _LArbysImageAnaBase_ptr->Initialize();
  }

  const std::vector<larcv::Image2D>& LArbysImage::get_image2d(IOManager& mgr, std::string producer) {

    LARCV_DEBUG() << "Extracting " << producer << " Image\n" << std::endl;
    if(!producer.empty()) {
      auto ev_image = (EventImage2D*)(mgr.get_data(kProductImage2D,producer));
      if(!ev_image) {
	LARCV_CRITICAL() << "Image by producer " << producer << " not found..." << std::endl;
	throw larbys();
      }
      return ev_image->Image2DArray();
    }
    return _empty_image_v;
  }
  
  bool LArbysImage::process(IOManager& mgr)
  {
    LARCV_DEBUG() << "Process index " << mgr.current_entry() << std::endl;

    bool status = true;

    if(_LArbysImageAnaBase_ptr) {
      auto const& event_id = mgr.event_id();
      _LArbysImageAnaBase_ptr->EventID(mgr.current_entry(),
				       event_id.run(), event_id.subrun(), event_id.event());
    }

    if(_roi_producer.empty()) {

      auto const& adc_image_v    = get_image2d(mgr,_adc_producer);
      auto const& track_image_v  = get_image2d(mgr,_track_producer);
      auto const& shower_image_v = get_image2d(mgr,_shower_producer);

      assert(adc_image_v.size());
      assert(track_image_v.empty()  || adc_image_v.size() == track_image_v.size());
      assert(shower_image_v.empty() || adc_image_v.size() == shower_image_v.size());

      status = Reconstruct(adc_image_v,track_image_v,shower_image_v);

      if(_LArbysImageAnaBase_ptr) _LArbysImageAnaBase_ptr->Analyze(_alg_mgr);

      return status;
							       
    }else{

      auto const& adc_image_v    = get_image2d(mgr,_adc_producer);
      auto const& track_image_v  = get_image2d(mgr,_track_producer);
      auto const& shower_image_v = get_image2d(mgr,_shower_producer);

      assert(adc_image_v.size());
      assert(track_image_v.empty()  || adc_image_v.size() == track_image_v.size());
      assert(shower_image_v.empty() || adc_image_v.size() == shower_image_v.size());

      LARCV_DEBUG() << "Extracting " << _roi_producer << " ROI\n" << std::endl;
      if( !(mgr.get_data(kProductROI,_roi_producer)) ) {
	LARCV_CRITICAL() << "ROI by producer " << _roi_producer << " not found..." << std::endl;
	throw larbys();
      }
      auto const& roi_v = ((EventROI*)(mgr.get_data(kProductROI,_roi_producer)))->ROIArray();

      for(auto const& roi : roi_v) {

	auto const& bb_v = roi.BB();
	assert(bb_v.size() == adc_image_v.size());

	std::vector<larcv::Image2D> crop_adc_image_v;
	std::vector<larcv::Image2D> crop_track_image_v;
	std::vector<larcv::Image2D> crop_shower_image_v;

	for(size_t plane=0; plane<bb_v.size(); ++plane) {

	  auto const& bb           = bb_v[plane];
	  
	  auto const& adc_image    = adc_image_v[plane];
	  crop_adc_image_v.emplace_back(std::move(adc_image.crop(bb)));

	  if(!track_image_v.empty()) {
	    auto const& track_image  = track_image_v[plane];
	    crop_track_image_v.emplace_back(std::move(track_image.crop(bb)));
	  }

	  if(!shower_image_v.empty()) {
	    auto const& shower_image = shower_image_v[plane];
	    crop_shower_image_v.emplace_back(std::move(shower_image.crop(bb)));
	  }
	}

	status = status && Reconstruct(crop_adc_image_v, crop_track_image_v, crop_shower_image_v);

	if(_LArbysImageAnaBase_ptr) _LArbysImageAnaBase_ptr->Analyze(_alg_mgr);
      
      }
    }
    return status;
  }

  bool LArbysImage::Reconstruct(const std::vector<larcv::Image2D>& adc_image_v,
				const std::vector<larcv::Image2D>& track_image_v,
				const std::vector<larcv::Image2D>& shower_image_v)
  {
    _adc_img_mgr.clear();
    _track_img_mgr.clear();
    _shower_img_mgr.clear();
    _alg_mgr.ClearData();

    ::larocv::Watch watch_all, watch_one;
    watch_all.Start();
    watch_one.Start();

    for(auto& img_data : _LArbysImageMaker.ExtractImage(adc_image_v))
      _adc_img_mgr.emplace_back(std::move(std::get<0>(img_data)),std::move(std::get<1>(img_data)));
    
    for(auto& img_data : _LArbysImageMaker.ExtractImage(track_image_v)) 
      _track_img_mgr.emplace_back(std::move(std::get<0>(img_data)),std::move(std::get<1>(img_data)));
    
    
    for(auto& img_data : _LArbysImageMaker.ExtractImage(shower_image_v))
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
     
    _process_time_cluster_storage += watch_one.WallTime();

    _process_time_analyze += watch_all.WallTime();

    ++_process_count;

    _tree->Fill();
    
    return true;
  }

  void LArbysImage::finalize()
  {
    if ( has_ana_file() ) 
      _alg_mgr.Finalize(&(ana_file()));

    if ( _tsanalyze ) 
      _TrackShowerAna.Finalize(&(ana_file()));

    if ( _LArbysImageAnaBase_ptr ) {
      if( has_ana_file() )
	_LArbysImageAnaBase_ptr->Finalize(&(ana_file()));
      else
	_LArbysImageAnaBase_ptr->Finalize();
    }

  }

}
#endif
