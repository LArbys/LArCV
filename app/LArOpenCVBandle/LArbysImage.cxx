#ifndef __LARBYSIMAGE_CXX__
#define __LARBYSIMAGE_CXX__

#include "LArbysImage.h"
#include "Base/ConfigManager.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static LArbysImageProcessFactory __global_LArbysImageProcessFactory__;

  LArbysImage::LArbysImage(const std::string name)
    : ProcessBase(name)
  {}
    
  void LArbysImage::configure(const PSet& cfg)
  {
    _track_producer  = cfg.get<std::string>("TrackImageProducer");
    _shower_producer = cfg.get<std::string>("ShowerImageProducer","");
    _output_producer = cfg.get<std::string>("OutputImageProducer","");

    _process_count = 0;
    _process_time_image_extraction = 0;
    _process_time_analyze = 0;
    _process_time_cluster_storage = 0;
    
    _charge_to_gray_scale = cfg.get<double>("Q2Gray");
    _charge_max = cfg.get<double>("QMax");
    _charge_min = cfg.get<double>("QMin");
    _plane_weights = cfg.get<std::vector<float> >("MatchPlaneWeights");
    _debug = cfg.get<bool>("Debug");

    ::fcllite::PSet copy_cfg(_alg_mgr.Name(),cfg.get_pset(_alg_mgr.Name()).data_string());
    _alg_mgr.Configure(copy_cfg.get_pset(_alg_mgr.Name()));
    _alg_mgr.MatchPlaneWeights() = _plane_weights;

    auto const output_cluster_alg_name = cfg.get<std::string>("OutputClusterAlgName","");
    _output_cluster_alg_id = ::larocv::kINVALID_ALGO_ID;
    if(!output_cluster_alg_name.empty())
      _output_cluster_alg_id = _alg_mgr.GetClusterAlgID(output_cluster_alg_name);
  }

  void LArbysImage::initialize()
  {
    //_eui = new ::larlite::event_user;
    //_tree = new TTree("tree","");
    //_tree->Branch("user_info",&_eui);
  }

  bool LArbysImage::process(IOManager& mgr)
  {
    //_eui->clear_data();
    
    _track_img_mgr.clear();
    _shower_img_mgr.clear();
    _alg_mgr.ClearData();

    ::larocv::Watch watch_all, watch_one;
    watch_all.Start();
    watch_one.Start();

    this->extract_image(mgr);

    _process_time_image_extraction += watch_one.WallTime();

    for (size_t plane = 0; plane < _track_img_mgr.size(); ++plane) {

      auto const& img  = _track_img_mgr.img_at(plane);
      auto      & meta = _track_img_mgr.meta_at(plane);
      auto const& roi  = _track_img_mgr.roi_at(plane);

      //if (_debug) meta.set_ev_user(_eui);

      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;

      _alg_mgr.Add(img, meta, roi, 0);

    }

    for (size_t plane = 0; plane < _shower_img_mgr.size(); ++plane) {

      auto const& img  = _shower_img_mgr.img_at(plane);
      auto      & meta = _shower_img_mgr.meta_at(plane);
      auto const& roi  = _shower_img_mgr.roi_at(plane);

      //if (_debug) meta.set_ev_user(_eui);

      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;

      _alg_mgr.Add(img, meta, roi, 1);

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

    auto ev_track_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_track_producer));

    if(!ev_track_image) {
      LARCV_CRITICAL() << "Image by " << _track_producer << " not found..." << std::endl;
      throw larbys();
    }

    EventImage2D* ev_shower_image = nullptr;

    if(!_shower_producer.empty()) {
      ev_shower_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_shower_producer));
      if(!ev_shower_image) {
	LARCV_CRITICAL() << "Image by " << _shower_producer << " not found..." << std::endl;
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
  
  void LArbysImage::extract_image(IOManager& mgr) {
    LARCV_DEBUG() << "Extracting Image\n" << std::endl;

    auto ev_track_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_track_producer));

    if(!ev_track_image) {
      LARCV_CRITICAL() << "Image by " << _track_producer << " not found..." << std::endl;
      throw larbys();
    }

    EventImage2D* ev_shower_image = nullptr;
    
    if(!_shower_producer.empty()) {
      ev_shower_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_shower_producer));
      if(!ev_shower_image) {
	LARCV_CRITICAL() << "Image by " << _shower_producer << " not found..." << std::endl;
	throw larbys();
      }
    }

    auto const& track_image_v = ev_track_image->Image2DArray();
    
    for(size_t i=0; i<track_image_v.size(); ++i) {

      auto const& cvmeta = track_image_v[i].meta();

      LARCV_DEBUG() << "Reading image (rows,cols) = (" << cvmeta.rows() << "," << cvmeta.cols() << ") "
		    << " ... (height,width) = (" << cvmeta.height() << "," << cvmeta.width() << ")" << std::endl;
	

      ::larocv::ImageMeta meta(cvmeta.width(), cvmeta.height(),
			       cvmeta.cols(),  cvmeta.rows(),
			       cvmeta.min_y(), cvmeta.min_x(), i);

      LARCV_DEBUG() << "LArOpenCV meta @ plane " << i << " ... "
		    << "Reading image (rows,cols) = (" << meta.num_pixel_row() << "," << meta.num_pixel_column() << ") "
		    << " ... (height,width) = (" << meta.height() << "," << meta.width() << ")" << std::endl;
      
      _track_img_mgr.push_back(::cv::Mat(cvmeta.cols(),cvmeta.rows(),CV_8UC1,cvScalar(0.)),meta,::larocv::ROI());

      auto& mat = _track_img_mgr.img_at(i);

      for(size_t row=0; row<cvmeta.rows(); ++row) {
	for(size_t col=0; col<cvmeta.cols(); ++col) {

	  float charge = track_image_v[i].pixel(row,col);
	  charge -= _charge_min;
	  if(charge < 0) charge = 0;
	  if(charge > _charge_max) charge = _charge_max;
	  charge /= _charge_to_gray_scale;
	  mat.at<unsigned char>(col,cvmeta.rows()-1-row) = (unsigned char)((int)charge);
	  //mat.at<unsigned char>(cvmeta.rows()-1-row,col) = (unsigned char)((int)charge);
	}
      }
    }

    if(ev_shower_image) {

      auto const& shower_image_v = ev_shower_image->Image2DArray();
      for(size_t i=0; i<shower_image_v.size(); ++i) {
	
	auto const& cvmeta = shower_image_v[i].meta();
	
	LARCV_DEBUG() << "Reading image (rows,cols) = (" << cvmeta.rows() << "," << cvmeta.cols() << ") "
		      << " ... (height,width) = (" << cvmeta.height() << "," << cvmeta.width() << ")" << std::endl;
	
	
	::larocv::ImageMeta meta(cvmeta.width(), cvmeta.height(),
				 cvmeta.cols(),  cvmeta.rows(),
				 cvmeta.min_y(), cvmeta.min_x(), i);
	
	LARCV_DEBUG() << "LArOpenCV meta @ plane " << i << " ... "
		      << "Reading image (rows,cols) = (" << meta.num_pixel_row() << "," << meta.num_pixel_column() << ") "
		      << " ... (height,width) = (" << meta.height() << "," << meta.width() << ")" << std::endl;
	
	_shower_img_mgr.push_back(::cv::Mat(cvmeta.cols(),cvmeta.rows(),CV_8UC1,cvScalar(0.)),meta,::larocv::ROI());
	
	auto& mat = _shower_img_mgr.img_at(i);
	
	for(size_t row=0; row<cvmeta.rows(); ++row) {
	  for(size_t col=0; col<cvmeta.cols(); ++col) {
	    
	    float charge = shower_image_v[i].pixel(row,col);
	    charge -= _charge_min;
	    if(charge < 0) charge = 0;
	    if(charge > _charge_max) charge = _charge_max;
	    charge /= _charge_to_gray_scale;
	    mat.at<unsigned char>(col,cvmeta.rows()-1-row) = (unsigned char)((int)charge);
	    //mat.at<unsigned char>(cvmeta.rows()-1-row,col) = (unsigned char)((int)charge);
	  }
	}
      }
    }
  }

  void LArbysImage::finalize()
  {
    if ( has_ana_file() ) {
      _alg_mgr.Finalize(&(ana_file()));
      _tree->Write();
    }
  }

}
#endif
