#ifndef __LARBYSIMAGE_CXX__
#define __LARBYSIMAGE_CXX__

#include "LArbysImage.h"
#include "Base/ConfigManager.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include "LArbysImageOut.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/Contour2DAnalysis.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/ImagePatchAnalysis.h"
#include <array>

namespace larcv {

  static LArbysImageProcessFactory __global_LArbysImageProcessFactory__;

  LArbysImage::LArbysImage(const std::string name)
    : ProcessBase(name),
      _PreProcessor(),
      _LArbysImageMaker(),
      _reco_holder()
  {}
      
  void LArbysImage::configure(const PSet& cfg)
  {
    _adc_producer         = cfg.get<std::string>("ADCImageProducer");
    _track_producer       = cfg.get<std::string>("TrackImageProducer","");
    _shower_producer      = cfg.get<std::string>("ShowerImageProducer","");
    _thrumu_producer      = cfg.get<std::string>("ThruMuImageProducer","");
    _stopmu_producer      = cfg.get<std::string>("StopMuImageProducer","");
    _roi_producer         = cfg.get<std::string>("ROIProducer","");
    _output_producer      = cfg.get<std::string>("OutputImageProducer","");

    _mask_thrumu_pixels = cfg.get<bool>("MaskThruMu",false);
    _mask_stopmu_pixels = cfg.get<bool>("MaskStopMu",false);

    _LArbysImageMaker.Configure(cfg.get<larcv::PSet>("LArbysImageMaker"));

    _write_reco = cfg.get<bool>("WriteAnaReco");
    _reco_holder.Configure(cfg.get<larcv::PSet>("LArbysRecoHolder"));

    _preprocess = cfg.get<bool>("PreProcess",true);
    if (_preprocess) {
      LARCV_INFO() << "Preprocessing image" << std::endl;
      _PreProcessor.Configure(cfg.get<larcv::PSet>("PreProcessor"));
    }
    
    _process_count = 0;
    _process_time_image_extraction = 0;
    _process_time_analyze = 0;
    _process_time_cluster_storage = 0;
    
    _plane_weights = cfg.get<std::vector<float> >("MatchPlaneWeights",{1,1,1});

    ::fcllite::PSet copy_cfg(_alg_mgr.Name(),cfg.get_pset(_alg_mgr.Name()).data_string());
    _alg_mgr.Configure(copy_cfg.get_pset(_alg_mgr.Name()));
    _alg_mgr.MatchPlaneWeights() = _plane_weights;
  }
  
  void LArbysImage::initialize()
  {
    _thrumu_image_v.clear();
    _stopmu_image_v.clear();
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

  void LArbysImage::construct_cosmic_image(IOManager& mgr, std::string producer,
					   const std::vector<larcv::Image2D>& adc_image_v,
					   std::vector<larcv::Image2D>& mu_image_v) {
    LARCV_DEBUG() << "Constructing " << _thrumu_producer << " Pixel2D => Image2D" << std::endl;
    if(!producer.empty()) {
      auto ev_pixel2d = (EventPixel2D*)(mgr.get_data(kProductPixel2D,producer));
      if(!ev_pixel2d) {
	LARCV_CRITICAL() << "Pixel2D by producer " << producer << " not found..." << std::endl;
	throw larbys();
      }
      LARCV_DEBUG() << "Using Pixel2D producer " << producer << std::endl;
      auto const& pixel2d_m = ev_pixel2d->Pixel2DClusterArray();
      for(size_t img_idx=0; img_idx<adc_image_v.size(); ++img_idx) {
	auto const& meta = adc_image_v[img_idx].meta();
	if(mu_image_v.size() <= img_idx)
	  mu_image_v.emplace_back(larcv::Image2D(meta));
	if(mu_image_v[img_idx].meta() != meta)
	  mu_image_v[img_idx] = larcv::Image2D(meta);
	auto& mu_image = mu_image_v[img_idx];
	mu_image.paint(0);
	auto itr = pixel2d_m.find(img_idx);
	if(itr == pixel2d_m.end()) {
	  LARCV_DEBUG() << "No Pixel2D found for plane " << img_idx << std::endl;
	  continue;
	}
	else{
	  size_t npx_x = meta.cols();
	  size_t npx_y = meta.rows();
	  for(auto const& pixel_cluster : (*itr).second) {
	    for(auto const& pixel : pixel_cluster) {
	      if(pixel.X() >= npx_x || pixel.Y() >= npx_y) {
		LARCV_WARNING() << "Ignoring cosmic pixel (row,col) = ("
				<< pixel.Y() << "," << pixel.X() << ")"
				<< " as it is out of bounds (ncol=" << npx_x << ", nrow=" << npx_y << ")" << std::endl;
		continue;
	      }
	      mu_image.set_pixel( (pixel.X() * meta.rows() + pixel.Y()), 100 );
	    }
	  }
	}
      }
    }
  }

  void LArbysImage::mask_image(Image2D& target, const Image2D& ref)
  {
    LARCV_DEBUG() << "Masking: " << target.meta().dump() << std::flush;
    if(target.meta() != ref.meta()) {
      LARCV_CRITICAL() << "Cannot mask images w/ different meta!" << std::endl;
      throw larbys();
    }
    auto meta = target.meta();
    std::vector<float> data = target.move();
    auto const& ref_vec = ref.as_vector();

    for(size_t i=0; i<data.size(); ++i) { if(ref_vec[i]>0) data[i]=0; }	

    target.move(std::move(data));
  }
  
  bool LArbysImage::process(IOManager& mgr)
  {
    LARCV_DEBUG() << "Process index " << mgr.current_entry() << std::endl;

    bool status = true;

    if(_roi_producer.empty()) {
      size_t pidx = 0;          
      auto const& adc_image_v    = get_image2d(mgr,_adc_producer);
      auto const& track_image_v  = get_image2d(mgr,_track_producer);
      auto const& shower_image_v = get_image2d(mgr,_shower_producer);
      construct_cosmic_image(mgr, _thrumu_producer, adc_image_v, _thrumu_image_v);
      construct_cosmic_image(mgr, _stopmu_producer, adc_image_v, _stopmu_image_v);
      auto const& thrumu_image_v = _thrumu_image_v;
      auto const& stopmu_image_v = _stopmu_image_v;
      
      assert(adc_image_v.size());
      assert(track_image_v.empty()  || adc_image_v.size() == track_image_v.size());
      assert(shower_image_v.empty() || adc_image_v.size() == shower_image_v.size());
      assert(thrumu_image_v.empty() || adc_image_v.size() == thrumu_image_v.size());
      assert(stopmu_image_v.empty() || adc_image_v.size() == stopmu_image_v.size());

      bool mask_thrumu = _mask_thrumu_pixels && !thrumu_image_v.empty();
      bool mask_stopmu = _mask_stopmu_pixels && !stopmu_image_v.empty();

      if(!mask_thrumu && !mask_stopmu) {
	LARCV_DEBUG() << "Reconstruct no mask thrumu and no mask stopmu" << std::endl;
	status = Reconstruct(adc_image_v,
			     track_image_v,shower_image_v,
			     thrumu_image_v, stopmu_image_v);
	status = status && StoreParticles(mgr,_alg_mgr,adc_image_v,pidx);
      }else{
	auto copy_adc_image_v    = adc_image_v;
	auto copy_track_image_v  = track_image_v;
	auto copy_shower_image_v = shower_image_v;
	LARCV_DEBUG() << "ADC image size " << adc_image_v.size() << std::endl;
	for(size_t plane=0; plane<adc_image_v.size(); ++plane) {
	  
	  if(mask_thrumu) {
	    LARCV_DEBUG() << "Masking thrumu plane " << plane << std::endl;
	    mask_image(copy_adc_image_v[plane], thrumu_image_v[plane]);
	    if(!copy_track_image_v.empty() ) mask_image(copy_track_image_v[plane],  thrumu_image_v[plane]);
	    if(!copy_shower_image_v.empty()) mask_image(copy_shower_image_v[plane], thrumu_image_v[plane]);
	  }
	  
	  if(mask_stopmu) {
	    LARCV_DEBUG() << "Masking stopmu plane " << plane << std::endl;
	    mask_image(copy_adc_image_v[plane], stopmu_image_v[plane]);
	    if(!copy_track_image_v.empty() ) mask_image(copy_track_image_v[plane],  stopmu_image_v[plane]);
	    if(!copy_shower_image_v.empty()) mask_image(copy_shower_image_v[plane], stopmu_image_v[plane]);
	  }

	}
	LARCV_DEBUG() << "Reconstruct" << std::endl;
	status = Reconstruct(copy_adc_image_v,
			     copy_track_image_v,copy_shower_image_v,
			     thrumu_image_v, stopmu_image_v);
	status = status && StoreParticles(mgr,_alg_mgr,copy_adc_image_v,pidx);
      }
    }else{
      size_t pidx = 0;
      auto const& adc_image_v    = get_image2d(mgr,_adc_producer);
      auto const& track_image_v  = get_image2d(mgr,_track_producer);
      auto const& shower_image_v = get_image2d(mgr,_shower_producer);
      construct_cosmic_image(mgr, _thrumu_producer, adc_image_v, _thrumu_image_v);
      construct_cosmic_image(mgr, _stopmu_producer, adc_image_v, _stopmu_image_v);
      auto const& thrumu_image_v = _thrumu_image_v;
      auto const& stopmu_image_v = _stopmu_image_v;

      assert(adc_image_v.size());
      assert(track_image_v.empty()  || adc_image_v.size() == track_image_v.size());
      assert(shower_image_v.empty() || adc_image_v.size() == shower_image_v.size());
      assert(thrumu_image_v.empty() || adc_image_v.size() == thrumu_image_v.size());
      assert(stopmu_image_v.empty() || adc_image_v.size() == stopmu_image_v.size());

      LARCV_DEBUG() << "Extracting " << _roi_producer << " ROI\n" << std::endl;
      if( !(mgr.get_data(kProductROI,_roi_producer)) ) {
	LARCV_CRITICAL() << "ROI by producer " << _roi_producer << " not found..." << std::endl;
	throw larbys();
      }
      auto const& roi_v = ((EventROI*)(mgr.get_data(kProductROI,_roi_producer)))->ROIArray();
      
      for(auto const& roi : roi_v) {
	LARCV_DEBUG() << " @ fresh roi " << &roi << std::endl;
	auto const& bb_v = roi.BB();
	assert(bb_v.size() == adc_image_v.size());

	std::vector<larcv::Image2D> crop_adc_image_v;
	std::vector<larcv::Image2D> crop_track_image_v;
	std::vector<larcv::Image2D> crop_shower_image_v;
	std::vector<larcv::Image2D> crop_thrumu_image_v;
	std::vector<larcv::Image2D> crop_stopmu_image_v;

	for(size_t plane=0; plane<bb_v.size(); ++plane) {

	  auto const& bb           = bb_v[plane];
	  auto const& adc_image    = adc_image_v[plane];
	  
	  crop_adc_image_v.emplace_back(adc_image.crop(bb));

	  if(!track_image_v.empty()) {
	    auto const& track_image  = track_image_v[plane];
	    crop_track_image_v.emplace_back(track_image.crop(bb));
	  }

	  if(!shower_image_v.empty()) {
	    auto const& shower_image = shower_image_v[plane];
	    crop_shower_image_v.emplace_back(shower_image.crop(bb));
	  }

	  if(!thrumu_image_v.empty()) {
	    auto const& thrumu_image = thrumu_image_v[plane];
	    crop_thrumu_image_v.emplace_back(thrumu_image.crop(bb));
	  }

	  if(!stopmu_image_v.empty()) {
	    auto const& stopmu_image = stopmu_image_v[plane];
	    crop_stopmu_image_v.emplace_back(stopmu_image.crop(bb));
	  }

	  if(!crop_thrumu_image_v.empty() && _mask_thrumu_pixels) {
	    LARCV_DEBUG() << "Masking thrumu plane " << plane << std::endl;
	    mask_image(crop_adc_image_v[plane], crop_thrumu_image_v[plane]);
	    if(!crop_track_image_v.empty() ) mask_image(crop_track_image_v[plane],  crop_thrumu_image_v[plane]);
	    if(!crop_shower_image_v.empty()) mask_image(crop_shower_image_v[plane], crop_thrumu_image_v[plane]);
	  }

	  if(!crop_stopmu_image_v.empty() && _mask_stopmu_pixels) {
	    LARCV_DEBUG() << "Masking stopmu plane " << plane << std::endl;
	    mask_image(crop_adc_image_v[plane], crop_stopmu_image_v[plane]);
	    if(!crop_track_image_v.empty() ) mask_image(crop_track_image_v[plane],  crop_stopmu_image_v[plane]);
	    if(!crop_shower_image_v.empty()) mask_image(crop_shower_image_v[plane], crop_stopmu_image_v[plane]);
	  }

	}
	
	LARCV_DEBUG() << "Reconstruct" << std::endl;

	status = status && Reconstruct(crop_adc_image_v,
				       crop_track_image_v, crop_shower_image_v,
				       crop_thrumu_image_v, crop_stopmu_image_v);
	status = status && StoreParticles(mgr,_alg_mgr,crop_adc_image_v,pidx);
      }
    }

    if (_write_reco) {
      LARCV_DEBUG() << "Writing RecoHolder tree & reset" << std::endl;
      _reco_holder.Write();
      _reco_holder.ResetOutput();
    }

    LARCV_DEBUG() << "return " << status << std::endl;
    return status;
  }

  bool LArbysImage::StoreParticles(IOManager& iom,
				   larocv::ImageClusterManager& mgr,
				   const std::vector<Image2D>& adc_image_v,
				   size_t& pidx) {
    
    LARCV_DEBUG() << iom.event_id().run()<<","<<iom.event_id().subrun()<<","<<iom.event_id().event()<<","<<std::endl;
    const auto& adc_cvimg_v_ = mgr.InputImages(0);

    std::vector<cv::Mat> adc_cvimg_v;
    adc_cvimg_v.resize(3);

    for(size_t plane=0;plane<3;++plane)
      adc_cvimg_v[plane] = adc_cvimg_v_[plane].clone();
    
    auto event_pgraph        = (EventPGraph*)  iom.get_data(kProductPGraph,_output_producer);
    auto event_ctor_pixel    = (EventPixel2D*) iom.get_data(kProductPixel2D,_output_producer+"_ctor");
    auto event_img_pixel     = (EventPixel2D*) iom.get_data(kProductPixel2D,_output_producer+"_img");

    _reco_holder.ShapeData(mgr);

    bool _filter_reco=true;
    if (_filter_reco)
      _reco_holder.Filter();
    
    const auto& vtx_ana = _reco_holder.ana();
      
    LARCV_DEBUG() << "Matching... " << _reco_holder.Verticies().size() << " vertices" << std::endl;
    for(size_t vtxid=0;vtxid<_reco_holder.Verticies().size();++vtxid) {
      const auto& vtx3d = *(_reco_holder.Vertex(vtxid));
      const auto& pcluster_vv = _reco_holder.PlaneParticles(vtxid);
      const auto& tcluster_vv = _reco_holder.PlaneTracks(vtxid);
      
      auto match_vv = _reco_holder.Match(vtxid,adc_cvimg_v_);
      
      if (match_vv.empty()) {
	LARCV_DEBUG() << "NO match for vertex id " << vtxid << std::endl;
	continue;
      }
      
      PGraph pgraph;
      for( auto match_v : match_vv ) {
	std::array<size_t,3> plane_arr{{kINVALID_SIZE,kINVALID_SIZE,kINVALID_SIZE}};
	std::array<size_t,3> id_arr{{kINVALID_SIZE,kINVALID_SIZE,kINVALID_SIZE}};
	std::array<const larocv::data::ParticleCluster*,3> pcluster_arr{{nullptr,nullptr,nullptr}};
	std::array<const larocv::data::TrackClusterCompound*,3> tcluster_arr{{nullptr,nullptr,nullptr}};

	larocv::data::ParticleType_t partype;
	LARCV_DEBUG() << "Examining " << match_v.size() << " match." << std::endl;
	for (auto match : match_v) {
	  
	  auto plane = match.first;
	  auto id    = match.second;
	  plane_arr[plane] = plane;
	  id_arr[plane]    = id;
	  
	  pcluster_arr[plane]=pcluster_vv[plane][id];
	  tcluster_arr[plane]=tcluster_vv[plane][id];
	  
	  partype=pcluster_arr[plane]->type;
	}
	
	bool endok=false;

	larocv::data::Vertex3D endpt3d;
	if (partype==larocv::data::ParticleType_t::kTrack) 
	  endok = vtx_ana.MatchEdge(tcluster_arr,endpt3d);
	
	LARCV_DEBUG() << "Storing particle of type " << (uint)partype << std::endl;
	  
	ROI proi;

	if      (partype==larocv::data::ParticleType_t::kTrack)  proi.Shape(kShapeTrack);
	else if (partype==larocv::data::ParticleType_t::kShower) proi.Shape(kShapeShower);
	else throw larbys("Unknown particle type matched");

	proi.Position(vtx3d.x,vtx3d.y,vtx3d.z,kINVALID_DOUBLE);
	if (endok)
	  proi.EndPosition(endpt3d.x,endpt3d.y,endpt3d.z,kINVALID_DOUBLE);
	
	LARCV_DEBUG() << " @ pg array index " << pidx << std::endl;
	  
	for(size_t plane=0;plane<3;++plane) 
	  proi.AppendBB(adc_image_v[plane].meta());
	  
	pgraph.Emplace(std::move(proi),pidx);
	pidx++;
	  
	for(size_t plane=0;plane<3;++plane) {
	  const auto& pmeta = adc_image_v[plane].meta();
	    
	  std::vector<Pixel2D> pixel_v;
	    
	  const auto& par = pcluster_arr[plane];
	  const auto& img2d = adc_image_v[plane];
	  auto& cvimg = adc_cvimg_v[plane];

	  larocv::GEO2D_Contour_t par_pixel_v;
	  if(par) {
	    auto masked = larocv::MaskImage(cvimg,(*par)._ctor,0,false);
	    par_pixel_v = larocv::FindNonZero(masked);
	    pixel_v.reserve(par_pixel_v.size());
	    cvimg = larocv::MaskImage(cvimg,(*par)._ctor,0,true);
	  }
	    
	  float isum=0;
	  for (const auto& px : par_pixel_v) {
	    auto col = cvimg.cols-px.x;
	    auto row = px.y;
	    auto iii = img2d.pixel(col,row);
	    pixel_v.emplace_back(col,row);
	    pixel_v.back().Intensity(iii);
	    isum+=iii;
	  }
	    
	  Pixel2DCluster pixcluster(std::move(pixel_v));
	  event_img_pixel->Emplace(plane,std::move(pixcluster),pmeta);
	    
	  //store the contour at the same index along size the pixels themselves
	  std::vector<Pixel2D> ctor_v;
	  if (par) {
	    ctor_v.reserve(par->_ctor.size());
	    for(const auto& pt : (*par)._ctor)  {
	      auto col=cvimg.cols-pt.x;
	      auto row=pt.y;
	      ctor_v.emplace_back(row,col);
	      ctor_v.back().Intensity(1.0);
	    }
	  }
	  Pixel2DCluster pixctor(std::move(ctor_v));
	  event_ctor_pixel->Emplace(plane,std::move(pixctor),pmeta);
	}
      }//end this match
      event_pgraph->Emplace(std::move(pgraph));
    }//end vertex


    _reco_holder.FilterMatches();
    if (_write_reco) {
      const auto& eid = iom.event_id();
      _reco_holder.StoreEvent(eid.run(),eid.subrun(),eid.event(),iom.current_entry());
    }
    
    _reco_holder.Reset();
    return true;
  }
  
  bool LArbysImage::Reconstruct(const std::vector<larcv::Image2D>& adc_image_v,
				const std::vector<larcv::Image2D>& track_image_v,
				const std::vector<larcv::Image2D>& shower_image_v,
				const std::vector<larcv::Image2D>& thrumu_image_v,
				const std::vector<larcv::Image2D>& stopmu_image_v)
  {
    _adc_img_mgr.clear();
    _track_img_mgr.clear();
    _shower_img_mgr.clear();
    _thrumu_img_mgr.clear();
    _stopmu_img_mgr.clear();
    _alg_mgr.ClearData();

    ::larocv::Watch watch_all, watch_one;
    watch_all.Start();
    watch_one.Start();

    static int adcctr=0;
    for(auto& img_data : _LArbysImageMaker.ExtractImage(adc_image_v)) {
      /*
	cv::Mat thresholded;
	cv::threshold( std::get<0>(img_data), thresholded, 1, 255, 0);
	std::stringstream ss;
	ss << "adc_plane_" << std::get<1>(img_data).plane() << "_" << adcctr << ".png";
	cv::imwrite(std::string(ss.str()),thresholded);
	adcctr++;
      */
      _adc_img_mgr.emplace_back(std::move(std::get<0>(img_data)),std::move(std::get<1>(img_data)));
    }

    static int trkctr=0;
    for(auto& img_data : _LArbysImageMaker.ExtractImage(track_image_v))  {
      /*
	cv::Mat thresholded;
	cv::threshold( std::get<0>(img_data), thresholded, 1, 255, 0);
	std::stringstream ss;
	ss << "track_plane_" << std::get<1>(img_data).plane() << "_" << trkctr << ".png";
	cv::imwrite(std::string(ss.str()),thresholded);
	trkctr++;
      */
      _track_img_mgr.emplace_back(std::move(std::get<0>(img_data)),std::move(std::get<1>(img_data)));
    }

    static int shrctr=0;
    for(auto& img_data : _LArbysImageMaker.ExtractImage(shower_image_v)) {
      /*
	cv::Mat thresholded;
	cv::threshold( std::get<0>(img_data), thresholded, 1, 255, 0);
	std::stringstream ss;
	ss << "shower_plane_" << std::get<1>(img_data).plane() << "_" << shrctr << ".png";
	cv::imwrite(std::string(ss.str()),thresholded);
	shrctr++;
      */
      _shower_img_mgr.emplace_back(std::move(std::get<0>(img_data)),std::move(std::get<1>(img_data)));
    }

    for(auto& img_data : _LArbysImageMaker.ExtractImage(thrumu_image_v)) {
      _thrumu_img_mgr.emplace_back(std::move(std::get<0>(img_data)),std::move(std::get<1>(img_data)));
    }

    for(auto& img_data : _LArbysImageMaker.ExtractImage(stopmu_image_v)) {
      _stopmu_img_mgr.emplace_back(std::move(std::get<0>(img_data)),std::move(std::get<1>(img_data)));
    }
    
    _process_time_image_extraction += watch_one.WallTime();

    for (size_t plane = 0; plane < _adc_img_mgr.size(); ++plane) {

      auto       & img  = _adc_img_mgr.img_at(plane);
      const auto & meta = _adc_img_mgr.meta_at(plane);
      const auto & roi  = _adc_img_mgr.roi_at(plane);

      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
      _alg_mgr.Add(img, meta, roi, 0);
    }

    for (size_t plane = 0; plane < _track_img_mgr.size(); ++plane) {
      
      auto       & img  = _track_img_mgr.img_at(plane);
      const auto & meta = _track_img_mgr.meta_at(plane);
      const auto & roi  = _track_img_mgr.roi_at(plane);

      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
      _alg_mgr.Add(img, meta, roi, 1);
    }
    
    for (size_t plane = 0; plane < _shower_img_mgr.size(); ++plane) {

      auto       & img  = _shower_img_mgr.img_at(plane);
      const auto & meta = _shower_img_mgr.meta_at(plane);
      const auto & roi  = _shower_img_mgr.roi_at(plane);

      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
      _alg_mgr.Add(img, meta, roi, 2);
    }

    for (size_t plane = 0; plane < _thrumu_img_mgr.size(); ++plane) {

      auto       & img  = _thrumu_img_mgr.img_at(plane);
      const auto & meta = _thrumu_img_mgr.meta_at(plane);
      const auto & roi  = _thrumu_img_mgr.roi_at(plane);

      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
      _alg_mgr.Add(img, meta, roi, 3);
    }

    for (size_t plane = 0; plane < _stopmu_img_mgr.size(); ++plane) {

      auto       & img  = _stopmu_img_mgr.img_at(plane);
      const auto & meta = _stopmu_img_mgr.meta_at(plane);
      const auto & roi  = _stopmu_img_mgr.roi_at(plane);

      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
      _alg_mgr.Add(img, meta, roi, 4);
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

    _alg_mgr.Process();

    watch_one.Start();
     
    _process_time_cluster_storage += watch_one.WallTime();

    _process_time_analyze += watch_all.WallTime();

    ++_process_count;

    //_tree->Fill();
    
    return true;
  }

  void LArbysImage::finalize()
  {
    if ( has_ana_file() ) 
      _alg_mgr.Finalize(&(ana_file()));
    
    if (_write_reco)
      _reco_holder.WriteOut(&(ana_file()));

  }
  
}
#endif
