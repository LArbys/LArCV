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

namespace larcv {

  static LArbysImageProcessFactory __global_LArbysImageProcessFactory__;

  LArbysImage::LArbysImage(const std::string name)
    : ProcessBase(name),
      _PreProcessor(),
      _LArbysImageMaker(),
      _LArbysImageAnaBase_ptr(nullptr)
      //_geo()
  {}
      
  void LArbysImage::configure(const PSet& cfg)
  {
    _adc_producer    = cfg.get<std::string>("ADCImageProducer");
    _track_producer  = cfg.get<std::string>("TrackImageProducer","");
    _shower_producer = cfg.get<std::string>("ShowerImageProducer","");
    _roi_producer    = cfg.get<std::string>("ROIProducer","");
    _output_producer = cfg.get<std::string>("OutputImageProducer","");
    _output_module_name   = cfg.get<std::string>("OutputModuleName","");
    _output_module_offset = cfg.get<size_t>("OutputModuleOffset",kINVALID_SIZE);
    
    _LArbysImageMaker.Configure(cfg.get<larcv::PSet>("LArbysImageMaker"));
    
    _preprocess = cfg.get<bool>("PreProcess",true);
    if (_preprocess) {
      LARCV_INFO() << "Preprocessing image" << std::endl;
      _PreProcessor.Configure(cfg.get<larcv::PSet>("PreProcessor"));
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
	  crop_adc_image_v.emplace_back(adc_image.crop(bb));

	  if(!track_image_v.empty()) {
	    auto const& track_image  = track_image_v[plane];
	    crop_track_image_v.emplace_back(track_image.crop(bb));
	  }

	  if(!shower_image_v.empty()) {
	    auto const& shower_image = shower_image_v[plane];
	    crop_shower_image_v.emplace_back(shower_image.crop(bb));
	  }
	}

	status = status && Reconstruct(crop_adc_image_v, crop_track_image_v, crop_shower_image_v);

	if(_LArbysImageAnaBase_ptr) _LArbysImageAnaBase_ptr->Analyze(_alg_mgr);
      
      }
    }

    status = status && StoreParticles(mgr,_alg_mgr);
    
    return status;
  }

  bool LArbysImage::StoreParticles(IOManager& iom, const larocv::ImageClusterManager& mgr) {

    auto const& adc_image_v = get_image2d(iom,_adc_producer);
    
    auto event_pgraph   = (EventPGraph*) iom.get_data(kProductPGraph,_output_producer);
    auto event_pixel    = (EventPixel2D*) iom.get_data(kProductPixel2D,_output_producer);

    const larocv::data::AlgoDataManager& data_mgr   = mgr.DataManager();
    const larocv::data::AlgoDataAssManager& ass_man = data_mgr.AssManager();

    auto output_module_id = data_mgr.ID(_output_module_name);
    const auto vtx3d_array = (larocv::data::Vertex3DArray*) data_mgr.Data(output_module_id, 0);
    const auto& vertex3d_v = vtx3d_array->as_vector();

    for(size_t vtxid=0;vtxid<vertex3d_v.size();++vtxid) {

      const auto& vtx3d = vertex3d_v[vtxid];
	
      PGraph pgraph;

      size_t pidx=0;

      for(size_t plane=0;plane<3;++plane) {

    	//get the particle cluster array
    	const auto par_array = (larocv::data::ParticleClusterArray*)
    	  data_mgr.Data(output_module_id, plane+_output_module_offset);
	
    	//get the compound array
    	const auto comp_array = (larocv::data::TrackClusterCompoundArray*)
    	  data_mgr.Data(output_module_id, plane+_output_module_offset+3);

    	auto par_ass_idx_v = ass_man.GetManyAss(vtx3d,par_array->ID());
	
    	for(size_t ass_id=0;ass_id<par_ass_idx_v.size();++ass_id) {

	  auto ass_idx = par_ass_idx_v[ass_id];
	  if (ass_idx==kINVALID_SIZE)
	    throw larbys("Invalid vertex->particle association detected");
    	  const auto& par = par_array->as_vector()[ass_idx];

	  ROI proi;
	  if (par.type==larocv::data::ParticleType_t::kTrack)
	    proi.Shape(kShapeTrack);
	  if (par.type==larocv::data::ParticleType_t::kShower)
	    proi.Shape(kShapeShower);
	  else throw larbys("Unknown?");

	  
	  //set particle position
	  proi.Position(vtx3d.x,
			vtx3d.y,
			vtx3d.z,
			kINVALID_DOUBLE);
	  
	  // set particle meta (bbox)
	  const auto& pmeta = adc_image_v[plane].meta();
	  proi.AppendBB(pmeta);

	  /*
    	  auto comp_ass_id = ass_man.GetOneAss(par,comp_array->ID());
	  if (comp_ass_id==kINVALID_SIZE)
	    continue;
    	  const auto& comp = comp_array->as_vector()[comp_ass_id];
	  //set particle end point if exists
	  const auto& endpt = comp.end_pt();
	  //proi.EndPosition(x,y,z,t);
	  */
	  
	  pgraph.Emplace(std::move(proi),pidx);
	  event_pgraph->Emplace(std::move(pgraph));

	  Pixel2DCluster pcluster;
	  
	  const auto& img2d = adc_image_v[plane];
	  const auto& cvimg = mgr.InputImages(0)[plane];

	  auto par_pixel_v = larocv::FindNonZero(larocv::MaskImage(cvimg,par._ctor,0,false));
	  std::vector<Pixel2D> pixel_v;
	  pixel_v.reserve(par_pixel_v.size());
	  for (const auto& px : par_pixel_v) {
	    pixel_v.emplace_back(px.x,px.y);
	    pixel_v.back().Intensity(img2d.pixel(px.x,px.y));
	  }
	  Pixel2DCluster pixcluster(std::move(pixel_v));
	  event_pixel->Emplace(plane,std::move(pixcluster));
	  
	  pidx++;
	} // end this particle
	
      } // end this plane
      
    }// end this vertex
    
    return true;
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

    //update LArPlaneGeo
    // for (size_t plane = 0; plane < _adc_img_mgr.size(); ++plane)
    //   _geo.ResetPlaneInfo(_alg_mgr.mgr.meta_at(plane));
    
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

    if ( _LArbysImageAnaBase_ptr ) {
      if( has_ana_file() )
	_LArbysImageAnaBase_ptr->Finalize(&(ana_file()));
      else
	_LArbysImageAnaBase_ptr->Finalize();
    }

  }
  
}
#endif
