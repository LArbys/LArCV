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
      _LArbysImageAnaBase_ptr(nullptr),
      _vtx_ana()
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

      LARCV_DEBUG() << "Reconstruct" << std::endl;
      status = Reconstruct(adc_image_v,track_image_v,shower_image_v);
      
      if(_LArbysImageAnaBase_ptr) _LArbysImageAnaBase_ptr->Analyze(_alg_mgr);

      status = status && StoreParticles(mgr,_alg_mgr);
      
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
	LARCV_DEBUG() << "Reconstruct" << std::endl;
	status = status && Reconstruct(crop_adc_image_v, crop_track_image_v, crop_shower_image_v);

	if(_LArbysImageAnaBase_ptr) _LArbysImageAnaBase_ptr->Analyze(_alg_mgr);
	
	status = status && StoreParticles(mgr,_alg_mgr);
      }
    }
    
    return status;
  }

  bool LArbysImage::StoreParticles(IOManager& iom, const larocv::ImageClusterManager& mgr) {
    LARCV_DEBUG() << iom.event_id().run()<<","<<iom.event_id().subrun()<<","<<iom.event_id().event()<<","<<std::endl;
    const auto& adc_image_v = get_image2d(iom,_adc_producer);
    const auto& adc_cvimg_v = mgr.InputImages(0);
      
    auto event_pgraph   = (EventPGraph*) iom.get_data(kProductPGraph,_output_producer);
    auto event_pixel    = (EventPixel2D*) iom.get_data(kProductPixel2D,_output_producer);

    const larocv::data::AlgoDataManager& data_mgr   = mgr.DataManager();
    const larocv::data::AlgoDataAssManager& ass_man = data_mgr.AssManager();

    auto output_module_id = data_mgr.ID(_output_module_name);
    const auto vtx3d_array = (larocv::data::Vertex3DArray*) data_mgr.Data(output_module_id, 0);
    const auto& vertex3d_v = vtx3d_array->as_vector();

    LARCV_DEBUG() << "Observed " << vertex3d_v.size() << " verticies" << std::endl;

    std::vector<const larocv::data::Vertex3D*> vertex_ptr_v;
    std::vector<std::vector<std::vector<const larocv::data::ParticleCluster*> > > particle_cluster_ptr_vvv;
    std::vector<std::vector<std::vector<const larocv::data::TrackClusterCompound*> > > track_comp_ptr_vvv;
    
    for(size_t vtxid=0;vtxid<vertex3d_v.size();++vtxid) {
      const auto& vtx3d = vertex3d_v[vtxid];
      LARCV_DEBUG() << "On vertex " << vtxid << " of type " << (uint) vtx3d.type << std::endl;
      vertex_ptr_v.push_back(&vtx3d);

      std::vector<std::vector<const larocv::data::ParticleCluster* > > pcluster_vv;
      std::vector<std::vector<const larocv::data::TrackClusterCompound* > > tcluster_vv;
      pcluster_vv.resize(3);
      tcluster_vv.resize(3);
      
      for(size_t plane=0;plane<3;++plane) {
	
	auto& pcluster_v=pcluster_vv[plane];
	auto& tcluster_v=tcluster_vv[plane];
	
	auto output_module_id = data_mgr.ID(_output_module_name);
	
	const auto par_array = (larocv::data::ParticleClusterArray*)
	  data_mgr.Data(output_module_id, plane+_output_module_offset);

	const auto comp_array = (larocv::data::TrackClusterCompoundArray*)
	  data_mgr.Data(output_module_id, plane+_output_module_offset+3);
	
	auto par_ass_idx_v = ass_man.GetManyAss(vtx3d,par_array->ID());
	pcluster_v.resize(par_ass_idx_v.size());
	tcluster_v.resize(par_ass_idx_v.size());
	
	for(size_t ass_id=0;ass_id<par_ass_idx_v.size();++ass_id) {
	  auto ass_idx = par_ass_idx_v[ass_id];
	  if (ass_idx==kINVALID_SIZE) throw larbys("Invalid vertex->particle association detected");
	  const auto& par = par_array->as_vector()[ass_idx];
	  pcluster_v[ass_id] = &par;
	  auto comp_ass_id = ass_man.GetOneAss(par,comp_array->ID());
	  if (comp_ass_id==kINVALID_SIZE && par.type==larocv::data::ParticleType_t::kTrack)
	    throw larbys("Track particle with no track!");
	  const auto& comp = comp_array->as_vector()[comp_ass_id];
	  tcluster_v[ass_id] = &comp;
	} 
	_vtx_ana.ResetPlaneInfo(mgr.InputImageMetas(0)[plane]);
      }

      particle_cluster_ptr_vvv.emplace_back(std::move(pcluster_vv));
      track_comp_ptr_vvv.emplace_back(std::move(tcluster_vv));
    } //end this vertex
    
    

    size_t pidx=0;
    for(size_t vtxid=0;vtxid<vertex3d_v.size();++vtxid) {
      const auto& vtx3d = *vertex_ptr_v[vtxid];
      const auto& pcluster_vv = particle_cluster_ptr_vvv[vtxid];
      const auto& tcluster_vv = track_comp_ptr_vvv[vtxid];
      
      bool _require_two_multiplicity=true;
      if (_require_two_multiplicity) { 
	auto multiplicity=_vtx_ana.RequireParticleCount(pcluster_vv,2,2);
	if (!multiplicity) continue;
      }
      
      bool _require_fiducial=true;
      if (_require_fiducial) {
	auto fiduciality=_vtx_ana.CheckFiducial(vtx3d);
	if (!fiduciality) continue;
      }      
      
      auto match_vv = _vtx_ana.MatchClusters(pcluster_vv,adc_cvimg_v,0.5,2,2);
      if (match_vv.empty()) continue;

      PGraph pgraph;
      for( auto match_v : match_vv ) {
	//for this match
	if (match_v.size()==2) {
	  LARCV_DEBUG() << "2 plane match found" << std::endl;
	  auto& plane0 = match_v[0].first;
	  auto& id0    = match_v[0].second;
	  auto& plane1 = match_v[1].first;
	  auto& id1    = match_v[1].second;

	  const auto& cvimg0 = adc_cvimg_v[plane0];
	  const auto& cvimg1 = adc_cvimg_v[plane1];

	  const auto& par0   = *(pcluster_vv[plane0][id0]);
	  const auto& par1   = *(pcluster_vv[plane1][id1]);

	  auto partype=par0.type;
	  bool endok=false;
	  larocv::data::Vertex3D endpt3d;
	  
	  if (partype==larocv::data::ParticleType_t::kTrack) {
	    const auto& track0 = *(tcluster_vv[plane0][id0]);
	    const auto& track1 = *(tcluster_vv[plane1][id1]);
	    auto end0 = track0.end_pt();
	    auto end1 = track1.end_pt();
	    endok = _vtx_ana.Geo().YZPoint(end0,plane0,end1,plane1,endpt3d);
	  }	    
	  
	  ROI proi;

	  if (par0.type==larocv::data::ParticleType_t::kTrack) proi.Shape(kShapeTrack);
	  else if (par0.type==larocv::data::ParticleType_t::kShower) proi.Shape(kShapeShower);
	  else throw larbys("Unknown?");

	  // set particle position
	  proi.Position(vtx3d.x,vtx3d.y,vtx3d.z,kINVALID_DOUBLE);
	  // set end position
	  if (endok)
	    proi.EndPosition(endpt3d.x,endpt3d.y,endpt3d.z,kINVALID_DOUBLE);
	  
	  // set particle meta (bbox)
	  for(size_t plane=0;plane<3;++plane) {
	    const auto& pmeta = adc_image_v[plane].meta();
	    proi.AppendBB(pmeta);
	  }

	  pgraph.Emplace(std::move(proi),pidx);
	  pidx++;

	  std::array<const larocv::data::ParticleCluster*,3> pcluster_arr{{nullptr,nullptr,nullptr}};
	  pcluster_arr[plane0] = &par0;
	  pcluster_arr[plane1] = &par1;
	    
	  for(size_t plane=0;plane<3;++plane) {
	    std::vector<Pixel2D> pixel_v;
	    
	    const auto& par = pcluster_arr[plane];
	    const auto& img2d = adc_image_v[plane];
	    const auto& cvimg = adc_cvimg_v[plane];

	    larocv::GEO2D_Contour_t par_pixel_v;
	    if(par) {
	      par_pixel_v = larocv::FindNonZero(larocv::MaskImage(cvimg,(*par)._ctor,0,false));
	      pixel_v.reserve(par_pixel_v.size());
	    }
	    float isum=0;
	    for (const auto& px : par_pixel_v) {
	      auto row=cvimg.cols-px.x;
	      auto col=px.y;
	      auto iii=img2d.pixel(row,col);
	      pixel_v.emplace_back(row,col);
	      pixel_v.back().Intensity(iii);
	      isum+=iii;
	    }
	    LARCV_DEBUG() << "isum = " << isum << std::endl;
	    LARCV_CRITICAL() << " PUTTING IN EVENT PIXEL SIZE " << pixel_v.size() << std::endl;
	    Pixel2DCluster pixcluster(std::move(pixel_v));
	    event_pixel->Emplace(plane,std::move(pixcluster));
	  }
	  
	} // end match size 2
	else if (match_v.size()==3) {
	  LARCV_DEBUG() << "3 plane match found" << std::endl;

	  auto& plane0 = match_v[0].first;
	  auto& id0    = match_v[0].second;
	  auto& plane1 = match_v[1].first;
	  auto& id1    = match_v[1].second;
	  auto& plane2 = match_v[2].first;
	  auto& id2    = match_v[2].second;

	  const auto& cvimg0 = adc_cvimg_v[plane0];
	  const auto& cvimg1 = adc_cvimg_v[plane1];
	  const auto& cvimg2 = adc_cvimg_v[plane2];
	  
	  const auto& par0   = *(pcluster_vv[plane0][id0]);
	  const auto& par1   = *(pcluster_vv[plane1][id1]);
	  const auto& par2   = *(pcluster_vv[plane2][id2]);

	  auto partype=par0.type;
	  bool endok=false;
	  larocv::data::Vertex3D endpt3d;
	  
	  if (partype==larocv::data::ParticleType_t::kTrack) {

	    const auto& track0 = *(tcluster_vv[plane0][id0]);
	    const auto& track1 = *(tcluster_vv[plane1][id1]);
	    const auto& track2 = *(tcluster_vv[plane2][id2]);
	  
	    auto end0 = track0.end_pt();
	    auto end1 = track1.end_pt();
	    auto end2 = track2.end_pt();
	    
	    larocv::data::Vertex3D vertex;
	    endok = _vtx_ana.Geo().YZPoint(end0,plane0,end1,plane1,vertex);
	    LARCV_DEBUG() << "Testing end0 @ " << end0 << " on plane " << plane0 << " & end1 " << end1 << " @ plane " << plane1 << std::endl;
	    if (!endok) { 
	      endok = _vtx_ana.Geo().YZPoint(end0,plane0,end2,plane2,vertex);
	      LARCV_DEBUG() << "Testing end0 @ " << end0 << " on plane " << plane0 << " & end2 " << end2 << " @ plane " << plane2 << std::endl;
	    }
	    if (!endok) {
	      endok = _vtx_ana.Geo().YZPoint(end1,plane1,end2,plane2,vertex);
	      LARCV_DEBUG() << "Testing end1 @ " << end1 << " on plane " << plane1 << " & end2 " << end2 << " @ plane " << plane2 << std::endl;
	    }
	  }	    
	  
	  ROI proi;

	  if (par0.type==larocv::data::ParticleType_t::kTrack) proi.Shape(kShapeTrack);
	  else if (par0.type==larocv::data::ParticleType_t::kShower) proi.Shape(kShapeShower);
	  else throw larbys("Unknown?");

	  // set particle position
	  proi.Position(vtx3d.x,vtx3d.y,vtx3d.z,kINVALID_DOUBLE);
	  // set end position
	  if (endok)
	    proi.EndPosition(endpt3d.x,endpt3d.y,endpt3d.z,kINVALID_DOUBLE);
	  
	  // set particle meta (bbox)
	  for(size_t plane=0;plane<3;++plane) {
	    const auto& pmeta = adc_image_v[plane].meta();
	    proi.AppendBB(pmeta);
	  }

	  pgraph.Emplace(std::move(proi),pidx);
	  pidx++;

	  std::array<const larocv::data::ParticleCluster*,3> pcluster_arr{{nullptr,nullptr,nullptr}};
	  pcluster_arr[plane0] = &par0;
	  pcluster_arr[plane1] = &par1;
	  pcluster_arr[plane2] = &par2;
	    
	  for(size_t plane=0;plane<3;++plane) {
	    std::vector<Pixel2D> pixel_v;
	    
	    const auto& par = pcluster_arr[plane];
	    const auto& img2d = adc_image_v[plane];
	    const auto& cvimg = adc_cvimg_v[plane];

	    larocv::GEO2D_Contour_t par_pixel_v;
	    if(par) {
	      par_pixel_v = larocv::FindNonZero(larocv::MaskImage(cvimg,(*par)._ctor,0,false));
	      pixel_v.reserve(par_pixel_v.size());
	    }
	    float isum=0;
	    for (const auto& px : par_pixel_v) {
	      auto row=cvimg.cols-px.x;
	      auto col=px.y;
	      auto iii=img2d.pixel(row,col);
	      pixel_v.emplace_back(row,col);
	      pixel_v.back().Intensity(iii);
	      isum+=iii;
	    }
	    LARCV_DEBUG() << "isum = " << isum << std::endl;
	    Pixel2DCluster pixcluster(std::move(pixel_v));
	    event_pixel->Emplace(plane,std::move(pixcluster));
	  }
	} // end match 3
      }//end this match
      event_pgraph->Emplace(std::move(pgraph));
    }//end vertex
    
    
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

    //_tree->Fill();
    
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
