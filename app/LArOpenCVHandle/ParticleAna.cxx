#ifndef __PARTICLEANA_CXX__
#define __PARTICLEANA_CXX__

#include "ParticleAna.h"
#include "LArOpenCV/ImageCluster/AlgoClass/PixelChunk.h"
#include "LArbysUtils.h"
#include <numeric>

#define PI 3.14159265

namespace larcv {

  static ParticleAnaProcessFactory __global_ParticleAnaProcessFactory__;

  ParticleAna::ParticleAna(const std::string name)
    : ProcessBase(name), _LArbysImageMaker()
  {
    _particle_tree = nullptr;
    _angle_tree    = nullptr;
    _dqdx_tree     = nullptr;

    _ev_img_v      = nullptr;
    _ev_trk_img_v  = nullptr;
    _ev_shr_img_v  = nullptr;
    _ev_roi_v      = nullptr;
    _ev_pgraph_v   = nullptr;
    _ev_pcluster_v = nullptr;
    _ev_ctor_v     = nullptr;
  }
    
  void ParticleAna::configure(const PSet& cfg)
  {

    _analyze_particle = cfg.get<bool>("AnalyzeParticle",false);
    if(_analyze_particle) { }

    _analyze_dqdx     = cfg.get<bool>("AnalyzedQdX",false);
    if(_analyze_dqdx) {
      _maskradius     = cfg.get<double>("MaskCircleRadius");
      _bins           = cfg.get<int>("dQdxBins");
      _adc_threshold  = cfg.get<float>("ADCThreshold");
    }
    
    _analyze_angle    = cfg.get<bool>("AnalyzeAngle",false);
    if(_analyze_angle) {
      _open_angle_cut = cfg.get<float>("OpenAngleCut");
      _pradius        = cfg.get<double>("PclusterRadius");
    }

    _LArbysImageMaker.Configure(cfg.get<larcv::PSet>("LArbysImageMaker"));
    
    _img_prod           = cfg.get<std::string>("ImageProducer"       ,"tpc");
    _pgraph_prod        = cfg.get<std::string>("PGraphProducer"      ,"test");
    _pcluster_ctor_prod = cfg.get<std::string>("Pixel2DCtorProducer" ,"test_ctor");
    _pcluster_img_prod  = cfg.get<std::string>("Pixel2DImageProducer","test_img");
    _reco_roi_prod      = cfg.get<std::string>("RecoROIProducer"     ,"croimerge");
    
    _trk_img_prod = cfg.get<std::string>("TrackImageProducer" ,"");
    _shr_img_prod = cfg.get<std::string>("ShowerImageProducer","");
    
  }

  void ParticleAna::initialize()
  {

    //
    // Particle Tree
    //
    _particle_tree = new TTree("ParticleTree","ParticleTree");
    _particle_tree->Branch("entry"  , &_entry  ,"entry/I");
    _particle_tree->Branch("run"    , &_run    ,"run/I");
    _particle_tree->Branch("subrun" , &_subrun ,"subrun/I");
    _particle_tree->Branch("event"  , &_event  ,"event/I");
    _particle_tree->Branch("length"           , &_length          , "length/F");
    _particle_tree->Branch("width"            , &_width           , "width/F");
    _particle_tree->Branch("perimeter"        , &_perimeter       , "perimeter/F");
    _particle_tree->Branch("area"             , &_area            , "area/F");
    _particle_tree->Branch("npixel"           , &_npixel          , "npixel/i");
    _particle_tree->Branch("track_frac"       , &_track_frac      , "track_frac/F");
    _particle_tree->Branch("shower_frac"      , &_shower_frac     , "shower_frac/F");
    _particle_tree->Branch("mean_pixel_dist"  , &_mean_pixel_dist , "mean_pixel_dist/D");
    _particle_tree->Branch("sigma_pixel_dist" , &_sigma_pixel_dist, "sigma_pixel_dist/D");
    _particle_tree->Branch("angular_sum"      , &_angular_sum     , "angular_sum/D");
    _particle_tree->Branch("plane"            , &_plane           , "plane/I");    
 
    //
    // Angle Tree
    //
    _angle_tree = new TTree("AngleTree","AngleTree");
    _angle_tree->Branch("entry"  , &_entry  ,"entry/I");
    _angle_tree->Branch("run"    , &_run    ,"run/I");
    _angle_tree->Branch("subrun" , &_subrun ,"subrun/I");
    _angle_tree->Branch("event"  , &_event  ,"event/I");
    _angle_tree->Branch("vtxid"  , &_vtxid  ,"vtxid/I");
    _angle_tree->Branch("plane",&_plane,"plane/I");    
    _angle_tree->Branch("straight_lines",&_straight_lines,"straight_lines/I");
    _angle_tree->Branch("dir0_c",&_dir0_c);
    _angle_tree->Branch("dir1_c",&_dir1_c);
    _angle_tree->Branch("angle0",&_angle0_c);
    _angle_tree->Branch("angle1",&_angle1_c);
    _angle_tree->Branch("angle_diff",&_angle_diff);
    _angle_tree->Branch("dir0_p",&_dir0_p);
    _angle_tree->Branch("dir1_p",&_dir1_p);
    _angle_tree->Branch("mean0",&_mean0);//position of cluste0 to 2d vtx
    _angle_tree->Branch("mean1",&_mean1);//position of cluste1 to 2d vtx
    _angle_tree->Branch("meanl",&_meanl,"meanl/F");//dqdx mean of particle left to the vtx
    _angle_tree->Branch("meanr",&_meanr,"meanr/F");//dqdx mean of particle right to the vtx
    _angle_tree->Branch("stdl",&_stdl,"stdl/F");
    _angle_tree->Branch("stdr",&_stdr,"stdr/F");


    //
    // dQdX Tree
    //
    _dqdx_tree = new TTree("dqdxTree","dqdxTree");
    _dqdx_tree->Branch("entry"  , &_entry  ,"entry/I");
    _dqdx_tree->Branch("run"    , &_run    ,"run/I");
    _dqdx_tree->Branch("subrun" , &_subrun ,"subrun/I");
    _dqdx_tree->Branch("event"  , &_event  ,"event/I");
    _dqdx_tree->Branch("dqdxdelta",&_dqdxdelta,"dqdxdelta/F");
    _dqdx_tree->Branch("dqdxratio",&_dqdxratio,"dqdxratio/F");
    

  }

  bool ParticleAna::process(IOManager& mgr)
  {
    _ev_img_v      = (EventImage2D*)mgr.get_data(kProductImage2D,_img_prod);
    _ev_roi_v      = (EventROI*)    mgr.get_data(kProductROI,_reco_roi_prod);
    _ev_pgraph_v   = (EventPGraph*) mgr.get_data(kProductPGraph,_pgraph_prod);
    _ev_pcluster_v = (EventPixel2D*)mgr.get_data(kProductPixel2D,_pcluster_img_prod);
    _ev_ctor_v     = (EventPixel2D*)mgr.get_data(kProductPixel2D,_pcluster_ctor_prod);

    if(!_trk_img_prod.empty())
      _ev_trk_img_v = (EventImage2D*)mgr.get_data(kProductImage2D,_trk_img_prod);
    if(!_shr_img_prod.empty())
      _ev_shr_img_v = (EventImage2D*)mgr.get_data(kProductImage2D,_shr_img_prod);
    
    _run    = _ev_pgraph_v->run();
    _subrun = _ev_pgraph_v->subrun();
    _event  = _ev_pgraph_v->event();
    _entry  = mgr.current_entry();

    std::cout<< "run, subrun, event  "<<_run<<" | "<<_subrun<<" | "<<_event<<" | "<<_entry<<std::endl;
    
    if(_analyze_particle) AnalyzeParticle();
    if(_analyze_dqdx)     AnalyzedQdX();
    if(_analyze_angle)    AnalyzeAngle();

    return true;
  }

  void ParticleAna::finalize()
  {
    _particle_tree->Write();
    _dqdx_tree->Write();
    _angle_tree->Write();
  }  
  
  //
  // Particle Related Functionality (Vic reponsible)
  //
  void ParticleAna::AnalyzeParticle() {
    
    // Get particle contours and images
    auto const& ctor_m = _ev_ctor_v->Pixel2DClusterArray();
    auto const& pcluster_m = _ev_pcluster_v->Pixel2DClusterArray();
      
    // Iterate over reconstructed vertex (1 vertex == 1 ParticleGraph)
    for(auto const& pgraph : _ev_pgraph_v->PGraphArray()) {

      // Get a list of particles (each unique particle == 1 ROI)
      auto const& roi_v = pgraph.ParticleArray();

      // Peel off the first ROI to get the reconstructed vertex (the particle start point)
      // Each reconstructed PGraph is part of the same CROI, so vic expects
      // the meta to be the same for each ROI, lets get it out and store in vector
      std::vector<ImageMeta> meta_v;
      meta_v.resize(3);
      for(size_t plane=0; plane<3; ++plane) 
	meta_v[plane] = roi_v.front().BB(plane);
      
      
      // Now go retrieve the particle contours, and particle pixels
      // ...the indicies are stored in ClusterIndexArray
      auto const& cluster_idx_v = pgraph.ClusterIndexArray();
      
      // Loop per plane, get the particle contours and images for this plane
      for(size_t plane=0; plane<3; ++plane) {

	auto iter_pcluster = pcluster_m.find(plane);
	if(iter_pcluster == pcluster_m.end()) {
	  LARCV_DEBUG() << "No particle cluster found" << std::endl;
	  continue;
	}
	
	auto iter_ctor = ctor_m.find(plane);
	if(iter_ctor == ctor_m.end()) {
	  LARCV_DEBUG() << "No contour found" << std::endl;
	  continue;
	}

	// Retrieve the particle images and particle contours on this plane
	const auto& pcluster_v = (*iter_pcluster).second;
	const auto& ctor_v = (*iter_ctor).second;

	// Ges this planes meta
	const auto& meta = meta_v.at(plane);

	// Get this plane image2d, crop it to this ROI using the meta
	auto adc_img2d = _ev_img_v->Image2DArray().at(plane).crop(meta);

	// Get the cv::Mat for this image (in the same style as LArbysImage)
	// we have to transpose and flip it along the rows to match Image2D orientation
	auto adc_cvimg = _LArbysImageMaker.ExtractMat(adc_img2d);
	adc_cvimg = larocv::Transpose(adc_cvimg);
	adc_cvimg = larocv::Flip(adc_cvimg,0);

	// For each particle, get the contour and image on this plane (from pcluster_v/ctor_v)
	for(auto cluster_idx : cluster_idx_v) {
	  const auto& pcluster = pcluster_v.at(cluster_idx);
	  const auto& pctor    = ctor_v.at(cluster_idx);

	  // There is no particle cluster on this plane
	  if (pctor.empty()) continue;

	  // Convert the Pixel2DArray of contour points to a GEO2D_Contour_t
	  larocv::GEO2D_Contour_t ctor;
	  ctor.resize(pctor.size());
	  for(size_t i=0;i<ctor.size();++i) {
	    ctor[i].x = pctor[i].X();
	    ctor[i].y = pctor[i].Y();
	  }
	  
	  // Make a PixelChunk given this contour, and this adc image
	  larocv::PixelChunk pchunk(ctor,adc_cvimg);

	  _length           = pchunk.length;
	  _width            = pchunk.width;
	  _perimeter        = pchunk.perimeter;
	  _area             = pchunk.area;
	  _npixel           = pchunk.npixel;
	  _mean_pixel_dist  = pchunk.mean_pixel_dist;
	  _sigma_pixel_dist = pchunk.sigma_pixel_dist;
	  _angular_sum      = pchunk.angular_sum;
	  _plane            = plane;

	  // To be filled if shower image is sent in
	  // _track_frac = n/a
	  // _shower_frac = n/a

	  // Write it out per particles
	  _particle_tree->Fill();
	} // end this particle
      } // end this plane    
    } // end this vertex

    return;
  }

  //
  // Angle Related Functionality (Rui reponsible)
  //
  void ParticleAna::AnalyzeAngle() {

    _dir0_c.clear();
    _dir0_c.resize(3,-99999);
    _dir1_c.clear();
    _dir1_c.resize(3,-99999);
    _dir0_p.clear();
    _dir0_p.resize(3,-99999);
    _dir1_p.clear();
    _dir1_p.resize(3,-99999);
    _angle0_c.clear();
    _angle0_c.resize(3,-99999);
    _angle1_c.clear();
    _angle1_c.resize(3,-99999);
    _mean0.clear();
    _mean0.resize(3);
    _mean1.clear();
    _mean1.resize(3);
    _straight_lines = 0;
    
    auto const& ctor_m     = _ev_ctor_v->Pixel2DClusterArray();
    auto const& pcluster_m = _ev_pcluster_v->Pixel2DClusterArray();
    
    // Iterate over vertex
    
    auto vtx_counts = _ev_pgraph_v->PGraphArray().size();
    
    if (vtx_counts==0 ) std::cout<<"due to vtx_counts == 0 ?"<<std::endl;
    
    if(vtx_counts!=0) {
      
      for (int vtx_idx = 0; vtx_idx < vtx_counts; ++ vtx_idx){
	_vtxid = vtx_idx;
	
	std::cout<<"vertex id "<<_vtxid<<std::endl;
	
	auto pgraph = _ev_pgraph_v->PGraphArray().at(vtx_idx);
	//for(auto const& pgraph : _ev_pgraph_v->PGraphArray()) {
	
	// Get a list of particles (each unique particle == 1 ROI)
	auto const& roi_v = pgraph.ParticleArray();
	if(roi_v.size() != 2) continue;
	
	// Peel off the first ROI to get the reconstructed vertex (the particle start point)
	auto const& roi0 = roi_v.front();
	
	auto _x = roi0.X();
	auto _y = roi0.Y();
	auto _z = roi0.Z();
	
	bool done0=false;
	bool done1=false;
	
	// Now go retrieve the particle contours, and particles pixels
	// the indicies are stored in ClusterIndexArray
	auto const& cluster_idx_v = pgraph.ClusterIndexArray();
	// Particle 0
	auto const& cluster_idx0 = cluster_idx_v.at(0);
	// Particle 1
	auto const& cluster_idx1 = cluster_idx_v.at(1);
	
	// Playing with straight lines
	for(size_t plane=0; plane<3; ++plane) {
	  
	  auto iter_pcluster = pcluster_m.find(plane);
	  if(iter_pcluster == pcluster_m.end()) continue;
	  auto iter_ctor = ctor_m.find(plane);
	  if(iter_ctor == ctor_m.end()) continue;
	  
	  // Retrieve the contour
	  auto const& ctor_v = (*iter_ctor).second;
	  
	  // Projected the 3D start point onto this plane
	  double x_vtx2d(0), y_vtx2d(0);
	  Project3D(pgraph.ParticleArray().back().BB(plane), _x, _y, _z, 0, plane, x_vtx2d, y_vtx2d);
	  // Some shit related to axis rotation, check this by looking @ image in notebook
	  auto tmp = pgraph.ParticleArray().back().BB(plane).rows()-y_vtx2d;
	  y_vtx2d = tmp;
	  
	  std::cout<<"vtx id is "<<vtx_idx<<"plane,"<<plane<<"  x is "<<x_vtx2d<<"y is "<<y_vtx2d<<std::endl;

	  // Get the first particle contour
	  auto const& ctor0 = ctor_v.at(cluster_idx0);
	  if(ctor0.size()>2) {
	    // Converting Pixel2D to geo2d::VectorArray<int>
	    ::larocv::GEO2D_Contour_t ctor;
	    ctor.resize(ctor0.size());
	    
	    for(size_t i=0; i<ctor0.size(); ++i) {
	      ctor[i].x = ctor0[i].X();
	      ctor[i].y = ctor0[i].Y();
	    }
	    
	    // Get the mean position of the contour
	    auto mean = Getx2vtxmean(ctor, x_vtx2d, y_vtx2d);
	    _mean0[plane] = mean;

	    // Calculate the direction of particle 0
	    auto dir0_c = larocv::CalcPCA(ctor).dir;
	    if (dir0_c.x == 0 && dir0_c.y >0) _angle0_c[plane] = 90;
	    if (dir0_c.x == 0 && dir0_c.y <0) _angle0_c[plane] = 270;
	    if (dir0_c.y == 0 && dir0_c.x >0) _angle0_c[plane] = 0;
	    if (dir0_c.y == 0 && dir0_c.x <0) _angle0_c[plane] = 180;
	    if(dir0_c.x!=0 ) {
	      _dir0_c[plane] =  (dir0_c.y/dir0_c.x);
	      std::cout<<"dir0"<<_dir0_c[plane]<<" @ plane "<<plane<<std::endl;
	      _angle0_c[plane] = atan(_dir0_c[plane])*180/PI;
	      std::cout<<"before adding angle0 "<<_angle0_c[plane]<<std::endl;
	      if (_mean0[plane] < 0) _angle0_c[plane] = atan(_dir0_c[plane])*180/PI + 180; 
	      LARCV_DEBUG()<<"plane "<<plane<<"  angle0 "<<_angle0_c[plane]<<"  plane "<<plane<<std::endl;
	    }
	  }

	  // Get the contour of particle 1
	  auto const& ctor1 = ctor_v.at(cluster_idx1);
	  if(ctor1.size()>2) {
	  
	    ::larocv::GEO2D_Contour_t ctor;
	    ctor.resize(ctor1.size());
	    for(size_t i=0; i<ctor1.size(); ++i) {
	      ctor[i].x = ctor1[i].X();
	      ctor[i].y = ctor1[i].Y();
	    }

	    // Get the mean position of the contour
	    auto mean = Getx2vtxmean(ctor, x_vtx2d, y_vtx2d);
	    _mean1[plane] = mean;
	  
	    // Calculate the direction of particle 1
	    auto dir1_c = larocv::CalcPCA(ctor).dir;
	    if (dir1_c.x == 0 && dir1_c.y >0) _angle1_c[plane] = 90;
	    if (dir1_c.x == 0 && dir1_c.y <0) _angle1_c[plane] = 270;
	    if (dir1_c.y == 0 && dir1_c.x >0) _angle0_c[plane] = 0;
	    if (dir1_c.y == 0 && dir1_c.x <0) _angle0_c[plane] = 180;
	    if(dir1_c.x!=0 ) {
	      _dir1_c[plane] =  (dir1_c.y/dir1_c.x);
	      std::cout<<"dir1"<<_dir1_c[plane]<<" @ plane "<<plane<<std::endl;
	      _angle1_c[plane] = atan(_dir1_c[plane])*180/PI;
	      std::cout<<"before adding angle1  "<<_angle1_c[plane]<<std::endl;
	      if (_mean1[plane] < 0) _angle1_c[plane] = atan(_dir1_c[plane])*180/PI + 180; 
	      LARCV_DEBUG()<<"plane "<<plane<<"  angle1 "<<_angle1_c[plane]<<"  plane "<<plane<<std::endl;
	    }
	  }
	  
	  // Analyze the angle
	  if(_angle0_c[plane]!= -99999 && _angle1_c[plane] != -99999){
	    auto angle = (fabs( _angle0_c[plane] - _angle1_c[plane]));
	    if (angle > 180.0) angle = 360.0-angle;
	    if (angle >= _open_angle_cut ) _straight_lines+=1 ;
	  }
      } // end plane loop
	
	for (int planeid =2 ; planeid >= 0 ; planeid--){
	  std::cout<<"angle0 "<<_angle0_c[planeid]<<"angle1 "<<_angle1_c[planeid]<<" on planeid "<<planeid<<std::endl;
	  if (_angle0_c[planeid]!=-99999 && _angle1_c[planeid]!=-99999){
	    _angle_diff = std::abs(_angle0_c[planeid]-_angle1_c[planeid]);
	    if (_angle_diff>180) _angle_diff = 360 - _angle_diff;
	    std::cout<<">>>>>>Selected angle0 "<<_angle0_c[planeid]<<"angle1 "<<_angle1_c[planeid]<<" on  plane "<<planeid
		     <<" diff is "<<_angle_diff<<std::endl;
	    break;
	  }
	  //std::cout<<"angle diff is "<<_angle_diff<<" on plane"<<planeid<<std::endl;
	}
	LARCV_DEBUG()<<"straight_lines  "<<_straight_lines<<std::endl;
	
	// Loop over planes again to calculate the direction using points close to vertex
	for(size_t plane=0; plane<3; ++plane) {
	  
	  auto iter_pcluster = pcluster_m.find(plane);
	  if(iter_pcluster == pcluster_m.end()) continue;
	  
	  auto iter_ctor = ctor_m.find(plane);
	  if(iter_ctor == ctor_m.end()) continue;
	  
	  auto const& pcluster_v = (*iter_pcluster).second;
	  auto const& ctor_v = (*iter_ctor).second;
	  
	  auto const& pcluster0 = pcluster_v.at(cluster_idx0);
	  auto const& ctor0 = ctor_v.at(cluster_idx0);
	  
	  double x_vtx2d(0), y_vtx2d(0);
	  
	  Project3D(pgraph.ParticleArray().back().BB(plane), _x, _y, _z, 0, plane, x_vtx2d, y_vtx2d);
	  
	  auto tmp = pgraph.ParticleArray().back().BB(plane).rows()-y_vtx2d;
	  y_vtx2d = tmp;

	  if(!done0 && ctor0.size()>2) {
	    
	    ::larocv::GEO2D_Contour_t ctor;
	    ctor.resize(ctor0.size());
	    for(size_t i=0; i<ctor0.size(); ++i) {
	      ctor[i].x = ctor0[i].X();
	      ctor[i].y = ctor0[i].Y();
	    }
	    
	    auto mean = Getx2vtxmean(ctor, x_vtx2d, y_vtx2d);
	    _mean0[plane] = mean;
	    auto dir0_c = larocv::CalcPCA(ctor).dir;
	    if(dir0_c.x!=0 ) {
	    _dir0_c[plane] =  (dir0_c.y/dir0_c.x);
	    }
	    
	    ::larocv::GEO2D_Contour_t pclus;
	    pclus.clear();
	  
	    for(auto const& pt : pcluster0) {
	      geo2d::Vector<double> ppt(pt.X(),pt.Y());
	      double x = pt.X();
	      double y = pt.Y();
	      auto d = pow((pow(y-x_vtx2d,2)+pow(x-y_vtx2d,2)),0.5);
	      if (d < _pradius) pclus.emplace_back(ppt);
	    }
	  
	    if (pclus.size()>=2) {
	      auto dir0_p = larocv::CalcPCA(pclus).dir;
	      //geo2d::Vector<double> DIR(dir0_p);
	      if(dir0_p.x!=0  ) _dir0_p[plane] = dir0_p.y/ dir0_p.x;
	    }
	    done0 = true;
	    
	    auto const& pcluster1 = pcluster_v.at(cluster_idx1);
	    auto const& ctor1 = ctor_v.at(cluster_idx1);
	    
	    if(!done1 && ctor1.size()>2) {
	      
	      ::larocv::GEO2D_Contour_t ctor;
	      ctor.resize(ctor1.size());
	      for(size_t i=0; i<ctor1.size(); ++i) {
		ctor[i].x = ctor1[i].X();
		ctor[i].y = ctor1[i].Y();
	      }
	      
	      auto mean = Getx2vtxmean(ctor, x_vtx2d, y_vtx2d);
	      _mean1[plane] = mean;
	      auto dir1_c = larocv::CalcPCA(ctor).dir;
	      if(dir1_c.x!=0 ) {
		_dir1_c[plane] =  (dir1_c.y/dir1_c.x);
	      }
	      
	      ::larocv::GEO2D_Contour_t pclus;
	      pclus.clear();
	      
	      for(auto const& pt : pcluster1) {
		auto x = pt.X();
		auto y = pt.Y();
		geo2d::Vector<double> ppt(pt.X(),pt.Y());
		auto d = pow((pow(x-x_vtx2d,2)+pow(y-y_vtx2d,2)),0.5);
		if (d < _pradius) pclus.emplace_back(ppt);
	      }
	      
	      if (pclus.size()>=2) {
		auto dir1_p = larocv::CalcPCA(pclus).dir;
		if(dir1_p.x!=0 ) _dir1_p[plane] = dir1_p.y/dir1_p.x;
	      }
	      done1 = true;
	    }
	    _plane = plane;
	    if(done0 && done1) break;
	  }
	}
	_angle_tree->Fill();      
      }
    }
    return;
  }
  
  //
  // dQdX Related Functionality (Rui responsible)
  //
  
  void ParticleAna::AnalyzedQdX() {

    auto const& ctor_m = _ev_ctor_v->Pixel2DClusterArray();
    auto const& pcluster_m = _ev_pcluster_v->Pixel2DClusterArray();

    // Loop over vertices
    for(auto const& pgraph : _ev_pgraph_v->PGraphArray()) {

      // Get the list of particles
      auto const& roi_v = pgraph.ParticleArray();

      // Are there not two particles? If not skip...
      if(roi_v.size()!=2) continue;
      auto const& roi0 = roi_v.front();

      auto _x = roi0.X();
      auto _y = roi0.Y();
      auto _z = roi0.Z();

      bool done0=false;
      bool done1=false;

      auto const& cluster_idx_v = pgraph.ClusterIndexArray();
      auto const& cluster_idx0 = cluster_idx_v.at(0);
      auto const& cluster_idx1 = cluster_idx_v.at(1);
      
      //dqdx
      bool save = false;
      LARCV_DEBUG()<<"=========================dqdx starts here==========================="<<std::endl;
      for(size_t plane=0; plane < 3; ++plane) {
	if (plane !=2 ) continue;
	
	auto iter_pcluster = pcluster_m.find(plane);
	if(iter_pcluster == pcluster_m.end()) continue;

	auto iter_ctor = ctor_m.find(plane);
	if(iter_ctor == ctor_m.end()) continue;
	
	double x_vtx2d(0), y_vtx2d(0);

	// Mask Circle is centered at 2D vertex
	Project3D(pgraph.ParticleArray().back().BB(plane), _x, _y, _z, 0, plane, y_vtx2d, x_vtx2d);
	
	LARCV_DEBUG()<<"plane "<<plane<<std::endl;
	LARCV_DEBUG()<<_x<<" , "<<_y<<" , "<<_z<<std::endl;
	LARCV_DEBUG()<<"x_vtx2d, "<<x_vtx2d<<"y_vtx2d, "<<y_vtx2d<<std::endl;
	
	LARCV_DEBUG()<<"mask circle x<<  "<< x_vtx2d<<" circle y  "<<y_vtx2d<<std::endl; 
	geo2d::Circle<float> mask_circle(x_vtx2d, y_vtx2d, _maskradius);

	// Get the meta for this ROI
	auto crop_img = pgraph.ParticleArray().back().BB(plane);

	// Get the cv::Mat for this ROI
	auto cv_img = _LArbysImageMaker.ExtractMat(_ev_img_v->Image2DArray()[plane].crop(crop_img));

	// Mask away stuff outside this circle
	auto img = larocv::MaskImage(cv_img,mask_circle,0,false);
	
	LARCV_DEBUG()<<"rows  "<< img.rows<<"cols "<<img.cols<<std::endl;
	
	LARCV_DEBUG()<<"<<<<<<<<<<<<Found more than 2 pixel in Maskimage>>>>>>>>>>>>"<<std::endl;

	// Found the number of points inside the image
	if ( larocv::FindNonZero(img).size() < 2 ) continue ;

	// Get the PCA
	auto pca = larocv::CalcPCA(img);
	
	// The last point on the masked track pt00
	::cv::Point pt00;
	
	for (size_t row = 0; row < img.rows; row++){
	  for (size_t col = 0; col < img.cols; col++){
	    float q = (float)(img.at<uchar>(row,col));
	    if ( q > _adc_threshold ) {
	      ::cv::Point pt;
	      pt.y = row;
	      pt.x = col;
	      pt00 = PointShift(pt, pca);
	      LARCV_DEBUG()<<"pt00row "<<row<<"pt00col "<<col<<std::endl;
	      break;
	    }
	  }
	}
	
	LARCV_DEBUG()<<"lastptx "<<pt00.x<<" lastpt0y "<<pt00.y<<std::endl;
	
	std::vector<float> dist_v;
	dist_v.clear();
	std::vector<float> dq_v;
	dq_v.clear();
	
	for (size_t row = 0; row < img.rows; ++row)
	  {
	    for (size_t col = 0; col < img.cols; ++col)
	      {
		float q = (float)(img.at<uchar>(row,col));
		if ( q > _adc_threshold ) 
		  {
		    ::cv::Point pt;
		    ::cv::Point spt; //shifted point
		    pt.y = row;
		    pt.x = col;
		    spt = PointShift(pt,pca);
		    float dist = sqrt(pow(float(spt.x)-float(pt00.x), 2)+pow(float(spt.y)-float(pt00.y),2));
		    dist_v.emplace_back(dist);
		    dq_v.emplace_back(q);
		  }
	      }
	  }

	if (dist_v.size() != dq_v.size())
	  {
	    LARCV_CRITICAL()<<"Length of dist and dQ are different"<<std::endl;
	    larbys();
	  }
	
	LARCV_DEBUG()<<"<<<<<<<<Dist_v has >0 size>>>>>>>>"<<std::endl;
	if ( dist_v.size() < 2 ) continue;
	
	auto max  = *std::max_element(std::begin(dist_v), std::end(dist_v));
	auto min  = *std::min_element(std::begin(dist_v), std::end(dist_v)); 
	
	auto bin_width=  ( max - min ) / _bins;
	
	LARCV_DEBUG()<<"dist_v size  "<<dist_v.size()<<std::endl;
	LARCV_DEBUG()<<"max dist_v  "<<max<<std::endl;
	LARCV_DEBUG()<<"min dist_v  "<<min<<std::endl;
	LARCV_DEBUG()<<"bin length  "<<bin_width<<std::endl;
	
	std::vector<float> dqdx_v; //binned dqdx
	dqdx_v.resize(_bins+1, 0.0);
	std::vector<float> dqdx_l; //binned pixels on the LEFT side of vtx
	std::vector<float> dqdx_r; //binned dqdx on the RIGHT side of  vtx
	dqdx_l.clear();
	dqdx_r.clear();
	
	for (size_t i = 0; i< dist_v.size();++i)
	  {
	    int idx =  int(dist_v[i] / bin_width) ; 
	    dqdx_v[idx] += dq_v[i];
	  }
	  
	float vtx_dist;
	vtx_dist = sqrt(pow(x_vtx2d-pt00.x, 2)+pow(y_vtx2d-pt00.y,2));
	int vtx_id = int(vtx_dist / bin_width);
	
	for (size_t i = 0; i< _bins; ++i){
	  if (i <= vtx_id) dqdx_l.emplace_back(dqdx_v[i]);
	  if (i > vtx_id)  dqdx_r.emplace_back(dqdx_v[i]);
	}
	
	_meanl = Mean(dqdx_l);
	_meanr = Mean(dqdx_r);
	_stdl  = STD(dqdx_l);
	_stdr  = STD(dqdx_r);
	
	_dqdxdelta = fabs(_meanl - _meanr) ;
	
	if ( _meanl >= _meanr && _meanl !=0 ) _dqdxratio = _meanr/_meanl;
	if ( _meanl <  _meanr && _meanr !=0 ) _dqdxratio = _meanl/_meanr;

	LARCV_DEBUG()<<"meanl "<<_meanl<<std::endl;
	LARCV_DEBUG()<<"meanr "<<_meanr<<std::endl;
	LARCV_DEBUG()<<"diff  "<<_dqdxdelta<<std::endl;
	
	save = true;
	//if (save) break;
      }//dqdx_end
      _dqdx_tree->Fill();
    } // end vertex
  }
 
  double ParticleAna::Getx2vtxmean( ::larocv::GEO2D_Contour_t ctor, float x2d, float y2d)
  {
    double sum = 0;
    double mean = -999; 
    for(size_t idx= 0;idx < ctor.size(); ++idx){
      sum += ctor[idx].x - x2d;
      //sum += ctor[idx].y - y2d;
    }
    if (ctor.size()>0) mean = sum / ctor.size();
    return mean;
  }

  //Project pts to PCA by shifting.
  cv::Point ParticleAna::PointShift(::cv::Point pt, geo2d::Line<float> pca)
  {
    float slope ;
    slope = pca.dir.y / pca.dir.x;
    if (slope >-1 && slope <=1 ) pt.y = pca.y(pt.x);
    else pt.x = pca.x(pt.y);
    
    return pt;
  }

  double ParticleAna::Mean(const std::vector<float>& v)
  {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / (double) v.size();
    
    return mean;
  }

  double ParticleAna::STD(const std::vector<float>& v)
  {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / (double) v.size();
    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / (double) v.size() - mean * mean);
    
    return stdev;
  }
  
 }
#endif
