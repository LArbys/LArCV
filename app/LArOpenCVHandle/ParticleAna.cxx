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
    _angle_tree = nullptr;
    _dqdx_tree = nullptr;

   _ev_img_v = nullptr;
   _ev_trk_img_v = nullptr;
   _ev_shr_img_v = nullptr;
   _ev_roi_v = nullptr;
   _ev_pgraph_v = nullptr;
   _ev_pcluster_v = nullptr;
   _ev_ctor_v = nullptr;

  }
    
  void ParticleAna::configure(const PSet& cfg)
  {

    _analyze_particle = cfg.get<bool>("AnalyzeParticle",false);
    if(_analyze_particle) {

    }
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

    _img_prod           = cfg.get<std::string>("tpc");
    _pgraph_prod        = cfg.get<std::string>("test");
    _pcluster_ctor_prod = cfg.get<std::string>("test_ctor");
    _pcluster_img_prod  = cfg.get<std::string>("test_img");
    _reco_roi_prod      = cfg.get<std::string>("croi_merge");

    _trk_img_prod = cfg.get<std::string>("");
    _shr_img_prod = cfg.get<std::string>("");
    
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
    
    
    /*
    _tree->Branch("plane",&_plane,"plane/I");    
    _tree->Branch("straight_lines",&_straight_lines,"straight_lines/I");
    _tree->Branch("dir0_c",&_dir0_c);
    _tree->Branch("dir1_c",&_dir1_c);
    _tree->Branch("angle0",&_angle0_c);
    _tree->Branch("angle1",&_angle1_c);
    _tree->Branch("dir0_p",&_dir0_p);
    _tree->Branch("dir1_p",&_dir1_p);
    _tree->Branch("mean0",&_mean0,"mean0/D");//position of cluste0 to 2d vtx
    _tree->Branch("mean1",&_mean1,"mean1/D");//position of cluste1 to 2d vtx
    _tree->Branch("meanl",&_meanl,"meanl/F");//dqdx mean of particle left to the vtx
    _tree->Branch("meanr",&_meanr,"meanr/F");//dqdx mean of particle right to the vtx
    _tree->Branch("stdl",&_stdl,"stdl/F");
    _tree->Branch("stdr",&_stdr,"stdr/F");

    _tree->Branch("dqdxdelta",&_dqdxdelta,"dqdxdelta/F");
    _tree->Branch("dqdxratio",&_dqdxratio,"dqdxratio/F");
    */

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
    
    if(_analyze_particle) AnalyzeParticle();
    if(_analyze_dqdx)     AnalyzedQdX();
    if(_analyze_angle)    AnalyzeAngle();

    return true;
  }


  //
  // Particle Related Functionality (Vic reponsible)
  //
  void ParticleAna::AnalyzeParticle() {

    /*
    _length=...;
    _width=...;
    _perimeter=...;
    _area=...;
    _npixel=...;
    _track_frac=...;
    _shower_frac=...;
    _mean_pixel_dist=...;
    _sigma_pixel_dist=...;
    _angular_sum=...;
    */

    _particle_tree->Fill();
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
      
    _straight_lines = 0;

    auto const& ctor_m = _ev_ctor_v->Pixel2DClusterArray();
    auto const& pcluster_m = _ev_pcluster_v->Pixel2DClusterArray();
      
    for(auto const& pgraph : _ev_pgraph_v->PGraphArray()) {
      auto const& roi_v = pgraph.ParticleArray();
      if(roi_v.size()!=2) continue;

      auto const& roi0 = roi_v[0];
      
      auto _x = roi0.X();
      auto _y = roi0.Y();
      auto _z = roi0.Z();

      bool done0=false;
      bool done1=false;
      
      auto const& cluster_idx_v = pgraph.ClusterIndexArray();
      auto const& cluster_idx0 = cluster_idx_v[0];
      auto const& cluster_idx1 = cluster_idx_v[1];
      
      // Straight Lines Selection Start
      for(size_t plane=0; plane<3; ++plane) {

	auto iter_pcluster = pcluster_m.find(plane);
	if(iter_pcluster == pcluster_m.end()) continue;

	auto iter_ctor = ctor_m.find(plane);
	if(iter_ctor == ctor_m.end()) continue;

	auto const& ctor_v = (*iter_ctor).second;
	auto const& ctor0 = ctor_v.at(cluster_idx0);

	double x_vtx2d(0), y_vtx2d(0);
	  
	Project3D(pgraph.ParticleArray().back().BB(plane), _x, _y, _z, 0, plane, x_vtx2d, y_vtx2d);
	
	auto tmp = pgraph.ParticleArray().back().BB(plane).rows()-y_vtx2d;
	y_vtx2d = tmp;

	if(ctor0.size()>2) {
	  	  
	  ::larocv::GEO2D_Contour_t ctor;
	  ctor.resize(ctor0.size());
	  for(size_t i=0; i<ctor0.size(); ++i) {
	    ctor[i].x = ctor0[i].X();
	    ctor[i].y = ctor0[i].Y();
	  }
	  auto mean = Getx2vtxmean(ctor, x_vtx2d, y_vtx2d);
	  _mean0 = mean;
	  auto dir0_c = larocv::CalcPCA(ctor).dir;
	  if(dir0_c.x!=0 ) {
	    _dir0_c[plane] =  (dir0_c.y/dir0_c.x);
	    _angle0_c[plane] = atan(_dir0_c[plane])*180/PI;
	    if (_mean0 < 0) _angle0_c[plane] = atan(_dir0_c[plane])*180/PI + 180; 
	    LARCV_DEBUG()<<"plane "<<plane<<"  angle0 "<<_angle0_c[plane]<<"  plane "<<plane<<std::endl;
	  }
	}
	
	auto const& ctor1 = ctor_v.at(cluster_idx1);
	if(ctor1.size()>2) {
	  
	  ::larocv::GEO2D_Contour_t ctor;
	  ctor.resize(ctor1.size());
	  for(size_t i=0; i<ctor1.size(); ++i) {
	    ctor[i].x = ctor1[i].X();
	    ctor[i].y = ctor1[i].Y();
	  }
	  
	  auto mean = Getx2vtxmean(ctor, x_vtx2d, y_vtx2d);
	  _mean1 = mean;
	  auto dir1_c = larocv::CalcPCA(ctor).dir;
	  if(dir1_c.x!=0 ) {
	    _dir1_c[plane] =  (dir1_c.y/dir1_c.x);
	    _angle1_c[plane] = atan(_dir1_c[plane])*180/PI;
	    if (_mean1 < 0) _angle1_c[plane] = atan(_dir1_c[plane])*180/PI + 180; 
	    LARCV_DEBUG()<<"plane "<<plane<<"  angle1 "<<_angle1_c[plane]<<"  plane "<<plane<<std::endl;
	  }
	}
	if(_angle0_c[plane]!= -99999 && _angle1_c[plane] != -99999){
	  auto angle = (fabs( _angle0_c[plane] - _angle1_c[plane]));
	  if (angle > 180.0) angle = 360.0-angle;

	  if (angle >= _open_angle_cut ) _straight_lines+=1 ;
	}
      } //Straight Lines Selection End
      
      LARCV_DEBUG()<<"straight_lines  "<<_straight_lines<<std::endl;
      
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
	  _mean0 = mean;
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
	    _mean1 = mean;
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
    }
      
    _angle_tree->Fill();
    return;
  }
  
  //
  // dQdX Related Functionality (Rui responsible)
  //
  
  void ParticleAna::AnalyzedQdX() {

    auto const& ctor_m = _ev_ctor_v->Pixel2DClusterArray();
    auto const& pcluster_m = _ev_pcluster_v->Pixel2DClusterArray();
    
    for(auto const& pgraph : _ev_pgraph_v->PGraphArray()) {
      auto const& roi_v = pgraph.ParticleArray();
      if(roi_v.size()!=2) continue;
      auto const& roi0 = roi_v[0];
      auto const& roi1 = roi_v[1];

      auto _x = roi0.X();
      auto _y = roi0.Y();
      auto _z = roi0.Z();

      bool done0=false;
      bool done1=false;

      auto const& cluster_idx_v = pgraph.ClusterIndexArray();
      auto const& cluster_idx0 = cluster_idx_v[0];
      auto const& cluster_idx1 = cluster_idx_v[1];
      
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

	//Mask Circle is centered at 2D vertex
	Project3D(pgraph.ParticleArray().back().BB(plane), _x, _y, _z, 0, plane, y_vtx2d, x_vtx2d);
	
	LARCV_DEBUG()<<"plane "<<plane<<std::endl;
	LARCV_DEBUG()<<_x<<" , "<<_y<<" , "<<_z<<std::endl;
	LARCV_DEBUG()<<"x_vtx2d, "<<x_vtx2d<<"y_vtx2d, "<<y_vtx2d<<std::endl;
	
	LARCV_DEBUG()<<"mask circle x<<  "<< x_vtx2d<<" circle y  "<<y_vtx2d<<std::endl; 
	geo2d::Circle<float> mask_circle(x_vtx2d, y_vtx2d, _maskradius);
	
	auto crop_img = pgraph.ParticleArray().back().BB(plane);
	  
	auto cv_img = _LArbysImageMaker.ExtractMat(_ev_img_v->Image2DArray()[plane].crop(crop_img));
	
	auto img = larocv::MaskImage(cv_img,mask_circle,0,false);
	
	LARCV_DEBUG()<<"rows  "<< img.rows<<"cols "<<img.cols<<std::endl;
	
	LARCV_DEBUG()<<"<<<<<<<<<<<<Found more than 2 pixel in Maskimage>>>>>>>>>>>>"<<std::endl;
	
	if ( larocv::FindNonZero(img).size() < 2 ) continue ;
	
	auto pca = larocv::CalcPCA(img);
	
	//The last point on the masked track pt00
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
	//LARCV_DEBUG()<<std::endl;
	//for (auto dist : dist_v) LARCV_DEBUG()<<dist<<" , ";
	//LARCV_DEBUG()<<std::endl;
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
