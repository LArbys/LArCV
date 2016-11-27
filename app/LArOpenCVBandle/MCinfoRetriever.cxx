#ifndef __MCINFORETRIEVER_CXX__
#define __MCINFORETRIEVER_CXX__

#include "MCinfoRetriever.h"
#include "DataFormat/ROI.h"
#include "DataFormat/EventROI.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/LArProperties.h"
#include "DataFormat/EventImage2D.h"
#include "Core/Geo2D.h"

namespace larcv {

  static MCinfoRetrieverProcessFactory __global_MCinfoRetrieverProcessFactory__;

  MCinfoRetriever::MCinfoRetriever(const std::string name)
    : ProcessBase(name), _mc_tree(nullptr)
  {}
    
  void MCinfoRetriever::configure(const PSet& cfg)
  {
    _producer_roi       = cfg.get<std::string>("MCProducer");
    _producer_image2d   = cfg.get<std::string>("Image2DProducer");

    _min_nu_dep_e  = cfg.get<float>("MinNuDepE");
    _max_nu_dep_e  = cfg.get<float>("MaxNuDepE");
    _min_nu_init_e = cfg.get<float>("MinNuInitE");
    _max_nu_init_e = cfg.get<float>("MaxNuInitE");

    _min_n_proton = cfg.get<int>("MinNProton",0);
    _min_n_lepton = cfg.get<int>("MinNLepton",0);
    _min_n_meson  = cfg.get<int>("MinNMeson",0);
    _min_n_shower = cfg.get<int>("MinNShower",0);
    _min_n_neutron = cfg.get<int>("MinNNeutron",0);

    _check_vis         = cfg.get<bool>("CheckVisibility",false);
    _min_proton_init_e = cfg.get<float>("ProtonMinInitE");
    _min_lepton_init_e = cfg.get<float>("LeptonMinInitE");
    _do_not_reco       = cfg.get<bool>("DoNotReco");

  }

  cv::Rect MCinfoRetriever::Get2DRoi(const ImageMeta& meta,
				     const ImageMeta& bb) {

    //bb == ROI on plane META
    
    float x_compression = meta.width()  / (float) meta.cols();
    float y_compression = meta.height() / (float) meta.rows();
    

    int x=(bb.bl().x - meta.bl().x)/x_compression;
    int y=(bb.bl().y - meta.bl().y)/y_compression;
    
    int dx=(bb.tr().x-bb.bl().x)/x_compression;
    int dy=(bb.tr().y-bb.bl().y)/y_compression;

    return cv::Rect(x,y,dx,dy);
  }
  

  
  void MCinfoRetriever::Project3D(const ImageMeta& meta,
				  double _parent_x,double _parent_y,double _parent_z,uint plane,
				  double& xpixel, double& ypixel) 
  {
    
    auto geohelp = larutil::GeometryHelper::GetME();//Geohelper from LArLite
    auto larpro  = larutil::LArProperties::GetME(); //LArProperties from LArLite

    auto vtx_2d = geohelp->Point_3Dto2D(_parent_x, _parent_y, _parent_z, plane );
    
    double x_compression  = meta.width()  / meta.cols();
    double y_compression  = meta.height() / meta.rows();
    xpixel = (vtx_2d.w/geohelp->WireToCm() - meta.tl().x) / x_compression;
    ypixel = (((_parent_x/larpro->DriftVelocity() + _parent_t/1000.)*2+3200)-meta.br().y)/y_compression;
    
  }
  

  //Input is track line and 4 linesegment consists the ROI
  //Function intersection finds the intersecion point of track
  //and ROI in particle direction
  
  geo2d::Vector<float> MCinfoRetriever::Intersection (const geo2d::HalfLine<float>& hline,
						      const cv::Rect& rect)
  {
    
    geo2d::LineSegment<float> ls1(geo2d::Vector<float>(rect.x           ,rect.y            ),
				  geo2d::Vector<float>(rect.x+rect.width,rect.y            ));

    geo2d::LineSegment<float> ls2(geo2d::Vector<float>(rect.x+rect.width,rect.y            ),
				  geo2d::Vector<float>(rect.x+rect.width,rect.y+rect.height));

    geo2d::LineSegment<float> ls3(geo2d::Vector<float>(rect.x+rect.width,rect.y+rect.height),
				  geo2d::Vector<float>(rect.x           ,rect.y+rect.height));

    geo2d::LineSegment<float> ls4(geo2d::Vector<float>(rect.x           ,rect.y+rect.height),
				  geo2d::Vector<float>(rect.x           ,rect.y            ));
    

    geo2d::Vector<float> pt(-1,-1);
    
    try {
      auto x = hline.x(ls1.pt1.y);
      pt.x=x;
      pt.y=ls1.pt1.y;
      if ( pt.x <= ls1.pt2.x and pt.x >= ls1.pt1.x ) return pt;
    } catch(...){
      //std::cout << "No point found in this direction" << std::endl;
    }

    try {
      auto y = hline.y(ls2.pt1.x);
      pt.x=ls2.pt1.x;
      pt.y=y;
      if ( pt.y <= ls2.pt2.y and pt.y >= ls2.pt1.y ) return pt;
    } catch(...){
      //std::cout << "No point found in this direction" << std::endl;
    }

    try {
      auto x = hline.x(ls3.pt1.y);
      pt.x=x;
      pt.y=ls3.pt1.y;
      if ( pt.x >= ls3.pt2.x and pt.x <= ls3.pt1.x ) return pt;
    } catch(...){
      //std::cout << "No point found in this direction" << std::endl;
    }

    try {
      auto y= hline.y(ls4.pt1.x);
      pt.x=ls4.pt1.x;
      pt.y=y;
      if ( pt.y >= ls4.pt2.y and pt.y <= ls4.pt1.y ) return pt;
    } catch(...){
      //std::cout << "No point found in this direction" << std::endl;
    }
    
    // throw larbys("\nNo intersection point found?\n");
    return pt;
    
  }

  void MCinfoRetriever::Clear() {
    
    _vtx_2d_w_v.clear();
    _vtx_2d_t_v.clear();
    _vtx_2d_w_v.resize(3);
    _vtx_2d_t_v.resize(3);

    _image_v.clear();
    _image_v.resize(3);
    
    _daughter_pdg_v.clear();
    _daughter_trackid_v.clear();
    _daughter_parenttrackid_v.clear();

    _daughter_energyinit_v.clear();
    _daughter_energydep_v.clear();

    _daughter_length_vv.clear();

    _daughter_2dstartx_vv.clear();
    _daughter_2dstarty_vv.clear();

    _daughter_2dendx_vv.clear();
    _daughter_2dendy_vv.clear();

    _daughterPx_v.clear();
    _daughterPy_v.clear();
    _daughterPz_v.clear();
    
    _daughter_2dcosangle_vv.clear();
  }
  
  void MCinfoRetriever::initialize()
  {
    _mc_tree = new TTree("mctree","MC infomation");

    _mc_tree->Branch("run",&_run,"run/i");
    _mc_tree->Branch("subrun",&_subrun,"subrun/i");
    _mc_tree->Branch("event",&_event,"event/i");
    _mc_tree->Branch("parentPDG",&_parent_pdg,"parentPDG/I");

    _mc_tree->Branch("energyDeposit",&_energy_deposit,"energyDeposit/D");
    _mc_tree->Branch("energyInit",&_energy_init,"energyInit/D");

    _mc_tree->Branch("parentX",&_parent_x,"parentX/D");
    _mc_tree->Branch("parentY",&_parent_y,"parentY/D");
    _mc_tree->Branch("parentZ",&_parent_z,"parentZ/D");
    _mc_tree->Branch("parentT",&_parent_t,"parentT/D");

    _mc_tree->Branch("parentPx",&_parent_px,"parentPx/D");
    _mc_tree->Branch("parentPy",&_parent_py,"parentPy/D");
    _mc_tree->Branch("parentPz",&_parent_pz,"parentpz/D");

    _mc_tree->Branch("currentType",&_current_type,"currentType/S");
    _mc_tree->Branch("interactionType",&_current_type,"InteractionType/S");
    
    _mc_tree->Branch("vtx2d_w",&_vtx_2d_w_v);
    _mc_tree->Branch("vtx2d_t",&_vtx_2d_t_v);

    _mc_tree->Branch("nprimary", &_nprimary,"nprimary/i");
    _mc_tree->Branch("ntotal", &_ntotal,"ntotal/i");
    
    _mc_tree->Branch("nproton", &_nproton,"nproton/i");
    _mc_tree->Branch("nlepton", &_nlepton,"nlepton/i");
    _mc_tree->Branch("nmeson", &_nmeson,"nmeson/i");
    _mc_tree->Branch("nshower", &_nshower,"nshower/i");
    _mc_tree->Branch("nneutron", &_nneutron,"nneutron/i");    

    _mc_tree->Branch("hi_lep_pdg",&_hi_lep_pdg,"hi_lep_pdg/I");

    _mc_tree->Branch("dep_sum_lep",&_dep_sum_lep,"dep_sum_lep/D");
    _mc_tree->Branch("ke_sum_lep",&_ke_sum_lep,"ke_sum_lep/D");
    
    _mc_tree->Branch("dep_sum_proton",&_dep_sum_proton,"dep_sum_proton/D");
    _mc_tree->Branch("ke_sum_proton",&_ke_sum_proton,"ke_sum_proton/D");

    _mc_tree->Branch("dep_sum_meson",&_dep_sum_meson,"dep_sum_meson/D");
    _mc_tree->Branch("ke_sum_meson",&_ke_sum_meson,"ke_sum_meson/D");
    
    _mc_tree->Branch("dep_sum_shower",&_dep_sum_shower,"dep_sum_shower/D");
    _mc_tree->Branch("ke_sum_shower",&_ke_sum_shower,"ke_sum_shower/D");

    _mc_tree->Branch("dep_sum_neutron",&_dep_sum_neutron,"dep_sum_neutron/D");
    _mc_tree->Branch("ke_sum_neutron",&_ke_sum_neutron,"ke_sum_neutron/D");

    _mc_tree->Branch("daughterPx_v", &_daughterPx_v);
    _mc_tree->Branch("daughterPy_v", &_daughterPy_v);
    _mc_tree->Branch("daughterPz_v", &_daughterPz_v);
    
    _mc_tree->Branch("daughterPdg_v"           , &_daughter_pdg_v);
    _mc_tree->Branch("daughterTrackid_v"       , &_daughter_trackid_v);
    _mc_tree->Branch("daughterParentTrackid_v" , &_daughter_parenttrackid_v);
    _mc_tree->Branch("daughterLength_vv"       , &_daughter_length_vv);
    _mc_tree->Branch("daughterEnergyInit_v"    , &_daughter_energyinit_v);
    _mc_tree->Branch("daughterEnergyDep_v"     , &_daughter_energydep_v);
    _mc_tree->Branch("daughter2DStartX_vv"     , &_daughter_2dstartx_vv);
    _mc_tree->Branch("daughter2DStartY_vv"     , &_daughter_2dstarty_vv);
    _mc_tree->Branch("daughter2DEndX_vv"       , &_daughter_2dendx_vv);
    _mc_tree->Branch("daughter2DEndY_vv"       , &_daughter_2dendy_vv);
    _mc_tree->Branch("daughter2DCosAngle_vv"   , &_daughter_2dcosangle_vv);
    
  }

  bool MCinfoRetriever::process(IOManager& mgr)
  {
    Clear();
    
    auto ev_roi = (larcv::EventROI*)mgr.get_data(kProductROI,_producer_roi);
    auto const ev_image2d = (larcv::EventImage2D*)mgr.get_data(kProductImage2D,_producer_image2d);

    _run    = (uint) ev_roi->run();
    _subrun = (uint) ev_roi->subrun();
    _event  = (uint) ev_roi->event();

    ///////////////////////////////
    // Neutrino ROI
    auto roi = ev_roi->at(0);

    _parent_pdg     = roi.PdgCode();

    _energy_deposit = roi.EnergyDeposit();

    _energy_init    = roi.EnergyInit();

    _parent_x       = roi.X(); 
    _parent_y       = roi.Y(); 
    _parent_z       = roi.Z(); 
    _parent_t       = roi.T(); 

    _parent_px      = roi.Px(); 
    _parent_py      = roi.Py(); 
    _parent_pz      = roi.Pz(); 

    _current_type     = roi.NuCurrentType();
    _interaction_type = roi.NuInteractionType();
    
    //Get 2D projections from 3D
    for (uint plane = 0 ; plane<3;++plane){
      
      ///Convert [cm] to [pixel]
      const auto& img = ev_image2d->Image2DArray()[plane];
      const auto& meta = img.meta();
      
      double x_pixel(0), y_pixel(0);
      Project3D(meta,_parent_x,_parent_y,_parent_z,plane,x_pixel,y_pixel);
      
      _vtx_2d_w_v[plane] = x_pixel;
      _vtx_2d_t_v[plane] = y_pixel;
      
    }

    // for each ROI not nu, lets get the 3D line in direction of particle trajectory.
    // then project onto plane, and find the intersection with the edge of the particle
    // ROI box, I think this won't be such a bad proxy for the MC particle length and angle.

    ///////////////////////////////
    //Daughter ROI
    _nprimary=0;
    _ntotal=0;
    _nproton=0;
    _nneutron=0;
    _nlepton=0;
    _nmeson=0;
    _nshower=0;
    
    _hi_lep_pdg=-1;

    float hi_lep_e = 0;
    
    _dep_sum_lep=0;
    _ke_sum_lep=0;
    _dep_sum_proton=0;
    _ke_sum_proton=0;
    _dep_sum_meson=0;
    _ke_sum_meson=0;
    _dep_sum_shower=0;
    _ke_sum_shower=0;
    _dep_sum_neutron=0;
    _ke_sum_neutron=0;

    bool visibility=false;
    bool hadron_vis=false;
    bool lepton_vis=false;
    
    for(const auto& roi : ev_roi->ROIArray()) {

      //do not store super parent pdgcode 0? or any other neutrino 12 or 14 -- this means we may have 2 neutrino events
      if (roi.PdgCode() == 12 or
	  roi.PdgCode() == 14 or
	  roi.PdgCode() == 0) continue;

      LARCV_DEBUG() << "This particle is PDG code " << roi.ParentPdgCode() << std::endl;
      
      //get a unit vector for this pdg in 3 coordinates
      auto px = roi.Px();
      auto py = roi.Py();
      auto pz = roi.Pz();

      //length of p
      auto lenp = sqrt(px*px+py*py+pz*pz);
      
      px/=lenp;
      py/=lenp;
      pz/=lenp;

      // original location
      auto x0 = roi.X();
      auto y0 = roi.Y();
      auto z0 = roi.Z();
      //auto t  = roi.T();

      // here is another point in the direction of p.
      // Pxyz are info from genie(meaning that it won't be identical to PCA assumption).
      auto x1 = x0+px;
      auto y1 = y0+py;
      auto z1 = z0+pz;

      _daughter_length_vv.resize(3);
      _daughter_2dstartx_vv.resize(3);
      _daughter_2dstarty_vv.resize(3);
      _daughter_2dendx_vv.resize(3);
      _daughter_2dendy_vv.resize(3);
      _daughter_2dcosangle_vv.resize(3);
      
      //lets project both points
      for(uint plane=0; plane<3; ++plane) {

	auto& daughter_length_v = _daughter_length_vv[plane];
	auto& daughter_2dstartx_v = _daughter_2dstartx_vv[plane];
	auto& daughter_2dstarty_v = _daughter_2dstarty_vv[plane];
	auto& daughter_2dendx_v = _daughter_2dendx_vv[plane];
	auto& daughter_2dendy_v = _daughter_2dendy_vv[plane];
	auto& daughter_2dcosangle_v  = _daughter_2dcosangle_vv[plane];
	
	const auto& img  = ev_image2d->Image2DArray()[plane];
	const auto& meta = img.meta();
	
	double x_pixel0(0), y_pixel0(0);
	Project3D(meta,x0,y0,z0,plane,x_pixel0,y_pixel0);
	
	double x_pixel1(0), y_pixel1(0);
	Project3D(meta,x1,y1,z1,plane,x_pixel1,y_pixel1);

	// start and end in 2D
	geo2d::Vector<float> start(x_pixel0,y_pixel0);
	geo2d::Vector<float> end  (x_pixel1,y_pixel1);
	
	// get the line of particle
	geo2d::HalfLine<float> hline(start,end-start);
	
	//here is the bbox on this plane --> we need to get the single intersection point for the half line
	//and this bbox
	ImageMeta bb;
	try {
	  bb = roi.BB(plane);
	}
	catch(...) {
	  continue;
	}
	
	cv::Rect roi_on_plane = Get2DRoi(meta,roi.BB(plane));

	// the start point will be inside the 2D ROI
	// we need to intersection point between the edge and this half line, find it

	auto pt_edge = Intersection(hline,roi_on_plane);

	// std::cout << "ax.plot(" << pt_edge.y << "," << pt_edge.x << ",'o',markersize=15)"  << std::endl;
	
	daughter_length_v.push_back(geo2d::length(pt_edge - start));

	daughter_2dstartx_v.push_back(start.x);
	daughter_2dstarty_v.push_back(start.y);
	daughter_2dendx_v.push_back(pt_edge.x);
	daughter_2dendy_v.push_back(pt_edge.y);

	auto dir=pt_edge - start;
	double cosangle = dir.x / sqrt(dir.x*dir.x + dir.y*dir.y);
	
	daughter_2dcosangle_v.push_back(cosangle);
      }

      _daughterPx_v.push_back( roi.Px() );
      _daughterPy_v.push_back( roi.Py() );
      _daughterPz_v.push_back( roi.Pz() );
      
      _daughter_pdg_v.push_back((uint) roi.PdgCode());
      _daughter_trackid_v.push_back((uint) roi.TrackID());
      _daughter_parenttrackid_v.push_back((uint) roi.ParentTrackID());
      _daughter_energyinit_v.push_back(roi.EnergyInit());
      _daughter_energydep_v.push_back(roi.EnergyDeposit());

      auto pdgcode = roi.PdgCode();
      
      //check if this particle is primary...

      _ntotal+=1;
      
      //its not move on
      if (roi.TrackID() != roi.ParentTrackID()) continue;
      
      // it is
      _nprimary+=1;
      
      //this is proton
      if (pdgcode==2212) {

	_nproton++;
	_dep_sum_proton += roi.EnergyDeposit();
	_ke_sum_proton  += roi.EnergyInit() - 938.0;

	if (roi.EnergyInit() - 938.0 > _min_proton_init_e) hadron_vis = true;
      }

      //this is neutron
      if (pdgcode==2112 or pdgcode==-2112) {
	_nneutron++;
	_dep_sum_neutron += roi.EnergyDeposit();
	_ke_sum_neutron  += roi.EnergyInit() - 939.5;
      }

      //mesons are pion,kaon,...
      if (pdgcode==211 or pdgcode==-211 or
	  pdgcode==321 or pdgcode==-321) {
	_nmeson++;
	_dep_sum_meson += roi.EnergyDeposit();
	_ke_sum_meson  += roi.EnergyInit();
      }

      //leptons are electron, muon also (anti...)
      if (pdgcode==11 or pdgcode==-11 or
	  pdgcode==13 or pdgcode==-13) {

	_nlepton++;
	_dep_sum_lep += roi.EnergyDeposit();
	_ke_sum_lep  += roi.EnergyInit();

	if (roi.EnergyInit() > hi_lep_e) _hi_lep_pdg = pdgcode;

	if (roi.EnergyInit() > _min_lepton_init_e) lepton_vis = true;
	
      }
      
      //shower are electron, gamma, pi0
      if (pdgcode==11 or pdgcode==-11 or
	  pdgcode==22 or
	  pdgcode==111) {
	_nshower++;
	_dep_sum_shower += roi.EnergyDeposit();
	_ke_sum_shower  += roi.EnergyInit();
      }

    }
    
    //the analysis filter
    if ( _energy_deposit < _min_nu_dep_e  or _energy_deposit > _max_nu_dep_e)  return false;
    if ( _energy_init    < _min_nu_init_e or _energy_init    > _max_nu_init_e) return false;
    
    if ( _nproton  < _min_n_proton  ) return false;
    if ( _nneutron < _min_n_neutron ) return false;
    if ( _nlepton  < _min_n_lepton  ) return false;
    if ( _nmeson   < _min_n_meson   ) return false;
    if ( _nshower  < _min_n_shower  ) return false;


    visibility = hadron_vis && lepton_vis;
    
    if ( !visibility and _check_vis) return false;
    
    //Fill tree
    _mc_tree->Fill();

    if (_do_not_reco) return false;
    
    return true;
  }
  

  void MCinfoRetriever::finalize()
  {
    _mc_tree->Write();
  }

}
#endif
