#ifndef __RMEMPTYEVTS_CXX__
#define __RMEMPTYEVTS_CXX__

#include "RmEmptyEvts.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include "CVUtil/CVUtil.h"
#include "GeoAlgo/GeoAlgo.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/DetectorProperties.h"
#include "LArUtil/GeometryHelper.h"

namespace larcv {

  static RmEmptyEvtsProcessFactory __global_RmEmptyEvtsProcessFactory__;

  RmEmptyEvts::RmEmptyEvts(const std::string name)
    : ProcessBase(name)
  {
    _image_tree = nullptr; 
    _plane = -1; 
    _event = -1 ; 

  }
    
  void RmEmptyEvts::configure(const PSet& cfg)
  {
    _image_name = cfg.get<std::string>("ImageName");
    _roi_name = cfg.get<std::string>("ROIName");
  }

  void RmEmptyEvts::initialize()
  {

    if(!_image_tree){
      _image_tree = new TTree("image_tree","Tree for simple analysis");
      _image_tree->Branch( "event", &_event, "event/I" );
      _image_tree->Branch( "plane", &_plane, "plane/I");
      _image_tree->Branch( "vtx_x", &_vtx_x, "vtx_x/F");
      _image_tree->Branch( "vtx_y", &_vtx_y, "vtx_y/F");
      _image_tree->Branch( "vtx_z", &_vtx_z, "vtx_z/F");
      _image_tree->Branch( "dist_to_wall", &_dist_to_wall, "dist_to_wall/F");
      _image_tree->Branch( "e_dep", &_e_dep, "e_dep/F");
      _image_tree->Branch( "e_vis", &_e_vis, "e_vis/F");
      _image_tree->Branch( "pixel_count", &_pixel_count, "pixel_count/F");
      _image_tree->Branch( "child_e_ratio", &_child_e_ratio, "child_e_ratio/F");
      _image_tree->Branch( "nu_e_ratio", &_nu_e_ratio, "nu_e_ratio/F");
      _image_tree->Branch( "child_pdg_v", &_child_pdg_v, "child_pdg_v/F");
      _image_tree->Branch( "child_ratio_v", &_child_ratio_v, "child_ratio_v/F");
      _image_tree->Branch( "worst_ratio", &_worst_ratio, "worst_ratio/F");
    }
  }


 void RmEmptyEvts::reset(){

    _plane = -1;
    _vtx_x = -999;
    _vtx_y = -999;
    _vtx_z = -999;
    _dist_to_wall = -999;
    _e_dep = -999.;
    _e_vis = 0.;
    _pixel_count = 0;
    _child_e_ratio = 0.;
    _nu_e_ratio = 0.;
    _worst_ratio = 100000.;

    _child_pdg_v.clear() ;
    _child_ratio_v.clear() ;

  }

  bool RmEmptyEvts::process(IOManager& mgr)
  {

    auto ev_image2d = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_name));
    auto ev_roi = (EventROI*)(mgr.get_data(kProductROI,_roi_name));

    if (!ev_roi || !ev_image2d ) return false; 

    auto const& img2d_v = ev_image2d->Image2DArray();
    auto const& roi_v   = ev_roi->ROIArray();

    _event++; 

    if(!img2d_v.size() || !roi_v.size() ) return true ;

    const ::geoalgo::AABox_t tpc(0,-116,0,256.35,116,1036.8);
    ::geoalgo::GeoAlgo GeoAlg ;
    auto geomHelper = larutil::GeometryHelper::GetME();

    auto t2cm = geomHelper->TimeToCm();

    int index = 2;

      reset() ;

      _plane = index ;

      auto img2d = img2d_v[index];
      auto roi   = roi_v[0]; 

      _vtx_x = roi.X();
      _vtx_y = roi.Y();
      _vtx_z = roi.Z();

      const ::geoalgo::Point_t vtx(_vtx_x, _vtx_y, _vtx_z) ;
      _dist_to_wall = sqrt(GeoAlg.SqDist(tpc, vtx)) ;

      auto const& pixel_array = img2d.as_vector();
      auto const& meta = img2d.meta() ;

      for(size_t i = 0; i < pixel_array.size(); i++){
        auto const & v = pixel_array[i] ;
	if( v > 0.5 ) _pixel_count ++ ;
	}

      //Store rest mass of particles 
      std::map<int,float> pdg_m ;
      pdg_m[11] = 0.511;
      pdg_m[22] = 0.;
      pdg_m[12] = 0.;
      pdg_m[14] = 0.;
      pdg_m[2212] = 938.272;
      pdg_m[2112] = 939.565;
      pdg_m[13] = 105.658;
      pdg_m[111]  = 134.977;
      pdg_m[211]  = 139.570;
      pdg_m[321]  = 493.648;

      _nu_e_ratio = roi_v.at(0).EnergyDeposit() / roi_v.at(0).EnergyInit() ;
      _e_dep = roi_v.at(0).EnergyDeposit() ;

      float e_dep = 0.;
      float e_init = 0. ;

//      std::cout<<"\n\nEvent is : "<<_event<<", "<<_dist_to_wall<<std::endl ;//("<<_vtx_x<<", "<<_vtx_y<<", "<<_vtx_z<<")"<<std::endl ;
      _child_pdg_v.reserve(roi_v.size() - 1);
      _child_ratio_v.reserve(roi_v.size() - 1);

      for( int j = 1; j < roi_v.size(); j++){

        auto r = roi_v[j];

        if ( r.MCTIndex() == kINVALID_INDEX )
          std::cout<<"Track status ? "<<r.MCTIndex() <<std::endl; 

        //std::cout<<"PDG: "<<r.PdgCode()<<", "<<r.ParentPdgCode()
	//         <<", ratio: "<<r.EnergyDeposit()/(r.EnergyInit() - pdg_m[std::abs(r.PdgCode())])
	//	 << ", ("<<r.EnergyDeposit()<<", "<<r.EnergyInit()<<")"<<std::endl ; 

	_child_pdg_v.emplace_back(r.PdgCode()) ;


	e_dep += r.EnergyDeposit();
	e_init += r.EnergyInit() - pdg_m[std::abs(r.PdgCode())] ;

	_child_ratio_v.emplace_back(e_dep/e_init) ;

	if( (e_dep / e_init) < _worst_ratio )
	  _worst_ratio = e_dep/e_init ;

//        geoalgo::HalfLine roi_HL(r.X(),r.Y(),r.Z(),r.Px(),r.Py(),r.Pz());
//	geoalgo::Point_t vtx(r.X(),r.Y(),r.Z());
//        auto i_pt = GeoAlg.Intersection(tpc,roi_HL) ;
//	float t_dist = 0;
//	if (i_pt.size() )
//	  t_dist = sqrt ( pow(i_pt.at(0)[0] - r.X(),2) + pow(i_pt.at(0)[1] - r.Y(),2) + pow(i_pt.at(0)[2] - r.Z(),2) ) ;

      }

      _child_e_ratio = ( e_dep / e_init ) ;

      //std::cout<<"Fractional energy deposition: "<<_child_e_ratio<<", vs "<<_nu_e_ratio<<std::endl ;
      _image_tree->Fill();


  
  return true; 

  }

  void RmEmptyEvts::finalize()
  {
    if(has_ana_file()){
      _image_tree->Write();
      }
  }

}
#endif
