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
    _event = 0 ; 

    /// From LinearEnergy algorithm in larlite ShowerReco3D
    //_caloAlg = ::calo::CalorimetryAlg();
    //_caloAlg.setUseModBox(true);
    _e_to_eV = 23.6;  // ionization energy of Ar in eV
    _eV_to_MeV = 1e-6; // eV -> MeV conversion
  }
    
  void RmEmptyEvts::configure(const PSet& cfg)
  {
    _image_name = cfg.get<std::string>("ImageName");
    _roi_name = cfg.get<std::string>("ROIName");
  }

  void RmEmptyEvts::initialize()
  {

    double MeV_to_fC = 1. / ( _e_to_eV * _eV_to_MeV );
    double MIP = 2.3; // MeV/cm
    _recomb_factor = larutil::LArProperties::GetME()->ModBoxInverse( MIP ) / ( MIP * MeV_to_fC );
    _timetick = larutil::DetectorProperties::GetME()->SamplingRate() * 1.e-3;
    _tau = larutil::LArProperties::GetME()->ElectronLifetime(); 

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

  }

  bool RmEmptyEvts::process(IOManager& mgr)
  {

    auto ev_image2d = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_name));
    auto ev_roi = (EventROI*)(mgr.get_data(kProductROI,_roi_name));
    auto const& img2d_v = ev_image2d->Image2DArray();
    auto const& roi_v   = ev_roi->ROIArray();

    const ::geoalgo::AABox_t tpc(0,-116,0,256.35,116,1036.8);
    ::geoalgo::GeoAlgo GeoAlg ;
    auto geomHelper = larutil::GeometryHelper::GetME();

    auto t2cm = geomHelper->TimeToCm();

    for(size_t index=0; index < img2d_v.size(); ++index) {

      if (index != 2) continue ;

      reset() ;

      _plane = index ;

      auto const& img2d = img2d_v[index];
      auto const& roi   = roi_v[index]; 

      _vtx_x = roi.X();
      _vtx_y = roi.Y();
      _vtx_z = roi.Z();

      const ::geoalgo::Point_t vtx(_vtx_x, _vtx_y, _vtx_z) ;
      _dist_to_wall = sqrt(GeoAlg.SqDist(tpc, vtx)) ;

      _e_dep = roi.EnergyDeposit();

      auto const& pixel_array = img2d.as_vector();
      auto const& meta = img2d.meta() ;

      for(size_t i = 0; i < pixel_array.size(); i++){
        auto const & v = pixel_array[i] ;
	if(v > 0.9 )
	  _e_vis += v ;
           /////////////////////////////////////////////////////////////
           // store calculated energy
//           double E  = 0.;
//           double dQ = 0.;
        //   for (auto const &h : hits) {

        //     // lifetime correction
        //     double hit_tick = h.t / t2cm;
        //     double lifetimeCorr = exp( hit_tick * _timetick / _tau );

        //     dQ = _caloAlg.ElectronsFromADCArea(h.charge, pl);
        //     E += dQ * lifetimeCorr * _e_to_eV * _eV_to_MeV;

        //     }

        //   E /= _recomb_factor;

           // correct for plane-dependent shower reco energy calibration
           ///////////////////////////////////////////////////
	}

      _image_tree->Fill();
   }

   _event++; 
  
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
