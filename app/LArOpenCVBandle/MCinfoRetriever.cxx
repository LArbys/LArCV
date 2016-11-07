#ifndef __MCINFORETRIEVER_CXX__
#define __MCINFORETRIEVER_CXX__

#include "MCinfoRetriever.h"
#include "DataFormat/ROI.h"
#include "DataFormat/EventROI.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/LArProperties.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static MCinfoRetrieverProcessFactory __global_MCinfoRetrieverProcessFactory__;

  MCinfoRetriever::MCinfoRetriever(const std::string name)
    : ProcessBase(name)
  {}
    
  void MCinfoRetriever::configure(const PSet& cfg)
  {
    _producer_roi       = cfg.get<std::string>("MCProducer");
    _producer_image2d   = cfg.get<std::string>("Image2DProducer");
  }

  void MCinfoRetriever::initialize()
  {
    _mc_tree = new TTree("mctree","MC infomation");
    _mc_tree->Branch("run",&_run,"sun/I");
    _mc_tree->Branch("subrun",&_subrun,"subrun/I");
    _mc_tree->Branch("event",&_event,"event/I");
    _mc_tree->Branch("parentPDG",&_parent_pdg,"parentPDG/I");
    _mc_tree->Branch("energyDeposit",&_energy_deposit,"energyDeposit/D");
    _mc_tree->Branch("parentX",&_parent_x,"parentX/D");
    _mc_tree->Branch("parentY",&_parent_y,"parentY/D");
    _mc_tree->Branch("parentZ",&_parent_z,"parentZ/D");
    _mc_tree->Branch("parentT",&_parent_t,"parentT/D");
    _mc_tree->Branch("parentPx",&_parent_px,"parentPx/D");
    _mc_tree->Branch("parentPy",&_parent_py,"parentPy/D");
    _mc_tree->Branch("parentPz",&_parent_pz,"parentpz/D");
    _mc_tree->Branch("currentType",&_current_type,"currentType/S");
    _mc_tree->Branch("interactionType",&_current_type,"InteractionType/S");
    _mc_tree->Branch("vtx2d_w","std::vector<double>",&_vtx_2d_w_v);
    _mc_tree->Branch("vtx2d_t","std::vector<double>",&_vtx_2d_t_v);
    
  }

  bool MCinfoRetriever::process(IOManager& mgr)
  {

    _vtx_2d_w_v.clear();
    _vtx_2d_t_v.clear();
    _vtx_2d_w_v.resize(3);
    _vtx_2d_t_v.resize(3);
    _image_v.clear();
    _image_v.resize(3);
    
    auto ev_roi = (larcv::EventROI*)mgr.get_data(kProductROI,_producer_roi);
    auto const ev_image2d = (larcv::EventImage2D*)mgr.get_data(kProductImage2D,_producer_image2d);

    _run    = ev_roi->run();
    _subrun = ev_roi->subrun();
    _run    = ev_roi->event();
    
    auto roi = ev_roi->at(0);
    
    _parent_pdg = roi.PdgCode();
    _energy_deposit = roi.EnergyDeposit();
    _parent_x = roi.X(); 
    _parent_y = roi.Y(); 
    _parent_z = roi.Z(); 
    _parent_t = roi.T(); 
    _parent_px = roi.Px(); 
    _parent_py = roi.Py(); 
    _parent_pz = roi.Pz(); 

    _current_type = roi.NuCurrentType();
    _interaction_type  =roi.NuInteractionType();
    
    //Get 2D projections from 3D
    
    auto geohelp = larutil::GeometryHelper::GetME();//Geohelper from LArLite
    auto larpro  = larutil::LArProperties::GetME(); //LArProperties from LArLite
    
    for (int plane = 0 ; plane<3;++plane){
      
      auto vtx_2d = geohelp->Point_3Dto2D(_parent_x, _parent_y, _parent_z, plane );
      
      //auto vtx_w = vtx_2d.w / geohelp->WireToCm();
      //auto vtx_t = vtx_2d.t / geohelp->TimeToCm();

      ///Convert [cm] to [pixel]

      _image_v[plane] = ev_image2d->Image2DArray()[plane];
      
      _meta = _image_v[plane].meta();
      double x_compression, y_compression;
      x_compression  = _meta.width()  / _meta.cols();
      y_compression  = _meta.height() / _meta.rows();
      
      double x_pixel  ,  y_pixel;
      x_pixel = (vtx_2d.w/geohelp->WireToCm() - _meta.tl().x) / x_compression;
      y_pixel = (((_parent_x/larpro->DriftVelocity() + _parent_t/1000.)*2+3200)-_meta.br().y)/y_compression;
   
      _vtx_2d_w_v[plane] = x_pixel;
      _vtx_2d_t_v[plane] = y_pixel;
    
      
      
    }
    
    //Fill tree
    _mc_tree->Fill();
    return true;
    
  }

  void MCinfoRetriever::finalize()
  {
    _mc_tree->Write();
  }

}
#endif
