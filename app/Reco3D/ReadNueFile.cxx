#ifndef __READNUEFILE_CXX__
#define __READNUEFILE_CXX__

#include "ReadNueFile.h"

// larcv
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"

// larlite
#include "DataFormat/track.h"
#include "DataFormat/hit.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/wire.h"
#include "DataFormat/vertex.h"

// ROOT
#include "TTree.h"
#include "TVector3.h"

// AStar
#include "AStarUtils.h"
#include "AStar3DAlgoProton.h"

// test
#include <cassert>

namespace larcv {

  static ReadNueFileProcessFactory __global_ReadNueFileProcessFactory__;

  ReadNueFile::ReadNueFile(const std::string name)
    : ProcessBase(name), 
      _foutll(""),
      _spline_file(""), 
      _recoTree(nullptr)
  {}
  
  void ReadNueFile::configure(const PSet& cfg)
  {

    _img2d_producer    = cfg.get<std::string>("Image2DProducer");
    _pgraph_producer   = cfg.get<std::string>("PGraphProducer");
    _par_pix_producer  = cfg.get<std::string>("ParPixelProducer");
    _true_roi_producer = cfg.get<std::string>("TrueROIProducer");
    _mask_shower       = cfg.get<bool>("MaskShower",true);
  }

  void ReadNueFile::initialize()
  {
    LARCV_INFO() << "[ReadNueFile]" << std::endl;
    assert(!_spline_file.empty());
    tracker.SetDrawOutputs(false);
    tracker.SetOutputDir("png");
    tracker.SetSplineFile(_spline_file);
    tracker.initialize();
    tracker.SetCompressionFactors(1,6);
    tracker.SetVerbose(2);
    
    _recoTree = new TTree("_recoTree","_recoTree");
    _recoTree->Branch("run"     , &_run   , "_run/I");
    _recoTree->Branch("subrun"  , &_subrun, "_subrun/I");
    _recoTree->Branch("event"   , &_event , "_event/I");
    _recoTree->Branch("entry"   , &_entry , "_entry/I");

    //
    // Reco information
    // 
    _recoTree->Branch("vtx_id", &_vtx_id, "vtx_id/I");
    _recoTree->Branch("vtx_x" , &_vtx_x , "vtx_x/F");
    _recoTree->Branch("vtx_y" , &_vtx_y , "vtx_y/F");
    _recoTree->Branch("vtx_z" , &_vtx_z , "vtx_z/F");

    _recoTree->Branch("E_muon_v"   , &_E_muon_v);
    _recoTree->Branch("E_proton_v" , &_E_proton_v);
    _recoTree->Branch("Length_v"   , &_Length_v);
    _recoTree->Branch("Avg_Ion_v"  , &_Avg_Ion_v);
    _recoTree->Branch("Angle_v"    , &_Angle_v);
    _recoTree->Branch("Reco_goodness_v" , &_Reco_goodness_v);
    _recoTree->Branch("GoodVertex" , &_GoodVertex , "GoodVertex/I");
    _recoTree->Branch("Nreco" , &_Nreco , "Nreco/I");
    _recoTree->Branch("missingTrack",&_missingTrack);
    _recoTree->Branch("nothingReconstructed",&_nothingReconstructed);
    _recoTree->Branch("tooShortDeadWire",&_tooShortDeadWire);
    _recoTree->Branch("tooShortFaintTrack",&_tooShortFaintTrack);
    _recoTree->Branch("tooManyTracksAtVertex",&_tooManyTracksAtVertex);
    _recoTree->Branch("possibleCosmic",&_possibleCosmic);
    _recoTree->Branch("possiblyCrossing",&_possiblyCrossing);
    _recoTree->Branch("branchingTracks",&_branchingTracks);
    _recoTree->Branch("jumpingTracks",&_jumpingTracks);


    //
    // MC Information
    //
    _recoTree->Branch("Ep_t"       , &_Ep_t       , "Ep_t/D");
    _recoTree->Branch("Em_t"       , &_Em_t       , "Em_t/D");
    _recoTree->Branch("Ee_t"       , &_Ee_t       , "Ee_t/D");

    _recoTree->Branch("MuonStartPoint_X", &_MuonStartPoint_X, "MuonStartPoint_X/D");
    _recoTree->Branch("ProtonStartPoint_X", &_ProtonStartPoint_X, "ProtonStartPoint_X/D");
    _recoTree->Branch("ElectronStartPoint_X", &_ElectronStartPoint_X, "ElectronStartPoint_X/D");

    _recoTree->Branch("MuonStartPoint_Y", &_MuonStartPoint_Y, "MuonStartPoint_Y/D");
    _recoTree->Branch("ProtonStartPoint_Y", &_ProtonStartPoint_Y, "ProtonStartPoint_Y/D");
    _recoTree->Branch("ElectronStartPoint_Y", &_ElectronStartPoint_Y, "ElectronStartPoint_Y/D");

    _recoTree->Branch("MuonStartPoint_Z", &_MuonStartPoint_Z, "MuonStartPoint_Z/D");
    _recoTree->Branch("ProtonStartPoint_Z", &_ProtonStartPoint_Z, "ProtonStartPoint_Z/D");
    _recoTree->Branch("ElectronStartPoint_Z", &_ElectronStartPoint_Z, "ElectronStartPoint_Z/D");

    _recoTree->Branch("MuonEndPoint_X", &_MuonEndPoint_X, "MuonEndPoint_X/D");
    _recoTree->Branch("ProtonEndPoint_X", &_ProtonEndPoint_X, "ProtonEndPoint_X/D");
    _recoTree->Branch("ElectronEndPoint_X", &_ElectronEndPoint_X, "ElectronEndPoint_X/D");

    _recoTree->Branch("MuonEndPoint_Y", &_MuonEndPoint_Y, "MuonEndPoint_Y/D");
    _recoTree->Branch("ProtonEndPoint_Y", &_ProtonEndPoint_Y, "ProtonEndPoint_Y/D");
    _recoTree->Branch("ElectronEndPoint_Y", &_ElectronEndPoint_Y, "ElectronEndPoint_Y/D");

    _recoTree->Branch("MuonEndPoint_Z", &_MuonEndPoint_Z, "MuonEndPoint_Z/D");
    _recoTree->Branch("ProtonEndPoint_Z", &_ProtonEndPoint_Z, "ProtonEndPoint_Z/D");
    _recoTree->Branch("ElectronEndPoint_Z", &_ElectronEndPoint_Z, "ElectronEndPoint_Z/D");
    
    if (_foutll.empty()) throw larbys("specify larlite file output name");

    _storage.set_io_mode(larlite::storage_manager::kWRITE);
    _storage.set_out_filename(_foutll);

    if(!_storage.open()) {
      LARCV_CRITICAL() << "ERROR, larlite output file could not open" << std::endl;
      throw larbys("die");
    }


  }

  bool ReadNueFile::process(IOManager& mgr)
  {
    ClearEvent();
    LARCV_INFO() << std::endl;
    LARCV_INFO() << "============================================" << std::endl;
    LARCV_INFO() << "Entry " << mgr.current_entry() << " / " << mgr.get_n_entries() << std::endl;
    LARCV_INFO() << "============================================" << std::endl;
    gROOT->SetBatch(kTRUE);
    TVector3 vertex;

    //
    // Loop per vertex (larcv type is PGraph "Particle Graph")
    //
    
    auto ev_img_v       = (EventImage2D*) mgr.get_data(kProductImage2D,_img2d_producer);
    auto ev_pgraph_v    = (EventPGraph*)  mgr.get_data(kProductPGraph,_pgraph_producer);
    auto ev_pix_v       = (EventPixel2D*) mgr.get_data(kProductPixel2D,_par_pix_producer);
    EventROI* ev_partroi_v = nullptr;
    if (!_true_roi_producer.empty())
      ev_partroi_v = (EventROI*) mgr.get_data(kProductROI,_true_roi_producer);

    // auto tag_img_thru_v = (EventImage2D*) mgr.get_data(kProductImage2D,"thrumutags");
    // auto tag_img_stop_v = (EventImage2D*) mgr.get_data(kProductImage2D,"stopmutags");
    

    auto track_ptr = (larlite::event_track*)_storage.get_data(larlite::data::kTrack,"trackReco");
    auto vertex_ptr = (larlite::event_vertex*)_storage.get_data(larlite::data::kVertex,"trackReco");

    _run    = (int) ev_pgraph_v->run();
    _subrun = (int) ev_pgraph_v->subrun();
    _event  = (int) ev_pgraph_v->event();
    _entry  = (int) mgr.current_entry();

    // get the event clusters and full images
    const auto& full_adc_img_v = ev_img_v->Image2DArray();
    const auto& pix_m          = ev_pix_v->Pixel2DClusterArray();
    const auto& pix_meta_m     = ev_pix_v->ClusterMetaArray();

    // const auto& full_tag_img_thru_v = tag_img_thru_v->Image2DArray();
    // const auto& full_tag_img_stop_v = tag_img_stop_v->Image2DArray();


    //
    // Fill MC if exists
    //
    if (ev_partroi_v) {
      const auto& mc_roi_v = ev_partroi_v->ROIArray();
      FillMC(mc_roi_v);
    }


    //
    // No vertex, continue (but fill)
    //
    if (ev_pgraph_v->PGraphArray().empty()) {
      ClearVertex();
      return true;
    }


    static double wireRange = 5000;
    static double tickRange = 8502;

    if (_Full_meta_v.empty()) {
      assert (_Full_image_v.empty());
      assert (_Tagged_Image.empty());
      _Full_meta_v.resize(3);
      _Full_image_v.resize(3);
      _Tagged_Image.resize(3);
      for(size_t iPlane=0;iPlane<3;iPlane++){
	_Full_meta_v[iPlane]  = larcv::ImageMeta(wireRange,tickRange,(int)(tickRange)/6,(int)(wireRange),0,tickRange);
	_Full_image_v[iPlane] = larcv::Image2D(_Full_meta_v[iPlane]);
	_Tagged_Image[iPlane] = larcv::Image2D(_Full_meta_v[iPlane]);
      }
    } else {
      assert (full_adc_img_v.size()==3);
      for(size_t iPlane=0;iPlane<3;iPlane++){
	_Full_image_v[iPlane].paint(0);
	_Tagged_Image[iPlane].paint(0);
	_Full_image_v[iPlane].overlay( full_adc_img_v[iPlane] );
      }
      //if(full_tag_img_thru_v->size() == 3)Tagged_Image[iPlane].overlay( (*full_tag_img_thru_v)[iPlane] );
      //if(full_tag_img_stop_v->size() == 3)Tagged_Image[iPlane].overlay( (*full_tag_img_stop_v)[iPlane] );
    }

    //
    // loop over found vertices
    //
    static std::vector<TVector3> vertex_v;
    vertex_v.clear();
    vertex_v.reserve(ev_pgraph_v->PGraphArray().size());

    for(size_t pgraph_id = 0; pgraph_id < ev_pgraph_v->PGraphArray().size(); ++pgraph_id) {

      iTrack++;

      auto const& pgraph        = ev_pgraph_v->PGraphArray().at(pgraph_id);
      auto const& roi_v         = pgraph.ParticleArray();
      auto const& cluster_idx_v = pgraph.ClusterIndexArray();

      //
      // Get Estimated 3D Start and End Points
      vertex_v.emplace_back(pgraph.ParticleArray().front().X(),
			    pgraph.ParticleArray().front().Y(),
			    pgraph.ParticleArray().front().Z());

      if (_mask_shower) {
	//
	// mask shower particle pixels 
	// method : pixels stored via larbys image
	//
	for(size_t roid=0; roid < roi_v.size(); ++roid) {
	  const auto& roi = roi_v[roid];
	  auto cidx = cluster_idx_v.at(roid);

	  if (roi.Shape() == kShapeTrack) continue;
	
	  for(size_t plane=0; plane<3; ++plane) {
	  
	    auto iter_pix = pix_m.find(plane);
	    if(iter_pix == pix_m.end())
	      continue;

	    auto iter_pix_meta = pix_meta_m.find(plane);
	    if(iter_pix_meta == pix_meta_m.end())
	      continue;
	  
	    auto const& pix_v      = (*iter_pix).second;
	    auto const& pix_meta_v = (*iter_pix_meta).second;
	  
	    auto const& pix      = pix_v.at(cidx);
	    auto const& pix_meta = pix_meta_v.at(cidx);
	  
	    auto& plane_img = _Full_image_v.at(plane);

	    for(const auto& px : pix) {
	      auto posx = pix_meta.pos_x(px.Y());
	      auto posy = pix_meta.pos_y(px.X());
	      auto row = plane_img.meta().row(posy);
	      auto col = plane_img.meta().col(posx);
	      plane_img.set_pixel(row,col,0);
	    }
	  } // end plane
	} // end ROI
      } // end mask shower

    } // end vertex


    tracker.SetOriginalImage(_Full_image_v);
    tracker.SetTaggedImage(_Tagged_Image);
    tracker.SetTrackInfo(_run, _subrun, _event, 0);

    for(size_t ivertex = 0;ivertex<vertex_v.size();ivertex++){

      const auto& vtx = vertex_v[ivertex];
      double xyz[3] = {vtx.X(),vtx.Y(),vtx.Z()};
      vertex_ptr->push_back(larlite::vertex(xyz,ivertex));

      _vtx_id = (int) ivertex;
      _vtx_x  = (float) vtx.X();
      _vtx_y  = (float) vtx.Y();
      _vtx_z  = (float) vtx.Z();

      tracker.SetSingleVertex(vtx);
      tracker.ReconstructVertex();
      
      auto recoedVertex = tracker.GetReconstructedVertexTracks();
     
      for(const auto& itrack : recoedVertex)
	track_ptr->emplace_back(std::move(itrack));
     
      auto Energies_v = tracker.GetEnergies();
      _E_muon_v.resize(Energies_v.size());
      _E_proton_v.resize(Energies_v.size());
      
      for(size_t trackid=0; trackid<Energies_v.size(); ++trackid) {
	_E_proton_v[trackid] = Energies_v[trackid].front();
	_E_muon_v[trackid]   = Energies_v[trackid].back();
      }
      
      _Length_v  = tracker.GetVertexLength();;
      _Avg_Ion_v = tracker.GetAverageIonization();
      _Angle_v   = tracker.GetVertexAngle(15); 

      auto Reco_goodness_v = tracker.GetRecoGoodness();
      _Reco_goodness_v.resize(Reco_goodness_v.size(),kINVALID_INT);

      for(size_t gid=0; gid< _Reco_goodness_v.size(); ++gid) 
	_Reco_goodness_v[gid] = (int) Reco_goodness_v[gid];

      assert (_Reco_goodness_v.size() == 9);
      _missingTrack          = (int) _Reco_goodness_v[0];
      _nothingReconstructed  = (int) _Reco_goodness_v[1];
      _tooShortDeadWire      = (int) _Reco_goodness_v[2];
      _tooShortFaintTrack    = (int) _Reco_goodness_v[3];
      _tooManyTracksAtVertex = (int) _Reco_goodness_v[4];
      _possibleCosmic        = (int) _Reco_goodness_v[5];
      _possiblyCrossing      = (int) _Reco_goodness_v[6];
      _branchingTracks       = (int) _Reco_goodness_v[7];
      _jumpingTracks         = (int) _Reco_goodness_v[8];

      auto GoodVertex = false;
      GoodVertex  = tracker.IsGoodVertex();
      _GoodVertex = (int) GoodVertex;
      _Nreco++;
      
      _recoTree->Fill();
      ClearVertex();
    }
    _storage.set_id(_run,_subrun,_event);
    _storage.next_event(true);
    
    std::cout << "...Reconstruted..." << std::endl;
    return true;
  }

  void ReadNueFile::finalize() {
    tracker.finalize();

    if(has_ana_file()) {
      ana_file().cd();
      _recoTree->Write();
    }
    
    _storage.close();
  } 
   
  void ReadNueFile::SetSplineLocation(const std::string& fpath) {
    LARCV_INFO() << "setting spline loc @ " << fpath << std::endl;
    tracker.SetSplineFile(fpath);
    _spline_file = fpath;
    LARCV_DEBUG() << "end" << std::endl;
  }

  void ReadNueFile::ClearEvent() {
    _run    = kINVALID_INT;
    _subrun = kINVALID_INT;
    _event  = kINVALID_INT;
    _Nreco = 0;

    _MuonStartPoint_X = -1.0*kINVALID_DOUBLE;
    _ProtonStartPoint_X = -1.0*kINVALID_DOUBLE;
    _ElectronStartPoint_X = -1.0*kINVALID_DOUBLE;

    _MuonStartPoint_Y = -1.0*kINVALID_DOUBLE;
    _ProtonStartPoint_Y = -1.0*kINVALID_DOUBLE;
    _ElectronStartPoint_Y = -1.0*kINVALID_DOUBLE;

    _MuonStartPoint_Z = -1.0*kINVALID_DOUBLE;
    _ProtonStartPoint_Z = -1.0*kINVALID_DOUBLE;
    _ElectronStartPoint_Z = -1.0*kINVALID_DOUBLE;

    
    _MuonEndPoint_X = -1.0*kINVALID_DOUBLE;
    _ProtonEndPoint_X = -1.0*kINVALID_DOUBLE;
    _ElectronEndPoint_X = -1.0*kINVALID_DOUBLE;

    _MuonEndPoint_Y = -1.0*kINVALID_DOUBLE;
    _ProtonEndPoint_Y = -1.0*kINVALID_DOUBLE;
    _ElectronEndPoint_Y = -1.0*kINVALID_DOUBLE;

    _MuonEndPoint_Z = -1.0*kINVALID_DOUBLE;
    _ProtonEndPoint_Z = -1.0*kINVALID_DOUBLE;
    _ElectronEndPoint_Z = -1.0*kINVALID_DOUBLE;

    _Ep_t = -1.0*kINVALID_DOUBLE;
    _Em_t = -1.0*kINVALID_DOUBLE;
    _Ee_t = -1.0*kINVALID_DOUBLE;
  }

  void ReadNueFile::ClearVertex() {
    _vtx_id = -1.0 * kINVALID_INT;
    _vtx_x  = -1.0 * kINVALID_FLOAT;
    _vtx_y  = -1.0 * kINVALID_FLOAT;
    _vtx_z  = -1.0 * kINVALID_FLOAT;
    _E_muon_v.clear();
    _E_proton_v.clear();
    _Length_v.clear();
    _Avg_Ion_v.clear();
    _Angle_v.clear();
    _Reco_goodness_v.clear();
    _GoodVertex = -1.0*kINVALID_INT;
    _missingTrack = -1.0*kINVALID_INT;
    _nothingReconstructed = -1.0*kINVALID_INT;
    _tooShortDeadWire = -1.0*kINVALID_INT;
    _tooShortFaintTrack = -1.0*kINVALID_INT;
    _tooManyTracksAtVertex = -1.0*kINVALID_INT;
    _possibleCosmic = -1.0*kINVALID_INT;
    _possiblyCrossing = -1.0*kINVALID_INT;
    _branchingTracks = -1.0*kINVALID_INT;
    _jumpingTracks = -1.0*kINVALID_INT;

  }

  void ReadNueFile::FillMC(const std::vector<ROI>& mc_roi_v) {
    bool found_muon = false;
    bool found_proton = false;
    bool found_electron = false;

    for(size_t iMC = 0;iMC<mc_roi_v.size();iMC++){
      if(mc_roi_v[iMC].PdgCode() == 13){
	if (found_muon) continue;
	LARCV_INFO() << "muon.....@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
	_MuonStartPoint_X = mc_roi_v[iMC].X();
	_MuonStartPoint_Y = mc_roi_v[iMC].Y();
	_MuonStartPoint_Z = mc_roi_v[iMC].Z();

	_MuonEndPoint_X = mc_roi_v[iMC].EndPosition().X();
	_MuonEndPoint_Y = mc_roi_v[iMC].EndPosition().Y();
	_MuonEndPoint_Z = mc_roi_v[iMC].EndPosition().Z();
	_Em_t = mc_roi_v[iMC].EnergyDeposit();
	found_muon = true;
      }
      if(mc_roi_v[iMC].PdgCode() == 2212){
	if (found_proton) continue;
	LARCV_INFO() << "proton...@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
	_ProtonStartPoint_X = mc_roi_v[iMC].X();
	_ProtonStartPoint_Y = mc_roi_v[iMC].Y();
	_ProtonStartPoint_Z = mc_roi_v[iMC].Z();

	_ProtonEndPoint_X = mc_roi_v[iMC].EndPosition().X();
	_ProtonEndPoint_Y = mc_roi_v[iMC].EndPosition().Y();
	_ProtonEndPoint_Z = mc_roi_v[iMC].EndPosition().Z();
	_Ep_t = mc_roi_v[iMC].EnergyDeposit();
	found_proton = true;
      }
      if(mc_roi_v[iMC].PdgCode() == 11){
	if(found_electron) continue;
	LARCV_INFO() << "electron.@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
	_ElectronStartPoint_X = mc_roi_v[iMC].X();
	_ElectronStartPoint_Y = mc_roi_v[iMC].Y();
	_ElectronStartPoint_Z = mc_roi_v[iMC].Z();

	_ElectronEndPoint_X = mc_roi_v[iMC].EndPosition().X();
	_ElectronEndPoint_Y = mc_roi_v[iMC].EndPosition().Y();
	_ElectronEndPoint_Z = mc_roi_v[iMC].EndPosition().Z();
	_Ee_t = mc_roi_v[iMC].EnergyDeposit();
	found_electron = true;
      }
    } // end rois
    return;
  } // end mc


}
#endif
