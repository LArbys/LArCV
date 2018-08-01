#ifndef __TRACKEREVENTDISPLAY_CXX__
#define __TRACKEREVENTDISPLAY_CXX__

#include "TrackerEventDisplay.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/track.h"
#include "DataFormat/vertex.h"
#include "DataFormat/hit.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/wire.h"
#include "DataFormat/EventROI.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/TimeService.h"
#include "TFile.h"
#include "TTree.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TLine.h"
#include "TF1.h"
#include "TVector3.h"
#include "TRandom3.h"
#include "TSpline.h"

//#include "LArCV/core/DataFormat/ChStatus.h"
#include "AStarUtils.h"

#include "AStar3DAlgo.h"
#include "AStar3DAlgoProton.h"

//#include "SCE/SpaceChargeMicroBooNE.h"
#include "AStarTracker.h"


//#include "../../core/DataFormat/EventPGraph.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"

#include <cassert>

namespace larcv {

    static TrackerEventDisplayProcessFactory __global_TrackerEventDisplayProcessFactory__;

    TrackerEventDisplay::TrackerEventDisplay(const std::string name)
    : ProcessBase(name),
    _spline_file("Proton_Muon_Range_dEdx_LAr_TSplines.root")
    {}

    void TrackerEventDisplay::configure(const PSet& cfg)
    {
        _input_pgraph_producer     = cfg.get<std::string>("InputPgraphProducer");
        _img2d_producer            = cfg.get<std::string>("Image2DProducer");
        _par_pix_producer          = cfg.get<std::string>("ParPixelProducer");
        _true_roi_producer         = cfg.get<std::string>("TrueROIProducer");
        _mask_shower               = cfg.get<bool>("MaskShower",false);

    }

    void TrackerEventDisplay::initialize()
    {
        LARCV_INFO() << "[TrackerEventDisplay]" << std::endl;
        assert(!_spline_file.empty());
        tracker.SetDrawOutputs(false);
        tracker.SetOutputDir(out_dir);
        tracker.SetSplineFile(_spline_file);
        tracker.initialize();
        tracker.SetVerbose(0);

        std::string filename;

        std::cout << filename << std::endl;

        if (_finll.empty()) throw larbys("specify larlite file input name");

        _storage.set_io_mode(larlite::storage_manager::kREAD);
        _storage.add_in_filename(_finll);

        if(!_storage.open()) {
            LARCV_CRITICAL() << "ERROR, larlite input file could not open" << std::endl;
            throw larbys("die");
        }

    }

    bool TrackerEventDisplay::process(IOManager& mgr)
    {
        ClearEvent();
        std::cout << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Entry " << mgr.current_entry() << " / " << mgr.get_n_entries() << std::endl;
        std::cout << "============================================" << std::endl;
        gStyle->SetOptStat(0);

        TVector3 vertex(-1,-1,-1);

        auto ev_pgraph_v     = (EventPGraph*) mgr.get_data(kProductPGraph,_input_pgraph_producer);
        _run    = (int) ev_pgraph_v->run();
        _subrun = (int) ev_pgraph_v->subrun();
        _event  = (int) ev_pgraph_v->event();
        _entry  = (int) mgr.current_entry();

        _storage.next_event();
        auto ev_track  = (larlite::event_track*)  _storage.get_data(larlite::data::kTrack,"trackReco");
        auto ev_vertex = (larlite::event_vertex*) _storage.get_data(larlite::data::kVertex,"trackReco");
        auto ev_ass    = (larlite::event_ass*)    _storage.get_data(larlite::data::kAssociation,"trackReco");


        if(ev_pgraph_v->PGraphArray().size()==0){std::cout << "ev_pgraph_v->PGraphArray().size()==0" << std::endl;return true;}
        if(ev_track->size()==0){std::cout << "ev_track->size()==0" << std::endl;return true;}

        if((int)(_storage.run_id()) != _run){std::cout << "run# larlite and larcv don't match" << std::endl;return true;}
        if((int)(_storage.subrun_id()) != _subrun){std::cout << "subrun# larlite and larcv don't match" << std::endl;return true;}
        if((int)(_storage.event_id()) != _event){std::cout << "event# larlite and larcv don't match" << std::endl;return true;}


        auto ev_img_v           = (EventImage2D*)mgr.get_data(kProductImage2D,_img2d_producer);
        //auto tag_img_thru_v     = (EventImage2D*)mgr.get_data(kProductImage2D,"thrumutags");
        //auto tag_img_stop_v     = (EventImage2D*)mgr.get_data(kProductImage2D,"stopmutags");

        EventPixel2D* ev_pix_v = nullptr;
        if (!_par_pix_producer.empty())
            ev_pix_v = (EventPixel2D*) mgr.get_data(kProductPixel2D,_par_pix_producer);


        //auto ev_pcluster_v = (EventPixel2D*)mgr.get_data(kProductPixel2D,"test_img");
        //auto ev_ctor_v     = (EventPixel2D*)mgr.get_data(kProductPixel2D,"test_ctor");


        // get the event clusters and full images
        //auto const& ctor_m = ev_ctor_v->Pixel2DClusterArray();
        auto full_adc_img_v = &(ev_img_v->Image2DArray());
        //auto full_tag_img_thru_v = &(tag_img_thru_v->Image2DArray());
        //auto full_tag_img_stop_v = &(tag_img_stop_v->Image2DArray());



        //
        // Fill MC if exists
        //
        EventROI* ev_partroi_v = nullptr;
        if (!_true_roi_producer.empty()) ev_partroi_v = (EventROI*) mgr.get_data(kProductROI,_true_roi_producer);
        if (ev_partroi_v) {
            const auto& mc_roi_v = ev_partroi_v->ROIArray();
            FillMC(mc_roi_v);
        }


        // loop over found vertices
        //auto const& pcluster_m = ev_pcluster_v->Pixel2DClusterArray();

        std::vector<larcv::ImageMeta> Full_meta_v(3);
        std::vector<larcv::Image2D> Tagged_Image(3);
        std::vector<larcv::Image2D> Full_image_v(3);

        double wireRange = 5000;
        double tickRange = 8502;

        // Create base image2D with the full view, fill it with the input image 2D, we will crop it later
        for(size_t iPlane=0;iPlane<3;iPlane++){
            Full_meta_v[iPlane] = larcv::ImageMeta(wireRange,tickRange,(int)(tickRange)/6,(int)(wireRange),0,tickRange);
            Full_image_v[iPlane] = larcv::Image2D(Full_meta_v[iPlane]);
            Tagged_Image[iPlane] = larcv::Image2D(Full_meta_v[iPlane]);
            if(full_adc_img_v->size() == 3)Full_image_v[iPlane].overlay( (*full_adc_img_v)[iPlane] );
            //if(full_tag_img_thru_v->size() == 3)Tagged_Image[iPlane].overlay( (*full_tag_img_thru_v)[iPlane] );
            //if(full_tag_img_stop_v->size() == 3)Tagged_Image[iPlane].overlay( (*full_tag_img_stop_v)[iPlane] );
        }
        tracker.SetOriginalImage(Full_image_v);
        tracker.SetTaggedImage(Tagged_Image);
        tracker.SetTrackInfo(_run, _subrun, _event, 0);

        std::cout << _run << " " << _subrun << " " << _event <<  std::endl;


        static std::vector<TVector3> vertex_v;
        if(vertex_v.size()!=0)vertex_v.clear();


        larlite::event_track* ev_trk=nullptr;
        auto const& vtx_to_trk = _storage.find_one_ass(ev_vertex->id(), ev_trk, ev_vertex->name());
        if(!ev_trk || ev_trk->size() == 0) throw larlite::DataFormatException("Could not find associated track data product!");


        for(int vertex_index=0;vertex_index<ev_vertex->size();vertex_index++){
            larlite::event_track TracksAtVertex;
            std::cout << "vertex #" << vertex_index << std::endl;
            tracker.SetSingleVertex(TVector3(ev_vertex->at(vertex_index).X(),ev_vertex->at(vertex_index).Y(),ev_vertex->at(vertex_index).Z()));
            tracker.SetVertexID(ev_vertex->at(vertex_index).ID());

            for(auto const& trk_index : vtx_to_trk[vertex_index]) {
                TracksAtVertex.push_back( (*ev_trk)[trk_index]);
                std::cout << "\t => trk#" << trk_index << ", " << TracksAtVertex.back().Length() << " cm" << std::endl;
            }

            tracker.FeedLarliteVertexTracks(TracksAtVertex);
            tracker.DrawVertex();
        }

        return true;

    }

    void TrackerEventDisplay::MCevaluation(){
        std::cout << "MCevaluation()" << std::endl;

        std::vector< std::vector<double> > Energies_v = tracker.GetEnergies();
        //_Length_v = tracker.GetVertexLength();

        if(!tracker.IsGoodVertex()){std::cout << "error: is not good vertex" << std::endl;return;}
        if(_Avg_Ion_v.size()!=2){std::cout << "error: _Avg_Ion_v.size() = " << _Avg_Ion_v.size() << std::endl;return;}

        std::cout << "in store energies : " << std::endl;
        std::cout << Energies_v[0][0] << ":" << Energies_v[0][1] << std::endl;
        std::cout << Energies_v[1][0] << ":" << Energies_v[1][1] << std::endl;

        double Epreco;
        double Emreco;

        if(_Avg_Ion_v[0] > _Avg_Ion_v[1]){
            Epreco = Energies_v[0][0];// first track is proton
            Emreco = Energies_v[1][1];// second track is muon
        }
        else{
            Epreco = Energies_v[1][0]; // second track is proton
            Emreco = Energies_v[0][1]; // first track is muon
        }

        if( std::abs((Epreco-Ep_t)/Ep_t) > 0.1 || std::abs((Emreco-Em_t)/Em_t) > 0.1 ){
            checkEvents.push_back(Form("%d_%d_%d",_run,_subrun,_event));
        }

        std::cout << "proton : Epreco = " << Epreco << " MeV, Epth : " << Ep_t << " MeV..." << 100*(Epreco-Ep_t)/Ep_t << " %" << std::endl;
        std::cout << "muon   : Emreco = " << Emreco << " MeV, Emth : " << Em_t << " MeV..." << 100*(Emreco-Em_t)/Em_t << " %" << std::endl;

        std::cout << "E nu th   : " << NeutrinoEnergyTh << " MeV" << std::endl;
        std::cout << "E nu Reco : " << Epreco+Emreco << " MeV" << std::endl;
        std::cout << "relative Enu diff: " << 100*((Epreco+Emreco+146)-NeutrinoEnergyTh)/NeutrinoEnergyTh << " %" << std::endl;
        std::cout << "relative Enu_p+m diff: " << 100*((Epreco+Emreco)-(Ep_t+Em_t))/(Ep_t+Em_t) << " %" << std::endl;
    }

    void TrackerEventDisplay::finalize()
    {
        tracker.finalize();
        std::cout << "finalized tracker" << std::endl;

        _storage.close();

    }

    void TrackerEventDisplay::SetSplineLocation(const std::string& fpath) {
        LARCV_INFO() << "setting spline loc @ " << fpath << std::endl;
        tracker.SetSplineFile(fpath);
        _spline_file = fpath;
        LARCV_DEBUG() << "end" << std::endl;
    }

    void TrackerEventDisplay::FillMC(const std::vector<ROI>& mc_roi_v) {
        bool found_muon = false;
        bool found_proton = false;
        bool found_electron = false;

        std::vector<TVector3> MuonVertices;
        std::vector<TVector3> ProtonVertices;
        std::vector<TVector3> ElectronVertices;
        std::vector<TVector3> MuonEndPoint;
        std::vector<TVector3> ProtonEndPoint;
        std::vector<TVector3> ElectronEndPoint;

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

            // Adrien old version
            if(mc_roi_v[iMC].PdgCode() == 14){
                NeutrinoEnergyTh = mc_roi_v[iMC].EnergyDeposit();
                //std::cout << "Neutrino : " << NeutrinoEnergyTh << " MeV" << std::endl;
            }
            if(mc_roi_v[iMC].PdgCode() == 13){
                //std::cout << "muon.....@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
                Em_t = mc_roi_v[iMC].EnergyDeposit();
                MuonVertices.push_back(TVector3(mc_roi_v[iMC].X(),mc_roi_v[iMC].Y(),mc_roi_v[iMC].Z()));
                MuonEndPoint.push_back(TVector3(mc_roi_v[iMC].EndPosition().X(), mc_roi_v[iMC].EndPosition().Y(), mc_roi_v[iMC].EndPosition().Z()));
            }
            if(mc_roi_v[iMC].PdgCode() == 2212){
                if(mc_roi_v[iMC].EnergyDeposit() < 60) continue;
                //std::cout << "proton...@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
                Ep_t = mc_roi_v[iMC].EnergyDeposit();
                ProtonVertices.push_back(TVector3(mc_roi_v[iMC].X(),mc_roi_v[iMC].Y(),mc_roi_v[iMC].Z()));
                ProtonEndPoint.push_back(TVector3(mc_roi_v[iMC].EndPosition().X(), mc_roi_v[iMC].EndPosition().Y(), mc_roi_v[iMC].EndPosition().Z()));
            }
            if(mc_roi_v[iMC].PdgCode() == 11){
                //std::cout << "electron.@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
                ElectronVertices.push_back(TVector3(mc_roi_v[iMC].X(),mc_roi_v[iMC].Y(),mc_roi_v[iMC].Z()));
                ElectronEndPoint.push_back(TVector3(mc_roi_v[iMC].EndPosition().X(), mc_roi_v[iMC].EndPosition().Y(), mc_roi_v[iMC].EndPosition().Z()));
            }

            bool isVertex = false;
            bool isNumu = false;
            bool isNue = false;
            std::vector<int> goodMuon, goodElectron;
            for(size_t iProton = 0;iProton<ProtonVertices.size();iProton++){
                isVertex = false;
                isNumu = false;
                isNue = false;
                for(size_t iMuon = 0;iMuon<MuonVertices.size();iMuon++){
                    if(MuonVertices[iMuon] == ProtonVertices[iProton]){isVertex = true;isNumu = true;goodMuon.push_back(iMuon);}
                }
                for(size_t iElectron = 0;iElectron<ElectronVertices.size();iElectron++){
                    if(ProtonVertices[iProton] == ElectronVertices[iProton]){isVertex = true;isNue = true;goodElectron.push_back(iElectron);}
                }
                if(isVertex && MCVertices.size()!=0 && ProtonVertices[iProton] == MCVertices[MCVertices.size()-1])continue;
                if(isVertex){
                    MCvertex = ProtonVertices[iProton];
                    MCVertices.push_back(MCvertex);
                }
            }

            // end Adrien
        } // end rois
        return;
    } // end mc

    void TrackerEventDisplay::ClearEvent() {
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
        
        MCvertex.SetXYZ(-1,-1,-1);
        if(MCVertices.size()!=0)MCVertices.clear();
        
        ClearVertex();
    }
    
    void TrackerEventDisplay::ClearVertex() {
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
        GoodVertex = -1.0*kINVALID_INT;
        _missingTrack = -1.0*kINVALID_INT;
        _nothingReconstructed = -1.0*kINVALID_INT;
        _tooShortDeadWire = -1.0*kINVALID_INT;
        _tooShortFaintTrack = -1.0*kINVALID_INT;
        _tooManyTracksAtVertex = -1.0*kINVALID_INT;
        _possibleCosmic = -1.0*kINVALID_INT;
        _possiblyCrossing = -1.0*kINVALID_INT;
        _branchingTracks = -1.0*kINVALID_INT;
        _jumpingTracks = -1.0*kINVALID_INT;
        _trackQ50_v.clear();
        _trackQ30_v.clear();
        _trackQ20_v.clear();
        _trackQ10_v.clear();
        _trackQ5_v.clear();
        _trackQ3_v.clear();
        
    }
    
}
#endif
