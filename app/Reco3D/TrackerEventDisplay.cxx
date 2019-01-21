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

//#include "AStar3DAlgo.h"
//#include "AStar3DAlgoProton.h"

//#include "SCE/SpaceChargeMicroBooNE.h"
#include "AStarTracker.h"


//#include "../../core/DataFormat/EventPGraph.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"

#include <cassert>
#include <fstream>

namespace larcv {

    static TrackerEventDisplayProcessFactory __global_TrackerEventDisplayProcessFactory__;

    TrackerEventDisplay::TrackerEventDisplay(const std::string name)
    : ProcessBase(name),
    _spline_file("Proton_Muon_Range_dEdx_LAr_TSplines.root"){}

    void TrackerEventDisplay::configure(const PSet& cfg){
        _input_pgraph_producer     = cfg.get<std::string>("InputPgraphProducer");
        _img2d_producer            = cfg.get<std::string>("Image2DProducer");
        _par_pix_producer          = cfg.get<std::string>("ParPixelProducer");
        _true_roi_producer         = cfg.get<std::string>("TrueROIProducer");
        _mask_shower               = cfg.get<bool>("MaskShower",false);

    }

    void TrackerEventDisplay::initialize(){
        LARCV_INFO() << "[TrackerEventDisplay]" << std::endl;
        assert(!_spline_file.empty());
        tracker.SetDrawOutputs(false);
        tracker.SetOutputDir(out_dir);
        tracker.SetSplineFile(_spline_file);
        tracker.initialize();
        tracker.SetDrawVertical(true);
        tracker.SetDrawBlack(false);
        tracker.SetVerbose(0);

        std::string filename;

        std::cout << filename << std::endl;

        if (_finll.empty()) throw larbys("specify larlite file input name");
        if (_fana.empty())  throw larbys("specify root ana file input name");

        _storage.set_io_mode(larlite::storage_manager::kREAD);
        _storage.add_in_filename(_finll);

        if(!_storage.open()) {
            LARCV_CRITICAL() << "ERROR, larlite input file could not open" << std::endl;
            throw larbys("die");
        }

        TFile *fINana = TFile::Open(Form("%s",_fana.c_str()),"READ");
        if(!(fINana->IsOpen())){
            LARCV_CRITICAL() << "ERROR, root ana file could not open" << std::endl;
            throw larbys("die");
        }
        _recoTree = nullptr;
        _recoTree = (TTree*)fINana->Get("_recoTree");
        if(_recoTree==0){
            LARCV_CRITICAL() << "ERROR, could not access _recoTree" << std::endl;
            throw larbys("die");
        }
        _track_Goodness_v = 0;
        _Length_v = 0;
        _Reco_goodness_v = 0;
        _recoTree->SetBranchAddress("run",&_run_tree);
        _recoTree->SetBranchAddress("subrun",&_subrun_tree);
        _recoTree->SetBranchAddress("event",&_event_tree);
        _recoTree->SetBranchAddress("vtx_id",&_vtx_id_tree);
        _recoTree->SetBranchAddress("track_Goodness_v",&_track_Goodness_v);
        _recoTree->SetBranchAddress("Length_v",&_Length_v);
        _recoTree->SetBranchAddress("Reco_goodness_v",&_Reco_goodness_v);


        CreateSelectedList();
        MapTree();

    }

    bool TrackerEventDisplay::process(IOManager& mgr){
        //ClearEvent();
        gStyle->SetOptStat(0);

        TVector3 vertex(-1,-1,-1);

        auto ev_img_v        = (EventImage2D*)mgr.get_data(kProductImage2D,_img2d_producer);
        _run    = (int) ev_img_v->run();
        _subrun = (int) ev_img_v->subrun();
        _event  = (int) ev_img_v->event();
        _entry  = (int) mgr.current_entry();

        std::cout << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Entry " << mgr.current_entry() << " / " << mgr.get_n_entries() << std::endl;
        std::cout << "rse " << _run << "\t" << _subrun << "\t" << _event << std::endl;
        std::cout << "============================================" << std::endl;

        _storage.next_event();
        auto ev_track  = (larlite::event_track*)  _storage.get_data(larlite::data::kTrack,"trackReco");
        auto ev_vertex = (larlite::event_vertex*) _storage.get_data(larlite::data::kVertex,"trackReco");
        auto ev_ass    = (larlite::event_ass*)    _storage.get_data(larlite::data::kAssociation,"trackReco");


        if(ev_img_v->Image2DArray().size()==0){std::cout << "ev_img_v->Image2DArray().size()==0" << std::endl;return true;}
        if(ev_track->size()==0){std::cout << "ev_track->size()==0" << std::endl;return true;}

        if((int)(_storage.run_id()) != _run){std::cout << "run# larlite and larcv don't match" << std::endl;return true;}
        if((int)(_storage.subrun_id()) != _subrun){std::cout << "subrun# larlite and larcv don't match" << std::endl;return true;}
        if((int)(_storage.event_id()) != _event){std::cout << "event# larlite and larcv don't match" << std::endl;return true;}



        //auto tag_img_thru_v     = (EventImage2D*)mgr.get_data(kProductImage2D,"thrumutags");
        //auto tag_img_stop_v     = (EventImage2D*)mgr.get_data(kProductImage2D,"stopmutags");

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
        /*EventROI* ev_partroi_v = nullptr;
        if (!_true_roi_producer.empty()) ev_partroi_v = (EventROI*) mgr.get_data(kProductROI,_true_roi_producer);
        if (ev_partroi_v) {
            const auto& mc_roi_v = ev_partroi_v->ROIArray();
            FillMC(mc_roi_v);
        }*/


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
            _vtx_id = vertex_index;
            std::cout << "vertex #" << vertex_index << std::endl;
            int treeEntry = -1;
            //treeEntry = SearchMap();
            //if(treeEntry==-1){std::cout << "Not in the list...passing..." << std::endl;continue;}
            //_recoTree->GetEntry(treeEntry);

            bool GoodReco = true;
            int N5cm = 0;
            //std::cout  << "==> "<< _Length_v->size() << ",  " << _track_Goodness_v->size() << std::endl;
            //for(int i = 0;i<_Length_v->size();i++){
            //    if(_Length_v->at(i) > 5 && _track_Goodness_v->at(i) != 1)GoodReco =false;
            //    if(_Length_v->at(i) > 5)N5cm++;
            //}
            //if(N5cm != 2)GoodReco=false;
            //if(_Length_v->size() == 0)GoodReco=false;
            //if(!GoodReco)continue;

            //tracker.FeedVtxGoodness((*_Reco_goodness_v));

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

    void TrackerEventDisplay::finalize(){
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

    void TrackerEventDisplay::MapTree(){
        if(TreeMap.size() == 0)TreeMap.clear();
        for(int i=0;i<_recoTree->GetEntries();i++){
            _recoTree->GetEntry(i);
            std::vector<int> entryInfo(4);
            entryInfo[0] = _run_tree;
            entryInfo[1] = _subrun_tree;
            entryInfo[2] = _event_tree;
            entryInfo[3] = _vtx_id_tree ;
            TreeMap.push_back(entryInfo);
        }
    }

    int  TrackerEventDisplay::SearchMap(){
        int treeEntry=-1;
        for(int i=0;i<TreeMap.size();i++){
            if(_run != TreeMap[i][0])continue;
            if(_subrun != TreeMap[i][1])continue;
            if(_event  != TreeMap[i][2])continue;
            if(_vtx_id != TreeMap[i][3])continue;
            treeEntry=i;
            break;
        }
        if(treeEntry==-1){
            LARCV_CRITICAL() << "ERROR, could not find matching entry in _recoTree" << std::endl;
            throw larbys("die");
        }

        if(SelectedList.size()==0)return treeEntry;
        bool inList = false;
        for(int i=0;i<SelectedList.size();i++){
            if(_run != SelectedList[i][0])continue;
            if(_subrun != SelectedList[i][1])continue;
            if(_event  != SelectedList[i][2])continue;
            if(_vtx_id != SelectedList[i][3])continue;
            inList = true;
            break;
        }
        if(inList) return treeEntry;
        else return -1;
    }

    void TrackerEventDisplay::CreateSelectedList(){
        if(SelectedList.size()!=0)SelectedList.clear();
        if(eventListFile=="")return;
        std::ifstream listFile(eventListFile);
        if (!listFile) {
            LARCV_CRITICAL() << "ERROR, could not open " << eventListFile << " will die now..." << std::endl;
            throw larbys("die");
        }
        bool GoOn = true;
        int thisRun, thisSubRun, thisEvent, thisVertex;
        std::cout << "OK, event list file " << eventListFile << "  found!" << std::endl;
        while(GoOn == true){
            std::vector<int> eventInfo(4);
            listFile >> thisRun >> thisSubRun >> thisEvent >> thisVertex;
            //std::cout << thisRun << " " << thisSubRun << " " << thisEvent << " " << thisVertex;
            eventInfo[0] = thisRun;
            eventInfo[1] = thisSubRun;
            eventInfo[2] = thisEvent;
            eventInfo[3] = thisVertex;
            if(listFile.eof()){GoOn=false;break;}
            SelectedList.push_back(eventInfo);
            //std::cout << SelectedList.back()[0] << "\t" << SelectedList.back()[1] << "\t" << SelectedList.back()[2] << "\t" << SelectedList.back()[3] << std::endl;
        }
        std::cout << "Selected " << SelectedList.size() << " events" << std::endl;
    }
}
#endif
