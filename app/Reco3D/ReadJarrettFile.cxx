#ifndef __READJARRETTFILE_CXX__
#define __READJARRETTFILE_CXX__

#include "ReadJarrettFile.h"

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

namespace larcv {

    static ReadJarrettFileProcessFactory __global_ReadJarrettFileProcessFactory__;

    ReadJarrettFile::ReadJarrettFile(const std::string name)
    : ProcessBase(name)
    {}

    void ReadJarrettFile::configure(const PSet& cfg)
    {
      _input_pgraph_producer     = cfg.get<std::string>("InputPgraphProducer");
    }

    void ReadJarrettFile::initialize()
    {
        _isMC = false;
        std::cout << "[ReadJarrettFile]" << std::endl;
        tracker.SetSplineFile("Proton_Muon_Range_dEdx_LAr_TSplines.root");
        tracker.SetOutputDir("png");
        tracker.initialize();
        tracker.SetCompressionFactors(1,6);
        tracker.SetVerbose(0);
        NvertexSubmitted = 0;
        NgoodReco=0;

        std::string filename;
        //if(_isMC)filename="/Volumes/DataStorage/DeepLearningData/VertexedFiles/NuMuSelection_10-5.txt";
        //if(!_isMC)filename = "/Volumes/DataStorage/DeepLearningData/data_5e19/EXTBNB/EXTBNBSelected.txt";

        std::cout << filename << std::endl;
        _filename = filename;
        if(_isMC)ReadVertexFile(filename);// when using runall.sh
        //ReadVertexFile(filename);// when running on extBNB numu


        hEcomp             = new TH2D("hEcomp","hEcomp;E_th;E_reco",100,0,1000,100,0,1000);
        hEcomp_m           = new TH2D("hEcomp_m","hEcomp_m;E_th;E_reco",100,0,1000,100,0,1000);
        hEcomp_p           = new TH2D("hEcomp_p","hEcomp_p;E_th;E_reco",100,0,1000,100,0,1000);
        hEcomp1D           = new TH1D("hEcomp1D","hEcomp1D; (E_{reco}-E_{truth})/E_{truth}",500,-2,8);
        hEcomp1D_m         = new TH1D("hEcomp1D_m","hEcomp1D_m; (E_{reco}-E_{truth})/E_{truth}",1000,-2,2);
        hEcomp1D_p         = new TH1D("hEcomp1D_p","hEcomp1D_p; (E_{reco}-E_{truth})/E_{truth}",1000,-2,2);
        hAverageIonization = new TH1D("hAverageIonization","hAverageIonization",100,0,10000);
        hEnuReco           = new TH1D("hEnuReco","hEnuReco; E_{#nu, reco} (MeV)",200,0,2000);
        hEnuTh             = new TH1D("hEnuTh","hEnuTh;E_{#nu, th} (MeV)",200,0,2000);
        hEcompdQdx         = new TH2D("hEcompdQdx","hEcompdQdx;AverageADCperPix/L;(E_{reco}-E_{truth})/E_{truth}",100,0,1000,100,-2,8);
        hIonvsLength       = new TH2D("hIonvsLength","hIonvsLength;L [cm];AverageADCperPix",100,0,1000,100,0,10000);
        hEnuComp           = new TH2D("hEnuComp","hEnuComp;hEnuTh (MeV);hEnuReco (MeV)",200,0,2000,200,0,2000);
        hEnuComp1D         = new TH1D("hEnuComp1D","hEnuComp1D;(E_{reco}-E_{truth})/E_{truth}",1000,-2,2);
        hEnuvsPM_th        = new TH2D("hEnuvsPM_th","hEnuvsPM_th;E_{#nu};E_{p}+E_{m}",200,0,2000,200,0,2000);
        hPM_th_Reco_1D     = new TH1D("hPM_th_Reco_1D","hPM_th_Reco_1D;(E_{p+m reco}-E_{p+m th})/E_{p+m th}",1000,-2,2);
        hPM_th_Reco        = new TH2D("hPM_th_Reco","hPM_th_Reco;E_{p+m th};E_{p+m reco}",200,0,2000,200,0,2000);

        _recoTree = new TTree("_recoTree","_recoTree");
        _recoTree->Branch("run",&run);
        _recoTree->Branch("subrun",&subrun);
        _recoTree->Branch("event",&event);
        _recoTree->Branch("E_muon_v",&_E_muon_v);
        _recoTree->Branch("E_proton_v",&_E_proton_v);
        _recoTree->Branch("Length_v",&_Length_v);
        _recoTree->Branch("Avg_Ion_v",&_Avg_Ion_v);
        _recoTree->Branch("Angle_v",&_Angle_v);
        _recoTree->Branch("Ep_t", &Ep_t);
        _recoTree->Branch("Em_t", &Em_t);
        _recoTree->Branch("Reco_goodness_v",&_Reco_goodness_v);
        _recoTree->Branch("GoodVertex",&GoodVertex);
        _recoTree->Branch("missingTrack",&_missingTrack);
        _recoTree->Branch("nothingReconstructed",&_nothingReconstructed);
        _recoTree->Branch("tooShortDeadWire",&_tooShortDeadWire);
        _recoTree->Branch("tooShortFaintTrack",&_tooShortFaintTrack);
        _recoTree->Branch("tooManyTracksAtVertex",&_tooManyTracksAtVertex);
        _recoTree->Branch("possibleCosmic",&_possibleCosmic);
        _recoTree->Branch("possiblyCrossing",&_possiblyCrossing);
        _recoTree->Branch("branchingTracks",&_branchingTracks);
        _recoTree->Branch("jumpingTracks",&_jumpingTracks);
        _recoTree->Branch("RecoVertex",&RecoVertex);
        _recoTree->Branch("MCvertex",&MCvertex);
        _recoTree->Branch("vertexPhi",&_vertexPhi);
        _recoTree->Branch("vertexTheta",&_vertexTheta);
        _recoTree->Branch("closestWall",&_closestWall);
        


        _storage.set_io_mode(larlite::storage_manager::kWRITE);
        _storage.set_out_filename("larlite_reco3DTracks.root");
        if(!_storage.open())std::cout << "ERROR, larlite output file could not open" << std::endl;
    }

    bool ReadJarrettFile::process(IOManager& mgr)
    {
        std::cout << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Entry " << mgr.current_entry() << " / " << mgr.get_n_entries() << std::endl;
        std::cout << "============================================" << std::endl;
        gStyle->SetOptStat(0);

        TVector3 vertex(-1,-1,-1);
        TVector3 endPoint[2];
        MCvertex.SetXYZ(-1,-1,-1);


        auto ev_pgraph_v     = (EventPGraph*) mgr.get_data(kProductPGraph,_input_pgraph_producer); // for BNB 5e19, comment when EXTBNB
        run    = ev_pgraph_v->run();
        subrun = ev_pgraph_v->subrun();
        event  = ev_pgraph_v->event();




        larlite::event_track* track_ptr = (larlite::event_track*)_storage.get_data(larlite::data::kTrack,"trackReco");
        larlite::event_vertex* vertex_ptr = (larlite::event_vertex*)_storage.get_data(larlite::data::kVertex,"trackReco");

        if(ev_pgraph_v->PGraphArray().size()==0){_storage.set_id(run,subrun,event);_storage.next_event(true); return true;}
        if(_isMC && !IsGoodEntry(run,subrun,event)){_storage.set_id(run,subrun,event);_storage.next_event(true); return true;}

        auto ev_img_v           = (EventImage2D*)mgr.get_data(kProductImage2D,"wire");
        auto tag_img_thru_v     = (EventImage2D*)mgr.get_data(kProductImage2D,"thrumutags");
        auto tag_img_stop_v     = (EventImage2D*)mgr.get_data(kProductImage2D,"stopmutags");


        //auto ev_pcluster_v = (EventPixel2D*)mgr.get_data(kProductPixel2D,"test_img");
        //auto ev_ctor_v     = (EventPixel2D*)mgr.get_data(kProductPixel2D,"test_ctor");






        // get the event clusters and full images
        //auto const& ctor_m = ev_ctor_v->Pixel2DClusterArray();
        auto full_adc_img_v = &(ev_img_v->Image2DArray());
        auto full_tag_img_thru_v = &(tag_img_thru_v->Image2DArray());
        auto full_tag_img_stop_v = &(tag_img_stop_v->Image2DArray());

        Ep_t = -1;
        Em_t = -1;

        EventROI *ev_partroi_v = 0;
        std::vector<larcv::ROI> mc_roi_v;
        //____________________
        // get MC vertex info
        //--------------------

        if(_isMC){
            ev_partroi_v= (EventROI*)mgr.get_data(kProductROI,"segment");
            mc_roi_v = ev_partroi_v->ROIArray();
            std::vector<TVector3> MuonVertices;
            std::vector<TVector3> ProtonVertices;
            std::vector<TVector3> ElectronVertices;
            std::vector<TVector3> MuonEndPoint;
            std::vector<TVector3> ProtonEndPoint;
            std::vector<TVector3> ElectronEndPoint;
            for(size_t iMC = 0;iMC<mc_roi_v.size();iMC++){
                if(mc_roi_v[iMC].PdgCode() == 14){
                    NeutrinoEnergyTh = mc_roi_v[iMC].EnergyDeposit();
                    hEnuTh->Fill(NeutrinoEnergyTh);
                    std::cout << "Neutrino : " << NeutrinoEnergyTh << " MeV" << std::endl;
                }
                if(mc_roi_v[iMC].PdgCode() == 13){
                    std::cout << "muon.....@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
                    Em_t = mc_roi_v[iMC].EnergyDeposit();
                    MuonVertices.push_back(TVector3(mc_roi_v[iMC].X(),mc_roi_v[iMC].Y(),mc_roi_v[iMC].Z()));
                    MuonEndPoint.push_back(TVector3(mc_roi_v[iMC].EndPosition().X(), mc_roi_v[iMC].EndPosition().Y(), mc_roi_v[iMC].EndPosition().Z()));
                }
                if(mc_roi_v[iMC].PdgCode() == 2212){
                    std::cout << "proton...@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
                    Ep_t = mc_roi_v[iMC].EnergyDeposit();
                    ProtonVertices.push_back(TVector3(mc_roi_v[iMC].X(),mc_roi_v[iMC].Y(),mc_roi_v[iMC].Z()));
                    ProtonEndPoint.push_back(TVector3(mc_roi_v[iMC].EndPosition().X(), mc_roi_v[iMC].EndPosition().Y(), mc_roi_v[iMC].EndPosition().Z()));
                }
                if(mc_roi_v[iMC].PdgCode() == 11){
                    std::cout << "electron.@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
                    ElectronVertices.push_back(TVector3(mc_roi_v[iMC].X(),mc_roi_v[iMC].Y(),mc_roi_v[iMC].Z()));
                    ElectronEndPoint.push_back(TVector3(mc_roi_v[iMC].EndPosition().X(), mc_roi_v[iMC].EndPosition().Y(), mc_roi_v[iMC].EndPosition().Z()));
                }
            }
            std::vector<TVector3> MCVertices;
            std::vector<TVector3> MCEndPoint;
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
                    MCVertices.push_back(ProtonVertices[iProton]);
                    MCEndPoint.push_back(ProtonEndPoint[iProton]);
                    MCvertex = ProtonVertices[iProton];
                    if(isNumu){
                        for(size_t imu = 0;imu<goodMuon.size();imu++){
                            MCEndPoint.push_back(MuonEndPoint[goodMuon[imu]]);
                        }
                    }
                    if(isNue){
                        for(size_t ie = 0;ie<goodElectron.size();ie++){
                            MCEndPoint.push_back(ElectronEndPoint[goodElectron[ie]]);
                        }
                    }
                }
            }
        }
        //______________
        // End MC
        //--------------


        // loop over found vertices
        //auto const& pcluster_m = ev_pcluster_v->Pixel2DClusterArray();
        std::vector<TVector3> vertex_v;
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
        tracker.SetTrackInfo(run, subrun, event, 0);
        std::cout << run << " " << subrun << " " << event <<  std::endl;
        //if(!(subrun == 107 && event == 96242)){_storage.set_id(run,subrun,event);_storage.next_event(true); return true;}

        for(size_t pgraph_id = 0; pgraph_id < ev_pgraph_v->PGraphArray().size(); ++pgraph_id) {// comment when running on EXTBNB

            if(_isMC && !IsGoodVertex(run,subrun,event,pgraph_id)){continue;}
            //if(!IsGoodVertex(run,subrun,event,pgraph_id)){ continue;}

            auto const& pgraph = ev_pgraph_v->PGraphArray().at(pgraph_id);
            TVector3 vertex(pgraph.ParticleArray().front().X(),pgraph.ParticleArray().front().Y(),pgraph.ParticleArray().front().Z());
            vertex_v.push_back(vertex);
            RecoVertex=vertex;

        }

        //vertex_v = GetJarretVertex(run, subrun, event);// for BNBEXT

        NvertexSubmitted+=vertex_v.size();
        if(vertex_v.size()!=0){
            for(size_t ivertex = 0;ivertex<vertex_v.size();ivertex++){
                double xyz[3] = {vertex_v[ivertex].X(),vertex_v[ivertex].Y(),vertex_v[ivertex].Z()};
                vertex_ptr->push_back(larlite::vertex(xyz,ivertex));
                tracker.SetSingleVertex(vertex_v[ivertex]);
                tracker.ReconstructVertex();
                tracker.DrawVertex();
                larlite::event_track recoedVertex = tracker.GetReconstructedVertexTracks();
                *(track_ptr) = recoedVertex;
                if(_isMC)MCevaluation();
                std::vector< std::vector<double> > Energies_v = tracker.GetEnergies();
                _Length_v = tracker.GetVertexLength();
                _vertexPhi =   tracker.GetVertexPhi();
                _vertexTheta = tracker.GetVertexTheta();


                _E_muon_v.resize(Energies_v.size());
                _E_proton_v.resize(Energies_v.size());

                for(size_t trackid=0; trackid<Energies_v.size(); ++trackid) {
                    _E_proton_v[trackid] = Energies_v[trackid].front();
                    _E_muon_v[trackid]   = Energies_v[trackid].back();
                }

                _Length_v  = _Length_v;
                _Avg_Ion_v = tracker.GetAverageIonization();
                _Angle_v   = tracker.GetVertexAngle(15); // average over 15 cm to estimate the angles
                _Reco_goodness_v = tracker.GetRecoGoodness();

                _missingTrack          = _Reco_goodness_v.at(0);
                _nothingReconstructed  = _Reco_goodness_v.at(1);
                _tooShortDeadWire      = _Reco_goodness_v.at(2);
                _tooShortFaintTrack    = _Reco_goodness_v.at(3);
                _tooManyTracksAtVertex = _Reco_goodness_v.at(4);
                _possibleCosmic        = _Reco_goodness_v.at(5);
                _possiblyCrossing      = _Reco_goodness_v.at(6);
                _branchingTracks       = _Reco_goodness_v.at(7);
                _jumpingTracks         = _Reco_goodness_v.at(8);
                
                GoodVertex = false;
                GoodVertex = tracker.IsGoodVertex();
                if(GoodVertex)NgoodReco++;
                
                _recoTree->Fill();
                
            }
        }
        _storage.set_id(run,subrun,event);
        _storage.next_event(true);
        std::cout << "...Reconstruted..." << std::endl;

        return true;
    }

    bool ReadJarrettFile::IsGoodVertex(int run, int subrun, int event/*, int ROIid*/, int vtxID)
    {
        bool okVertex = false;
        for(size_t ivertex = 0;ivertex<_vertexInfo.size();ivertex++){
            if(   run    == _vertexInfo[ivertex][0]
               && subrun == _vertexInfo[ivertex][1]
               && event  == _vertexInfo[ivertex][2]
               //&& ROIid  == _vertexInfo[ivertex][4]
               && vtxID  == _vertexInfo[ivertex][5]
               )okVertex = true;
        }
        return okVertex;
    }

    bool ReadJarrettFile::IsGoodEntry(int run, int subrun, int event){
        bool okVertex = false;
        for(size_t ivertex = 0;ivertex<_vertexInfo.size();ivertex++){
            if(   run    == _vertexInfo[ivertex][0]
               && subrun == _vertexInfo[ivertex][1]
               && event  == _vertexInfo[ivertex][2]
               )okVertex = true;
        }
        return okVertex;
    }

    void ReadJarrettFile::ReadVertexFile(std::string filename)
    {
        std::cout << "ReadVertexFile" << std::endl;
        if(_vertexInfo.size()!=0)_vertexInfo.clear();
        std::vector<int> thisVertexInfo;
        std::ifstream file(Form("%s",filename.c_str()));
        if(!file){std::cout << "ERROR, could not open file of selected vertices to sort through" << std::endl;return;}
        std::string firstline;
        getline(file, firstline);
        bool goOn = true;
        int Run,SubRun,Event,Entry,ROI_ID,vtxid,rescale_vtxid;
        double x,y,z;
        char coma;
        while(goOn){
            //file >> Run >> coma >> SubRun >> coma >> Event >> coma >> Entry >> coma >> ROI_ID >> coma >> vtxid >> coma >> x >> coma >> y >> coma >> z >> coma >> rescale_vtxid;
            file >> Run >> SubRun >> Event >> Entry >> ROI_ID >> vtxid >> rescale_vtxid >> x >> y >> z ;
            //file >> Run >> coma >> SubRun >> coma >> Event >> coma >> Entry >> coma >> ROI_ID >> coma >> vtxid >> coma >> x >> coma >> y >> coma >> z ;
            if(thisVertexInfo.size()!=0)thisVertexInfo.clear();
            thisVertexInfo.push_back(Run);      //0
            thisVertexInfo.push_back(SubRun);   //1
            thisVertexInfo.push_back(Event);    //2
            thisVertexInfo.push_back(Entry);    //3
            thisVertexInfo.push_back(ROI_ID);   //4
            //thisVertexInfo.push_back(vtxid);    //5
            thisVertexInfo.push_back(rescale_vtxid);//5
            _vertexInfo.push_back(thisVertexInfo);
            if(file.eof()){goOn=false;break;}
        }
        std::cout << _vertexInfo.size() << " vertices to loop through" << std::endl;

    }

    std::vector<TVector3> ReadJarrettFile::GetJarretVertex(int run, int subrun, int event){
        std::vector<TVector3> vertex_vector;
        std::ifstream file(Form("%s",_filename.c_str()));
        if(!file){std::cout << "ERROR, could not open file of selected vertices to sort through" << std::endl;return vertex_vector;}
        std::string firstline;
        getline(file, firstline);
        bool goOn = true;
        int Run,SubRun,Event,Entry,ROI_ID,vtxid,rescale_vtxid;
        double x,y,z;
        char coma;
        while(goOn){
            file >> Run >> SubRun >> Event >> Entry >> ROI_ID >> vtxid >> rescale_vtxid >> x >> y >> z ;
            if(run == Run && SubRun == subrun && event == Event){
                vertex_vector.push_back(TVector3(x,y,z));
            }
            if(file.eof()){goOn=false;break;}
        }
        return vertex_vector;
    }

    void ReadJarrettFile::MCevaluation(){
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

        hEcomp->Fill(Ep_t,Epreco);
        hEcomp->Fill(Em_t,Emreco);
        hEcomp_p->Fill(Ep_t,Epreco);
        hEcomp_m->Fill(Em_t,Emreco);
        hEcomp1D->Fill((Epreco-Ep_t)/Ep_t);
        hEcomp1D->Fill((Emreco-Em_t)/Em_t);
        hEcomp1D_p->Fill((Epreco-Ep_t)/Ep_t);
        hEcomp1D_m->Fill((Emreco-Em_t)/Em_t);
        hEnuvsPM_th->Fill(NeutrinoEnergyTh,Ep_t+Em_t);
        hPM_th_Reco_1D->Fill(((Epreco+Emreco)-(Ep_t+Em_t))/(Ep_t+Em_t));
        hPM_th_Reco->Fill(Ep_t+Em_t,Epreco+Emreco);
        hEcompdQdx->Fill(_Avg_Ion_v[1]/_Length_v[1],(Epreco-Ep_t)/Ep_t);
        hEcompdQdx->Fill(_Avg_Ion_v[0]/_Length_v[0],(Emreco-Em_t)/Em_t);
        if( std::abs((Epreco-Ep_t)/Ep_t) > 0.1 || std::abs((Emreco-Em_t)/Em_t) > 0.1 ){
            checkEvents.push_back(Form("%d_%d_%d",run,subrun,event));
        }

        std::cout << "proton : Epreco = " << Epreco << " MeV, Epth : " << Ep_t << " MeV..." << 100*(Epreco-Ep_t)/Ep_t << " %" << std::endl;
        std::cout << "muon   : Emreco = " << Emreco << " MeV, Emth : " << Em_t << " MeV..." << 100*(Emreco-Em_t)/Em_t << " %" << std::endl;

        std::cout << "E nu th   : " << NeutrinoEnergyTh << " MeV" << std::endl;
        std::cout << "E nu Reco : " << Epreco+Emreco << " MeV" << std::endl;
        std::cout << "relative Enu diff: " << 100*((Epreco+Emreco)-NeutrinoEnergyTh)/NeutrinoEnergyTh << " %" << std::endl;
        std::cout << "relative Enu_p+m diff: " << 100*((Epreco+Emreco)-(Ep_t+Em_t))/(Ep_t+Em_t) << " %" << std::endl;
        hEnuReco->Fill(Epreco+Emreco);
        hEnuComp->Fill(NeutrinoEnergyTh,Epreco+Emreco);
        hEnuComp1D->Fill(((Epreco+Emreco)-NeutrinoEnergyTh)/NeutrinoEnergyTh);


        for(size_t itrack = 0;itrack<_Avg_Ion_v.size();itrack++){
            hAverageIonization->Fill(_Avg_Ion_v[itrack]);
            hIonvsLength->Fill(_Length_v[itrack],_Avg_Ion_v[itrack]);
            
        }
    }
    
    void ReadJarrettFile::finalize()
    {
        std::cout << NvertexSubmitted << " vertex submitted" << std::endl;
        std::cout << NgoodReco << " well reconstructed vertices" << std::endl;
        std::cout << NgoodReco*100./NvertexSubmitted << " % efficiency" << std::endl;
        std::cout << "saving root files?...";
        if(hEcomp->GetEntries() > 1){
            std::cout << "... yes" << std::endl;
            hEcomp->SaveAs(Form("hEcomp_%d_%d_%d.root",run,subrun,event));
            hEcomp_p->SaveAs(Form("hEcomp_p_%d_%d_%d.root",run,subrun,event));
            hEcomp_m->SaveAs(Form("hEcomp_m_%d_%d_%d.root",run,subrun,event));
            hEcompdQdx->SaveAs(Form("hEcompdQdx_%d_%d_%d.root",run,subrun,event));
            hEcomp1D->SaveAs(Form("hEcomp1D_%d_%d_%d.root",run,subrun,event));
            hEcomp1D_m->SaveAs(Form("hEcomp1D_m_%d_%d_%d.root",run,subrun,event));
            hEcomp1D_p->SaveAs(Form("hEcomp1D_p_%d_%d_%d.root",run,subrun,event));
            hIonvsLength->SaveAs(Form("hIonvsLength_%d_%d_%d.root",run,subrun,event));
            hAverageIonization->SaveAs(Form("hAverageIonization_%d_%d_%d.root",run,subrun,event));
            hEnuReco->SaveAs(Form("hEnuReco_%d_%d_%d.root",run,subrun,event));
            hEnuTh->SaveAs(Form("hEnuTh_%d_%d_%d.root",run,subrun,event));
            hEnuComp->SaveAs(Form("hEnuComp_%d_%d_%d.root",run,subrun,event));
            hEnuComp1D->SaveAs(Form("hEnuComp1D_%d_%d_%d.root",run,subrun,event));
            hEnuvsPM_th->SaveAs(Form("hEnuvsPM_th_%d_%d_%d.root",run,subrun,event));
            hPM_th_Reco_1D->SaveAs(Form("hPM_th_Reco_1D_%d_%d_%d.root",run,subrun,event));
            hPM_th_Reco->SaveAs(Form("hPM_th_Reco_%d_%d_%d.root",run,subrun,event));
        }
        else{std::cout << "... no" << std::endl;}
        tracker.finalize();
        std::cout << "finalized tracker" << std::endl;

        if(has_ana_file()) {
            ana_file().cd();
            _recoTree->Write();
        }
        std::cout << "wrote _recoTree" << std::endl;
        _storage.close();
        std::cout << "finalized storage" << std::endl;
    }
    
}
#endif
