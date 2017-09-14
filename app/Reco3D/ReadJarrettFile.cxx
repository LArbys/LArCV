#ifndef __READJARRETTFILE_CXX__
#define __READJARRETTFILE_CXX__

#include "ReadJarrettFile.h"

#include "DataFormat/track.h"
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


#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"

namespace larcv {

    static ReadJarrettFileProcessFactory __global_ReadJarrettFileProcessFactory__;

    ReadJarrettFile::ReadJarrettFile(const std::string name)
    : ProcessBase(name)
    {}

    void ReadJarrettFile::configure(const PSet& cfg)
    {}

    void ReadJarrettFile::initialize()
    {
        std::cout << "[ReadJarrettFile]" << std::endl;
        tracker.initialize();
        tracker.SetCompressionFactors(1,6);
        tracker.SetVerbose(0);
        iTrack = 0;

        ReadVertexFile("data/numuSelected.txt");
    }

    bool ReadJarrettFile::process(IOManager& mgr)
    {
        std::cout << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Entry " << mgr.current_entry() << " / " << mgr.get_n_entries() << std::endl;
        std::cout << "============================================" << std::endl;
        gStyle->SetOptStat(0);

        TVector3 vertex, endPoint[2];

        //
        // Loop per vertex (larcv type is PGraph "Particle Graph")
        //

        auto ev_img_v      = (EventImage2D*)mgr.get_data(kProductImage2D,"wire");
        //auto tag_img_v     = (EventImage2D*)mgr.get_data(kProductImage2D,"combinedtags");
        auto tag_img_thru_v     = (EventImage2D*)mgr.get_data(kProductImage2D,"thrumutags");
        auto tag_img_stop_v     = (EventImage2D*)mgr.get_data(kProductImage2D,"stopmutags");
        //auto tag_img_v     = (EventImage2D*)mgr.get_data(kProductImage2D,"containedtags");
        auto ev_pgraph_v   = (EventPGraph*) mgr.get_data(kProductPGraph,"test");
        //auto ev_pcluster_v = (EventPixel2D*)mgr.get_data(kProductPixel2D,"test_img");
        //auto ev_ctor_v     = (EventPixel2D*)mgr.get_data(kProductPixel2D,"test_ctor");
        auto ev_partroi_v  = (EventROI*)mgr.get_data(kProductROI,"segment");

        int run    = ev_pgraph_v->run();
        int subrun = ev_pgraph_v->subrun();
        int event  = ev_pgraph_v->event();

        // get the event clusters and full images
        //auto const& ctor_m = ev_ctor_v->Pixel2DClusterArray();
        auto full_adc_img_v = &(ev_img_v->Image2DArray());
        auto full_tag_img_thru_v = &(tag_img_thru_v->Image2DArray());
        auto full_tag_img_stop_v = &(tag_img_stop_v->Image2DArray());
        auto mc_roi_v = ev_partroi_v->ROIArray();

        // get MC vertex
        std::vector<TVector3> MuonVertices;
        std::vector<TVector3> ProtonVertices;
        std::vector<TVector3> ElectronVertices;
        std::vector<TVector3> MuonEndPoint;
        std::vector<TVector3> ProtonEndPoint;
        std::vector<TVector3> ElectronEndPoint;
        for(int iMC = 0;iMC<mc_roi_v.size();iMC++){
            if(mc_roi_v[iMC].PdgCode() == 13){
                std::cout << "muon.....@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
                MuonVertices.push_back(TVector3(mc_roi_v[iMC].X(),mc_roi_v[iMC].Y(),mc_roi_v[iMC].Z()));
                MuonEndPoint.push_back(TVector3(mc_roi_v[iMC].EndPosition().X(), mc_roi_v[iMC].EndPosition().Y(), mc_roi_v[iMC].EndPosition().Z()));
            }
            if(mc_roi_v[iMC].PdgCode() == 2212){
                std::cout << "proton...@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
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
        for(int iProton = 0;iProton<ProtonVertices.size();iProton++){
            isVertex = false;
            isNumu = false;
            isNue = false;
            for(int iMuon = 0;iMuon<MuonVertices.size();iMuon++){
                if(MuonVertices[iMuon] == ProtonVertices[iProton]){isVertex = true;isNumu = true;goodMuon.push_back(iMuon);}
            }
            for(int iElectron = 0;iElectron<ElectronVertices.size();iElectron++){
                if(ProtonVertices[iProton] == ElectronVertices[iProton]){isVertex = true;isNue = true;goodElectron.push_back(iElectron);}
            }
            if(isVertex && MCVertices.size()!=0 && ProtonVertices[iProton] == MCVertices[MCVertices.size()-1])continue;
            if(isVertex){
                MCVertices.push_back(ProtonVertices[iProton]);
                MCEndPoint.push_back(ProtonEndPoint[iProton]);
                if(isNumu){
                    for(int imu = 0;imu<goodMuon.size();imu++){
                        MCEndPoint.push_back(MuonEndPoint[goodMuon[imu]]);
                    }
                }
                if(isNue){
                    for(int ie = 0;ie<goodElectron.size();ie++){
                        MCEndPoint.push_back(ElectronEndPoint[goodElectron[ie]]);
                    }
                }
            }
        }


        // loop over found vertices
        //auto const& pcluster_m = ev_pcluster_v->Pixel2DClusterArray();
        std::vector<TVector3> vertex_v;
        std::vector<larcv::ImageMeta> Full_meta_v(3);
        std::vector<larcv::Image2D> Tagged_Image(3);
        std::vector<larcv::Image2D> Full_image_v(3);
        double wireRange = 5000;
        double tickRange = 8502;

        // Create base image2D with the full view, fill it with the input image 2D, we will crop it later
        for(int iPlane=0;iPlane<3;iPlane++){
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
        for(size_t pgraph_id = 0; pgraph_id < ev_pgraph_v->PGraphArray().size(); ++pgraph_id) {

            iTrack++;
            if(!IsGoodVertex(run,subrun,event,pgraph_id))continue;

            auto const& pgraph        = ev_pgraph_v->PGraphArray().at(pgraph_id);

            //
            // Get Estimated 3D Start and End Points
            std::vector<TVector3> EndPoints;
            TVector3 vertex(pgraph.ParticleArray().front().X(),pgraph.ParticleArray().front().Y(),pgraph.ParticleArray().front().Z());
            EndPoints.push_back(vertex);

            bool WrongEndPoint = false;
            for(int iPoint = 0;iPoint<EndPoints.size();iPoint++){
                if(!tracker.CheckEndPointsInVolume(EndPoints[iPoint]) ){std::cout << "=============> ERROR! End point " << iPoint << " outside of volume" << std::endl; WrongEndPoint = false;}
            }
            if(WrongEndPoint)continue;
            vertex_v.push_back(vertex);

        }
        //tracker.SetEventVertices(MCVertices);
        tracker.SetEventVertices(vertex_v);
        tracker.ReconstructEvent();
        std::cout << std::endl << std::endl;
        
        return true;
    }

    bool ReadJarrettFile::IsGoodVertex(int run, int subrun, int event/*, int ROIid*/, int vtxID)
    {
        bool okVertex = false;
        for(int ivertex = 0;ivertex<_vertexInfo.size();ivertex++){
            if(   run    == _vertexInfo[ivertex][0]
               && subrun == _vertexInfo[ivertex][1]
               && event  == _vertexInfo[ivertex][2]
               //&& ROIid  == _vertexInfo[ivertex][4]
               && vtxID  == _vertexInfo[ivertex][5])okVertex = true;
        }
        return okVertex;
    }

    void ReadJarrettFile::ReadVertexFile(std::string filename)
    {
        if(_vertexInfo.size()!=0)_vertexInfo.clear();
        std::vector<int> thisVertexInfo;
        std::ifstream file(Form("%s",filename.c_str()));
        if(!file){std::cout << "ERROR, could not open file of selected vertices to sort through" << std::endl;return;}
        std::string firstline;
        getline(file, firstline);
        bool goOn = true;
        int Run,SubRun,Event,Entry,ROI_ID,vtxid;
        double x,y,z;
        char coma;
        while(goOn){
            file >> Run >> coma >> SubRun >> coma >> Event >> coma >> Entry >> coma >> ROI_ID >> coma >> vtxid >> coma >> x >> coma >> y >> coma >> z;
            if(thisVertexInfo.size()!=0)thisVertexInfo.clear();
            thisVertexInfo.push_back(Run);      //0
            thisVertexInfo.push_back(SubRun);   //1
            thisVertexInfo.push_back(Event);    //2
            thisVertexInfo.push_back(Entry);    //3
            thisVertexInfo.push_back(ROI_ID);   //4
            thisVertexInfo.push_back(vtxid);    //5
            _vertexInfo.push_back(thisVertexInfo);
            if(file.eof()){goOn=false;break;}
        }
        std::cout << _vertexInfo.size() << " vertices to loop through" << std::endl;

    }
    
    void ReadJarrettFile::finalize()
    {
        tracker.finalize();
    }
    
}
#endif
