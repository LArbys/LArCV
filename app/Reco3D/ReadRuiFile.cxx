#ifndef __READRUIFILE_CXX__
#define __READRUIFILE_CXX__

#include "ReadRuiFile.h"

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

    static ReadRuiFileProcessFactory __global_ReadRuiFileProcessFactory__;

    ReadRuiFile::ReadRuiFile(const std::string name)
    : ProcessBase(name)
    {}

    void ReadRuiFile::configure(const PSet& cfg)
    {}

    void ReadRuiFile::initialize()
    {
        std::cout << "[ReadRuiFile]" << std::endl;
        tracker.initialize();
        tracker.SetCompressionFactors(1,6);
        tracker.SetVerbose(0);
        iTrack = 0;
    }

    bool ReadRuiFile::process(IOManager& mgr)
    {
        std::cout << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "Entry " << mgr.current_entry() << " / " << mgr.get_n_entries() << std::endl;
        std::cout << "============================================" << std::endl;
        gStyle->SetOptStat(0);
        //TCanvas *cImage = new TCanvas("cImage","cImage",900,300);
        //cImage->Divide(3,1);
        //TH2D *hImage[3];
        //TGraph *gEndPoint[3];
        //TGraph *gROI[3];

        TVector3 vertex, endPoint[2];

        //
        // Loop per vertex (larcv type is PGraph "Particle Graph")
        //

        auto ev_img_v      = (EventImage2D*)mgr.get_data(kProductImage2D,"wire");
        auto ev_pgraph_v   = (EventPGraph*) mgr.get_data(kProductPGraph,"test");
        auto ev_pcluster_v = (EventPixel2D*)mgr.get_data(kProductPixel2D,"test_img");
        auto ev_ctor_v     = (EventPixel2D*)mgr.get_data(kProductPixel2D,"test_ctor");
        auto ev_partroi_v  = (EventROI*)mgr.get_data(kProductROI,"segment");

        int run    = ev_pgraph_v->run();
        int subrun = ev_pgraph_v->subrun();
        int event  = ev_pgraph_v->event();

        // get the event clusters and full images
        auto const& ctor_m = ev_ctor_v->Pixel2DClusterArray();
        auto full_adc_img_v = &(ev_img_v->Image2DArray());
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
        //if(MCVertices.size() > 1)std::cout << "found " << MCVertices.size() << " MC vertices" << std::endl;
        //else{std::cout << "found " << MCVertices.size() << " MC vertex" << std::endl;}


        // loop over found vertices
        auto const& pcluster_m = ev_pcluster_v->Pixel2DClusterArray();
        //std::cout  << ev_pgraph_v->PGraphArray().size() << " vertices in entry" << std::endl;
        for(size_t pgraph_id = 0; pgraph_id < ev_pgraph_v->PGraphArray().size(); ++pgraph_id) {
            //std::cout << "vertex " << pgraph_id << " / " << ev_pgraph_v->PGraphArray().size() << std::endl;

            iTrack++;
            int i = iTrack;

            auto const& pgraph        = ev_pgraph_v->PGraphArray().at(pgraph_id);
            auto const& roi_v         = pgraph.ParticleArray();
            auto const& cluster_idx_v = pgraph.ClusterIndexArray();

            size_t nparticles = cluster_idx_v.size();
            //std::cout << nparticles << " particles in vertex" << std::endl;

            //
            // Get Estimated 3D Start and End Points
            std::vector<TVector3> EndPoints;
            TVector3 vertex(pgraph.ParticleArray().front().X(),pgraph.ParticleArray().front().Y(),pgraph.ParticleArray().front().Z());
            EndPoints.push_back(vertex);

            /*TVector3 newPoint;
            for(int ipart = 0;ipart<pgraph.ParticleArray().size();ipart++){
                newPoint.SetXYZ(pgraph.ParticleArray()[ipart].EndPosition().X(),pgraph.ParticleArray()[ipart].EndPosition().Y(),pgraph.ParticleArray()[ipart].EndPosition().Z());
                EndPoints.push_back(newPoint);
            }
            if(!(EndPoints.size() == nparticles+1)){std::cout << "ERROR : not the right number of end points : shoudl be Nparticles + 1 for the vertex" << std::endl; std::cin.get();}*/

            bool WrongEndPoint = false;
            for(int iPoint = 0;iPoint<EndPoints.size();iPoint++){
                if(!tracker.CheckEndPointsInVolume(EndPoints[iPoint]) ){std::cout << "=============> ERROR! End point " << iPoint << " outside of volume" << std::endl; WrongEndPoint = false;}
            }
            if(WrongEndPoint)continue;

            // Check Vertex is close to a MC one
            //bool vertexOK = false;
            //for(int iMC = 0;iMC<MCVertices.size();iMC++){
            //    if(( EndPoints.front() - MCVertices[iMC] ).Mag() < 5)vertexOK = true;
            //}
            //if(!vertexOK) continue;// try another vertex, this one is too far away


            std::vector<larcv::ImageMeta> Full_meta_v(3);
            std::vector<larcv::Image2D> Full_image_v(3);
            double wireRange = 5000;
            double tickRange = 8502;

            // Create base image2D with the full view, fill it with the input image 2D, we will crop it later
            for(int iPlane=0;iPlane<3;iPlane++){
                Full_meta_v[iPlane] = larcv::ImageMeta(wireRange,tickRange,(int)(tickRange)/6,(int)(wireRange),0,tickRange);
                Full_image_v[iPlane] = larcv::Image2D(Full_meta_v[iPlane]);
                if(full_adc_img_v->size() == 3)Full_image_v[iPlane].overlay( (*full_adc_img_v)[iPlane] );
            }

            std::vector<larcv::Image2D> data_images = tracker.CropFullImage2bounds(EndPoints,Full_image_v);
            tracker.SetOriginalImage(Full_image_v);
            tracker.SetTrackInfo(run, subrun, event, 20*i);
            tracker.SetImages(data_images);
            tracker.SetVertexEndPoints(EndPoints);
            tracker.SetEndPoints(EndPoints[0],EndPoints[1]);
            std::cout << "Reconstruct Vertex" << std::endl;
            tracker.ReconstructVertex();

            /*for(int iPart = 0;iPart < nparticles; iPart++){// start loop over particles in vertex

                // make smaller image by inverting and cropping the full one
                std::vector<larcv::Image2D> data_images = tracker.CropFullImage2bounds(EndPoints,Full_image_v);
                tracker.SetOriginalImage(Full_image_v);
                // configure and run tracker
                tracker.SetTrackInfo(run, subrun, event, 2*i+iPart);
                tracker.SetImages(data_images);

                std::vector<TVector3> points;
                points.push_back(EndPoints[0]);
                points.push_back(EndPoints[iPart+1]);
                tracker.SetVertexEndPoints(points);

                tracker.SetEndPoints(EndPoints[0],EndPoints[iPart+1]);
                tracker.ImprovedCluster();
                std::cout << "ImproveCluster done" << std::endl;
                std::cout << "track size = " << tracker.GetTrack().size() << std::endl;
                tracker.DrawTrack();

            }*/// end loop over particles in vertex

            std::cout << std::endl << std::endl;
        }
        
        return true;
    }
    
    void ReadRuiFile::finalize()
    {
        tracker.finalize();
    }
    
}
#endif
