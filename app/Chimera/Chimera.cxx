#ifndef __CHIMERA_CXX__
#define __CHIMERA_CXX__

#include "Chimera.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/IOManager.h"
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
#include "../Reco3D/AStarUtils.h" // ** remove

//#include "AStar3DAlgo.h"
//#include "AStar3DAlgoProton.h"

//#include "SCE/SpaceChargeMicroBooNE.h"
#include "ChimeraMachinery.h"


//#include "../../core/DataFormat/EventPGraph.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"

#include <cassert>
#include <fstream>
#include <tuple>

namespace larcv {

    static ChimeraProcessFactory __global_ChimeraProcessFactory__;

    Chimera::Chimera(const std::string name)
    : ProcessBase(name){}

    void Chimera::configure(const PSet& cfg){
        _input_pgraph_producer     = cfg.get<std::string>("InputPgraphProducer");
        _img2d_producer            = cfg.get<std::string>("Image2DProducer");
        _chimera_producer            = cfg.get<std::string>("ChimeraProducer");
        _par_pix_producer          = cfg.get<std::string>("ParPixelProducer");
        _true_roi_producer         = cfg.get<std::string>("TrueROIProducer");
        _mask_shower               = cfg.get<bool>("MaskShower",false);

    }

    void Chimera::initialize(){
        LARCV_INFO() << "[Chimera]" << std::endl;
        assert(!_spline_file.empty());
	mach.SetDrawOutputs(false); 
	mach.SetOutputDir(out_dir);
        mach.SetSplineFile(_spline_file);
        mach.initialize();
        mach.SetDrawVertical(true);
        mach.SetDrawBlack(false);
        mach.SetVerbose(0);

        std::string filename;

        std::cout << filename << std::endl;

        if (_finll.empty()) throw larbys("specify larlite file input name");

        _storage.set_io_mode(larlite::storage_manager::kREAD);
        _storage.add_in_filename(_finll);

        if(!_storage.open()) {
	  LARCV_CRITICAL() << "ERROR, larlite input file could not open" << std::endl;
	  throw larbys("die");

	}

	// Create tree that will hold vertex point for each track
	f1 = new TFile("testVertices_july10.root","RECREATE");
	_tree = new TTree("tree","Vertex Tree");
	_tree->Branch("run", &run, "run/I");
	_tree->Branch("subrun", &subrun, "subrun/I");
	_tree->Branch("event", &event, "event/I");
	//	_tree->Branch("vtxPts", &vtxPts);
	_tree->Branch("vtxPt_plane0_x", &vtxPt_plane0_x, "vtxPt_plane0_x/D");
	_tree->Branch("vtxPt_plane0_y", &vtxPt_plane0_y, "vtxPt_plane0_y/D");
	_tree->Branch("vtxPt_plane1_x", &vtxPt_plane1_x, "vtxPt_plane1_x/D");
	_tree->Branch("vtxPt_plane1_y", &vtxPt_plane1_y, "vtxPt_plane1_y/D");
	_tree->Branch("vtxPt_plane2_x", &vtxPt_plane2_x, "vtxPt_plane2_x/D");
	_tree->Branch("vtxPt_plane2_y", &vtxPt_plane2_y, "vtxPt_plane2_y/D");

	//	std::cout << "Initialized, amde tree" << std::endl;
	
	out_iom = new larcv::IOManager( larcv::IOManager::kWRITE );
	out_iom->set_verbosity((larcv::msg::Level_t)0);
	out_iom->set_out_file("test_july10.root");
	out_iom->initialize();	  
	
	  

    }
  
    bool Chimera::process(IOManager& mgr){
        //ClearEvent();
        gStyle->SetOptStat(0);

        TVector3 vertex(-1,-1,-1);

        auto ev_img_v        = (EventImage2D*)mgr.get_data(kProductImage2D,_img2d_producer);
        _run    = (int) ev_img_v->run();
        _subrun = (int) ev_img_v->subrun();
        _event  = (int) ev_img_v->event();
        _entry  = (int) mgr.current_entry();

	/*
	if (_run != myRun && _subrun != mySubrun && _event != myEvent ) {
	  std::cout << "We don't have a match! QUITTING..." << std::endl; 
	  return true;
	}
	*/

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

        if((int)(_storage.run_id())    != _run){std::cout << "run# larlite and larcv don't match" << std::endl;return true;}
        if((int)(_storage.subrun_id()) != _subrun){std::cout << "subrun# larlite and larcv don't match" << std::endl;return true;}
        if((int)(_storage.event_id())  != _event){std::cout << "event# larlite and larcv don't match" << std::endl;return true;}



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

        //double wireRange = 5000;
        double wireRange[3] = {2500,2500,3500};
        double tickRange = 8502;
        //double tickRange = 10000;

        // Create base image2D with the full view, fill it with the input image 2D, we will crop it later
        for(size_t iPlane=0;iPlane<3;iPlane++){
            Full_meta_v[iPlane] = larcv::ImageMeta(wireRange[iPlane],tickRange,(int)(tickRange)/6,(int)(wireRange[iPlane]),0,tickRange);
            Full_image_v[iPlane] = larcv::Image2D(Full_meta_v[iPlane]);
            Tagged_Image[iPlane] = larcv::Image2D(Full_meta_v[iPlane]);
            if(full_adc_img_v->size() == 3)Full_image_v[iPlane].overlay( (*full_adc_img_v)[iPlane] );
            //if(full_tag_img_thru_v->size() == 3)Tagged_Image[iPlane].overlay( (*full_tag_img_thru_v)[iPlane] );
            //if(full_tag_img_stop_v->size() == 3)Tagged_Image[iPlane].overlay( (*full_tag_img_stop_v)[iPlane] );
        }
        mach.SetOriginalImage(Full_image_v); 
        mach.SetTaggedImage(Tagged_Image);
        mach.SetTrackInfo(_run, _subrun, _event, 0);

	std::cout << _run << " " << _subrun << " " << _event <<  std::endl;


        static std::vector<TVector3> vertex_v;
        if(vertex_v.size()!=0)vertex_v.clear();


        larlite::event_track* ev_trk=nullptr;
        auto const& vtx_to_trk = _storage.find_one_ass(ev_vertex->id(), ev_trk, ev_vertex->name());
        if(!ev_trk || ev_trk->size() == 0) throw larlite::DataFormatException("Could not find associated track data product!");

	
	// Going thru the different vertices found
        for(int vertex_index=0;vertex_index<ev_vertex->size();vertex_index++){
            larlite::event_track TracksAtVertex;
            _vtx_id = vertex_index;
	    //	    if (_vtx_id != myVtxid) continue;

	    //            std::cout << "vertex #" << vertex_index << std::endl;
	    std::cout << "vertex #" << vertex_index+1 << " / " << ev_vertex->size() << std::endl;
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

            //mach.FeedVtxGoodness((*_Reco_goodness_v));

            mach.SetSingleVertex(TVector3(ev_vertex->at(vertex_index).X(),ev_vertex->at(vertex_index).Y(),ev_vertex->at(vertex_index).Z()));
            mach.SetVertexID(ev_vertex->at(vertex_index).ID());

            for(auto const& trk_index : vtx_to_trk[vertex_index]) {
                TracksAtVertex.push_back( (*ev_trk)[trk_index]);
                std::cout << "\t => trk#" << trk_index << ", " << TracksAtVertex.back().Length() << " cm" << std::endl;
            }

            mach.FeedLarliteVertexTracks(TracksAtVertex);
            mach.Get3DtracksFromLarlite();
	    
	    std::cout << "Right before DrawVertex" << std::endl;

	    // "crop" around the selected track (in _vertexTracks)
	    // set every pixel outside that box to zero
	    // then loop through each pixel in the cropped region and ask: is it within ~3 pixels of what's in _vertexTracks?
	    // if yes, then skip; if no, then set pixel value to 0
	    //	    mach.GetImageOneTrack(myTrack);
	    //mach.DrawVertex(); 

	    //    std::cout << "Does it even get to this point? Yes." << std::endl;
	    //std::cout << TracksAtVertex.size() << std::endl;
	    
	    
	    //	    mach.DrawVertex3D();

	    // For each track in the vertex:
	    for (int i=0; i < TracksAtVertex.size(); i++) {
	      //	      std::cout << "In the loop!" << std::endl;
	      auto ev_img_out        = (EventImage2D*)out_iom->get_data(kProductImage2D,_chimera_producer);
	      //	      std::cout << "Made our producer first..." << std::endl;

	      std::vector<larcv::Image2D> output;
	      std::vector<std::pair<double, double>> vtxVector;
	      tie(output, vtxVector) = mach.GetImageOneTrack(i); 
	      
	      //	      std::cout << "Cleaned tracks for index " << i << std::endl;

	      //	      std::cout << "Image items follow... " << std::endl;
	      for (auto const& img : output) {
		ev_img_out->Append(img);
	      }

	      run = _run;
	      subrun = _subrun;
	      event = _event*100+vertex_index*10+i;
	      vtxPt_plane0_x = vtxVector[0].first;
	      vtxPt_plane0_y = vtxVector[0].second;
	      vtxPt_plane1_x = vtxVector[1].first;
	      vtxPt_plane1_y = vtxVector[1].second;
	      vtxPt_plane2_x = vtxVector[2].first;
	      vtxPt_plane2_y = vtxVector[2].second;

	      //	      std::cout << "RSE: " << run << " " << subrun << " " << event << std::endl;
	      /*
	      std::cout << "All the vertex variables!!!!" << std::endl;
	      std::cout << "Track U plane x: " << vtxPt_plane0_x << std::endl;
	      std::cout << "Track U plane y: " << vtxPt_plane0_y << std::endl;
	      std::cout << "Track V plane x: " << vtxPt_plane1_x << std::endl;
	      std::cout << "Track V plane y: " << vtxPt_plane1_y << std::endl;
	      std::cout << "Track Y plane x: " << vtxPt_plane2_x << std::endl;
	      std::cout << "Track Y plane y: " << vtxPt_plane2_y << std::endl;
	      */
	      _tree->Fill();

	      //	      std::cout << "Done... " << std::endl;

	      out_iom->set_id( _run, _subrun, _event*100+vertex_index*10+i); // 100+i at the end to identify uniquely based on track
	      out_iom->save_entry();

	      //f1->Write();
	      
	    }

	    //	    mgr.finalize();

        }
	
	
        return true;

    }

    void Chimera::finalize(){
      mach.finalize();
      std::cout << "finalized mach" << std::endl;
      out_iom->finalize();
      f1->cd();
      f1->Write();
      f1->Close();
      _storage.close();
      delete out_iom;

    }
  
  void Chimera::SetSplineLocation(const std::string& fpath) {
    LARCV_INFO() << "setting spline loc @ " << fpath << std::endl;
    mach.SetSplineFile(fpath);
    _spline_file = fpath;
    LARCV_DEBUG() << "end" << std::endl;
  }

}
#endif
