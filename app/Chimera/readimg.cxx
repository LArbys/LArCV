#include <iostream>

// ROOT
#include "TFile.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "TStyle.h"

// larlite
#include "DataFormat/storage_manager.h"
#include "LArUtil/LArProperties.h"

// larcv
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventChStatus.h"

/**
 * simple macro to read an Image2D and plot it in TH2D
 *
 */
int main( int nargs, char** argv ) {

  gStyle->SetOptStat(0);

  std::string input_larcv  = argv[1]; // full file path of input larcv file 
  
  int run, subrun, event;  
  
  // input larcv
  larcv::IOManager iolarcv( larcv::IOManager::kREAD );
  iolarcv.add_in_file( input_larcv );
  iolarcv.initialize();

  /* code below shows how I would save the output
  // larcv:
  larcv::IOManager outlarcv( larcv::IOManager::kWRITE, "", larcv::IOManager::kTickBackward );
  outlarcv.set_out_file( output_larcv );
  outlarcv.initialize();
  //larlite:
  larlite::storage_manager outlarlite( larlite::storage_manager::kWRITE );
  outlarlite.set_out_filename( output_larlite );
  outlarlite.open();
  */
  std::string output_hist = "originalVertex.root";
  TFile fout( output_hist.c_str(), "new" ); // do not rewrite
  TH2D* hadc[3];

  int nentries = iolarcv.get_n_entries();
  for (int i=0; i<nentries; i++) {
    iolarcv.read_entry(i);

    std::cout << "entry " << i << std::endl;

    // in
    auto ev_img = (larcv::EventImage2D*)iolarcv.get_data( larcv::kProductImage2D, "wire" );

    // out: if I am saving Image2Ds
    //auto evout_wire            = (larcv::EventImage2D*)outlarcv.get_data( larcv::kProductImage2D, "testout");
    
    run    = iolarcv.event_id().run();
    subrun = iolarcv.event_id().subrun();
    event  = iolarcv.event_id().event();
    //std::cout <<"run "<< run <<" subrun " << subrun <<" event " << event << std::endl;

    for ( int planeid=0; planeid<3; planeid++ ) {
      auto const& img = ev_img->Image2DArray().at( planeid );
      auto const& meta = img.meta();
      char histname_event[100];
      sprintf(histname_event,"hadc_run%d_subrun%d_event%d_plane%d",run,subrun,event,planeid);
      hadc[planeid] = new TH2D(histname_event,"",meta.cols(), 0, meta.cols(), meta.rows(),0,meta.rows());

      for(int row = 0; row<meta.rows(); row++){
        for(int col=0; col<meta.cols(); col++){
          float pixamp = img.pixel( row, col );
          hadc[planeid]->SetBinContent(col+1, row+1, pixamp);
        }
      }
    }
    // if I am savinf Image2D output: transfer image and chstatus data to output file
    //for ( auto const& img : ev_img->Image2DArray() ) {
    //  evout_wire->Append( img );
    //}

    //for larlite output
    //outlarlite.set_id( run, subrun, event );
    //outlarlite.next_event();

    // for larcv output
    //outlarcv.set_id( run, subrun, event );
    //outlarcv.save_entry();

    fout.cd();
    for ( int p=0; p<3; p++ ) {    
      hadc[p]->Write();
    }

  }//end of entry loop
  
  iolarcv.finalize();
  //outlarcv.finalize();
  //outlarlite.close();
  fout.Close();

  
  return 0;
}
