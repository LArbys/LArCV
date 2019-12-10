#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <math.h>
// ROOT
#include "THStack.h"
#include "TEfficiency.h"
#include "TFile.h"
#include "TLine.h"
#include "TTree.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TH3F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TAttMarker.h"
#include "TVector3.h"
#include "TError.h"

// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/hit.h"

// larcv
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"


void remove_from_hit(larcv::Image2D& removal_img, const larlite::hit& hit, int padding=2);

int main(int nargs, char** argv){
  //to run:
  /*
  Run through all files in list: (Will break if opening too many files due to
  ROOT Problems)
  ./thrumu_maker filelist.txt
  Run through specific subset of files (inclusive):
  ./thrumu_maker filelist.txt startfile_num endfile_num
  */

  int start_file_num;
  int end_file_num;
  std::string filelist = argv[1];
  std::cout << filelist << "\n";
  if (nargs < 3) {
    start_file_num = 0;
    end_file_num = 0;
  }
  else{
    start_file_num = std::atoi(argv[2]);
    end_file_num = std::atoi(argv[3]);
  }
  int total_events_run_over =0;
	int total_files_run_over = 0;

	// TH2D removed_h =TH2D("removed_h","removed_h ",3456,0.,3456,1008,0.,1008.);

  std::string STRING;
  std::ifstream infile;
  infile.open(filelist);
  int file_idx = -1;
  gStyle->SetOptStat(0);

  while(std::getline(infile,STRING)){ // To get you all the lines.
    file_idx++;
    if (file_idx > end_file_num){
      continue;
    }
    else if ( file_idx < start_file_num){
      continue;
    }
		std::cout << "\n";
    std::cout << "Doing file number:" << file_idx << "\n";
		std::cout << "/////////////////////////////////////////////////////////////\n";

		total_files_run_over++;
    std::cout<<STRING << "\n"; // Prints our STRING.
    std::string supera_file       = STRING;
    // Get Supera File as cstr if you want to make a normal TFile from it
		// char supera_file_cstr[supera_file.size() + 1];
		// supera_file.copy(supera_file_cstr,supera_file.size()+1);
		// supera_file_cstr[supera_file.size()] = '\0';
		// larlite instantiation
    larlite::storage_manager* io_ll  = new larlite::storage_manager(larlite::storage_manager::kREAD);
    io_ll->add_in_filename(supera_file);
    io_ll->open();
		// larcv instantiation
    larcv::IOManager* io_cv  = new larcv::IOManager(larcv::IOManager::kREAD,"IOManager");
    io_cv->add_in_file(supera_file);
    io_cv->initialize();

    // larcv instantiation
    std::string outfile_str = "thrumu_outfile.root";
    larcv::IOManager* io_out_cv  = new larcv::IOManager(larcv::IOManager::kWRITE,"IOManager2");
    io_out_cv->set_out_file(outfile_str);
    io_out_cv->initialize();

		int nentries_mc_larlite = io_ll->get_entries();
    total_events_run_over += nentries_mc_larlite;
    for (int entry=0; entry < nentries_mc_larlite; entry++){
      std::cout << "Entry : " << entry << "\n";
      io_ll->go_to(entry);
			io_cv->read_entry(entry);

			// LArLite Imports
      const larlite::event_hit& ev_hit_wc           = *((larlite::event_hit*)io_ll->get_data(larlite::data::kHit,  "portedThresholdhit" ));
			// LArCV Imports
      larcv::EventImage2D* ev_in_adc  = (larcv::EventImage2D*)(io_cv->get_data(larcv::kProductImage2D, "wire"));
      std::vector< larcv::Image2D > img_adc_v = ev_in_adc->Image2DArray();
			std::cout << "\n";
      io_out_cv->set_id(ev_in_adc->run(),ev_in_adc->subrun(),ev_in_adc->event());
      larcv::EventImage2D* const ev_thrumu_out = (larcv::EventImage2D * )(io_out_cv->get_data(larcv::kProductImage2D, "thrumu"));

      // Copy Images
      larcv::Image2D thrumu_u(img_adc_v[0]);
      larcv::Image2D thrumu_v(img_adc_v[1]);
      larcv::Image2D thrumu_y(img_adc_v[2]);

      for (int idx=0;idx < ev_hit_wc.size();idx++){
        larlite::hit hit = ev_hit_wc[idx];
        int plane = hit.View();
        if (plane == 0){
          //These functions have a 3rd arg that does padding, default is 2
          //around the hit's col and row vals.
          remove_from_hit(thrumu_u, hit);
        }
        else if (plane == 1){
          remove_from_hit(thrumu_v, hit);
        }
        else {
          remove_from_hit(thrumu_y, hit);
        }
      }
      ev_thrumu_out->Emplace(std::move(thrumu_u));
      ev_thrumu_out->Emplace(std::move(thrumu_v));
      ev_thrumu_out->Emplace(std::move(thrumu_y));

      io_out_cv->save_entry();
    } //End of entry loop
    io_ll->close();
    io_cv->finalize();
    io_out_cv->finalize();

    delete io_ll;
    delete io_cv;
    delete io_out_cv;
    // To Test Output:
    larcv::IOManager io_test( larcv::IOManager::kREAD, "");
    io_test.add_in_file( outfile_str );
    io_test.initialize();
		io_test.read_entry(0);
		const auto ev_wire_test           = (larcv::EventImage2D*)io_test.get_data( larcv::kProductImage2D, "thrumu");
		std::vector<larcv::Image2D> const& wire_v_test = ev_wire_test->Image2DArray();
    std::cout << wire_v_test.size() << " Output wire v test size \n";

    // // To Draw Stuff:
    // for (int i = 0;i<3; i++){
    //   for (int row = 0;row<1008;row++){
    //     for (int col = 0;col<3456; col++){
    //       // orig_h.SetBinContent(col,row,img_adc_v[i].pixel(row,col));
    //       removed_h.SetBinContent(col,row,wire_v_test[i].pixel(row,col));
    //     }
    //   }
    //   TCanvas can("can", "histograms ", 3456, 1008);
    //   removed_h.SetTitle("Test Save Removed WC Image");
    //   removed_h.SetXTitle("column");
    //   removed_h.SetYTitle("row");
    //   removed_h.SetOption("COLZ");
    //   removed_h.Draw();
    //   can.SaveAs(Form("Removed_SaveTest_WC_%d.png",i));
    //   removed_h.Reset();
    // }

  }
	std::cout << "Files Ran Over:	" <<  total_files_run_over << "\n";
	std::cout << "Events Ran Over:	" <<  total_events_run_over << "\n";
	std::cout << "\n";

return 0;
}//End of main

//Functions:
void remove_from_hit(larcv::Image2D& removed_img, const larlite::hit& hit, int padding){
  int row;
  int col;
  int plane;
  int minrow;
  int maxrow;
  float offset = -3;
  larcv::ImageMeta meta = removed_img.meta();
  if(hit.View()==2){
    plane = 2;
    minrow = meta.row(hit.PeakTime()+2400-hit.SigmaPeakTime());
    maxrow = meta.row(hit.PeakTime()+2400+hit.SigmaPeakTime());
    row = meta.row(hit.PeakTime()+2400+offset);
    col = meta.col(hit.Channel()-2*2400);
  }
  else if (hit.View()==0){
    minrow = meta.row(hit.PeakTime()+2400-hit.SigmaPeakTime());
    maxrow = meta.row(hit.PeakTime()+2400+hit.SigmaPeakTime());
    row = meta.row(hit.PeakTime()+2400+offset);
    col = meta.col(hit.Channel());
    plane = 0;
  }
  else if (hit.View()==1){
    minrow = meta.row(hit.PeakTime()+2400-hit.SigmaPeakTime());
    maxrow = meta.row(hit.PeakTime()+2400+hit.SigmaPeakTime());
    row = meta.row(hit.PeakTime()+2400+offset);
    col = meta.col(hit.Channel()-2400);
    plane = 1;
  }
  for (int r = minrow-padding; r <= maxrow+padding; r++){
    for(int c = col-padding; c <= col+padding; c++){
      removed_img.set_pixel(r,c,0);
    }
  }
  return;
}
