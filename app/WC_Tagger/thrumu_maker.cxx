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
void overlay_from_hit(larcv::Image2D& new_img, const larcv::Image2D& old_img, const larlite::hit& hit, int padding=2);

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
      larcv::EventImage2D* const ev_masked_out = (larcv::EventImage2D * )(io_out_cv->get_data(larcv::kProductImage2D, "masked"));

      // Copy Images for removal for thrumu
      larcv::Image2D thrumu_u(img_adc_v[0]);
      larcv::Image2D thrumu_v(img_adc_v[1]);
      larcv::Image2D thrumu_y(img_adc_v[2]);
      // copy Images for overlay for masked
      larcv::Image2D masked_u(img_adc_v[0]);
      larcv::Image2D masked_v(img_adc_v[1]);
      larcv::Image2D masked_y(img_adc_v[2]);
      masked_u.paint(0.);
      masked_v.paint(0.);
      masked_y.paint(0.);

      for (int idx=0;idx < ev_hit_wc.size();idx++){
        larlite::hit hit = ev_hit_wc[idx];
        int plane = hit.View();
        if (plane == 0){
          //These functions have a 3rd arg that does padding, default is 2
          //around the hit's col and row vals.
          remove_from_hit(thrumu_u, hit);
          overlay_from_hit(masked_u,img_adc_v[plane], hit);
        }
        else if (plane == 1){
          remove_from_hit(thrumu_v, hit);
          overlay_from_hit(masked_v,img_adc_v[plane], hit);
        }
        else {
          remove_from_hit(thrumu_y, hit);
          overlay_from_hit(masked_y,img_adc_v[plane], hit);
        }
      }
      //Emplace images into output file
      ev_thrumu_out->Emplace(std::move(thrumu_u));
      ev_thrumu_out->Emplace(std::move(thrumu_v));
      ev_thrumu_out->Emplace(std::move(thrumu_y));

      ev_masked_out->Emplace(std::move(masked_u));
      ev_masked_out->Emplace(std::move(masked_v));
      ev_masked_out->Emplace(std::move(masked_y));

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
    const auto ev_masked_test           = (larcv::EventImage2D*)io_test.get_data( larcv::kProductImage2D, "masked");
    std::vector<larcv::Image2D> const& masked_v_test = ev_masked_test->Image2DArray();
    std::cout << masked_v_test.size() << " Output masked v test size \n";

    // To Draw Stuff:
    // TH2D removed_h =TH2D("removed_h","removed_h ",3456,0.,3456,1008,0.,1008.);
    // TH2D kept_h =TH2D("kept_h","kept_h ",3456,0.,3456,1008,0.,1008.);
    // for (int i = 0;i<3; i++){
    //   for (int row = 0;row<1008;row++){
    //     for (int col = 0;col<3456; col++){
    //       double val = masked_v_test[i].pixel(row,col);
    //       if (val == 50.0) val = 500;
    //       kept_h.SetBinContent(col,row,val);
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
    //
    //   kept_h.SetTitle("Test Save Kept WC Image");
    //   kept_h.SetXTitle("column");
    //   kept_h.SetYTitle("row");
    //   kept_h.SetOption("COLZ");
    //   kept_h.Draw();
    //   can.SaveAs(Form("Kept_SaveTest_WC_%d.png",i));
    //   kept_h.Reset();
    // }

  }
	std::cout << "Files Ran Over:	" <<  total_files_run_over << "\n";
	std::cout << "Events Ran Over:	" <<  total_events_run_over << "\n";
	std::cout << "\n";

return 0;
}//End of main

//Functions:
void remove_from_hit(larcv::Image2D& removed_img, const larlite::hit& hit, int padding){
  /*
  this function takes in an image and a hit, and removes charge in the Image
  based on the hit's position, with some padding around it (default padding of 2)
  */
  int row;
  int col;
  int plane;
  int minrow;
  int maxrow;
  float offset = -3;
  larcv::ImageMeta meta = removed_img.meta();

  float maxtick = hit.PeakTime()+2400+hit.SigmaPeakTime();
  float mintick = hit.PeakTime()+2400-hit.SigmaPeakTime();
  //std::cout << "mintick1: " << mintick << " " << maxtick << std::endl;
  if ( maxtick >= meta.max_y() )
    maxtick = meta.max_y()-meta.pixel_height();
  if ( mintick <= meta.min_y() )
    mintick = meta.min_y()+meta.pixel_height();
  //std::cout << "mintick2: " << mintick << " " << maxtick << std::endl;

  if (mintick>=meta.max_y()) return;
  if (maxtick<=meta.min_y()) return;

  float peaktick = hit.PeakTime()+2400+offset;
  if ( peaktick>=meta.max_y() ) return;
  if ( peaktick<=meta.min_y() ) return;
  
  if(hit.View()==2){
    plane = 2;
    minrow = meta.row(mintick,__FILE__,__LINE__);
    maxrow = meta.row(maxtick,__FILE__,__LINE__);
    row = meta.row(hit.PeakTime()+2400+offset,__FILE__,__LINE__);
    col = meta.col(hit.Channel()-2*2400,__FILE__,__LINE__);
  }
  else if (hit.View()==0){
    minrow = meta.row(mintick,__FILE__,__LINE__);
    maxrow = meta.row(maxtick,__FILE__,__LINE__);
    row = meta.row(hit.PeakTime()+2400+offset, __FILE__, __LINE__);
    col = meta.col(hit.Channel(), __FILE__, __LINE__);
    plane = 0;
  }
  else if (hit.View()==1){
    minrow = meta.row(mintick,__FILE__,__LINE__);
    maxrow = meta.row(maxtick,__FILE__,__LINE__);
    row = meta.row(hit.PeakTime()+2400+offset,__FILE__,__LINE__);
    col = meta.col(hit.Channel()-2400,__FILE__,__LINE__);
    plane = 1;
  }

  if ( minrow>maxrow ) {
    int tmp = minrow;
    minrow = maxrow;
    maxrow = tmp;
  }
  
  for (int r = minrow-padding; r <= maxrow+padding; r++){
    if (r<0 ) continue;
    if (r>=(int)removed_img.meta().rows()) continue;
    for(int c = col-padding; c <= col+padding; c++){
      if (c<0 ) continue;
      if (c>=(int)removed_img.meta().cols()) continue;

      removed_img.set_pixel(r,c,0);
    }
  }
  return;
}

void overlay_from_hit(larcv::Image2D& new_img, const larcv::Image2D& old_img, const larlite::hit& hit, int padding){
  /*
  this image takes in a new image, and an old image, and overlays the old img
  on top of the new img where the hit is located in row and col with some padding
  around the hit location
  if the column is all 0s in the padded region then instead you overlay a
  value of 50 at the hit location
  */
  int row;
  int col;
  int plane;
  float showerscore;
  float hit_dist;
  int minrow;
  int maxrow;
  float offset = -3;
  larcv::ImageMeta meta = old_img.meta();

  float maxtick = hit.PeakTime()+2400+hit.SigmaPeakTime();
  float mintick = hit.PeakTime()+2400-hit.SigmaPeakTime();
  if ( maxtick >= meta.max_y() )
    maxtick = meta.max_y()-meta.pixel_height();
  if ( mintick <= meta.min_y() )
    mintick = meta.min_y()+meta.pixel_height();

  if (mintick>=meta.max_y()) return;
  if (maxtick<=meta.min_y()) return;

  float peaktick = hit.PeakTime()+2400+offset;
  if ( peaktick>=meta.max_y() ) return;
  if ( peaktick<=meta.min_y() ) return;  
  
  if(hit.View()==2){
    plane = 2;
    minrow = meta.row(mintick,__FILE__,__LINE__);
    maxrow = meta.row(maxtick,__FILE__,__LINE__);
    row = meta.row(hit.PeakTime()+2400+offset,__FILE__,__LINE__);
    col = meta.col(hit.Channel()-2*2400,__FILE__,__LINE__);
  }
  else if (hit.View()==0){
    minrow = meta.row(mintick,__FILE__,__LINE__);
    maxrow = meta.row(maxtick,__FILE__,__LINE__);
    row = meta.row(hit.PeakTime()+2400+offset,__FILE__,__LINE__);
    col = meta.col(hit.Channel(),__FILE__,__LINE__);
    plane = 0;
  }
  else if (hit.View()==1){
    minrow = meta.row(mintick,__FILE__,__LINE__);
    maxrow = meta.row(maxtick,__FILE__,__LINE__);
    row = meta.row(hit.PeakTime()+2400+offset,__FILE__,__LINE__);
    col = meta.col(hit.Channel()-2400,__FILE__,__LINE__);
    plane = 1;
  }

  if ( minrow>maxrow ) {
    int tmp = minrow;
    minrow = maxrow;
    maxrow = tmp;
  }
  
  for(int c = col-padding; c <= col+padding; c++){
    double col_sum = 0;
    if ( c<0 ) continue;
    if ( c>=(int)new_img.meta().cols() ) continue;
    for (int r = minrow-padding; r <= maxrow+padding; r++){
      if ( r<0) continue;
      if ( r>=new_img.meta().rows() ) continue;
      double val = old_img.pixel(r,c);
      col_sum += val;
      new_img.set_pixel(r,c,val);
    }
    if (col_sum == 0) {
      // std::cout << "FLAG Hit Placed in Dead Region\n";
      new_img.set_pixel(row,c,50.000);
    }
  }
  return;
}
