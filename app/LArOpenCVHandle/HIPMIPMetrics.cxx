#ifndef __HIPMIPMETRICS_CXX__
#define __HIPMIPMETRICS_CXX__

#include "HIPMIPMetrics.h"

#include "DataFormat/EventImage2D.h"
#include "CVUtil/CVUtil.h"

#include <array>
#include <cassert>


namespace larcv {

  static HIPMIPMetricsProcessFactory __global_HIPMIPMetricsProcessFactory__;

  HIPMIPMetrics::HIPMIPMetrics(const std::string name)
    : ProcessBase(name)
  {}
    
  void HIPMIPMetrics::configure(const PSet& cfg)
  {
		_particle_segment_id = cfg.get<int>("ParticleSegmentID",9); 
	}

  void HIPMIPMetrics::initialize()
  {
		_tree = new TTree("hipmip","hipmip");
		
		_tree->Branch("run",&_run,"run/I");		
		_tree->Branch("subrun",&_subrun,"subrun/I");		
		_tree->Branch("event",&_event,"event/I");		
		_tree->Branch("entry",&_entry,"entry/I");		
	
		_tree->Branch("totalProtonPix_plane0",&_totalProtonPix_plane0,"totalProtonPix_plane0/I");
		_tree->Branch("totalProtonADC_plane0",&_totalProtonADC_plane0,"totalProtonADC_plane0/I");
		_tree->Branch("totalProtonPix_plane1",&_totalProtonPix_plane1,"totalProtonPix_plane1/I");
		_tree->Branch("totalProtonADC_plane1",&_totalProtonADC_plane1,"totalProtonADC_plane1/I");	
		_tree->Branch("totalProtonPix_plane2",&_totalProtonPix_plane2,"totalProtonPix_plane2/I");
		_tree->Branch("totalProtonADC_plane2",&_totalProtonADC_plane2,"totalProtonADC_plane2/I");
	
		// set our pixel counter and adc counter to zero!
		for(int i = 0; i < 3; i++){
			protonPixelCt[i] = 0;
			protonADCCt[i] = 0;
		}
	}

  bool HIPMIPMetrics::process(IOManager& mgr)
  {
		// load up that img
		auto ev_image = (EventImage2D*)mgr.get_data(kProductImage2D,"wire");
    auto ev_segment = (larcv::EventImage2D*)mgr.get_data(larcv::kProductImage2D,"segment");

		_run = (int) ev_image->run();
		_subrun = (int) ev_image->subrun();
		_event = (int) ev_image->event();
		_entry = (int) mgr.current_entry();

		// Loop through all planes!
		for(size_t img = 0; img < 3; img++){

			// loop through all pixels and find the proton ones!
			auto v_img = ev_image->Image2DArray().at(img);
			auto v_seg = ev_segment->Image2DArray().at(img);

			protonPixelCt[img] = 0;
			protonADCCt[img] = 0;

			for(size_t i = 0; i < v_seg.size(); i++){
				if(v_seg.as_vector().at(i) == _particle_segment_id){
					protonPixelCt[img] += 1;
					protonADCCt[img] += v_img.as_vector().at(i);
				}
			}
		}	

		_totalProtonPix_plane0 = protonPixelCt[0];
		_totalProtonADC_plane0 = protonADCCt[0];
		_totalProtonPix_plane1 = protonPixelCt[1];
		_totalProtonADC_plane1 = protonADCCt[1];
		_totalProtonPix_plane2 = protonPixelCt[2];
		_totalProtonADC_plane2 = protonADCCt[2];
		
		_tree->Fill();

		return true;
	}

  void HIPMIPMetrics::finalize()
  {
		_tree->Write();
	}

}
#endif
