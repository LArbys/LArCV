#ifndef __COSMICPIXELANA_CXX__
#define __COSMICPIXELANA_CXX__

#include "CosmicPixelAna.h"
#include "LArbysImageMaker.h"
#include "DataFormat/EventROI.h"

#include <array>
#include <cassert>

namespace larcv {
  
  static CosmicPixelAnaProcessFactory __global_CosmicPixelAnaProcessFactory__;
  
  CosmicPixelAna::CosmicPixelAna(const std::string name) :
    ProcessBase(name),
    _LArbysImageMaker(),
    _tree(nullptr)
  {}
    
  void CosmicPixelAna::configure(const PSet& cfg)
  {
    _ev_img2d_prod   = cfg.get<std::string>("EventImage2DProducer"); 
    _seg_img2d_prod  = cfg.get<std::string>("SegmentImage2DProducer");
    _thrumu_img_prod = cfg.get<std::string>("ThruMuProducer");
    _stopmu_img_prod = cfg.get<std::string>("StopMuProducer");
    _roi_prod        = cfg.get<std::string>("ROIProducer");
    
    _LArbysImageMaker.Configure(cfg.get<PSet>("LArbysImageMaker"));
  }
  
  void CosmicPixelAna::initialize()
  {
    
    _tree = new TTree("EventCosmicPixelTree","");
    _tree->Branch("run",&_run,"run/I");
    _tree->Branch("subrun",&_subrun,"subrun/I");
    _tree->Branch("event",&_event,"event/I");
    _tree->Branch("entry",&_entry,"entry/I");

    _tree->Branch("nupixel0",&_nupixel0,"nupixel0/I");
    _tree->Branch("nupixel1",&_nupixel1,"nupixel1/I");
    _tree->Branch("nupixel2",&_nupixel2,"nupixel2/I");

    _tree->Branch("nupixelsum",&_nupixelsum,"nupixelsum/F");
    _tree->Branch("nupixelavg",&_nupixelavg,"nupixelavg/F");

    _tree->Branch("cosmicpixel0",&_cosmicpixel0,"cosmicpixel0/I");
    _tree->Branch("cosmicpixel1",&_cosmicpixel1,"cosmicpixel1/I");
    _tree->Branch("cosmicpixel2",&_cosmicpixel2,"cosmicpixel2/I");

    _tree->Branch("cosmicpixelsum",&_cosmicpixelsum,"cosmicpixelsum/F");
    _tree->Branch("cosmicpixelavg",&_cosmicpixelavg,"cosmicpixelavg/F");
    
    _tree->Branch("ratiopixel0",&_ratiopixel0,"ratiopixel0/F");
    _tree->Branch("ratiopixel1",&_ratiopixel1,"ratiopixel1/F");
    _tree->Branch("ratiopixel2",&_ratiopixel2,"ratiopixel2/F");

    _tree->Branch("ratiopixelsum",&_ratiopixelsum,"ratiopixelsum/F");
    _tree->Branch("ratiopixelavg",&_ratiopixelavg,"ratiopixelavg/F");
    
  }

  bool CosmicPixelAna::process(IOManager& mgr)
  {
    if (_ev_img2d_prod.empty()) {
      LARCV_INFO() << "No event image provided nothing to do" << std::endl;
      return false;
    }
    
    auto const ev_img2d  = (EventImage2D*)(mgr.get_data(kProductImage2D,_ev_img2d_prod));
    if (!ev_img2d) throw larbys("Invalid event image producer provided");

    if (_seg_img2d_prod.empty()) {
      LARCV_INFO() << "No segment image provided nothing to do" << std::endl;
      return false;
    }
    
    auto const seg_img2d  = (EventImage2D*)(mgr.get_data(kProductImage2D,_seg_img2d_prod));
    if (!seg_img2d) throw larbys("Invalid segment image producer provided");

    assert(ev_img2d->Image2DArray().size() == seg_img2d->Image2DArray().size());
    for(size_t img_id=0; img_id<ev_img2d->Image2DArray().size(); ++img_id) {
      const auto& ev_img  = ev_img2d->Image2DArray().at(img_id);
      const auto& seg_img = seg_img2d->Image2DArray().at(img_id);

      if (ev_img.meta().rows() != seg_img.meta().rows()) {
	LARCV_WARNING() << "Event image and segmentation image differ in row size @ plane=" << img_id
			<< " ("<<ev_img.meta().rows()<<"!="<<seg_img.meta().rows()<<")"<<std::endl;
	
	return false;
      }

      if (ev_img.meta().cols() != seg_img.meta().cols()) {
	LARCV_WARNING() << "Event image and segmentation image differ in col size @ plane=" << img_id
			<< " ("<<ev_img.meta().cols()<<"!="<<seg_img.meta().cols()<<")"<<std::endl;
	
	return false;
      }
    }
    
    auto const ev_thrumu = (EventPixel2D*)(mgr.get_data(kProductPixel2D,_thrumu_img_prod));
    if (!ev_thrumu) throw larbys("Invalid ThruMu producer provided!");
    
    auto const ev_stopmu = (EventPixel2D*)(mgr.get_data(kProductPixel2D,_stopmu_img_prod));
    if (!ev_stopmu) throw larbys("Invalid StopMu producer provided!");

    auto const ev_roi = (EventROI*)(mgr.get_data(kProductROI,_roi_prod));
    if (!ev_roi) throw larbys("Invalid ROI producer provided!");

    if (ev_roi->ROIArray().size() != 1) throw larbys("Too many ROI!");

    const auto& roi = ev_roi->ROIArray().front();
      
    _run    = (int) ev_roi->run();
    _subrun = (int) ev_roi->subrun();
    _event  = (int) ev_roi->event();
    _entry  = (int) mgr.current_entry();
    
    std::array<size_t,3> neutrino_pixel_v;
    std::array<size_t,3> cosmic_pixel_v;
    
    for (size_t plane=0; plane<3; ++plane) {
      
      const auto& img2d = seg_img2d->Image2DArray().at(plane);
      
      auto thrumu_img2d = _LArbysImageMaker.ConstructCosmicImage(ev_thrumu,img2d,plane,1);
      auto stopmu_img2d = _LArbysImageMaker.ConstructCosmicImage(ev_stopmu,img2d,plane,1);

      const auto& bb = roi.BB(plane);
      
      auto crop_img2d  = img2d.crop(bb);
      auto crop_thrumu = thrumu_img2d.crop(bb);
      auto crop_stopmu = stopmu_img2d.crop(bb);
      
      auto crop_cosmic_img = crop_thrumu;
      crop_cosmic_img += crop_stopmu;

      crop_cosmic_img.eltwise(crop_img2d);
      
      size_t cosmic_pixels = 0;
      size_t neutrino_pixels = 0;
      
      assert( crop_cosmic_img.as_vector().size() == crop_img2d.as_vector().size());
      
      for(size_t px_id = 0; px_id < crop_cosmic_img.as_vector().size(); ++px_id) {
	
	auto cosmic_px   = crop_cosmic_img.as_vector()[px_id];
	auto neutrino_px = crop_img2d.as_vector()[px_id];
	
	if (cosmic_px) cosmic_pixels++;
	if (neutrino_px) neutrino_pixels++;
      }

      cosmic_pixel_v[plane] = cosmic_pixels;
      neutrino_pixel_v[plane] = neutrino_pixels;
    }

    _nupixel0 = neutrino_pixel_v[0];
    _nupixel1 = neutrino_pixel_v[1];
    _nupixel2 = neutrino_pixel_v[2];

    _cosmicpixel0 = cosmic_pixel_v[0];
    _cosmicpixel1 = cosmic_pixel_v[1];
    _cosmicpixel2 = cosmic_pixel_v[2];

    _ratiopixel0 = _nupixel0 ? (float)_cosmicpixel0 / (float)_nupixel0 : 0;
    _ratiopixel1 = _nupixel1 ? (float)_cosmicpixel1 / (float)_nupixel1 : 0;
    _ratiopixel2 = _nupixel2 ? (float)_cosmicpixel2 / (float)_nupixel2 : 0;

    _nupixelsum = _nupixel0 + _nupixel1 + _nupixel2;
    _nupixelavg = _nupixelsum / 3.0;

    _cosmicpixelsum = _cosmicpixel0 + _cosmicpixel1 + _cosmicpixel2;
    _cosmicpixelavg = _cosmicpixelsum / 3.0;

    _ratiopixelsum = _ratiopixel0 + _ratiopixel1 + _ratiopixel2;
    _ratiopixelavg = _ratiopixelsum / 3.0;

    _tree->Fill();
    return true;
  }

  void CosmicPixelAna::finalize()
  {
    if(has_ana_file()) {
      _tree->Write();
    }
  }

}
#endif
