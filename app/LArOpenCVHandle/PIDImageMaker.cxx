#ifndef __PIDIMAGEMAKER_CXX__
#define __PIDIMAGEMAKER_CXX__

#include "PIDImageMaker.h"

namespace larcv {

  static PIDImageMakerProcessFactory __global_PIDImageMakerProcessFactory__;

  PIDImageMaker::PIDImageMaker(const std::string name)
    : ProcessBase(name)
  {}
    
  void PIDImageMaker::configure(const PSet& cfg)
  {

    _roi_input_producer = cfg.get<std::string>("ROIInputProducer");
    _p0_roi_output_producer = cfg.get<std::string>("P0ROIOutputProducer");
    _p1_roi_output_producer = cfg.get<std::string>("P1ROIOutputProducer");
    _pgraph_producer = cfg.get<std::string>("RecoPGraphProducer");
    _pixel2d_ctor_producer = cfg.get<std::string>("Pixel2DContourProducer");
    _pixel2d_img_producer = cfg.get<std::string>("Pixel2DImageProducer");
    
    _p0_image_producer = cfg.get<std::string>("P0OutImageProducer");
    _p1_image_producer = cfg.get<std::string>("P1OutImageProducer");
    
    _LArbysImageMaker.Configure(cfg.get<larcv::PSet>("LArbysImageMaker"));
	
    _nevents = 0;
    _nevents_selected = 0;
  }

  void PIDImageMaker::initialize()
  {}

  bool PIDImageMaker::process(IOManager& mgr)
  {
    _nevents++;
    
    //auto const event_roi           = (EventROI*)(mgr.get_data(kProductROI, _roi_input_producer));
    auto const event_pgraph        = (EventPGraph*) (mgr.get_data(kProductPGraph,_pgraph_producer));
    auto const event_img_pixel2d  = (EventPixel2D*)(mgr.get_data(kProductPixel2D,_pixel2d_img_producer));
    auto const event_ctor_pixel2d   = (EventPixel2D*)(mgr.get_data(kProductPixel2D,_pixel2d_ctor_producer));
    
    auto event_p0_image      = (EventImage2D*)(mgr.get_data(kProductImage2D,_p0_image_producer));  
    auto event_p1_image      = (EventImage2D*)(mgr.get_data(kProductImage2D,_p1_image_producer));  
    auto event_p0_roi        = (EventROI*)(mgr.get_data(kProductROI, _p0_roi_output_producer));
    auto event_p1_roi        = (EventROI*)(mgr.get_data(kProductROI, _p1_roi_output_producer));

    auto run    = (uint) event_pgraph->run();
    auto subrun = (uint) event_pgraph->subrun();
    auto event  = (uint) event_pgraph->event();
    auto entry  = (uint) mgr.current_entry();    

    _alg_mgr.SetRSEE(run,subrun,event,entry);
    
    //auto ev_pgraph = event_pgraph->PGraphArray();
    //auto ev_ctor_meta_array = event_ctor_pixel2d->MetaArray();
    //auto ev_roi              = event_roi->ROIArray();
    
    auto ev_p_array            = event_img_pixel2d->Pixel2DArray();
    auto ev_pcluster_array     = event_img_pixel2d->Pixel2DClusterArray(); //std::vector< ::larcv::Pixel2DCluster >
    auto ev_clustermeta_array  = event_img_pixel2d->ClusterMetaArray(); // std::vector< ::larcv::ImageMeta>
    auto ev_imagemeta_array    = event_img_pixel2d->MetaArray();
    
    std::vector<Image2D> p0_img_v;
    p0_img_v.clear();
    std::vector<Image2D> p1_img_v;
    p1_img_v.clear();
    
    if (ev_pcluster_array.size() == 3) {
      RecoImgFiller(ev_pcluster_array, p0_img_v, p1_img_v);
    }else{
      //LARCV_CRITICAL()<<"No pixel2d clusters found. "<<std::endl;
      VoidImgFiller(p0_img_v,p1_img_v);
    }
    
    LARCV_DEBUG()<<"p0_img_v size "<<p0_img_v.size()<<std::endl;
    LARCV_DEBUG()<<"p1_img_v size "<<p1_img_v.size()<<std::endl;

    ROI proi;
    event_p0_roi->clear();
    event_p0_roi->Append(proi);
    event_p1_roi->clear();
    event_p1_roi->Append(proi);
    
    //event_p0_image->Emplace(std::move(p0_img_v));
    //event_p1_image->Emplace(std::move(p1_img_v));
    
    for(auto each : p0_img_v) event_p0_image->Append(each);
    for(auto each : p1_img_v) event_p1_image->Append(each);
	
    return true;
  }

  void PIDImageMaker::finalize()
  {
    std::cout<<"===========>>>>>>total events : "<<_nevents<<std::endl;
    std::cout<<"========>>>>>>selected events : "<<_nevents_selected<<std::endl;
  }

  void PIDImageMaker::RecoImgFiller(std::map<larcv::PlaneID_t, std::vector<larcv::Pixel2DCluster>> ev_pcluster_array,
				    std::vector<larcv::Image2D>& p0_img_v,
				    std::vector<larcv::Image2D>& p1_img_v){
    
    _nevents_selected++;

    for (size_t plane = 0; plane< 3; ++plane){
      
      auto pcluster_v = ev_pcluster_array[plane];
      
      if (pcluster_v.size()!=2) {
	LARCV_CRITICAL()<<"Not 2 particles on plane on"<<plane<<std::endl;
	Image2D img(576,576);
	img.resize(576,576, 0.0);
	
	p0_img_v.emplace_back(std::move(img));
	p1_img_v.emplace_back(std::move(img));
	continue;
      }
      
      for (size_t pid = 0; pid < 2; ++pid ){
	auto pcluster = pcluster_v[pid];
	Image2D img(576,576);
	if (!pcluster.size()) {
	  img.resize(576, 576, 0.0);
	}
	else{
	  LARCV_DEBUG()<<"max_x "<<pcluster.max_x()
		       <<" min_x "<<pcluster.min_x()
		       <<" max_y "<<pcluster.max_y()
		       <<" min_y "<<pcluster.min_y()
		       <<std::endl;
	  /*	size_t rows = std::abs(pcluster.max_x() - pcluster.min_x());
		size_t cols = std::abs(pcluster.max_y() - pcluster.min_y());
		LARCV_DEBUG()<<"creating rows is "<<rows<< " and creating cols is "<<cols;
		Image2D img(rows+1, cols+1);*/
	  
	//img.clear_data();
	  LARCV_DEBUG()<<"created pixel 2d image size is "<<img.size()<<std::endl;
	
	  for (auto pixel: pcluster) {
	    //LARCV_DEBUG()<<"raw x "<<pixel.X()<<" raw y "<<pixel.Y()<<std::endl;
	    //LARCV_DEBUG()<<"cal x "<<pixel.X() - pcluster.min_x()<<"cal y "<<pixel.Y() - pcluster.min_y()<<std::endl;
	    if(pixel.X() - pcluster.min_x() > 575 ||
	       pixel.Y() - pcluster.min_y() > 575) continue;
	    
	    img.set_pixel(pixel.X() - pcluster.min_x(), 
			  pixel.Y() - pcluster.min_y(),
			  pixel.Intensity());
	    
	    //if(plane==2)std::cout<<pixel.X() - pcluster.min_x()<<", "<<pixel.Y() - pcluster.min_y()<<", "<<pixel.Intensity()<<std::endl;
	  }
	}

	if (pid == 0 ) p0_img_v.emplace_back(std::move(img));
	if (pid == 1 ) p1_img_v.emplace_back(std::move(img));
      }
    }
  }
  


  void PIDImageMaker::VoidImgFiller(std::vector<larcv::Image2D>& p0_img_v,
				    std::vector<larcv::Image2D>& p1_img_v){
    for (size_t plane = 0; plane< 3; ++plane){
      
      Image2D img(576,576);
      img.resize(576,576, 0.0);
      
      p0_img_v.emplace_back(std::move(img));
      p1_img_v.emplace_back(std::move(img));
    }
  }
}
#endif
