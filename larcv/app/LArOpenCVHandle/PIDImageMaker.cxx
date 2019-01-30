#ifndef __PIDIMAGEMAKER_CXX__
#define __PIDIMAGEMAKER_CXX__

#include "PIDImageMaker.h"
#include "fstream"

namespace larcv {

  static PIDImageMakerProcessFactory __global_PIDImageMakerProcessFactory__;

  PIDImageMaker::PIDImageMaker(const std::string name)
    : ProcessBase(name)
  {}
    
  void PIDImageMaker::configure(const PSet& cfg)
  {
    //input
    _roi_input_producer     = cfg.get<std::string>("ROIInputProducer");
    _pgraph_producer        = cfg.get<std::string>("RecoPGraphProducer");
    _pixel2d_ctor_producer  = cfg.get<std::string>("Pixel2DContourProducer");
    _pixel2d_img_producer   = cfg.get<std::string>("Pixel2DImageProducer");
    
    //output
    _p0_roi_output_producer = cfg.get<std::string>("P0ROIOutputProducer");
    _p1_roi_output_producer = cfg.get<std::string>("P1ROIOutputProducer");
    _p0_image_producer      = cfg.get<std::string>("P0OutImageProducer");
    _p1_image_producer      = cfg.get<std::string>("P1OutImageProducer");
    _multi_image_producer   = cfg.get<std::string>("MultiOutImageProducer");
    _outimage_dim           = cfg.get<std::pair<int, int>>("OutputImageDim");
    
    _LArbysImageMaker.Configure(cfg.get<larcv::PSet>("LArbysImageMaker"));
      
    _nevents = 0;
    _nevents_passing_nueLL = 0;
    
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
    auto event_multi_image   = (EventImage2D*)(mgr.get_data(kProductImage2D,_multi_image_producer));  
    auto event_p0_roi        = (EventROI*)(mgr.get_data(kProductROI, _p0_roi_output_producer));
    auto event_p1_roi        = (EventROI*)(mgr.get_data(kProductROI, _p1_roi_output_producer));

    auto run    = (uint) event_pgraph->run();
    auto subrun = (uint) event_pgraph->subrun();
    auto event  = (uint) event_pgraph->event();
    auto entry  = (uint) mgr.current_entry();    

    _alg_mgr.SetRSEE(run,subrun,event,entry);
    
    auto pgraph_v = event_pgraph->PGraphArray();
    if(pgraph_v.size()){
      ROI croi = pgraph_v[0].ParticleArray().front();
      std::vector<Image2D> p01_croi_img;
      CROIImgFiller(croi, p01_croi_img);
    }
    
    
    
    //if(pgraph_v.size()>1)
      //LARCV_DEBUG()<<"size of pgraph_v"<<pgraph_v.size()<<std::endl;
    //ROI croi = event_pgraph->ParticleArray();
    	
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
    std::vector<Image2D> p01_img_v;
    p01_img_v.clear();
    
    // checking how many planes have reco pgrapgh
    //if (ev_pcluster_array.size() ==3) {
    if (ev_pcluster_array.size() > 0 ) {
    
      ROI proi;
      event_p0_roi->clear();
      event_p0_roi->Append(proi);
      event_p1_roi->clear();
      event_p1_roi->Append(proi);
      
      SPIDRecoImgFiller(ev_pcluster_array, p0_img_v, p1_img_v);
      MPIDRecoImgFiller(ev_pcluster_array, p01_img_v);
      for(auto each : p0_img_v ) event_p0_image->Append(each);
      for(auto each : p1_img_v ) event_p1_image->Append(each);
      for(auto each : p01_img_v) event_multi_image->Append(each);

      return true;
    }else return false;
    
    /*else{
      //LARCV_CRITICAL()<<"No pixel2d clusters found. "<<std::endl;
      LARCV_DEBUG()<<"run"<<run
	       <<"subrun"<<subrun
	       <<"event"<<event
	       <<"pixel2dclusterarraysize"<<ev_pcluster_array.size()<<std::endl;
      VoidImgFiller(p0_img_v,p1_img_v);
    }
    
    //event_p0_image->Emplace(std::move(p0_img_v));
    //event_p1_image->Emplace(std::move(p1_img_v));
    
    for(auto each : p0_img_v) event_p0_image->Append(each);
    for(auto each : p1_img_v) event_p1_image->Append(each);
    */
    //return true;
  }

  void PIDImageMaker::finalize()
  {
    LARCV_DEBUG()<<"================>>>>>>total events : "<<_nevents<<std::endl;
    LARCV_DEBUG()<<"========>>>>>>passing NueLL events : "<<_nevents_passing_nueLL<<std::endl;
  }

  void PIDImageMaker::SPIDRecoImgFiller(std::map<larcv::PlaneID_t, 
					std::vector<larcv::Pixel2DCluster>> ev_pcluster_array,
					std::vector<larcv::Image2D>& p0_img_v,
					std::vector<larcv::Image2D>& p1_img_v){
    
    _nevents_passing_nueLL++;

    for (size_t plane = 0; plane< 3; ++plane){
      
      auto pcluster_v = ev_pcluster_array[plane];

      if (pcluster_v.size()!=2) {
	//Need improve
	//LARCV_CRITICAL()<<"Not 2 particles on plane "<<plane<<std::endl;
	Image2D img(_outimage_dim.first,_outimage_dim.second);
	img.resize(_outimage_dim.first,_outimage_dim.second, 0.0);
	
	p0_img_v.emplace_back(std::move(img));
	p1_img_v.emplace_back(std::move(img));
	continue;
      }
            
      for (size_t pid = 0; pid < 2; ++pid ){
	auto pcluster = pcluster_v[pid];
	//LARCV_DEBUG()<<"pid is "<<pid<<" with size of "<<pcluster.size()<<std::endl;
	Image2D img(_outimage_dim.first,_outimage_dim.second);
	if (!pcluster.size()) {
	  img.resize(_outimage_dim.first,_outimage_dim.second, 0.0);
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
	    if(pixel.X() - pcluster.min_x() > (_outimage_dim.first  -1) ||
	       pixel.Y() - pcluster.min_y() > (_outimage_dim.second -1)) continue;
	    
	    img.set_pixel(pixel.X() - pcluster.min_x(), 
			  pixel.Y() - pcluster.min_y(),
			  pixel.Intensity());
	    
	    //if(plane==2)LARCV_DEBUG()<<pixel.X() - pcluster.min_x()<<", "<<pixel.Y() - pcluster.min_y()<<", "<<pixel.Intensity()<<std::endl;
	  }
	}
	if (pid == 0 ) p0_img_v.emplace_back(std::move(img));
	if (pid == 1 ) p1_img_v.emplace_back(std::move(img));
      }
    }
  }

  void PIDImageMaker::MPIDRecoImgFiller(std::map<larcv::PlaneID_t, std::vector<larcv::Pixel2DCluster>> ev_pcluster_array,
					std::vector<larcv::Image2D>& p01_img_v){
    
    _nevents_passing_nueLL++;
    
    for (size_t plane = 0; plane< 3; ++plane){
      
      auto pcluster_v = ev_pcluster_array[plane];
      //IF there are not exaxtly 2 particles reconstructed
      // Here needs upates!
      if ( pcluster_v.size()!=2  || !(pcluster_v[0].size() * pcluster_v[1].size())) {
	//Need improve
	//LARCV_CRITICAL()<<"Not 2 particles on plane on"<<plane<<std::endl;
	Image2D img(_outimage_dim.first,_outimage_dim.second);
	img.resize(_outimage_dim.first,_outimage_dim.second, 0.0);
	
	p01_img_v.emplace_back(std::move(img));
	continue;
      }
      
      //IF there are 2 particles reconstructed
      Image2D img(_outimage_dim.first,_outimage_dim.second);
      img.resize(_outimage_dim.first,_outimage_dim.second, 0.0);
     
      //if(!(pcluster_v[0].size() * pcluster_v[1].size())) continue;

      int max_x = (pcluster_v[0].max_x() > pcluster_v[1].max_x() ? pcluster_v[0].max_x() : pcluster_v[1].max_x());
      int min_x = (pcluster_v[0].min_x() < pcluster_v[1].min_x() ? pcluster_v[0].min_x() : pcluster_v[1].min_x());
      int max_y = (pcluster_v[0].max_y() > pcluster_v[1].max_y() ? pcluster_v[0].max_y() : pcluster_v[1].max_y());
      int min_y = (pcluster_v[0].min_y() < pcluster_v[1].min_y() ? pcluster_v[0].min_y() : pcluster_v[1].min_y());

      LARCV_DEBUG()<<"created pixel 2d image size is "<<img.size()<<std::endl;
      for (size_t pid = 0; pid < 2; ++pid ){
	auto pcluster = pcluster_v[pid];
	//LARCV_DEBUG()<<"In MPID pid is "<<pid<<" with size of "<<pcluster.size()<<std::endl;
	for (auto pixel: pcluster) {
	  //LARCV_DEBUG()<<"raw x "<<pixel.X()<<" raw y "<<pixel.Y()<<std::endl;
	  //LARCV_DEBUG()<<"cal x "<<pixel.X() - pcluster.min_x()<<"cal y "<<pixel.Y() - pcluster.min_y()<<std::endl;
	  if(pixel.X() - min_x > (_outimage_dim.first  -6) ||
	     pixel.Y() - min_y > (_outimage_dim.second -6)) continue;
	  


	  img.set_pixel(pixel.X() - min_x+5, 
			pixel.Y() - min_y+5,
			pixel.Intensity());
	  
	}
      }
      p01_img_v.emplace_back(std::move(img));
    }
    //LARCV_DEBUG()<<"p01_img_v has size of "<<p01_img_v.size()<<std::endl;
  }
  
  void PIDImageMaker::VoidImgFiller(std::vector<larcv::Image2D>& p0_img_v,
				    std::vector<larcv::Image2D>& p1_img_v){
    for (size_t plane = 0; plane< 3; ++plane){
      
      Image2D img(_outimage_dim.first,_outimage_dim.second);
      img.resize(_outimage_dim.first,_outimage_dim.second, 0.0);
      
      p0_img_v.emplace_back(std::move(img));
      p1_img_v.emplace_back(std::move(img));
    }
  }
  
  void PIDImageMaker::CROIImgFiller(ROI croi, std::vector<larcv::Image2D>& p01_img_v){
    //croi
    
    for(size_t plane =0; plane <=2 ;++plane){
      ImageMeta this_plane_meta = croi.BB(plane);
    }
    
  }

}
#endif
