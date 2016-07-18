#ifndef __FINDEMPTIES_CXX__
#define __FINDEMPTIES_CXX__

#include "FindEmpties.h"
#include "DataFormat/EventImage2D.h"
#include "CVUtil/CVUtil.h"

namespace larcv {

  static FindEmptiesProcessFactory __global_FindEmptiesProcessFactory__;

  FindEmpties::FindEmpties(const std::string name)
    : ProcessBase(name)
  {
    _image_tree = nullptr; 
    _pixel_tree = nullptr;

    _image_index = 0;
    _plane = -1;
    _pixel_count = 0;
    _pixel_less_count = 0;
    _pixel_sum   = 0.;
    _max_pixel   = 0.;
    _pixel_intens = 0;
    _max_dist = -1. ;
    
    _n_contours = 0;
    _tot_area = 0.;
    _tot_height = 0.;
    _tot_width = 0.; 

    _max_area = 0.;
    _max_height = 0.;
    _width_at_max_height = 0.; 
    _max_charge = 0.;
  }
    
  void FindEmpties::configure(const PSet& cfg)
  {
    _image_name = cfg.get<std::string>("ImageName");
    _pixel_count_threshold = cfg.get<float>("PixelCountThreshold");
  }

  void FindEmpties::initialize()
  {
    _event = 0 ;

    if(!_image_tree){
      _image_tree = new TTree("image_tree","Tree for simple analysis");
      _image_tree->Branch( "event", &_event, "event/s" );
      _image_tree->Branch( "image_index", &_image_index, "image_index/s" );
      _image_tree->Branch( "plane", &_plane, "plane/I");
      _image_tree->Branch( "pixel_count", &_pixel_count, "pixel_count/I" );
      _image_tree->Branch( "pixel_less_count", &_pixel_less_count, "pixel_less_count/I" );
      _image_tree->Branch( "pixel_sum", &_pixel_sum, "pixel_sum/F" );
      _image_tree->Branch( "max_pixel",   &_max_pixel,   "max_pixel/F"   );
      _image_tree->Branch( "pix_intens_v","std::vector<float>",&_pix_intens_v);
      _image_tree->Branch( "dist_v","std::vector<float>",&_dist_v);
      _image_tree->Branch( "max_dist",   &_max_dist,   "max_dist/F"   );
      _image_tree->Branch( "n_contours",   &_n_contours,   "n_contours/I"   );
      _image_tree->Branch( "tot_area",   &_tot_area,   "tot_area/F"   );
      _image_tree->Branch( "tot_height",   &_tot_height,   "tot_height/F"   );
      _image_tree->Branch( "tot_width",   &_tot_width,   "tot_width/F"   );
      _image_tree->Branch( "max_area",   &_max_area,   "max_area/F"   );
      _image_tree->Branch( "max_height",   &_max_height,   "max_height/F"   );
      _image_tree->Branch( "width_at_max_height",   &_width_at_max_height,   "width_at_max_height/F"   );
      _image_tree->Branch( "max_charge",   &_max_charge,   "max_charge/F"   );
     
    }

    if(!_pixel_tree){
      _pixel_tree = new TTree("pixel_tree","pixel_tree");
      _pixel_tree->Branch( "event", &_event, "event/s" );
      _pixel_tree->Branch( "plane", &_plane, "plane/I" );
      _pixel_tree->Branch( "pixel_intens",   &_pixel_intens,   "pixel_intens/F"   );  
      _pixel_tree->Branch( "pixel_dist",   &_pixel_dist,   "pixel_dist/F"   );  
    }   
    
  }

  void FindEmpties::reset(){
  
    _image_index = 0;
    _plane = -1; 
    _pixel_count = 0;
    _pixel_less_count = 0;
    _pixel_sum = 0;
    _max_pixel   = 0.;
    _dist_v.clear();
    _pix_intens_v.clear();
    _max_dist = -1. ;

    _pixel_intens = 0. ;
    _n_contours = 0;
    _tot_area = 0;
    _tot_height = 0.;
    _tot_width = 0.;
    _max_area = 0.;
    _max_height = 0.;
    _width_at_max_height = 0.; 
    _max_charge = 0.; 
    }

  bool FindEmpties::process(IOManager& mgr)
  {

    auto my_event_image2d = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_name));
    auto const& img2d_v = my_event_image2d->Image2DArray();

    //std::cout<<"\n\nEvent number: "<<_event <<std::endl;
    //std::cout<<std::endl ;

    for(size_t index=0; index < img2d_v.size(); ++index) {
      //std::cout<<"Plane : "<<index<<std::endl;
      //if (index != 0 ) continue ;

      reset() ;

      auto const& img2d = img2d_v[index];
      auto const& pixel_array = img2d.as_vector(); 
      auto const& meta = img2d.meta() ;
      //std::cout<<"Pixel width and height :" <<meta.pixel_height() <<", "<<meta.pixel_width() <<std::endl ;

      auto img = as_mat(img2d) ; 
      ::cv::cvtColor(img, img, CV_RGB2GRAY);
      
      ::cv::Mat sb_img; //(s)mooth(b)inary image

      auto _dilation_size = 5 ;
      auto _dilation_iter = 2 ;

      auto kernel = ::cv::getStructuringElement(cv::MORPH_ELLIPSE,::cv::Size(_dilation_size,_dilation_size));
      ::cv::dilate(img,sb_img,kernel,::cv::Point(-1,-1),_dilation_iter);
      ::cv::blur(sb_img,sb_img,::cv::Size(5,5));
      ::cv::threshold(sb_img,sb_img,0.75,255,CV_THRESH_BINARY); //return type is "threshold value?"

      //Contour finding
      std::vector<std::vector<::cv::Point>> ctor_v;
      std::vector<::cv::Vec4i> cv_hierarchy_v;
      ctor_v.clear(); cv_hierarchy_v.clear();
    
      ::cv::findContours(sb_img,ctor_v,cv_hierarchy_v,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
      //std::cout<<" Number of contours: "<<ctor_v.size()<<std::endl ;
      _n_contours = ctor_v.size();

      int max_area_index = -1 ;     
 
      for(size_t j = 0; j < ctor_v.size(); ++j){
        //std::cout<<::cv::contourArea(ctor_v[j]) <<", " ;

        if ( ::cv::contourArea(ctor_v[j]) > _max_area ){
          _max_area = ::cv::contourArea(ctor_v[j])  ;
          max_area_index = j ;
         }

        _tot_area+=::cv::contourArea(ctor_v[j]);
        cv::RotatedRect rect0 = ::cv::minAreaRect(cv::Mat(ctor_v[j]));
        cv::Point2f vertices[4];
        rect0.points(vertices);
        auto rect = rect0.size;

        auto temp_height = rect.height > rect.width ? rect.height : rect.width;
        auto temp_width  = rect.height > rect.width ? rect.width  : rect.height;
        
        if ( temp_height > _max_height ){
          _max_height = temp_height ;
          _width_at_max_height = temp_width ;
          }
        
        _tot_height += temp_height; 
        _tot_width += temp_width; 

       //_perimeter = ::cv::arcLength(ctor_v[j],1);
       }

     if (ctor_v.size() ){
     std::vector<::cv::Point> all_locations;
     ::cv::findNonZero(img, all_locations); // get the non zero points

     for( const auto& loc: all_locations ) { 

       if ( ::cv::pointPolygonTest(ctor_v[max_area_index],loc,false) < 0 ) 
           continue;
       
       _max_charge += (int) img.at<uchar>(loc.y, loc.x);
       }
      }


      _plane = int(img2d.meta().plane());
      _image_index = index;

      int max_pixel_index = -1; 
      _dist_v.reserve(pixel_array.size());
      _pix_intens_v.reserve(pixel_array.size());

      for(size_t i = 0; i < pixel_array.size(); i++){
        auto const & v = pixel_array[i] ;
        if( v > 0.5) 
	_pixel_sum += v ;

        if(v > _max_pixel){
          _max_pixel = v;
          max_pixel_index = i;
          }   

	if(v > 0.5 && v <=_pixel_count_threshold) _pixel_less_count++ ;
        if(v > 0.5 ) _pixel_count++;

          }   

       int pix_row = max_pixel_index % meta.rows() ; 
       int pix_col = max_pixel_index / meta.rows() ; 

        std::map<int,int> intens_count ;

        _pixel_dist = -1; 
        /// At this point, have foudn max pixel
        for(int r = 0; r < meta.rows(); r++){
          for(int c = 0; c < meta.cols(); c++){

            _pixel_intens = img2d.pixel(r,c);

            if (_pixel_intens >  1){
              intens_count[int(_pixel_intens)] ++ ;
	      //std::cout<<"Row and column? "<<meta.pos_y(r)<<", "<<meta.pos_x(c) <<std::endl ;
	      }

            if( _pixel_intens > 0.5 ){
              _pixel_dist = sqrt( pow((r - pix_row) * meta.pixel_height(),2)
                          + pow((c - pix_col) * meta.pixel_width(),2) ) ;
              if ( _pixel_dist > _max_dist )
                _max_dist = _pixel_dist ;
              
              _dist_v.emplace_back(_pixel_dist);
              _pix_intens_v.emplace_back(_pixel_intens);

              _pixel_tree->Fill();

               }
           }
         }

	 //if ( _plane == 0 && _pixel_count < 6 && _pixel_less_count < 60 && (_max_dist < 1002 || _max_dist > 3390) )
	  // return false ;


        // for(auto const & m : intens_count)
        //   std::cout<<"Intensity "<<m.first<<" has  "<<m.second <<" entries; " <<std::endl; 

      //LARCV_DEBUG() << "pixel max value: " << _max_pixel << std::endl;
      _image_tree->Fill();

      if ( _pixel_count == 0) 
        std::cout<<"0 Pixels for event: "<<_event<<",  in plane :"<<index <<std::endl ;

      //std::cout<<_pixel_count<<" count for event: "<<_event<<",  in plane :"<<index <<std::endl ;

    }
    

    _event++; 

    return true;
  }

  void FindEmpties::finalize()
  {
    // If an analysis output file is configured to exist, write TTree into an output
    if(has_ana_file()){
      _image_tree->Write();
      _pixel_tree->Write();
      }
  }

}
#endif
