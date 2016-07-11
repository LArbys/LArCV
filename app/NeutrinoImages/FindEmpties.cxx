#ifndef __FINDEMPTIES_CXX__
#define __FINDEMPTIES_CXX__

#include "FindEmpties.h"
#include "DataFormat/EventImage2D.h"
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
    _max_pixel   = 0.;
    _pixel_intens = 0;
    _max_dist = -1. ;
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
      _image_tree->Branch( "max_pixel",   &_max_pixel,   "max_pixel/F"   );
      _image_tree->Branch( "pix_intens_v","std::vector<float>",&_pix_intens_v);
      _image_tree->Branch( "dist_v","std::vector<float>",&_dist_v);
      _image_tree->Branch( "max_dist",   &_max_dist,   "max_dist/F"   );
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
    _max_pixel   = 0.;
    _dist_v.clear();
    _pix_intens_v.clear();
    _max_dist = -1. ;

    _pixel_intens = 0. ;
    }

  bool FindEmpties::process(IOManager& mgr)
  {

    auto my_event_image2d = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_name));
    auto const& img2d_v = my_event_image2d->Image2DArray();

    std::cout<<"\n\nEvent number: "<<_event <<std::endl;

    for(size_t index=0; index < img2d_v.size(); ++index) {
      //std::cout<<"Plane : "<<index<<std::endl;
     
      reset() ;

      auto const& img2d = img2d_v[index];
      auto const& pixel_array = img2d.as_vector(); 
      auto const& meta = img2d.meta() ;

      _plane = int(img2d.meta().plane());
      _image_index = index;

      int max_pixel_index = -1; 
      _dist_v.reserve(pixel_array.size());
      _pix_intens_v.reserve(pixel_array.size());

      for(size_t i = 0; i < pixel_array.size(); i++){
        auto const & v = pixel_array[i] ;

        if(v > _max_pixel){
          _max_pixel = v;
          max_pixel_index = i;
          }   

	if(v > 0.5 && v <=_pixel_count_threshold) _pixel_less_count++ ;
        if(v > _pixel_count_threshold) _pixel_count++;

          }   

       int pix_row = max_pixel_index % meta.rows() ; 
       int pix_col = max_pixel_index / meta.rows() ; 

        std::map<int,int> intens_count ;

        _pixel_dist = -1; 
        /// At this point, have foudn max pixel
        for(int r = 0; r < meta.rows(); r++){
          for(int c = 0; c < meta.cols(); c++){

            _pixel_intens = img2d.pixel(r,c);

	   //if(_pixel_intens > 0.5 && index == 0) std::cout<<"Intensity at ("<<r<<", "<<c<<"): "<<_pixel_intens<<std::endl ;

            if (_pixel_intens >  1)
              intens_count[int(_pixel_intens)] ++ ;

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

         for(auto const & m : intens_count)
         //  std::cout<<"Intensity "<<m.first<<" has  "<<m.second <<" entries; " <<std::endl; 

   //  std::cout<<"Numbers: "<< _image_index<<", "<< _plane<<", "<<_pixel_count <<", "
   //                   <<_max_pixel<<", "<< _dist_v.size()<<", "<< _pix_intens_v.size()<<", "
   //                   <<_max_dist <<std::endl;

      //LARCV_DEBUG() << "pixel max value: " << _max_pixel << std::endl;
      _image_tree->Fill();
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
