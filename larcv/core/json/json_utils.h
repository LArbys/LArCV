#include "nlohmann/json.hpp"
#include "larcv/core/DataFormat/Image2D.h"

#ifdef HASPYUTIL
#include "larcv/core/PyUtil/PyUtils.h"
#endif

namespace larcv {

  namespace json {

    // for convenience
    typedef nlohmann::json json;
    
    json as_json( const larcv::Image2D& img,
                  int run=0, int subrun=0, int event=0, int id=0 );
    json as_json( const larcv::ImageMeta& meta );

    std::vector<std::uint8_t> as_bson( const larcv::Image2D& img,
                                       int run=0, int subrun=0, int event=0, int id=0 );
    std::string as_json_str( const larcv::Image2D& img,
                             int run=0, int subrun=0, int event=0, int id=0 );
    larcv::Image2D   image2d_from_json( const json& j );
    larcv::ImageMeta imagemeta_from_json( const json& j );
    void rseid_from_json( const json& j,
                          int& run, int& subrun, int& event, int& id );
         

    larcv::Image2D image2d_from_bson( const std::vector<std::uint8_t>& b );
    larcv::Image2D image2d_from_json_str( const std::string& s );    

    json as_json_pixelarray( const larcv::Image2D&, const float );
    json as_json_pixelarray_withselection( const larcv::Image2D&,
                                           const larcv::Image2D&,
                                           const float );
    std::vector<std::uint8_t> as_bson_pixelarray( const larcv::Image2D& img,
                                                  const float threshold );
    std::vector<std::uint8_t> as_bson_pixelarray_withselection( const larcv::Image2D& img,
                                                                const larcv::Image2D& selectimg,
                                                                const float threshold );
    larcv::Image2D image2d_from_bson_pixelarray( const std::vector<std::uint8_t>& bson );
    json json_from_bson( const std::vector<std::uint8_t>& bson );
   
#ifdef HASPYUTIL
    PyObject* as_pystring( const larcv::Image2D& img );
    larcv::Image2D image2d_from_pystring( PyObject* str );
    larcv::Image2D image2d_from_pystring( PyObject* str,
                                          int& run, int& subrun, int& event, int& id);
#endif
    
    // this hack is needed for some reason
    class load_jsonutils {
    public:
      load_jsonutils(){};
      ~load_jsonutils(){};
    };
    
  }
}

