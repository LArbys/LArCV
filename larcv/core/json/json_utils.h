#include "nlohmann/json.hpp"
#include "larcv/core/DataFormat/Image2D.h"

namespace larcv {

  namespace json {

    // for convenience
    typedef nlohmann::json json;
    
    json as_json( const larcv::Image2D& img );
    json as_json( const larcv::ImageMeta& meta );

    std::vector<std::uint8_t> as_bson( const larcv::Image2D& img );
    std::string as_json_str( const larcv::Image2D& img );

    larcv::Image2D   image2d_from_json( const json& j );
    larcv::ImageMeta imagemeta_from_json( const json& j );

    larcv::Image2D image2d_from_bson( const std::vector<std::uint8_t>& b );
    larcv::Image2D image2d_from_json_str( const std::string& s );    

    // this hack is needed for some reason
    class load_jsonutils {
    public:
      load_jsonutils(){};
      ~load_jsonutils(){};
    };
  }

}
