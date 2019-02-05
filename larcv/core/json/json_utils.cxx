#include "json_utils.h"

namespace larcv {

  namespace json {

    /**
     * create a json object from an image2d object
     *
     */
    json as_json( const larcv::Image2D& img ) {

      // we stuff 
      
      json j;
      j["meta"] = as_json( img.meta() );
      j["data"] = img.as_vector();
      return j;
    }

    /** 
     * create a json object from an imagemeta object
     *
     */

    json as_json( const larcv::ImageMeta& meta ) {
      json j;
      j["min_x"] = meta.min_x();
      j["max_x"] = meta.max_x();
      j["min_y"] = meta.min_y();
      j["max_y"] = meta.max_y();
      j["rows"]  = (int)meta.rows();
      j["cols"]  = (int)meta.cols();
      j["pixheight"] = meta.pixel_height();
      j["pixwidth"]  = meta.pixel_width();
      j["id"]        = (int)meta.plane();
      return j;
    }

    /*
     * create a binary json message from an image2d
     *
     * @param[in] img Image2D to be serialized
     * @return vector<uint8_t> binary vector containing serialized img
     */
    std::vector<std::uint8_t> as_bson( const larcv::Image2D& img ) {
      return json::to_bson( as_json(img) );
    }

    /*
     * create a json message from an image2d (useful for python)
     *
     * @param[in] img Image2D to be serialized
     * @return std::string in JSON format
     */
    std::string as_json_str( const larcv::Image2D& img ) {
      return as_json(img).dump();
    }
    
    /**
     * create image2d from json
     *
     */
    larcv::Image2D image2d_from_json( const json& j ) {
      larcv::ImageMeta meta = imagemeta_from_json( j["meta"] );
      larcv::Image2D img2d( meta, j["data"].get<std::vector<float>>() );
      return img2d;
    }


    /**
     * create imagemeta from json
     * 
     * 
     */
    larcv::ImageMeta imagemeta_from_json( const json& j ) {
      larcv::ImageMeta meta( (j["max_x"].get<double>() - j["min_x"].get<double>()),
                             (j["max_y"].get<double>() - j["min_y"].get<double>()),
                             j["rows"].get<int>(),
                             j["cols"].get<int>(),
                             j["min_x"].get<double>(),
                             j["min_y"].get<double>(),
                             (larcv::PlaneID_t)j["id"].get<int>() );
      return meta;
    }

    larcv::Image2D image2d_from_bson( const std::vector<std::uint8_t>& b ) {
      json j = json::from_bson(b);
      return image2d_from_json( j );
    }

    /*
     * create an image2d from a json string
     *
     * @param[in] img Image2D to be serialized
     * @return std::string in JSON format
     */
    larcv::Image2D image2d_from_json_str( const std::string& s ) {
      return image2d_from_json( json::parse(s) );
    }
    

#ifdef HASPYUTIL
    PyObject* as_pystring( const larcv::Image2D& img ) {
      std::vector<std::uint8_t> b_v = as_bson( img );
      return larcv::as_pystring( b_v );
    }
    
    larcv::Image2D image2d_from_pystring( PyObject* str ) {
      if ( PyString_Check( str )==0 ) {
        logger::get("json_utils").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__, "Error not a PyString object" );
      }
      size_t len = PyString_Size(str);
      std::vector<std::uint8_t> b_v(len,0);
      memcpy( b_v.data(), (unsigned char*)PyString_AsString(str), sizeof(unsigned char)*len );
      
      return image2d_from_bson( b_v );
    }
#endif

  }//end of json namespace
}//end of larcv namespace
