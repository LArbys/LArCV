#include "json_utils.h"

namespace larcv {

  namespace json {

    /**
     * create a json object from an image2d object
     *
     */
    json as_json( const larcv::Image2D& img, int run, int subrun, int event, int id ) {

      // we stuff the data into json format.
      // we provide optional run,subrun,event,id
      // to help track if the images coming back
      //  are the ones we sent for the event
      //
      // sometimes if we a client sends out a job, gets disconnet and resends the job,
      // it can get replies from the worker twice which can
      // cause ssnet jobs from going out of sync
      std::vector<int> rseid(4);
      rseid[0] = run;
      rseid[1] = subrun;
      rseid[2] = event;
      rseid[3] = id;

      json j;
      j["meta"]  = as_json( img.meta() );
      j["data"]  = img.as_vector();
      j["rseid"] = rseid;
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
    std::vector<std::uint8_t> as_bson( const larcv::Image2D& img,
                                       int run, int subrun, int event, int id) {
      return json::to_bson( as_json(img,run,subrun,event,id) );
    }

    /*
     * create a json message from an image2d (useful for python)
     *
     * @param[in] img Image2D to be serialized
     * @return std::string in JSON format
     */
    std::string as_json_str( const larcv::Image2D& img,
                             int run, int subrun, int event, int id ) {
      return as_json(img,run,subrun,event,id).dump();
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
     * get run, subrun, event, id from json
     *
     */
    void rseid_from_json( const json& j,
                          int& run, int& subrun, int& event, int& id ) {
      std::vector<int> rseid = j["rseid"].get< std::vector<int> >();
      run    = rseid[0];
      subrun = rseid[1];
      event  = rseid[2];
      id     = rseid[3];
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

    /*
     * create a json object for an image2d represented as a pixel list
     *
     * the pixel list is stored in a single 1D vector<float>
     *  which stores 3 numbers for each pixel: (row,col,value)
     *
     * @param[in] img Image2D to convert
     * @param[in] pixels with value>threshold are stored in the list
     * return json object with keys "meta" and "data"
     */
    json as_json_pixelarray( const larcv::Image2D& img, const float threshold ) {
      json j;
      j["meta"] = as_json( img.meta() );

      std::vector< float > pixelarray;
      pixelarray.reserve( img.as_vector().size() );
      for ( size_t c=0; c<img.meta().cols(); c++ ) {
        for ( size_t r=0; r<img.meta().rows(); r++ ) {
          float val = img.pixel(r,c);
          if ( val>threshold ) {
            pixelarray.push_back( (float)r );
            pixelarray.push_back( (float)c );
            pixelarray.push_back( val );
          }
        }
      }
      j["pixelarray"] = pixelarray;

      return j;
    }

    /*
     * create a json object stored as pixel list, chosing pixels with values from another image.
     *
     * the pixel list is stored in a single 1D vector<float>
     *  which stores 3 numbers for each pixel: (row,col,value)
     *
     * @param[in] img Image2D to convert
     * @param[in] selectimg Image2D with which we do the selection
     * @param[in] pixels with value>threshold are stored in the list
     * return json object with keys "meta" and "data"
     */
    json as_json_pixelarray_withselection( const larcv::Image2D& img,
                                           const larcv::Image2D& selectimg, float threshold ) {

      // check that the number of rows and columns are the same
      if ( selectimg.meta().rows()!=img.meta().rows()
           || selectimg.meta().cols()!=img.meta().cols() ) {
        std::stringstream ss;
        ss << __FUNCTION__ << ":" << __FILE__ << "." << __LINE__ << ":: "
           << "rows and cols of source and selection image must be the same" << std::endl;
        throw std::runtime_error(ss.str());
      }
      json j;
      j["meta"] = as_json( img.meta() );

      std::vector< float > pixelarray;
      pixelarray.reserve( img.as_vector().size() );
      for ( size_t c=0; c<img.meta().cols(); c++ ) {
        for ( size_t r=0; r<img.meta().rows(); r++ ) {
          float	selectval = selectimg.pixel(r,c);
          if ( selectval>threshold ) {
            pixelarray.push_back( (float)r );
            pixelarray.push_back( (float)c );
            pixelarray.push_back( img.pixel(r,c) );
          }
        }
      }
      j["pixelarray"] = pixelarray;

      return j;
    }

    /*
     * create bson object containing image2d data represented as pixel list
     *
     * @param[in] img source image2d to convert
     * @param[in] threshold only keep pixels above this threshold
     * @return vector of uint8_t that stores the binary data
     *
     */
    std::vector<std::uint8_t> as_bson_pixelarray( const larcv::Image2D& img,
                                                  const float threshold ) {
      return json::to_bson( as_json_pixelarray(img,threshold) );
    }

    /*
     * create bson object containing image2d data represented as pixel list, with selection
     *
     * @param[in] img source image2d to convert
     * @param[in] selectimg image we use to determine which pixels to keep
     * @param[in] threshold only keep pixels above this threshold
     * @return vector of uint8_t that stores the binary data
     *
     */
    std::vector<std::uint8_t> as_bson_pixelarray_withselection( const larcv::Image2D& img,
                                                                const larcv::Image2D& selectimg,
                                                                const float threshold ) {
      return json::to_bson( as_json_pixelarray_withselection(img,selectimg,threshold) );
    }

    larcv::Image2D image2d_from_bson_pixelarray( const std::vector<std::uint8_t>& bson ) {
      // get back to json
      json j = json::from_bson(bson);
      // get the image meta
      larcv::ImageMeta meta = imagemeta_from_json( j["meta"] );
      // get the vector of values
      std::vector<float> data = j["pixelarray"].get< std::vector<float> >();
      // make a blank image2d
      larcv::Image2D img2d(meta);
      // fill the image
      size_t npts = data.size()/3;
      if ( data.size()%3!=0 ) {
        std::stringstream ss;
        ss << __FUNCTION__ << ":" << __FILE__ << "." << __LINE__ << " :: "
           << "vector representing pixel array should be mutiple of 3 with (row,col,value)."
           << " vecsize=" << data.size()
           << " vec%3=" << data.size()%3
           << std::endl;
        throw std::runtime_error(ss.str());
      }
      for ( size_t ipt=0; ipt<npts; ipt++ ) {
        int row   = (int)data[ 3*ipt ];
        int col   = (int)data[ 3*ipt+1 ];
        float val = (float)data[ 3*ipt ];
        img2d.set_pixel( row, col, val );
      }

      return img2d;
    }

    /**
     * convert bson into json.
     *
     * providing wrapper in order to include into python
     *
     * @param[in] bson binary data in a vector<uint8_t>
     *
     * return json object
     *
     */
    json json_from_bson( const std::vector<std::uint8_t>& bson ) {
      return json::from_bson(bson);
    }

#ifdef HASPYUTIL
    PyObject* as_pystring( const larcv::Image2D& img,
      int run, int subrun, int event, int id ) {
      std::vector<std::uint8_t> b_v = as_bson( img, run, subrun, event, id );
      return larcv::as_pystring( b_v ); //from pyutils
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

    larcv::Image2D image2d_from_pystring( PyObject* str,
                                          int& run, int& subrun, int& event, int& id ) {
      if ( PyString_Check( str )==0 ) {
        logger::get("json_utils").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__, "Error not a PyString object" );
      }
      size_t len = PyString_Size(str);
      std::vector<std::uint8_t> b_v(len,0);
      memcpy( b_v.data(), (unsigned char*)PyString_AsString(str), sizeof(unsigned char)*len );

      json data = json::from_bson(b_v);
      rseid_from_json( data, run, subrun, event, id );

      return image2d_from_json( data );
    }
#endif

  }//end of json namespace
}//end of larcv namespace
