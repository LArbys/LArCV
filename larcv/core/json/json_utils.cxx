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
     * create a json object from an clustermask object.
     *
     * @param[in] mask ClusterMask object to serialize
     * @param[in] run Run number
     * @param[in] subrun Subrun number
     * @param[in] event Event number
     * @param[in] id Integer provided to user to label object in this event
     * @return json message
     */
    json as_json( const larcv::ClusterMask& mask, int run, int subrun, int event, int id ) {

      // we stuff the data into json format.
      // we provide optional run,subrun,event,id
      // to help track if the images coming back
      //  are the ones we sent for the event
      //
      // sometimes if we a client sends out a job, gets disconnet and resends the job,
      // it can get replies from the worker twice which can
      // cause  jobs from going out of sync
      std::vector<int> rseid(4);
      rseid[0] = run;
      rseid[1] = subrun;
      rseid[2] = event;
      rseid[3] = id;

      json j;
      j["prob"] = mask.probability_of_class;
      j["meta"]  = as_json( mask.meta );
      j["box"]  = mask.as_vector_box_no_convert();
      j["type"] = mask.type;
      j["points_v"] = mask.as_vector_mask_no_convert();
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

    std::vector<std::uint8_t> as_bson( const larcv::ClusterMask& mask,
                                       int run, int subrun, int event, int id) {
      return json::to_bson( as_json(mask,run,subrun,event,id) );
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
     * create image2d from json.
     *
     * @param[in] j JSON object with image2d info.
     * @return image2d from json message
     *
     */
    larcv::Image2D image2d_from_json( const json& j ) {
      larcv::ImageMeta meta = imagemeta_from_json( j["meta"] );
      larcv::Image2D img2d( meta, j["data"].get<std::vector<float>>() );
      return img2d;
    }

    /**
     * create ClusterMask from json.
     *
     * @param[in] j JSON object with image2d info.
     * @return ClusterMask from json message
     *
     */
    larcv::ClusterMask clustermask_from_json( const json& j ) {
      larcv::ImageMeta meta = imagemeta_from_json( j["meta"] );
      larcv::BBox2D dummy_box(1600,5000,1600,5000,kINVALID_PROJECTIONID);
      std::vector<larcv::Point2D> dummy_v(0,Point2D(0,0));
      larcv::ClusterMask cmask(dummy_box, meta, dummy_v, 0);

      cmask.probability_of_class = j["prob"].get<float>();
      cmask.meta = meta;
      std::vector<float> box_v = j["box"].get<std::vector<float>>();
      cmask.type = j["type"].get<InteractionID_t>();
      std::vector<float> pts_v_float = j["points_v"].get<std::vector<float>>();
      cmask.box = larcv::BBox2D((double)box_v.at(0) , (double)box_v.at(1) , (double)box_v.at(2), (double)box_v.at(3));
      size_t num_pts = pts_v_float.size() / 2 ;
      std::vector<Point2D> pts_v(num_pts, Point2D() );
      for (int i=0; i < num_pts; ++i){
        pts_v.at(i) = Point2D(pts_v_float[2*i], pts_v_float[2*i+1]);
      }
      cmask.points_v = pts_v;

      return cmask;
    }

    /**
     * 
     * create ClusterMask from json.
     *
     * provide function with same name as other types.
     *
     * @param[in] j JSON object with image2d info.
     * @param[out] mask_v container to return ClusterMask
     *
     */
    void from_json( const json& j, std::vector<larcv::ClusterMask>& mask_v ) {
      larcv::ClusterMask mask = clustermask_from_json( j );
      mask_v.emplace_back( std::move(mask) );
    }

    /**
     * get run, subrun, event, id from json
     * 
     * @param[in] j json object
     * @param[inout] run Run number
     * @param[inout] subrun Subrun number
     * @param[inout] event Event number
     * @param[inout] id Image ID number
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
     * @param[in] j JSON object with ImageMeta info in it
     * @return ImageMeta object
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

    /**
     * convert bson to image2d.
     *
     * @param[in]  b buffer of bytes containing binary json message
     * @return image2d serialized in binary json
     *
     */
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


    /**
     * convert json to image2d.
     *
     * provides function call with same name as the other types.
     *
     * @param[in]  j json object containing image2d data
     * @param[out] image2d serialized in binary json
     *
     */
    void from_json( const json& j, std::vector<larcv::Image2D>& img_v ) {
      larcv::Image2D img = image2d_from_json( j );
      img_v.emplace_back( std::move(img) );
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

    // =======================================================
    // larcv::SparseImage methods
    // =======================================================

    /**
       convert SparseImage object into json message

       @param[in] sparsedata The data to convert
       @param[in] run run number to store in bson message
       @param[in] subrun subrun number to store in bson message
       @param[in] event event number to store in bson message
       @param[in] id additional user index that can be stored in bson message

       @return json object
     */
    json as_json( const larcv::SparseImage& sparsedata,
                  int run, int subrun, int event, int id)
    {
      // get the parameters of the sparse data
      int nfeatures = sparsedata.nfeatures();
      int stride    = 2+nfeatures;
      int npts      = sparsedata.pixellist().size()/stride;

      if ( (int)sparsedata.pixellist().size()%stride!=0 ) {
        std::stringstream msg;
        msg << "json_utils.cxx::as_bson_pybytes:"
            << " calculated stride (" << stride << ") per point does "
            << " not divide evenly into "
            << " data array size (" << sparsedata.pixellist().size() << ")"
            << std::endl;
        logger::get("json_utils").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__, msg.str() );
        throw std::runtime_error(msg.str());
      }

      // each feature plane has a meta. we store a vector of metas for these

      std::vector<json> meta_v;
      for ( size_t ifeat=0; ifeat<nfeatures; ifeat++ ) {
        json jmeta = as_json( sparsedata.meta(ifeat) );
        meta_v.push_back( jmeta );
      }

      // output msg
      json j;

      // key params
      j["datatype"] = "larcv::SparseImage";
      j["nfeatures"] = nfeatures;
      j["npts"] = npts;
      j["index"] = sparsedata.index();

      // output meta
      j["meta_v"] = meta_v;

      // run, subrun, event, id
      std::vector<int> rseid(4);
      rseid[0] = run;
      rseid[1] = subrun;
      rseid[2] = event;
      rseid[3] = id;
      j["rseid"] = rseid;

      // the data
      j["data"] = sparsedata.pixellist();

      return j;
    }

    /**
       convert SparseImage object into bson (binary-json) message

       @param[in] sparsedata The data to convert
       @param[in] run run number to store in bson message
       @param[in] subrun subrun number to store in bson message
       @param[in] event event number to store in bson message
       @param[in] id additional user index that can be stored in bson message

       @return bson object as binary array
    */
    std::vector<std::uint8_t> as_bson( const larcv::SparseImage& sparsedata,
                                       int run, int subrun, int event, int id)
    {
      return json::to_bson( as_json(sparsedata,run,subrun,event,id) );
    }

    /**
     * convert SparseImage json message back into a SparseImage object
     *
     * @param[in] json message containng sparseimage data.
     * @return SparseImage from message.
     */
    larcv::SparseImage sparseimg_fromjson( const json& msg )
    {

      //int& run, int& subrun, int& event, int& id) {
      std::vector<float> data = msg["data"].get< std::vector<float> >();
      std::vector< json > jmeta_v = msg["meta_v"].get< std::vector<json> >();
      std::vector< larcv::ImageMeta > meta_v;
      for ( auto const& jmeta : jmeta_v ) {
        meta_v.push_back( imagemeta_from_json( jmeta ) );
        //std::cout << "  " << meta_v.back().plane() << ", " << meta_v.back().dump() << std::endl;
      }

      int nfeatures = msg["nfeatures"];
      int npts      = msg["npts"];
      int index     = msg["index"];

      larcv::SparseImage sparseimg( nfeatures, npts, data, meta_v, index );
      return sparseimg;
    }


    /**
     * convert SparseImage json message back into a SparseImage object w RSEid
     *
     * @param[in] json message containng sparseimage data
     * @param[inout] run    get run number from messaeg
     * @param[inout] subrun get subrun number from message
     * @param[inout] event  get event number from message
     * @param[inout] id     get id number from message
     */
    larcv::SparseImage sparseimg_fromjson( const json& msg,
                                           int& run, int& subrun, int& event, int& id )
    {
      larcv::SparseImage sparseimg = sparseimg_fromjson( msg );
      std::vector<int> rseid = msg["rseid"].get< std::vector<int> >();
      run    = rseid.at(0);
      subrun = rseid.at(1);
      event  = rseid.at(2);
      id     = rseid.at(3);
      return sparseimg;
    }


    /**
     * convert SparseImage json message back into a SparseImage object.
     *
     * provide function with same name as other types.
     *
     * @param[in] json message containng sparseimage data.
     * @return SparseImage from message.
     */
    void from_json( const json& msg, std::vector<larcv::SparseImage>& spimg_v )
    {
      larcv::SparseImage spimg = sparseimg_fromjson( msg );
      spimg_v.emplace_back( std::move(spimg) );
    }


#ifdef HASPYUTIL
    /**
     * function returns image2d as binary data in pybytes object
     * 
     */
    PyObject* as_pybytes( const larcv::Image2D& img,
      int run, int subrun, int event, int id ) {
      std::vector<std::uint8_t> b_v = as_bson( img, run, subrun, event, id );
      return larcv::as_pybytes( b_v ); //from pyutils
    }

    PyObject* as_pybytes( const larcv::ClusterMask& mask,
      int run, int subrun, int event, int id ) {
      std::vector<std::uint8_t> b_v = as_bson( mask, run, subrun, event, id );
      return larcv::as_pybytes( b_v ); //from pyutils
    }

    larcv::Image2D image2d_from_pybytes( PyObject* bytes ) {
      if ( PyBytes_Check( bytes )==0 ) {
        logger::get("json_utils").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__, "Error not a PyBytes object" );
      }
      size_t len = PyBytes_Size(bytes);
      std::vector<std::uint8_t> b_v(len,0);
      memcpy( b_v.data(), (unsigned char*)PyBytes_AsString(bytes), sizeof(unsigned char)*len );

      return image2d_from_bson( b_v );
    }

    larcv::Image2D image2d_from_pybytes( PyObject* bytes,
                                          int& run, int& subrun, int& event, int& id ) {
      if ( PyBytes_Check( bytes )==0 ) {
        logger::get("json_utils").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__, "Error not a PyBytes object" );
      }
      size_t len = PyBytes_Size(bytes);
      std::vector<std::uint8_t> b_v(len,0);
      memcpy( b_v.data(), (unsigned char*)PyBytes_AsString(bytes), sizeof(unsigned char)*len );

      json data = json::from_bson(b_v);
      rseid_from_json( data, run, subrun, event, id );

      return image2d_from_json( data );
    }


    // ======================
    // SPARSEIMAGE
    // ======================

    /**
       convert SparseImage object into pybytes object containing binary data stored in BSON format

       @param[in] sparsedata The data to convert
       @param[in] run run number to store in bson message
       @param[in] subrun subrun number to store in bson message
       @param[in] event event number to store in bson message
       @param[in] id additional user index that can be stored in bson message

       @return bson pybytes
     */
    PyObject* as_bson_pybytes( const larcv::SparseImage& sparsedata,
                                int run, int subrun, int event, int id)
    {
      return larcv::as_pybytes( as_bson( sparsedata, run, subrun, event, id ) );
    }

    /**
     * convert pybytes containing bson msg back into SparseImage object
     *
     * @param[in] sparsedata SparseImage object to convert
     * @param[in] run run number to store in message
     * @param[in] subrun subrun number to store in message
     * @param[in] event event number to store in message
     * @param[in] id use-defined index number to store in message
     *
     * @return SparseImage object
     */
    larcv::SparseImage sparseimg_from_bson_pybytes( PyObject* msg,
                                                    int& run, int& subrun, int& event, int& id)
    {

      if ( PyBytes_Check( msg )==0 ) {
        logger::get("json_utils").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__, "Error not a PyBytes object" );
      }
      size_t len = PyBytes_Size(msg);
      std::vector<std::uint8_t> b_v(len,0);
      memcpy( b_v.data(), (unsigned char*)PyBytes_AsString(msg), sizeof(unsigned char)*len );

      json data = json::from_bson(b_v);
      rseid_from_json( data, run, subrun, event, id );
      //std::cout << "rse: " << run << " " << subrun << " " << event << std::endl;

      return sparseimg_fromjson(data);
    }

    // ==========================
    // CLUSTERMASK
    // ==========================

    larcv::ClusterMask clustermask_from_pybytes( PyObject* bytes,
                                          int& run, int& subrun, int& event, int& id ) {
      if ( PyBytes_Check( bytes )==0 ) {
        logger::get("json_utils").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__, "Error not a PyBytes object" );
      }
      size_t len = PyBytes_Size(bytes);
      std::vector<std::uint8_t> b_v(len,0);
      memcpy( b_v.data(), (unsigned char*)PyBytes_AsString(bytes), sizeof(unsigned char)*len );

      json data = json::from_bson(b_v);
      rseid_from_json( data, run, subrun, event, id );

      return clustermask_from_json( data );
    }

    // END OF HASPYUTIL: Block for Python functions
#endif

  }//end of json namespace
}//end of larcv namespace
