#ifndef __DBSCAN_CXX__
#define __DBSCAN_CXX__

#include "DBSCAN.h"
#include "DataFormat/DataFormatTypes.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/ImageMeta.h"
#include "DBSCANAlgo.h"

namespace larcv {

  static DBSCANProcessFactory __global_DBSCANProcessFactory__;

  DBSCAN::DBSCAN(const std::string name)
    : ProcessBase(name)
  {
  }
    
  void DBSCAN::configure(const PSet& cfg)
  {
    finput_producer  = cfg.get<std::string>("InputImageProducer");
    foutput_producer = cfg.get<std::string>("OutputImageProducer");
    feps             = cfg.get< std::vector<double> >("planeNeighborRadius");
    fminpoints       = cfg.get< std::vector<int> >("planeMinPoints");
    fthreshold       = cfg.get< std::vector<double> >("adchitthreshold");
  }

  void DBSCAN::initialize()
  {}

  bool DBSCAN::process(IOManager& mgr)
  {
    larcv::EventImage2D* input_imgs  = (larcv::EventImage2D*)mgr.get_data( larcv::kProductImage2D, finput_producer );
    larcv::EventImage2D* output_imgs = (larcv::EventImage2D*)mgr.get_data( larcv::kProductImage2D, foutput_producer );

    // go through each plane.
    std::vector< larcv::Image2D > output_images;
    for ( auto const& img : input_imgs->Image2DArray() ) {
      const larcv::ImageMeta& meta = img.meta();
      int plane     = (int)meta.plane();
      double thresh = fthreshold.at(plane);
      double eps    = feps.at(plane);
      int minpoints = fminpoints.at(plane);

      // find hits.
      dbscan::dbPoints pixels;
      for (int c=0; c<meta.cols(); c++) {
	for (int r=0; r<meta.rows(); r++) {
	  if ( img.pixel( c, r )>thresh ) {
	    std::vector<double> pt(2,0.0);
	    pt.at(0) = c;
	    pt.at(1) = r;
	    pixels.emplace_back(pt);
	  }
	}
      }

      dbscan::DBSCANAlgo algo;
      dbscan::dbscanOutput clusterout = algo.scan( pixels, minpoints, eps );

      // make output image
      larcv::Image2D imgout( meta );
      imgout.paint(0.0);

      // label background as 1
      // rest increment cluster i+1
      for (int ic=0; ic<clusterout.clusters.size(); ic++) {
	const std::vector< int >& cl = clusterout.clusters.at(ic);
	for ( auto& idx : cl ) {
	  const std::vector<double>& pix = pixels.at(idx);
	  if ( cl.size()>1 )
	    imgout.set_pixel( (int)(pix.at(0)+0.1), (int)(pix.at(1)+0.1), ic+1 );
	  else
	    imgout.set_pixel( (int)(pix.at(0)+0.1), (int)(pix.at(1)+0.1), 1 );
	}
      }

      output_images.emplace_back( imgout );
    }

    output_imgs->Emplace( std::move(output_images) );
  }//end of analyze
  
  void DBSCAN::finalize()
  {}

}
#endif
