#ifndef LARBYSIMAGEMAKER_CXX
#define LARBYSIMAGEMAKER_CXX

#include "LArbysImageMaker.h"

namespace larcv {

  void
  LArbysImageMaker::Configure(const PSet& pset) {
    LARCV_DEBUG() << "start" << std::endl;
    this->set_verbosity((msg::Level_t)pset.get<int>("Verbosity",2));

    _charge_to_gray_scale = pset.get<double>("Q2Gray");
    _charge_max = pset.get<double>("QMax");
    _charge_min = pset.get<double>("QMin");
    
    LARCV_DEBUG() << "end" << std::endl;
    return;
  }

  
  std::vector<std::tuple<cv::Mat,larocv::ImageMeta> >
  LArbysImageMaker::ExtractImage(IOManager& mgr,std::string producer) {

    LARCV_DEBUG() << "Extracting " << producer << " Image\n" << std::endl;
    
    EventImage2D* ev_image = nullptr;
    
    if(!producer.empty()) {
      ev_image = (EventImage2D*)(mgr.get_data(kProductImage2D,producer));
      if(!ev_image) {
	LARCV_CRITICAL() << "Image by producer " << producer << " not found..." << std::endl;
	throw larbys();
      }
    }

    auto const& image_v = ev_image->Image2DArray();

    std::vector<std::tuple<cv::Mat,larocv::ImageMeta> > ret_v;
    ret_v.resize(image_v.size(),std::make_tuple(cv::Mat(),larocv::ImageMeta()));
    
    for(size_t i=0; i<image_v.size(); ++i) {
      auto& ret = ret_v[i];
      
      auto const& cvmeta = image_v[i].meta();
	
      LARCV_DEBUG() << "Reading  image (rows,cols) = (" << cvmeta.rows() << "," << cvmeta.cols() << ") "
		    << " ... (height,width) = (" << cvmeta.height() << "," << cvmeta.width() << ")" << std::endl;
	
      larocv::ImageMeta meta(cvmeta.width(), cvmeta.height(),
			     cvmeta.cols(),  cvmeta.rows(),
			     cvmeta.min_y(), cvmeta.min_x(), i);
	
      LARCV_DEBUG() << "LArOpenCV meta @ plane " << i << " ... "
		    << "Reading  image (rows,cols) = (" << meta.num_pixel_row() << "," << meta.num_pixel_column() << ") "
		    << " ... (height,width) = (" << meta.height() << "," << meta.width() << ")" << std::endl;

      std::get<0>(ret) = cv::Mat(cvmeta.cols(),cvmeta.rows(),CV_8UC1,cvScalar(0.));
      std::get<1>(ret) = meta;
      auto& mat  = std::get<0>(ret);
      
      for(size_t row=0; row<cvmeta.rows(); ++row) {
	for(size_t col=0; col<cvmeta.cols(); ++col) {
	  float charge = image_v[i].pixel(row,col);
	  charge -= _charge_min;
	  if(charge < 0) charge = 0;
	  if(charge > _charge_max) charge = _charge_max;
	  charge /= _charge_to_gray_scale;
	  mat.at<unsigned char>(col,cvmeta.rows()-1-row) = (unsigned char)((int)charge);
	}
      }
    } // end collection of images

    return ret_v;
  } // end extract image
}
#endif
