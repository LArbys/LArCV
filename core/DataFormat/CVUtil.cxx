#ifndef __LARCV_CVUTIL_CXX__
#define __LARCV_CVUTIL_CXX__

#include "CVUtil.h"

namespace larcv {

  cv::Mat as_mat(const Image2D& larcv_img)
  {
    auto const& meta = larcv_img.meta();
    cv::Mat img(meta.rows(),meta.cols(),CV_8UC3);
    
    unsigned char* px_ptr = (unsigned char*)img.data;
    int cn = img.channels();
    
    for(int i=0;i<meta.rows();i++) {
      for (int j=0;j<meta.cols();j++) {
	
	float q = larcv_img.pixel(i,j);
	px_ptr[i*img.cols*cn + j*cn + 0] = (unsigned char)(((int)(q+0.5)));
	px_ptr[i*img.cols*cn + j*cn + 1] = (unsigned char)(((int)(q+0.5))/256);
	px_ptr[i*img.cols*cn + j*cn + 2] = (unsigned char)(((int)(q+0.5))/256/256);
      }
    }
    return img;
  }

}

#endif
