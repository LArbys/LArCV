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

  Image2D imread(const std::string file_name)
  {
    ::cv::Mat image;
    image = ::cv::imread(file_name.c_str(), CV_LOAD_IMAGE_COLOR);

    ImageMeta meta(image.cols,image.rows,image.cols, image.rows, 0., 0.);
    Image2D larcv_img(meta);
      
    unsigned char* px_ptr = (unsigned char*)image.data;
    int cn = image.channels();
    
    for(int i=0;i<image.rows;i++) {
      for (int j=0;j<image.cols;j++) {
	float q = 0;
	q += (float)(px_ptr[i*image.cols*cn + j*cn + 0]);               //B
	q += (float)(px_ptr[i*image.cols*cn + j*cn + 1]) * 256.;        //G
	q += (float)(px_ptr[i*image.cols*cn + j*cn + 2]) * 256. * 256.; //R
	larcv_img.set_pixel(j,i,q);
      }
    }
    return larcv_img;
  }

  Image2D imread_gray(const std::string file_name)
  {
    ::cv::Mat image;
    image = ::cv::imread(file_name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

    ImageMeta meta(image.cols,image.rows,image.cols, image.rows, 0., 0.);
    Image2D larcv_img(meta);
      
    unsigned char* px_ptr = (unsigned char*)image.data;

    for(int i=0;i<image.rows;i++) {
      for (int j=0;j<image.cols;j++) {
	float q = 0;
	q += (float)(px_ptr[i*image.cols + j]);
	larcv_img.set_pixel(j,i,q);
      }
    }
    return larcv_img;
  }


}

#endif
