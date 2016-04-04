#ifndef __LARCV_IOMANAGERTEMPLATES_H__
#define __LARCV_IOMANAGERTEMPLATES_H__

#include "IOManager.h"

#include "Image2D.h"
template class larcv::IOManager<larcv::Image2D>;

#include "ImageMeta.h"
template class larcv::IOManager<larcv::ImageMeta>;

#endif
