/**
 * \file ImageModUtils.h
 *
 * \ingroup Package_Name
 * 
 * \brief header for utility functions
 *
 * @author kazuhiro
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __IMAGEMODUTILS_H__
#define __IMAGEMODUTILS_H__

#include "DataFormat/Image2D.h"
#include "DataFormat/Pixel2DCluster.h"

namespace larcv {

  Image2D as_image2d(const Pixel2DCluster& pcluster);
  Image2D as_image2d(const Pixel2DCluster& pcluster, size_t target_rows, size_t target_cols);

}

#endif
/** @} */ // end of doxygen group 

