#ifndef __INTERWRITEOUTLC_CXX__
#define __INTERWRITEOUTLC_CXX__

#include "InterWriteOutLC.h"

#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventPixel2D.h"

namespace larcv {

  static InterWriteOutLCProcessFactory __global_InterWriteOutLCProcessFactory__;

  InterWriteOutLC::InterWriteOutLC(const std::string name)
    : ProcessBase(name)
  {}
    
  bool InterWriteOutLC::process(IOManager& mgr) {

    larcv::EventPGraph* ev_inter_pgraph = nullptr;
    ev_inter_pgraph = (larcv::EventPGraph*)mgr.get_data(larcv::kProductPGraph,"inter_par");
    (void)ev_inter_pgraph;

    larcv::EventPixel2D* ev_inter_par_pixel;
    ev_inter_par_pixel = (larcv::EventPixel2D*)mgr.get_data(larcv::kProductPixel2D,"inter_par_pixel");
    (void)ev_inter_par_pixel;

    larcv::EventPixel2D* ev_inter_img_pixel;
    ev_inter_img_pixel = (larcv::EventPixel2D*)mgr.get_data(larcv::kProductPixel2D,"inter_img_pixel");
    (void)ev_inter_img_pixel;

    larcv::EventPixel2D* ev_inter_int_pixel;
    ev_inter_int_pixel = (larcv::EventPixel2D*)mgr.get_data(larcv::kProductPixel2D,"inter_int_pixel");
    (void)ev_inter_int_pixel;

    return true;
  }

}
#endif
