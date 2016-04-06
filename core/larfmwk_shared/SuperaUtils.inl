#ifndef __SUPERAUTILS_INL__
#define __SUPERAUTILS_INL__

#ifndef __CINT__

#include "FMWKInterface.h"
#include "Base/larbys.h"   // for exception handling
#include "DataFormat/ImageMeta.h"
#include "DataFormat/Image2D.h"
namespace larcv {
  namespace supera {

    template <class T>
    larcv::Image2D Extract(const ImageMeta& meta_in, const std::vector<T>& wires)
    {
      ImageMeta meta(meta_in.width(),meta_in.height(),
		     meta_in.rows(),meta_in.cols(),
		     meta_in.tl().x,meta_in.tl().y,//-3200,
		     meta_in.plane());
      //int nticks = meta.rows();
      //int nwires = meta.cols();

      size_t row_comp_factor = (size_t)(meta.pixel_height());
      int ymax = meta.max_y();
      int ymin = (meta.min_y() >= 0 ? meta.min_y() : 0);

      larcv::Image2D img(meta_in);

      for(auto const& wire : wires) {

	auto const& wire_id = ChannelToWireID(wire.Channel());
	
	if((int)(wire_id.Plane) != meta.plane()) continue;

	size_t col=0;
	try{
	  col = meta.col(wire_id.Wire);
	}catch(const larbys&){
	  continue;
	}

	for(auto const& range : wire.SignalROI().get_ranges()) {
	  
	  auto const& adcs = range.data();
	  size_t start_index = range.begin_index();
	  size_t end_index   = start_index + adcs.size() - 1;
	  if(start_index > ymax || end_index < ymin) continue;

	  if(row_comp_factor>1) {

	    for(size_t index=0; index<adcs.size(); ++index) {
	      if(index + start_index < ymin) continue;
	      if(index + start_index > ymax) break;
	      auto row = meta.row((double)(start_index+index));
	      img.set_pixel(row,col,adcs[index]);
	    }
	  }else{
	    if(end_index   > ymax) end_index   = ymax;
	    if(start_index < ymin) start_index = ymin;
	    /*
	    std::cout << "Calling a reverse_copy..." << std::endl
		      << "      Ticks       : " << range.begin_index() << " => "
		      << (range.begin_index() + adcs.size() - 1) << std::endl
		      << "      Meta Bounds : " << meta.min_y() << " => " << meta.max_y() << std::endl
		      << "      Index Used  : " << (start_index - range.begin_index()) << " => "
		      << (end_index - range.begin_index()) << std::endl
		      << "      Filling Row : " << (start_index-ymin+1) << " => "
		      << (end_index - ymin) << std::endl;
	    */
	    img.reverse_copy(end_index-ymin,
			     col,
			     adcs,
			     end_index - start_index+1);
	  }
	}
      }
      
      return img;
    }
  }
}

#endif
#endif
