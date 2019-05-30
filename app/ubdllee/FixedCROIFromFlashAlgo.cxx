#include "FixedCROIFromFlashAlgo.h"

#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"

namespace larcv {
  namespace ubdllee {

    FixedCROIFromFlashAlgo::FixedCROIFromFlashAlgo() {
    }

    FixedCROIFromFlashAlgo::FixedCROIFromFlashAlgo( const FixedCROIFromFlashConfig& config )
      : m_config(config)
    {
    }
    
    std::vector<larcv::ROI> FixedCROIFromFlashAlgo::findCROIfromFlash( const larlite::opflash& intimeflash ) {

      std::vector<larcv::ROI> roi_v;

      // constants
      
      
      // we make 2 ROIs. z-center is the flash pe center.
      // x-center splits the drift region
      float croi_xcenters[2]  = {  65.0, 190.0 };
      float croi_ycenters[2]  = { -60.0,  60.0 };
      float croi_dzcenters[2] = { -75.0,  75.0 };

      float flash_meanz  = getFlashMeanZ(intimeflash);
      float croi_zcenter = flash_meanz;
      
      if ( croi_zcenter>1036.0-m_config.croi_dzcenters[1] ){
	croi_zcenter = 1036.0-m_config.croi_dzcenters[1];
	std::cout << "z-center(" << flash_meanz << " above higher bound, adjusting to " << croi_zcenter << std::endl;
      }
      if ( croi_zcenter+croi_dzcenters[0]<0 ) {
	croi_zcenter = fabs(croi_dzcenters[0]);
	std::cout << "z-center(" << flash_meanz << " below lower bound, adjusting to " << croi_zcenter << std::endl;
      }

      
      bool split_z = m_config.split_fixed_ycroi;
      int icroi = 0;
      for (int ix=0; ix<2; ix++) {
	for (int iy=0; iy<2; iy++) {

	  // we define a croi that covers the edges
	  // we make a fake TaggerFlashMatchData object using 3 points of a larlite track
	  std::cout << "Making ROI with center: (" << croi_xcenters[ix] << "," << croi_ycenters[iy] << "," << croi_zcenter << ")" << std::endl;

          // define the 3D bounding boxes
	  Double_t ll[3] = { croi_xcenters[ix] - 66.0, croi_ycenters[iy]-60.0, croi_zcenter - 35.0 };
          Double_t ce[3] = { croi_xcenters[ix] +  0.0, croi_ycenters[iy]+ 0.0, croi_zcenter +  0.0 };
	  Double_t ur[3] = { croi_xcenters[ix] + 66.0, croi_ycenters[iy]+60.0, croi_zcenter + 35.0 };

          // get the 3D projections into the planes
          larcv::ROI croi( larcv::kROIBNB, larcv::kShapeUnknown );

          // need ticks
          int min_tick = 3200 + ll[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
          int max_tick = 3200 + ur[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
          if ( (max_tick-min_tick)!=512*6 ) {
            int center = (int)(min_tick+max_tick)/2;
            min_tick = center - 256*6;
            max_tick = center + 256*6;
          }
          
          for ( size_t p=0; p<3; p++ ) {
            int min_col = larutil::Geometry::GetME()->NearestWire( ll, p );
            int max_col = larutil::Geometry::GetME()->NearestWire( ur, p );
            // flip if needed
            if ( min_col>max_col ) {
              int tmp = min_col;
              min_col = max_col;
              max_col = tmp;
            }
            // set fixed size
            if ( (max_col-min_col) != 512 ) {
              int ave = (max_col+min_col)/2;
              min_col = (ave)-256;              
              max_col = (ave)+256;
            }
            // adjust if on image bounds
            if ( min_col<0 ) {
              min_col = 0;
              max_col = 512;
            }
            else if ( max_col>=3456 )  {
              max_col = 3456;
              min_col = 3456-512;
            }

            larcv::ImageMeta meta( max_col-min_col, max_tick-min_tick,
                                   512, 512, min_col, min_tick, (larcv::PlaneID_t)p );
            croi.AppendBB(meta);
          }

          
	  int nzrois = 1;
	  if ( split_z )
	    nzrois = 2;
          
	  for (int iz=0; iz<nzrois; iz++) {
	    
	    larcv::ROI croi_zmod( croi.Type(), croi.Shape() );
	    
	    // we expand the y-plane roi to be 512 pixels wide (they are smaller in order to keep 3D consistent box)
	    std::vector<larcv::ImageMeta> bb_zmod;
	    for ( auto const& meta : croi.BB() ) {
	      if ( (int)meta.plane()!=2 ) {
		// U,V just pass through
		bb_zmod.push_back( meta );
	      }
	      else {
		// Y, make an expanded roi in the wire dimension. keep tick dimension
		double postemp[3] = {0,0, croi_zcenter};

		// if we split, we move the z position
		if (split_z)
		  postemp[2] += croi_dzcenters[iz];
		
		// the new center
		float ywire = larutil::Geometry::GetME()->NearestWire( postemp, 2 );

		// adjust the origin
		float yp_origin = ywire-0.5*meta.width();
		
		// need to check the range
		float split_ymax = yp_origin + meta.width();

		if ( split_ymax>=3456 )
		  yp_origin = 3456 - meta.width();
		if ( yp_origin<0 )
		  yp_origin = 0;

		larcv::ImageMeta ymeta( meta.width(), meta.height(), 512, 512, yp_origin, min_tick, 2 );
		bb_zmod.push_back( ymeta );
	      }
	    }
	    // update the BB
	    croi_zmod.SetBB( bb_zmod );
            
	    roi_v.emplace_back( std::move(croi_zmod) );
	  }//end of z loop
	}//end of y loop
      }//end of x loop
      

      std::cout << " ------------------ " << std::endl;
      std::cout << "[Selected CROI]" << std::endl;
      for ( auto const& croi : roi_v ) {
        for ( size_t p=0; p<3; p++ ) {
          std::cout << "  " << croi.BB(p).dump() << std::endl;
        }
      }
      std::cout << " ------------------ " << std::endl;
      
      return roi_v;
    }
    
    float FixedCROIFromFlashAlgo::getFlashMeanZ( const larlite::opflash& flash ) {
      float meanz = 0.;
      float totq  = 0.;
      for ( size_t iopdet=0; iopdet<32; iopdet++ ) {
        Double_t xyz[3];
        //larutil::Geometry::GetME()->GetOpChannelPosition( ich, xyz );
        larutil::Geometry::GetME()->GetOpDetPosition( iopdet, xyz );
        meanz += flash.PE(iopdet)*xyz[2];
        totq  += flash.PE(iopdet);
      }
      if ( totq>0 )
        meanz /= totq;
      else
        meanz = 0;
      return meanz;
    }
    
  }//end of ubdllee namespace
}//end of ublarcvapp namespace
