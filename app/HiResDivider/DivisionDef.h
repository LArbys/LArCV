#ifndef __DIVISIONDEF__
#define __DIVISIONDEF__

//#include "Base/larcv_base.h"
#include "DataFormat/DataFormatTypes.h"
#include "DataFormat/ImageMeta.h"
#include <map>

namespace larcv {
  namespace hires {

    //class DivisionDef : larcv::larcv_base {
    class DivisionDef {
    public:
      DivisionDef( int plane0_wirebounds[], int plane1_wirebounds[], int plane2_wirebounds[], int tickbounds[],
		   float det_xbounds[], float det_ybounds[], float det_zbounds[] );
      DivisionDef( const DivisionDef& ); // copy constructor
      DivisionDef() {};
      ~DivisionDef() {};

      const larcv::ImageMeta& getPlaneMeta( PlaneID_t plane ) const;
      bool isInsideDetRegion( float x, float y, float z ) const;

    protected:

      void setPlaneMeta( PlaneID_t plane, int wirebounds[],int tickbounds[] );

      int mNPlanes;
      std::map< PlaneID_t, larcv::ImageMeta > m_planeMeta; // key is plane id
      float fDetBounds[3][2];

    };

  }
}


#endif
