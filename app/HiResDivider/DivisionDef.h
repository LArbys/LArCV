#ifndef __DIVISIONDEF__
#define __DIVISIONDEF__

#include "DataFormat/ImageMeta.h"
#include <map>

namespace larcv {
  namespace hires {

    class DivisionDef {
    public:
      DivisionDef( int plane0_wirebounds[], int plane1_wirebounds[], int plane2_wirebounds[], int tickbounds[],
		   float det_xbounds[], float det_ybounds[], float det_zbounds[] );
      DivisionDef( const DivisionDef& ); // copy constructor
      DivisionDef() {};
      ~DivisionDef() {};

      const larcv::ImageMeta& getPlaneMeta( int plane );
      bool isInsideDetRegion( float x, float y, float z ) const;

    protected:

      void setPlaneMeta( int plane, int wirebounds[],int tickbounds[] );

      int mNPlanes;
      std::map< int, larcv::ImageMeta > m_planeMeta; // key is plane id
      float fDetBounds[3][2];

    };

  }
}


#endif
