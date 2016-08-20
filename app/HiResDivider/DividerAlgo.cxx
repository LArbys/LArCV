#include "DividerAlgo.h"
#include <cstring>

namespace divalgo {

  DividerAlgo::DividerAlgo() {
    memset( fOffset, 0, sizeof(float)*3 );
    memset( fDetCenter, 0, sizeof(float)*3 );
  }

  DividerAlgo::~DividerAlgo() {
  }

  void DividerAlgo::getregionwires( const float z1, const float y1, const float z2, const float y2,
				    std::vector<int>& ywires, std::vector<int>& uwires, std::vector<int>& vwires ) {
    
  }

}
