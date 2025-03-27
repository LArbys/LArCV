#include "Voxel3DSet.h"

#include "Voxel3D.h"
#include "larcv/core/Base/larbys.h"

namespace larcv {

  Voxel3DSet::Voxel3DSet(const Voxel3DMeta& meta)
    : _meta(meta)
  {}

  void Voxel3DSet::Add(const Voxel3D& vox)
  {
    Voxel3D copy(vox);
    Emplace(std::move(copy));
  }
  
  void Voxel3DSet::Emplace(Voxel3D&& vox)
  {
    if(!_meta.valid())
      throw larbys("Voxel3DSet::Emplace cannot be called without a valid meta!");
    // In case it's empty or greater than the last one
    if(_voxel_v.empty() || _voxel_v.back() < vox) {
      _voxel_v.emplace_back(std::move(vox));
      return;
    }
    // In case it's smaller than the first one
    if(_voxel_v.front() > vox) {
      _voxel_v.emplace_back(std::move(vox));
      for(size_t idx=0; (idx+1)<_voxel_v.size(); ++idx) {
	auto& element1 = _voxel_v[ _voxel_v.size() - (idx+1) ];
	auto& element2 = _voxel_v[ _voxel_v.size() - (idx+2) ];
	std::swap( element1, element2 );
      }
      return;
    }
    
    // Else do log(N) search
    auto iter = std::lower_bound(_voxel_v.begin(), _voxel_v.end(), vox);

    // Cannot be the end
    if( iter == _voxel_v.end() )
      throw larbys("Voxel3DSet sorting logic error!");
    
    // If found, merge
    if( !(vox < (*iter)) ) {
      (*iter) += vox.Value();
      return;
    }
    
    // Else insert @ appropriate place
    else {
      size_t target_loc = iter - _voxel_v.begin();
      _voxel_v.emplace_back(std::move(vox));
      for(size_t idx=target_loc; (idx+1)<_voxel_v.size(); ++idx) {
	auto& element1 = _voxel_v[ _voxel_v.size() - (idx+1) ];
	auto& element2 = _voxel_v[ _voxel_v.size() - (idx+2) ];
	std::swap( element1, element2 );
      }
    }
    return;
  }
  
}//end of larcv namespace
