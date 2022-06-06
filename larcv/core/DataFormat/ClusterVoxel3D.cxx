#include "ClusterVoxel3D.h"

#include <stdexcept>

namespace larcv {

  void ClusterVoxel3D::meta(const larcv::Voxel3DMeta& meta)
  {
    for (auto const& vs : this->as_vector()) {
      for (auto const& vox : vs.as_vector()) {
	if (vox.id() < meta.size()) continue;
	std::cerr << "VoxelSet contains ID " << vox.id()
		  << " which cannot exists in Voxel3DMeta with size " << meta.size()
		  << std::endl;
	throw std::exception();
      }
    }
    _meta = meta;
  }

}
