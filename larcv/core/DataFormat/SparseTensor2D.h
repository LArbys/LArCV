#ifndef __LARCV_CORE_DATAFORMAT_SPARSETENSOR2D_H__
#define __LARCV_CORE_DATAFORMAT_SPARSETENSOR2D_H__

#include "Voxel.h"
#include "ImageMeta.h"

namespace larcv {
  
  /**
     \class SparseTensor2D
     @brief Container of multiple (2D-projected) voxel set array
  */
  class SparseTensor2D : public VoxelSet {
  public:
    /// Default ctor
    SparseTensor2D() {}
    /// Default dtor
    virtual ~SparseTensor2D() {}
    /// copy ctor w/ VoxelSet
    SparseTensor2D(const larcv::VoxelSet& vs)
      : VoxelSet(vs)
    {}
    /// move ctor
    SparseTensor2D(larcv::VoxelSet&& vs, larcv::ImageMeta&& meta);

    //
    // Read-access
    //
    /// Access ImageMeta of specific projection
    inline const larcv::ImageMeta& meta() const { return _meta; }
    /// Returns a const reference to voxel closest to a voxel with specified id. if no such voxel within distance, return invalid voxel.
    const Voxel& close(VoxelID_t id, double distance, const larcv::ImageMeta& meta) const;

    //
    // Write-access
    //
    /// Clear everything
    inline void clear_data() { VoxelSet::clear_data(); _meta = ImageMeta(); }
    /// Meta setter
    void meta(const larcv::ImageMeta& meta);

  private:
    larcv::ImageMeta _meta;

  };

}

#endif
