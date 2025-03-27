#ifndef __LARCV_CORE_DATAFORMAT_VOXEL3DSET_H__
#define __LARCV_CORE_DATAFORMAT_VOXEL3DSET_H__

/**
   \class Voxel3DSet
   @brief Container of multiple voxels consisting of ordered sparse vector and meta data
*/

#include "Voxel3D.h"
#include "Voxel3DMeta.h"

namespace larcv {
  
  class Voxel3DSet {
  public:
    /// Default ctor
    Voxel3DSet(const Voxel3DMeta& meta=Voxel3DMeta());
    /// Default dtor
    virtual ~Voxel3DSet(){}

    /// getter
    inline const std::vector<larcv::Voxel3D>& GetVoxelSet() const
    { return _voxel_v; }
    
    /// getter
    inline const Voxel3DMeta& GetVoxelMeta() const
    { return _meta; }
    /// clear
    inline void Clear() { _voxel_v.clear(); _meta.clear();}
    /// reset
    inline void Reset(const Voxel3DMeta& meta)
    { Clear(); _meta = meta; }
    /// adder
    void Add(const Voxel3D& vox);
    #ifndef __CINT__
    #ifndef __CLING__
    /// adder
    void Emplace(Voxel3D&& vox);
    /// mover
    inline void Move(Voxel3DSet&& vox_set)
    { _meta = std::move(vox_set._meta); _voxel_v = std::move(vox_set._voxel_v); }
    #endif
    #endif
  private:
    /// Meta data information
    Voxel3DMeta _meta;
    /// Ordered sparse vector of voxels 
    std::vector<larcv::Voxel3D> _voxel_v;
  };
  
}//end of larcv namespace

#endif
