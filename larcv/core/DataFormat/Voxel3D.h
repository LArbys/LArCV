/**
 * \file Voxel3D.h
 *
 * \ingroup core_DataFormat
 *
 * \brief Class def header for 3D specific extension of voxels
 *
 * @author kazuhiro
 */

/** \addtogroup core_DataFormat

    @{*/
#ifndef LARCV_VOXEL3D_H
#define LARCV_VOXEL3D_H

#include "Voxel.h"
#include "Voxel3DMeta.h"
#include "utils.h"
#include "SparseTensor3D.h"

namespace larcv {


  /**
     \class ClusterVoxel3D
     @brief Container of multiple (3D-projected) voxel set array
  */
  class ClusterVoxel3D : public VoxelSetArray {
  public:
    /// Default ctor
    ClusterVoxel3D() {}
    /// Default dtor
    virtual ~ClusterVoxel3D() {}

    //
    // Read-access
    //
    /// Access Voxel3DMeta of specific projection
    inline const larcv::Voxel3DMeta& meta() const { return _meta; }

    //
    // Write-access
    //
    /// Clear everything
    inline void clear_data() { VoxelSetArray::clear_data(); _meta = Voxel3DMeta(); }
    /// set VoxelSetArray
    inline void set(VoxelSetArray& vsa, const Voxel3DMeta& meta)
    { *((VoxelSetArray*)this) = vsa; this->meta(meta); }
    /// emplace VoxelSetArray
    inline void emplace(VoxelSetArray&& vsa, const Voxel3DMeta& meta)
    { *((VoxelSetArray*)this) = std::move(vsa); this->meta(meta); }
    /// Merge another ClusterVoxel3D
    inline void merge(const ClusterVoxel3D& vsa, bool check_meta_strict=true)
    { 
      if(!(this->meta().valid()))
        this->meta(vsa.meta());
      else if(check_meta_strict) {
        if(vsa.meta() != this->meta()) {
          std::cerr << "Meta mismatched (strict check)!" << std::endl
                    << this->meta().dump() << std::endl
                    << vsa.meta().dump() << std::endl;
          throw std::exception();
        }
      }
      else if(this->meta().num_voxel_x() != vsa.meta().num_voxel_x() || 
        this->meta().num_voxel_y() != vsa.meta().num_voxel_y() || 
        this->meta().num_voxel_z() != vsa.meta().num_voxel_z() ) {
        if(vsa.meta() != this->meta()) {
          std::cerr << "Meta mismatched (loose check)!" << std::endl
                    << this->meta().dump() << std::endl
                    << vsa.meta().dump() << std::endl;
          throw std::exception(); 
        }
      }
      for(auto vs : vsa.as_vector()) {
        vs.id(larcv::kINVALID_INSTANCEID);
        ((VoxelSetArray*)this)->emplace(std::move(vs));
      } 
    }
    /// Meta setter
    void meta(const larcv::Voxel3DMeta& meta);

  private:
    larcv::Voxel3DMeta _meta;
  };

}

#endif
/** @} */ // end of doxygen group
