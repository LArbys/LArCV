/**
 * \file Voxel3D.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class Voxel3D
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef VOXEL3D_H
#define VOXEL3D_H

#include <iostream>
#include <vector>
#include <array>
#include "DataFormatTypes.h"
namespace larcv {

  /**
     \class Voxel3D
     @brief 3D voxel definition element class consisting of ID and stored value
  */
  class Voxel3D{
    
  public:
    
    /// Default constructor
    Voxel3D(Voxel3DID_t id=kINVALID_VOXEL3DID, float value=kINVALID_FLOAT);
    /// Default destructor
    ~Voxel3D(){}
    
    /// ID getter
    inline Voxel3DID_t ID() const { return _id; }
    /// Value getter
    inline float  Value() const { return _value; }

    /// Value setter
    inline void Set(Voxel3DID_t id, float value) { _id = id; _value = value; }

    //
    // uniry operators
    //
    inline Voxel3D& operator += (float value)
    { _value += value; return (*this); }
    inline Voxel3D& operator -= (float value)
    { _value -= value; return (*this); }
    inline Voxel3D& operator *= (float factor)
    { _value *= factor; return (*this); }
    inline Voxel3D& operator /= (float factor)
    { _value /= factor; return (*this); }

    //
    // binary operators
    //
    inline bool operator == (const Voxel3D& rhs) const
    { return (_id == rhs._id); }
    inline bool operator <  (const Voxel3D& rhs) const
    {
      if( _id < rhs._id) return true;
      if( _id > rhs._id) return false;
      return false;
    }
    inline bool operator <= (const Voxel3D& rhs) const
    { return  ((*this) == rhs || (*this) < rhs); }
    inline bool operator >  (const Voxel3D& rhs) const
    { return !((*this) <= rhs); }
    inline bool operator >= (const Voxel3D& rhs) const
    { return !((*this) <  rhs); }

    inline bool operator == (const float& rhs) const
    { return _value == rhs; }
    inline bool operator <  (const float& rhs) const
    { return _value <  rhs; }
    inline bool operator <= (const float& rhs) const
    { return _value <= rhs; }
    inline bool operator >  (const float& rhs) const
    { return _value >  rhs; }
    inline bool operator >= (const float& rhs) const
    { return _value >= rhs; }

  private:
    Voxel3DID_t _id; ///< voxel id
    float  _value; ///< Pixel Value
  };

  /**
     \class Voxel3DMeta
     @brief Meta data for defining voxels (ID, size, position) and voxelized volume (coordinate, size)
  */
  class Voxel3DMeta {
  public:
    /// Default ctor
    Voxel3DMeta();
    /// Default dtor
    ~Voxel3DMeta(){}

    /// Define dimensions
    void Set(double xmin, double xmax,
	     double ymin, double ymax,
	     double zmin, double zmax,
	     size_t xnum,
	     size_t ynum,
	     size_t znum);
    /// Clear method
    void Clear();
    /// Checker if the meta parameters are set properly or not
    inline bool Valid() const { return _valid; }
    /// Returns size
    inline Voxel3DID_t Size() const { return _num_element; }
    /// Given a position, returns voxel ID
    Voxel3DID_t ID(double x, double y, double z) const;
    /// Given a valid voxel ID, returns a position array
    const std::array<double,3> Position(Voxel3DID_t id) const;
    /// Given a valid voxel ID, returns X position
    double X(Voxel3DID_t id) const;
    /// Given a valid voxel ID, returns Y position
    double Y(Voxel3DID_t id) const;
    /// Given a valid voxel ID, returns Z position
    double Z(Voxel3DID_t id) const;
    /// Returns voxel count along x-axis
    inline size_t NumVoxelX()  const { return _xnum; }
    /// Returns voxel count along y-axis
    inline size_t NumVoxelY()  const { return _ynum; }
    /// Returns voxel count along z-axis
    inline size_t NumVoxelZ()  const { return _znum; }
    /// Returns voxel size along x-axis;
    inline double SizeVoxelX() const { return _xlen; }
    /// Returns voxel size along y-axis;
    inline double SizeVoxelY() const { return _ylen; }
    /// Returns voxel size along z-axis;
    inline double SizeVoxelZ() const { return _zlen; }
    /// Returns voxel definition maximum x value
    inline double MaxX() const { return _xmax; }
    /// Returns voxel definition maximum y value
    inline double MaxY() const { return _ymax; }
    /// Returns voxel definition maximum z value
    inline double MaxZ() const { return _zmax; }
    /// Returns voxel definition minimum x value
    inline double MinX() const { return _xmin; }
    /// Returns voxel definition minimum y value
    inline double MinY() const { return _ymin; }
    /// Returns voxel definition minimum z value
    inline double MinZ() const { return _zmin; }

  private:

    bool   _valid; ///< Boolean set to true only if voxel parameters are properly set
    Voxel3DID_t _num_element; ///< Total number of voxel elements
    
    double _xmin; ///< X min value in voxel definition in [cm]
    double _xmax; ///< X max value in voxel definition in [cm]
    double _xlen; ///< X voxel size in [cm]
    
    double _ymin; ///< Y min value in voxel definition in [cm]
    double _ymax; ///< Y min value in voxel definition in [cm]
    double _ylen; ///< Y voxel size in [cm]
    
    double _zmin; ///< Z min value in voxel definition in [cm]
    double _zmax; ///< Z min value in voxel definition in [cm]
    double _zlen; ///< Z voxel size in [cm]
    
    size_t _xnum; ///< Number of voxels along X
    size_t _ynum; ///< Number of voxels along Y
    size_t _znum; ///< Number of voxels along Z
  };

  /**
     \class Voxel3DSet
     @brief Container of multiple voxels consisting of ordered sparse vector and meta data
   */
  class Voxel3DSet {
  public:
    /// Default ctor
    Voxel3DSet(const Voxel3DMeta& meta=Voxel3DMeta());
    /// Default dtor
    ~Voxel3DSet(){}

    /// adder
    void Add(const Voxel3D& vox);
    /// adder
    void Emplace(Voxel3D&& vox);
    /// getter
    inline const std::vector<larcv::Voxel3D>& Get() const
    { return _voxel_v; }
    /// clear
    inline void Clear() { _voxel_v.clear(); _meta.Clear();}
    /// reset
    inline void Reset(const Voxel3DMeta& meta)
    { Clear(); _meta = meta; }
    
  private:
    /// Meta data information
    Voxel3DMeta _meta;
    /// Ordered sparse vector of voxels 
    std::vector<larcv::Voxel3D> _voxel_v;
  };
  
}

#endif
/** @} */ // end of doxygen group 

