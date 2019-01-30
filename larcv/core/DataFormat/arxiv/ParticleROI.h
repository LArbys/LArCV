/**
 * \file ParticleROI.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class ParticleROI
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef __PARTICLEROI_H__
#define __PARTICLEROI_H__

#include <iostream>
#include "Vertex.h"
#include "ImageMeta.h"
#include "DataFormatTypes.h"
namespace larcv {

  /**
     \class ParticleROI
     User defined class ParticleROI ... these comments are used to generate
     doxygen documentation!
  */
  class ParticleROI{
    
  public:
    
    /// Default constructor
    ParticleROI(larcv::ParticleType_t type=larcv::kUnknownParticle,
	     larcv::ShapeType_t shape=larcv::kUnknownShape)
      : _index      (kINVALID_INDEX)
      , _shape      (shape)
      , _type       (type )
      , _mcst_index (kINVALID_INDEX)
      , _mct_index  (kINVALID_INDEX)
    {}
    
    /// Default destructor
    ~ParticleROI(){}

    ParticleROIIndex_t Index      () const { return _index;          }
    ParticleType_t Type          () const { return _type;           }
    ShapeType_t    Shape         () const { return _shape;          }
    MCSTIndex_t    MCSTIndex     () const { return _mcst_index;     }
    MCTIndex_t     MCTIndex      () const { return _mct_index;      }
    double         EnergyDeposit () const { return _energy_deposit; }
    double         EnergyInit    () const { return _energy_init;    }
    int            PdgCode       () const { return _pdg;            }
    larcv::Vertex  Position      () const { return _vtx;            }
    double         X() const { return _vtx.X(); }
    double         Y() const { return _vtx.Y(); }
    double         Z() const { return _vtx.Z(); }
    double         T() const { return _vtx.T(); }
    double         Px() const { return _px;      }
    double         Py() const { return _py;      }
    double         Pz() const { return _pz;      }
    const std::vector<larcv::ImageMeta>& BB() const { return _bb_v;  }
    const ImageMeta& BB(PlaneID_t plane) const;
    
    void Index         (ParticleROIIndex_t id )  { _index = id;         }
    void Type          (ParticleType_t type ) { _type  = type;       }
    void Shape         (ShapeType_t shape )   { _shape = shape;      }
    void MCSTIndex     (MCSTIndex_t id )      { _mcst_index = id;    }
    void MCTIndex      (MCTIndex_t id )       { _mct_index = id;     }
    void EnergyDeposit (double e)             { _energy_deposit = e; }
    void EnergyInit    (double e)             { _energy_init = e;    }
    void PdgCode       (int code)             { _pdg = code;         }
    void Position      (const larcv::Vertex& vtx) { _vtx = vtx;          }
    void Position      (double x, double y, double z, double t) { _vtx = Vertex(x,y,z,t); }
    void Momentum      (double px, double py, double pz) { _px = px; _py = py; _pz = pz; }
    void ROIIndex      (std::vector<ROIIndex_t> id_v)    { _roi_index_v = id_v; }
    void AppendROIIndex(ROIIndex_t id)        { _roi_index_v.push_back(id); }

    void AppendBB      (const larcv::ImageMeta& bb);
    void SetBB         (const std::vector<larcv::ImageMeta>& bb_v);
    
  private:

    ParticleROIIndex_t _index;
    ShapeType_t    _shape;
    ParticleType_t _type;
    MCSTIndex_t    _mcst_index;
    MCTIndex_t     _mct_index;

    double         _energy_deposit;
    double         _energy_init;
    int            _pdg;
    Vertex         _vtx;
    double         _px,_py,_pz;
    std::vector<larcv::ImageMeta> _bb_v;    
  };
}
#endif
/** @} */ // end of doxygen group 

