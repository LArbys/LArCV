/**
 * \file ROI.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class ROI
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef __ROI_H__
#define __ROI_H__

#include <iostream>
#include "Vertex.h"
#include "ImageMeta.h"
#include "DataFormatTypes.h"
namespace larcv {

  /**
     \class ROI
     User defined class ROI ... these comments are used to generate
     doxygen documentation!
  */
  class ROI{
    
  public:
    
    /// Default constructor
    ROI(larcv::ROIType_t type=larcv::kROIUnknown,
	larcv::ShapeType_t shape=larcv::kShapeUnknown)
      : _index      (kINVALID_INDEX)
      , _shape      (shape)
      , _type       (type )
      , _mcst_index (kINVALID_INDEX)
      , _mct_index  (kINVALID_INDEX)
      , _energy_deposit (0.)
      , _energy_init    (0.)
      , _pdg            (0)
      , _parent_pdg     (0)
      , _trackid        (kINVALID_UINT)
      , _parent_trackid (kINVALID_UINT)
      , _px             (0.)
      , _py             (0.)
      , _pz             (0.)
      , _current_type   (-1)
      , _interaction_type (-1)
      , _process ("")
    {}
    
    /// Default destructor
    ~ROI(){}

    ROIIndex_t     Index         () const { return _index;          }
    ROIType_t      Type          () const { return _type;           }
    ShapeType_t    Shape         () const { return _shape;          }
    MCSTIndex_t    MCSTIndex     () const { return _mcst_index;     }
    MCTIndex_t     MCTIndex      () const { return _mct_index;      }
    double         EnergyDeposit () const { return _energy_deposit; }
    double         EnergyInit    () const { return _energy_init;    }
    int            PdgCode       () const { return _pdg;            }
    int            ParentPdgCode () const { return _parent_pdg;     }
    unsigned int   TrackID       () const { return _trackid;        }
    unsigned int   ParentTrackID () const { return _parent_trackid; }
    const larcv::Vertex& Position() const { return _vtx;            }
    double         X  () const { return _vtx.X(); }
    double         Y  () const { return _vtx.Y(); }
    double         Z  () const { return _vtx.Z(); }
    double         T  () const { return _vtx.T(); }
    double         Px () const { return _px;      }
    double         Py () const { return _py;      }
    double         Pz () const { return _pz;      }
    double         ParentX  () const { return _parent_vtx.X(); }
    double         ParentY  () const { return _parent_vtx.Y(); }
    double         ParentZ  () const { return _parent_vtx.Z(); }
    double         ParentT  () const { return _parent_vtx.T(); }
    double         ParentPx () const { return _parent_px;      }
    double         ParentPy () const { return _parent_py;      }
    double         ParentPz () const { return _parent_pz;      }
    const larcv::Vertex& ParentPosition () const { return _parent_vtx;}
    const std::vector<larcv::ImageMeta>& BB() const { return _bb_v;  }
    const ImageMeta& BB(PlaneID_t plane) const;
    short NuCurrentType() const { return _current_type; }
    short NuInteractionType() const { return _interaction_type; }
    const std::string& CreationProcess() const { return _process; }
    const larcv::Vertex& EndPosition() const { return _end_pt; }
    
    void Index         (ROIIndex_t id  )    { _index = id;         }
    void Type          (ROIType_t type )    { _type  = type;       }
    void Shape         (ShapeType_t shape ) { _shape = shape;      }
    void MCSTIndex     (MCSTIndex_t id )    { _mcst_index = id;    }
    void MCTIndex      (MCTIndex_t id )     { _mct_index = id;     }
    void EnergyDeposit (double e)           { _energy_deposit = e; }
    void EnergyInit    (double e)           { _energy_init = e;    }
    void PdgCode       (int code)           { _pdg = code;         }
    void ParentPdgCode (int code)           { _parent_pdg = code;  }
    void TrackID       (unsigned int id )   { _trackid = id;       }
    void ParentTrackID (unsigned int id )   { _parent_trackid = id;}
    void Position      (const larcv::Vertex& vtx) { _vtx = vtx;    }
    void Position      (double x, double y, double z, double t) { _vtx = Vertex(x,y,z,t); }
    void Momentum      (double px, double py, double pz) { _px = px; _py = py; _pz = pz; }
    void ParentPosition (const larcv::Vertex& vtx) { _parent_vtx = vtx; }
    void ParentPosition (double x, double y, double z, double t) { _parent_vtx = Vertex(x,y,z,t); }
    void ParentMomentum (double px, double py, double pz) { _parent_px = px; _parent_py = py; _parent_pz = pz; }
    void AppendBB      (const larcv::ImageMeta& bb);
    void SetBB         (const std::vector<larcv::ImageMeta>& bb_v);
    void NuCurrentType (short curr) { curr = -1; _current_type = curr; }
    void NuInteractionType (short itype) { itype = -1; _interaction_type = itype; }
    void CreationProcess (const std::string& proc) { _process = proc; }
    void EndPosition   (const larcv::Vertex& vtx) { _end_pt = vtx; }
    void EndPosition   (double x, double y, double z, double t) { _end_pt = Vertex(x,y,z,t); }

    std::string dump() const;
    
  private:

    ROIIndex_t   _index;
    ShapeType_t  _shape;
    ROIType_t    _type;
    MCSTIndex_t  _mcst_index;
    MCTIndex_t   _mct_index;

    double       _energy_deposit;
    double       _energy_init;
    int          _pdg;
    int          _parent_pdg;
    unsigned int _trackid;
    unsigned int _parent_trackid;
    Vertex       _vtx;
    double       _px,_py,_pz;
    Vertex       _parent_vtx;
    double       _parent_px, _parent_py, _parent_pz;
    std::vector<larcv::ImageMeta> _bb_v;
    short _current_type;
    short _interaction_type;

    std::string _process;
    Vertex      _end_pt;
  };
}
#endif
/** @} */ // end of doxygen group 

