/**
 * \file LEE1e1p.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class LEE1e1p
 *
 * @author kazuhiro
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __LEE1E1PANA_H__
#define __LEE1E1PANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include <TTree.h>
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/Contour2DAnalysis.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/ImagePatchAnalysis.h"
#include "LArbysImageMaker.h"
#include <math.h>
#include <numeric>

namespace larcv {

  /**
     \class ProcessBase
     User defined class LEE1e1pAna ... these comments are used to generate
     doxygen documentation!
  */
  class LEE1e1pAna : public ProcessBase {

  public:
    
    /// Default constructor
    LEE1e1pAna(const std::string name="LEE1e1pAna");
    
    /// Default destructor
    ~LEE1e1pAna(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();
    
  private:

    TTree* _tree;
    TTree* _event_tree;
    int _entry;
    int _run;
    int _subrun;
    int _event;
    int _roid;
    int _vtxid;

    double _tx;
    double _ty;
    double _tz;
    double _scex;
    double _scey;
    double _scez;
    double _te;
    double _tt;
    double _x;
    double _y;
    double _z;
    double _dr;
    double _scedr;
    int _npar;
    int _shape0;
    int _shape1;
    std::vector<double> _score0;
    std::vector<double> _score1;
    double _score_shower0;
    double _score_shower1;
    double _score_track0;
    double _score_track1;
    double _score0_pi;
    double _score0_e;
    double _score0_g;
    double _score0_p;
    double _score0_mu;
    double _score1_pi;
    double _score1_e;
    double _score1_g;
    double _score1_p;
    double _score1_mu;
    int _npx0;
    int _npx1;
    double _q0;
    double _q1;
    double _area0;
    double _area1;
    double _len0;
    double _len1;
    double _area_croi0;
    double _area_croi1;
    double _area_croi2;
    int _good_croi0;
    int _good_croi1;
    int _good_croi2;
    int _good_croi_ctr;
    int _num_croi;
    double _min_vtx_dist;
    ::larutil::SpaceChargeMicroBooNE _sce;

    std::string _img2d_prod;
    std::string _pgraph_prod;        
    std::string _pcluster_ctor_prod;
    std::string _pcluster_img_prod;
    std::string _truth_roi_prod;  
    std::string _reco_roi_prod;    
    
  };

  /**
     \class larcv::LEE1e1pAnaFactory
     \brief A concrete factory class for larcv::LEE1e1pAna
  */
  class LEE1e1pAnaProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    LEE1e1pAnaProcessFactory() { ProcessFactory::get().add_factory("LEE1e1pAna",this); }
    /// dtor
    ~LEE1e1pAnaProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new LEE1e1pAna(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

