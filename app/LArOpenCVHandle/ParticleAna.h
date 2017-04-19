/**
 * \file ParticleAna.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class ParticleAna
 *
 * @author vgenty
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __PARTICLEANA_H__
#define __PARTICLEANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

#include "LArbysImageMaker.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/Contour2DAnalysis.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/ImagePatchAnalysis.h"
#include "TTree.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class ParticleAna ... these comments are used to generate
     doxygen documentation!
  */
  class ParticleAna : public ProcessBase {

  public:
    
    /// Default constructor
    ParticleAna(const std::string name="ParticleAna");
    
    /// Default destructor
    ~ParticleAna(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();


  private:

    EventImage2D* _ev_img_v;
    EventImage2D* _ev_trk_img_v;
    EventImage2D* _ev_shr_img_v;
    EventROI* _ev_roi_v;
    EventPGraph* _ev_pgraph_v;
    EventPixel2D* _ev_pcluster_v;
    EventPixel2D* _ev_ctor_v;

    int _run;
    int _subrun;
    int _event;
    int _entry;
    
    std::string _img_prod;       
    std::string _reco_roi_prod;
    std::string _pgraph_prod;
    std::string _pcluster_img_prod;
    std::string _pcluster_ctor_prod; 
    std::string _trk_img_prod;
    std::string _shr_img_prod;
    
    LArbysImageMaker _LArbysImageMaker;
    
    //
    // Particle Related Fucntionality
    //
    bool _analyze_particle;

    TTree* _particle_tree;

    void AnalyzeParticle();

    float _length;
    float _width;
    float _perimeter;
    float _area;
    uint  _npixel;
    float _track_frac;
    float _shower_frac;
    double _mean_pixel_dist;
    double _sigma_pixel_dist;
    double _angular_sum;
    
    //
    // Angle & dQdX Related Functionality
    //
    bool _analyze_angle;
    bool _analyze_dqdx;
    
    TTree* _angle_tree;
    TTree* _dqdx_tree;
    
    void AnalyzeAngle();
    void AnalyzedQdX();
    
    double Getx2vtxmean( ::larocv::GEO2D_Contour_t ctor, float x2d, float y2d);
    cv::Point PointShift(::cv::Point pt, geo2d::Line<float> pca);

    double Mean(const std::vector<float>& v);
    double STD(const std::vector<float>& v);
    
    float _meanl;
    float _meanr;
    float _stdl;
    float _stdr;
    float _dqdxdelta;
    float _dqdxratio;

    uint _plane;
    double _pradius;
    double _maskradius;
    uint _bins;
    float _open_angle_cut;
    float _adc_threshold;

    std::vector<double> _angle0_c;
    std::vector<double> _angle1_c;
    uint _straight_lines;
    
    double _mean0; //mean value of x in a ctor w.r.t the vertex
    double _mean1; //to determine the direction of PCA w.r.t to the vertex

    std::vector<double> _dir0_c;//particle direction from contour
    std::vector<double> _dir1_c;
    std::vector<double> _dir0_p;//particle direction from pixels close to vertex
    std::vector<double> _dir1_p;
    
  };

  /**
     \class larcv::ParticleAnaFactory
     \brief A concrete factory class for larcv::ParticleAna
  */
  class ParticleAnaProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    ParticleAnaProcessFactory() { ProcessFactory::get().add_factory("ParticleAna",this); }
    /// dtor
    ~ParticleAnaProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new ParticleAna(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

