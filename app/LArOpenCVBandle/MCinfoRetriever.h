/**
 * \file MCinfoRetriever.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class MCinfoRetriever
 *
 * @author Rui
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __MCINFORETRIEVER_H__
#define __MCINFORETRIEVER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

#include <vector>
#include "TH1D.h"
#include "TTree.h"
#include "LArUtil/PxUtils.h"
#include "DataFormat/Image2D.h"
#include "Core/HalfLine.h"
#include "Core/Line.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class MCinfoRetriever ... these comments are used to generate
     doxygen documentation!
  */
  class MCinfoRetriever : public ProcessBase {

  public:
    
    /// Default constructor
    MCinfoRetriever(const std::string name="MCinfoRetriever");
    
    /// Default destructor
    ~MCinfoRetriever(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    void Clear();
    
    //  private:
  protected:

    std::string _producer_roi;
    std::string _producer_image2d;
    
    TTree* _mc_tree;
    
    /// Event ID
    uint _run;
    uint _subrun;
    uint _event;

    /// Primary Particle Info
    int _parent_pdg;//primary particle pdg

    double _energy_deposit;
    double _energy_init;

    double _parent_x;
    double _parent_y;  
    double _parent_z;  
    double _parent_t;  
    double _parent_px;
    double _parent_py;  
    double _parent_pz;
    
    short _current_type;
    short _interaction_type;
    float _length_2d;

    uint _nprimary;
    uint _ntotal;
    
    uint _nproton;
    uint _nlepton;
    uint _nshower;
    uint _nmeson;
    uint _nneutron;

    int _hi_lep_pdg;
    
    double _dep_sum_lep;
    double _ke_sum_lep;

    double _dep_sum_proton;
    double _ke_sum_proton;

    double _dep_sum_meson;
    double _ke_sum_meson;

    double _dep_sum_shower;
    double _ke_sum_shower;

    double _dep_sum_neutron;
    double _ke_sum_neutron;
    
    geo2d::Vector<float> _start; //2d start point
    geo2d::Vector<float> _dir;   //2d dir

    std::vector<uint>   _daughter_pdg_v;
    std::vector<uint>   _daughter_trackid_v;
    std::vector<uint>   _daughter_parenttrackid_v;    

    std::vector<double> _daughter_energyinit_v;
    std::vector<double> _daughter_energydep_v;

    std::vector<std::vector<double> > _daughter_length_vv;
    std::vector<std::vector<double> > _daughter_2dstartx_vv;
    std::vector<std::vector<double> > _daughter_2dstarty_vv;
    std::vector<std::vector<double> > _daughter_2dendx_vv;
    std::vector<std::vector<double> > _daughter_2dendy_vv;
    std::vector<std::vector<double> > _daughter_2dcosangle_vv;
    
    std::vector<double> _daughterPx_v;
    std::vector<double> _daughterPy_v;
    std::vector<double> _daughterPz_v;


    /// 2D Vertex Info
    std::vector<double> _vtx_2d_w_v;
    std::vector<double> _vtx_2d_t_v;

    /// LARCV Image2D data
    std::vector<larcv::Image2D> _image_v;
    ImageMeta _meta;

    /// Filtering
    float _min_nu_dep_e;
    float _max_nu_dep_e;
    float _min_nu_init_e;
    float _max_nu_init_e;
    
    int _min_n_proton;
    int _min_n_neutron;
    int _min_n_lepton;
    int _min_n_meson; 
    int _min_n_shower;

    float _min_proton_init_e;
    float _min_lepton_init_e;
    
    bool _check_vis;

    bool _do_not_reco;
    
  private:
    ///Project 3D track into 2D Image(per plane) 
    void Project3D(const ImageMeta& meta,
		   double _parent_x,double _parent_y,double _parent_z,uint plane,
		   double& xpixel, double& ypixel);
    
    ///Calculate edge point on 2D ROI
    geo2d::Vector<float> Intersection (const geo2d::HalfLine<float>& hline,
				       const cv::Rect& rect);
    
    cv::Rect Get2DRoi(const ImageMeta& meta,
		      const ImageMeta& roi_meta);

    
  };

  /**
     \class larcv::MCinfoRetrieverFactory
     \brief A concrete factory class for larcv::MCinfoRetriever
  */
  class MCinfoRetrieverProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    MCinfoRetrieverProcessFactory() { ProcessFactory::get().add_factory("MCinfoRetriever",this); }
    /// dtor
    ~MCinfoRetrieverProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new MCinfoRetriever(instance_name); }

  };

}

#endif
/** @} */ // end of doxygen group 

