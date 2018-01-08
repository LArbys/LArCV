#ifndef __VERTEXFILTER_H__
#define __VERTEXFILTER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "DataFormat/EventPixel2D.h"

namespace larcv {

  class VertexFilter : public ProcessBase {

  public:
    VertexFilter(const std::string name="VertexFilter");
    ~VertexFilter(){}

    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();
    void clear();
    
    void SetIndexVector(const std::vector<bool>& vec)
    { _idx_v = vec; }

    void SetParticleType(const std::vector<std::pair<int,int> >& vec)
    { _par_v = vec; }
    
    bool FillParticles(const std::vector<size_t>& cid_v,
		       EventPixel2D* ictor_v,EventPixel2D* iimg_v,
		       EventPixel2D* octor_v,EventPixel2D* oimg_v);

  private:
    
    TTree* _tree;
    int _run;
    int _subrun;
    int _event;
    int _entry;
    
    int _cvtxid;
    int _fvtxid;

    bool _set_shape;
    
    std::string _in_pg_prod;

    std::string _in_ctor_prod;
    std::string _in_img_prod;

    std::string _in_super_ctor_prod;
    std::string _in_super_img_prod;

    std::string _out_pg_prod;

    std::string _out_ctor_prod;
    std::string _out_img_prod;

    std::string _out_super_ctor_prod;
    std::string _out_super_img_prod;

    std::vector<bool> _idx_v;
    std::vector<std::pair<int,int> > _par_v;
    
  };

  class VertexFilterProcessFactory : public ProcessFactoryBase {
  public:
    VertexFilterProcessFactory() { ProcessFactory::get().add_factory("VertexFilter",this); }
    ~VertexFilterProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new VertexFilter(instance_name); }
  };

}
#endif

