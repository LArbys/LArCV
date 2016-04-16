#ifndef __HIRESIMAGEDIVIDER_CXX__
#define __HIRESIMAGEDIVIDER_CXX__

#include "HiResImageDivider.h"
#include "TFile.h"
#include "TTree.h"

namespace larcv {
  namespace hires {
    static HiResImageDividerProcessFactory __global_HiResImageDividerProcessFactory__;
    
    HiResImageDivider::HiResImageDivider(const std::string name)
      : ProcessBase(name)
    {}
    
    void HiResImageDivider::configure(const PSet& cfg)
    {
      fDivisionFile      = cfg.get<std::string>("DivisionFile");
      fNPlanes           = cfg.get<int>( "NPlanes" );
      fTickImageWidth    = cfg.get<int>( "TickImageWidth" );
      fMaxWireImageWidth = cfg.get<int>( "MaxWireImageWidth" );
    }
    
    void HiResImageDivider::initialize()
    {
      TFile* f = new TFile( fDivisionFile.c_str(), "open" );
      TTree* t = (TTree*)f->Get("imagedivider/regionInfo");
      float **planebounds = new float*[fNPlanes];
      int planenwires[fNPlanes];
      for (int p=0; p<fNPlanes; p++) {
	planebounds[p] = new float[2];
	char bname1[100];
	sprintf( bname1, "plane%d_wirebounds", p );
	t->SetBranchAddress( bname1, planebounds[p] );

	char bname2[100];
	sprintf( bname2, "plane%d_nwires", p );
	t->SetBranchAddress( bname2, &(planenwires[p]) );

      }
      
      float zbounds[2];
      float xbounds[2];
      float ybounds[2];
      int tickbounds[2] = { 0, fTickImageWidth };

      t->SetBranchAddress( "zbounds", zbounds );
      t->SetBranchAddress( "ybounds", ybounds );
      t->SetBranchAddress( "xbounds", xbounds );

      fMaxWireInRegion = 0;
      size_t entry = 0;
      size_t bytes = t->GetEntry(0);
      while ( bytes>0 ) {
	for (int p=0; p<3; p++) {
	  if ( fMaxWireInRegion<planenwires[p] )
	    fMaxWireInRegion = planenwires[p];
	}
	int plane0[2], plane1[2], plane2[2];
	for (int i=0; i<2; i++) {
	  plane0[i] = (int)planebounds[0][i];
	  plane1[i] = (int)planebounds[1][i];
	  plane2[i] = (int)planebounds[2][i];
	}
	DivisionDef div( plane0, plane1, plane2, tickbounds, xbounds, ybounds, zbounds );
	
	m_divisions.emplace_back( div );
	entry++;
	bytes = t->GetEntry(entry);
      }

      for (int p=0; p<fNPlanes; p++) {
	delete [] planebounds[p];
      }
      delete [] planebounds;
      
      f->Close();
      
    }
    
    bool HiResImageDivider::process(IOManager& mgr)
    {
      
      
    }
    
    void HiResImageDivider::finalize(TFile* ana_file)
    {}
    
  }
}
#endif
