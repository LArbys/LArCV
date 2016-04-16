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
      fDivisionFile       = cfg.get<std::string>("DivisionFile");
      fNPlanes            = cfg.get<int>( "NPlanes", 3 );
      fTickStart          = cfg.get<int>( "TickStart", 2400 );
      fTickDownSample     = cfg.get<int>( "TickDownSampleFactor", 6 );
      fMaxWireImageWidth  = cfg.get<int>( "MaxWireImageWidth" );
      fInputImageProducer = cfg.get<std::string>( "InputImageProducer" );
      fInputROIProducer   = cfg.get<std::string>( "InputROIProducer" );
    }
    
    void HiResImageDivider::initialize()
    {
      // The image divisions are calculated before hand in the fixed grid model
      // we load the prefined region image definitions here
      
      TFile* f = new TFile( fDivisionFile.c_str(), "open" );
      TTree* t = (TTree*)f->Get("imagedivider/regionInfo");
      int **planebounds = new int*[fNPlanes];
      int planenwires[fNPlanes];
      for (int p=0; p<fNPlanes; p++) {
	planebounds[p] = new int[2];
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
      int tickbounds[2];

      t->SetBranchAddress( "zbounds", zbounds );
      t->SetBranchAddress( "ybounds", ybounds );
      t->SetBranchAddress( "xbounds", xbounds );
      t->SetBranchAddress( "tickbounds", tickbounds );

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
      // This processor does the following:
      // 1) read in hi-res images (from producer specified in config)
      // 2) (how to choose which one we clip?)
      
    }
    
    void HiResImageDivider::finalize(TFile* ana_file)
    {}

    // -------------------------------------------------------

    bool HiResImageDivider::decideToContinueBasedOnROI( const larcv::ROI& roi ) {
    }

    int HiResImageDivider::findVertexDivisionUsingROI( const larcv::ROI& roi ) {
      int regionindex = 0;
      for ( std::vector< larcv::hires::DivisionDef >::iterator it=m_divisions.begin(); it!=m_divisions.end(); it++) {
	DivisionDef const& div = (*it);
	if ( div.isInsideDetRegion( roi.X(), roi.Y(), roi.Z() ) )
	  return regionindex;
	regionindex++;
      }
      return -1;
    }
    
    bool HiResImageDivider::decideToKeepBasedOnROI( const larcv::ROI& roi ) {
    }

  }
}
#endif
