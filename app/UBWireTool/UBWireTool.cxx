#include "UBWireTool.h"
#include "TFile.h"
#include "TTree.h"

namespace larcv {

  UBWireTool* UBWireTool::_g_ubwiretool = NULL;

  UBWireTool::UBWireTool() {
    loadData();
  }

  void UBWireTool::loadData() {

    TFile fGeoFile( Form("%s/app/PMTWeights/dat/geoinfo.root",getenv("LARCV_BASEDIR")), "OPEN" );
    
    // Get the Wire Info
    TTree* fWireTree = (TTree*)fGeoFile.Get( "imagedivider/wireInfo" );
    int wireID;
    int planeID;
    float start[3];
    float end[3];
    fWireTree->SetBranchAddress( "wireID", &wireID );
    fWireTree->SetBranchAddress( "plane",  &planeID );
    fWireTree->SetBranchAddress( "start", start );
    fWireTree->SetBranchAddress( "end", end );
      
    int nentries = fWireTree->GetEntries();
    for ( int ientry=0; ientry<nentries; ientry++ ) {
      fWireTree->GetEntry(ientry);
      if ( m_WireData.find( planeID )==m_WireData.end() ) {
	// cannot find instance of wire data for plane id. make one.
	m_WireData[planeID] = larcv::WireData( planeID );
      }
      // start is always the one with the lowest y-value
      if ( start[1]>end[1] ) {
	// swap start end end
	for (int i=0; i<3; i++) {
	  float temp = start[i];
	  start[i] = end[i];
	  end[i] = temp;
	}
      }
      m_WireData[planeID].addWire( wireID, start, end );
    }
    // insert empty instance
    m_WireData[-1] = larcv::WireData( -1 );
    
    fGeoFile.Close();
    
  }

  UBWireTool* UBWireTool::_get_global_instance() {
    if ( _g_ubwiretool==NULL )
      _g_ubwiretool = new UBWireTool();
    return _g_ubwiretool;
  }

  const larcv::WireData& UBWireTool::getWireData(int p) {
    UBWireTool* _g = UBWireTool::_get_global_instance();
    auto it = _g->m_WireData.find(p);
    if ( it!=_g->m_WireData.end() ) {
      return it->second;
    }
    _g->LARCV_CRITICAL() << "Asked UBWireTool for wire data from a plane that doesn't exist (p=" << p << ")" << std::endl;
    throw larbys();
    return _g->m_WireData[-1]; // never gets here
  }
  
  
  
}
