#include "PMTWireWeights.h"
#include <iostream>

namespace larcv {
  namespace pmtweights {
    
    PMTWireWeights::PMTWireWeights( std::string geofile, int wire_rows ) {
      fNWires = wire_rows;

      //fGeoInfoFile = "configfiles/geoinfo.root";
      fGeoInfoFile = geofile;
      //std::cout << "Filling Weights using " << fGeoInfoFile << std::endl;
      fGeoFile = new TFile( Form("%s/app/PMTWeights/dat/%s",getenv("LARCV_BASEDIR"),fGeoInfoFile.c_str()), "OPEN" );

      // Get the PMT Info
      fNPMTs = 32;
      fPMTTree  = (TTree*)fGeoFile->Get( "imagedivider/pmtInfo" );
      int femch;
      float pos[3];
      fPMTTree->SetBranchAddress( "femch", &femch );
      fPMTTree->SetBranchAddress( "pos", pos );
      for (int n=0; n<fNPMTs; n++) {
	fPMTTree->GetEntry(n);
	//std::cout << "[pmt " << femch << "] ";
	for (int i=0; i<3; i++) {
	  pmtpos[femch][i] = pos[i];
	  //std::cout << pmtpos[femch][i] << " ";
	}
	//std::cout << std::endl;
      }

      // Get the Wire Info
      fWireTree = (TTree*)fGeoFile->Get( "imagedivider/wireInfo" );
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
	  m_WireData[planeID] = WireData( planeID );
	}
	m_WireData[planeID].addWire( wireID, start, end );
      }

//       std::cout << "Number of wire data stored (per plane)" << std::endl;
//       for ( std::map<int,WireData>::iterator it=m_WireData.begin(); it!=m_WireData.end(); it++) {
// 	std::cout << " [Plane " << (*it).first << "]: " << (*it).second.nwires() << std::endl;
//       }

      
      // Configure
      configure();

      fGeoFile->Close();
    }
    
    PMTWireWeights::~PMTWireWeights() {
    }
    
    void PMTWireWeights::configure() {
      // here build the set of weights
      // each PMT gets assigned a weight that is: w = d/(D)
      // d is the shortest distance between the wire and pmt center
      // D is the sum of all such distances such that sum(w,NPMTS=32) = 1.0
      // we make a weight matrix W = [ N, M ] where N is the number of wires and M is the number of PMTs
      // the way we will use this is to assign each PMT a value: z = q/Q so that z is the fraction of charge seen in the trigger window
      // the weight assigned to each wire will be W*Z, where Z is the vector of z values for all M PMTs.

      for ( std::map<int,WireData>::iterator it=m_WireData.begin(); it!=m_WireData.end(); it++ ) {
	int plane = (*it).first;
	WireData const& data = (*it).second;
	int nwires = data.nwires();
	if ( fNWires>0 ) 
	  nwires = fNWires;

	//cv::Mat mat( nwires, fNPMTs, CV_32F );
	larcv::Image2D mat( nwires, fNPMTs );

	int iwires = 0;
	for ( std::set< int >::iterator it_wire=data.wireIDs.begin(); it_wire!=data.wireIDs.end(); it_wire++ ) {
	  int wireid = (*it_wire);
	  // we first need to project the data into 2D: z,y -> (x,y)
	  std::vector< float > const& start = (*(data.wireStart.find(wireid))).second;
	  //std::vector< float > const& end   = (*(data.wireEnd.find(wireid))).second;
	  float s2[2] = { start.at(2), start.at(1) };
	  //float e2[2] = { end.at(2),   end.at(1)   };
	  float l2 = (*(data.wireL2.find(wireid))).second;

	  std::vector<float> dists(fNPMTs,0.0);
	  float denom = 1000.0;
	  //std::cout << "[plane " << plane << ", wire " << wireid << "] ";
	  for (int ipmt=0; ipmt<fNPMTs; ipmt++) {
	    float p2[2] = { pmtpos[ipmt][2], pmtpos[ipmt][1] };
	    float d = getDistance2D( s2, s2, p2, l2 );
	    dists[ipmt] = d;
	    //std::cout << d << " ";
	    //denom += d;
	  }
	  //std::cout << std::endl;

	  // populate the matrix
	  for (int ipmt=0; ipmt<fNPMTs; ipmt++) {
	    //mat.at<float>( wireid, ipmt ) = dists.at(ipmt)/denom;
	    mat.set_pixel( wireid, ipmt, dists.at(ipmt)/denom );
	  }
	  iwires++;
	}//end of wire loop
	planeWeights[plane] = mat;
      }//end of plane loop
    }//end of configure()
    
    float PMTWireWeights::getDistance2D( float s[], float e[], float p[], float l2 ) {
      // we assume that the user has projected the information into 2D
      // we calculate the projection of p (the pmt pos) onto the line segment formed by (e-s), the end point of the wires
      // where s is the origin of the coorindate system

      // since we want distance from wire to pmt, perpendicular to the wire, we can form
      // a right triangle. the distance is l = sqrt(P^2 - a^2), where a is the projection vector

      float ps[2]; // vector from wire start to PMT pos
      float es[2]; // vector from wire start to wire end
      float dot = 0.0;  // dot product of the above
      float psnorm = 0.0; // distane from wire start to pmt pos
      for (int i=0; i<2; i++) {
	ps[i] = p[i]-s[i];
	es[i] = e[i]-s[i];
	dot += ps[i]*es[i];
	psnorm += ps[i]*ps[i];
      }
      float dist = sqrt( psnorm - dot*dot/l2 );
      return dist;
    }

  }
}
