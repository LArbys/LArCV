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


  float UBWireTool::calculateIntersectionTriangle( const std::vector<float>& p0, const std::vector<float>& p1, const std::vector<float>& p2 ) {
    // returns area of intersection triangle for 3 wires
    // inputs
    // ------
    //  p0, p1, p3: (2D coordinate for wires from each plane (0,1,2))

    float d01[2] = { p1[0]-p0[0], p1[1]-p0[1] }; // line segment from p0 -> p1
    float d20[2] = { p0[0]-p2[0], p0[1]-p2[1] }; // line segment from p2 -> p0
    float base = sqrt( d01[0]*d01[0] + d01[1]*d01[1] ); // length of d01, acting as our base
    float vp[2] = { d01[1], -d01[0] }; // direction perpendicular to d01 ( d01.vp = 0.0 )
    float height = 0.0;
    // project d20 onto vp to get distance from d01 line to p2
    for (int i=0; i<2; i++) height += d20[i]*vp[i]; 
    height /= base;
    
    return 0.5*std::fabs(height*base); // return area of triangle
  }

  void UBWireTool::lineSegmentIntersection2D( const std::vector< std::vector<float> >& ls1, const std::vector< std::vector<float> >& ls2, 
					      std::vector<float>& insec, int& crosses ) {
    // find intersection for two line segments in 2D
    // inputs
    // ------
    // ls1,ls2: line segment 1 (x,y) for start and end point, ls1[ start or end ][ x or y ]
    // 
    // outputs
    // -------
    //  insec: intersection point (x,y)
    //  crosses: does the intersection occur, and if it does, is it within the bounds of the line segments

    //std::cout << "[lineSegmentIntersection2D] begin" << std::endl;
    //std::cout << "  testing: ls1=(" << ls1[0][0] << "," << ls1[0][1] << ") -> (" << ls1[1][0] << ","<< ls1[1][1] <<")" << std::endl;
    //std::cout << "  testing: ls2=(" << ls2[0][0] << "," << ls2[0][1] << ") -> (" << ls2[1][0] << ","<< ls2[1][1] <<")" << std::endl;
    insec.resize(2,0.0);
    float Y1 = ls1[1][1] - ls1[0][1];
    float X1 = ls1[0][0] - ls1[1][0];
    float C1 = Y1*ls1[0][0] + X1*ls1[0][1];

    float Y2 = ls2[1][1] - ls2[0][1];
    float X2 = ls2[0][0] - ls2[1][0];
    float C2 = Y2*ls2[0][0] + X2*ls2[0][1];

    float det = Y1*X2 - Y2*X1;
    if ( det==0 ) { 
      // parallel
      crosses = 0;
      //std::cout << "[lineSegmentIntersection2D] end.  parallel lines" << std::endl;
      return;
    }

    insec[0] = (X2*C1 - X1*C2)/det;
    insec[1] = (Y1*C2 - Y2*C1)/det;

    // check if interesction within line segments
    // padding needed for y-wire which is vertical
    crosses = 1;
    for (int i=0; i<2; i++) {
      if ( std::min( ls1[0][i]-0.15, ls1[1][i]-0.15 ) > insec[i] || std::max( ls1[0][i]+0.15, ls1[1][i]+0.15 )<insec[i] )
	crosses = 0;
      if ( std::min( ls2[0][i]-0.15, ls2[1][i]-0.15 ) > insec[i] || std::max( ls2[0][i]+0.15, ls2[1][i]+0.15 )<insec[i] )
	crosses = 0;
      if ( crosses==0 )
	break;
    }
    
    //std::cout << "  crosses=" << crosses << ": intersection=(" << insec[0] << "," << insec[1] << ")" << std::endl;
    //std::cout << "[lineSegmentIntersection2D] end." << std::endl;
    return;
  }

  void UBWireTool::findWireIntersections( const std::vector< std::vector<int> >& wirelists, 
					  const std::vector< std::vector<float> >& valid_range,
					  std::vector< std::vector<int> >& intersections3plane,
					  std::vector< std::vector<float> >& vertex3plane,
					  std::vector<float>& areas3plane,
					  std::vector< std::vector<int> >& intersections2plane, std::vector< std::vector<float> >& vertex2plane ) {
    // takes in lists of wire id numbers and returns intersection combinrations, the area of the triangle they make (if 3D), and 2 plane intersection candidats
    // we are going to go ahead and assume 3 planes for now. we can modify for generic number of planes later (never)

    // inputs
    // ------
    //  wirelists: list of wires to test, one for each plane wirelists[plane][ test wire ]
    //  valid_range: range for each plane for which intersections are valid (in real space), valid_range[ plane ][0] or valid_range[ plane ][1]
    //  
    // outputs
    // -------
    //  intersections3plane: list of intersections. each intersection is a list of 3 wires, one from each plane, intersections3plane[intersection #][ plane ]
    //  vertex3plane: list of (z,y) vertex
    //  areas3plane: list of intersection triangle area for each 3-plane intersection
    //  intersections2plane: list of intersections for 2 planes, intersections2plane[intersection #][ plane ]
    //  vertex2plane: list of 2-plane intersection (z,y) vertices

    // we're going to loop through all combination of wires. This can get bad fast. N^p problem...
    // expecting about O(10) endpoints per flash.  worst case, looking at 1000 combinations.
    // but we use the validity ranges to start removing intersections we don't care about

    const int nplanes = wirelists.size();
    std::set< std::vector<int> > checked2plane;
    std::set< std::vector<int> > checked3plane;

    UBWireTool* _g = UBWireTool::_get_global_instance();
    
    for (int p0=0;p0<nplanes;p0++) {
      // loop through first wire plane
      for (int idx0=0; idx0<wirelists.at(p0).size(); idx0++) {
      
	// get the first wire
	int wid0 = wirelists.at(p0).at(idx0);
	const std::vector<float>& start0 = _g->m_WireData[p0].wireStart.find(wid0)->second;
	const std::vector<float>& end0   = _g->m_WireData[p0].wireEnd.find(wid0)->second;
	
	// go to the other planes and check the wires there
	for (int p1=p0+1; p1<nplanes; p1++) {
	  // get wire on this plane
	  for (int idx1=0; idx1<wirelists.at(p1).size(); idx1++) {
	    int wid1 = wirelists.at(p1).at(idx1);

	    std::vector< int > combo2d(3,-1);
	    combo2d[p0] = wid0;
	    combo2d[p1] = wid1;
	    if ( checked2plane.find( combo2d )==checked2plane.end() )
	      checked2plane.insert( combo2d );
	    else {
	      //std::cout << "  .. already checked: (" << combo2d[0] << "," << combo2d[1] << "," << combo2d[2] << ")" << std::endl;
	      continue;
	    }

	    const std::vector<float>& start1 = _g->m_WireData[p1].wireStart.find(wid1)->second;
	    const std::vector<float>& end1   = _g->m_WireData[p1].wireEnd.find(wid1)->second;
	    
	    // change line end points from 3D to 2D (x,y,z) -> (z,y)
	    std::vector< std::vector<float> > ls0(2); // line segment 1
	    ls0[0].push_back( start0[2] );
	    ls0[0].push_back( start0[1] );
	    ls0[1].push_back( end0[2] );
	    ls0[1].push_back( end0[1] );
	    std::vector< std::vector<float> > ls1(2); // line segment 2
	    ls1[0].push_back( start1[2] );
	    ls1[0].push_back( start1[1] );
	    ls1[1].push_back( end1[2] );
	    ls1[1].push_back( end1[1] );
	    
	    // test intersection
	    std::vector<float> intersection01; 
	    int crosses01 = 0;
	    lineSegmentIntersection2D( ls0, ls1, intersection01, crosses01 );
	    if ( !crosses01 ) {
	      //std::cout << "  (" << p0 << "," << wid0 << ") and (" << p1 << "," << wid1 << ") does not cross" << std::endl;
	      continue; // move on if doesn't cross
	    }
	    bool valid = true;
	    for (int i=0; i<2; i++) {
	      if ( intersection01[i]<valid_range[i][0] || intersection01[i]>valid_range[i][1] ) {
		valid = false;
		break;
	      }
	    }
	    if ( !valid ) {
	      //std::cout << "  (" << p0 << "," << wid0 << ") and (" << p1 << "," << wid1 << ") crosses but not valid. "
	      //	<< " intersection(z,y)=(" << intersection01[0] << "," << intersection01[1] << ")"
	      //	<< std::endl;
	      continue; // not a valid intersection
	    }

	    // we got a 2plane intersection
	    std::vector<int> p2intersect(3,-1);
	    p2intersect[p0] = wid0;
	    p2intersect[p1] = wid1;
	    intersections2plane.emplace_back( p2intersect );
	    vertex2plane.push_back( intersection01 );

	    // we try for the 3plane intersection
	    int p2 = 2;
	    if ( p0==0 && p1==1 ) p2 = 2;
	    else if ( p0==0 && p1==2 ) p2 = 1;
	    else if ( p0==1 && p1==2 ) p2 = 0;
	    else
	      continue;


	    for (int idx2=0; idx2<(int)wirelists.at(p2).size(); idx2++) {
	      int wid2 = wirelists.at(p2).at(idx2);

	      std::vector< int > combo3d(3,-1);
	      combo3d[p0] = wid0;
	      combo3d[p1] = wid1;
	      combo3d[p2] = wid2;
	      if ( checked3plane.find( combo3d )==checked3plane.end() )
		checked3plane.insert( combo3d );
	      else {
		//std::cout << "    ... already checked: (" << combo3d[0] << ", " << combo3d[1] << ", " << combo3d[2] << ")" << std::endl;
		continue;
	      }

	      const std::vector<float>& start2 = _g->m_WireData[p2].wireStart.find(wid2)->second;
	      const std::vector<float>& end2   = _g->m_WireData[p2].wireEnd.find(wid2)->second;

	      std::vector< std::vector<float> > ls2(2); // line segment 2
	      ls2[0].push_back( start2[2] );
	      ls2[0].push_back( start2[1] );
	      ls2[1].push_back( end2[2] );
	      ls2[1].push_back( end2[1] );

	      std::vector<float> intersection02;
	      int crosses02 = 0;
	      lineSegmentIntersection2D( ls0, ls2, intersection02, crosses02 );

	      std::vector<float> intersection12;
	      int crosses12 = 0;
	      lineSegmentIntersection2D( ls1, ls2, intersection12, crosses12 );
	      
	      if ( !crosses02 || !crosses12 )  {
		//std::cout << "  3-plane check: one combination does not cross, crosses02=" << crosses02 << " crosses12=" << crosses12 << std::endl;
		continue;
	      }

	      bool valid2 = true;
	      for (int i=0; i<2; i++) {
		if ( intersection02[i]<valid_range[i][0] || intersection02[i]>valid_range[i][1] ) {
		  //std::cout << "  3-plane check: intersection02 not valid" << std::endl;
		  valid = false;
		  break;
		}
		if ( intersection12[i]<valid_range[i][0] || intersection12[i]>valid_range[i][1] ) {
		  //std::cout << "  3-plane check: intersection12 not valid" << std::endl;
		  valid = false;
		  break;
		}
	      }
	      if ( !valid2 )
		continue; // not a valid intersection
	      
	      // got a 3 plane intersection!
	      std::vector<int> p3intersect(3,-1);
	      p3intersect[p0] = wid0;
	      p3intersect[p1] = wid1;
	      p3intersect[p2] = wid2;
	      intersections3plane.emplace_back( p3intersect );
	      // get score for 3 plane intersections
	      float area = calculateIntersectionTriangle( intersection01, intersection02, intersection12 );
	      areas3plane.push_back(area);
	      // get the 3 plane vertex
	      std::vector<float> vert3(2,0.0);
	      for (int i=0; i<2; i++) 
		vert3[i] = (intersection01[i]+intersection02[i]+intersection12[i] )/3.0;
	      vertex3plane.emplace_back( vert3 );
	    }//end of loop over p2 wires
	  }//end of loop over p1 wires
	}//end of loop over p1 planes
      }//end of loop over p0 wires
    }//end of loop over p0 planes
  }//end of findWireIntersections(...)
  
  
  
}
