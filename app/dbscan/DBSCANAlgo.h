#ifndef __DBSCAN_ALGO__
#define __DBSCAN_ALGO__

#include "ANN/ANNAlgo.h"

#include <vector>
#include <utility> // for pair

#include "DataFormat/ImageMeta.h"

namespace dbscan {

  typedef std::vector< std::vector<double> > dbPoints; // list of (x,y,z,....) points
  typedef std::vector< int > dbCluster; // list of indices to dbPoints
  typedef std::vector< std::vector<int> >  dbClusters; // list of list of indices to provided dbPoints

  class dbscanOutput {
  public: 
    dbscanOutput() {
      clusters.clear();
      clusterid.clear();
    };
    virtual ~dbscanOutput() {};
    dbClusters clusters;
    std::vector<int> clusterid;
    std::vector<int> nneighbors;

    // utility functions. seems like useful, common tasks, but would make more sense to separate these from the data itself?
    int findMatchingCluster( const std::vector<double>& testpoint, const dbPoints& data, const double radius ) const;
    void closestHitsInCluster( const int clusterid, const std::vector<double>& test_pos, const dbPoints& src_data, 
			       const larcv::ImageMeta& meta, const float cm_per_tick, const float cm_per_wire,
			       std::vector< std::pair<int,double> >& hitlist, const int max_nhits=-1 ) const;

  };
  

  class DBSCANAlgo {

  public:
    DBSCANAlgo() {};
    virtual ~DBSCANAlgo() {};

    dbscanOutput scan( dbPoints input, int minPoints, double eps, bool borderPoints=false, double approx=0.0 );

  };


}

#endif
