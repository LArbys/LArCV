#ifndef __DBSCAN_ALGO__
#define __DBSCAN_ALGO__

#include "ANN/ANNAlgo.h"

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
    int findMatchingCluster( const std::vector<double>& testpoint, const dbPoints& data, const double radius );
        
  };
  

  class DBSCANAlgo {

  public:
    DBSCANAlgo() {};
    virtual ~DBSCANAlgo() {};

    dbscanOutput scan( dbPoints input, int minPoints, double eps, bool borderPoints=false, double approx=0.0 );

  };


}

#endif
