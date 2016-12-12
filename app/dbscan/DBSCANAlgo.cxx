#include "DBSCANAlgo.h"


namespace dbscan {

  dbscanOutput DBSCANAlgo::scan( dbPoints input, int minPts, double eps, bool borderPoints, double approx ) {

    // allocate output
    dbscanOutput output;

    int npts  = input.size();
    if ( npts==0 )
      return output;
    int ndims = input.at(0).size();
    std::vector<double> weights;

    // create bdtree structure object
    ann::ANNAlgo bdtree( npts, ndims );
    
    // now fill it
    for (int i=0; i<npts; i++) {
      bdtree.setPoint( i, input.at(i) );
    }

    // initialize structure
    bdtree.initialize();

    // kd-tree uses squared distances
    double eps2 = eps*eps;

    bool weighted = false;
    double Nweight = 0.0;
    if (weights.size() != 0) {
      if (weights.size() != npts) {
	std::cout << "length of weights vector is incompatible with data." << std::endl;
	return output;
      }
      weighted = true;
    }

    // DBSCAN
    std::vector<bool> visited(npts, false);
    std::vector<int>  N, N2;
    output.clusterid.resize(npts,-1);
    output.nneighbors.resize(npts,0);
    
    for (int i=0; i<npts; i++) {
      // check for interrupt
      //if (!(i % 100)) Rcpp::checkUserInterrupt(); //convert this to python
      
      if (visited[i]) continue;

      // ---------------------------------------------------------
      // Region Query
      // ---------------------------------------------------------
      // replacing this
      //N = regionQuery(i, dataPts, kdTree, eps2, approx);
      //if(frNN.size())   N = Rcpp::as< std::vector<int> >(frNN[i]);
      //else              N = regionQuery(i, dataPts, kdTree, eps2, approx);
      // ---------------------------------------------------------
      N = bdtree.regionQuery( i, eps2, approx );
      output.nneighbors.at( i ) = N.size();
      // Note: the points are not sorted by distance!
      // ---------------------------------------------------------

      // noise points stay unassigned for now
      //if (weighted) Nweight = sum(weights[IntegerVector(N.begin(), N.end())]) +
      //       if (weighted) {
      // 	// This should work, but Rcpp has a problem with the sugar expression!
      // 	// Assigning the subselection forces it to be materialized.
      // 	// Nweight = sum(weights[IntegerVector(N.begin(), N.end())]) +
      // 	// weights[i];
      // 	NumericVector w = weights[IntegerVector(N.begin(), N.end())];
      // 	Nweight = sum(w);
      //       } else Nweight = N.size();
      
      //       if (Nweight < minPts) continue;
      
      // start new cluster and expand
      std::vector<int> cluster;
      cluster.push_back(i);
      visited[i] = true;

      while (!N.empty()) {
	int j = N.back();
	N.pop_back();

	if (visited[j]) continue; // point already processed
	visited[j] = true;

// 	if(frNN.size())   N2 = Rcpp::as< std::vector<int> >(frNN[j]);
// 	else              N2 = regionQuery(j, dataPts, kdTree, eps2, approx);
	N2 = bdtree.regionQuery(j,eps2,approx);
		
	if (weighted) {
	  //NumericVector w = weights[IntegerVector(N2.begin(), N2.end())];
	  //Nweight = sum(w);
	  Nweight = 0.0;
	  for (int in2=0; in2<(int)N2.size(); in2++) {
	    Nweight += weights.at( N2.at(in2) );
	  }
	} else Nweight = (double)N2.size();

	if (Nweight >= minPts) { // expand neighborhood
	  // this is faster than set_union and does not need sort! visited takes
	  // care of duplicates.
	  std::copy(N2.begin(), N2.end(), std::back_inserter(N));
	}

	// for DBSCAN* (borderPoints==FALSE) border points are considered noise
	if(Nweight >= minPts || borderPoints) cluster.push_back(j);
      }

      // add cluster to list
      output.clusters.push_back(cluster);
    }

    // prepare cluster vector
    // unassigned points are noise (cluster 0)
    output.clusterid.resize(npts,0);
    for (std::size_t i=0; i<output.clusters.size(); i++) {
      for (std::size_t j=0; j<output.clusters[i].size(); j++) {
	output.clusterid[output.clusters[i][j]] = i; // R adds one probably because 1-indexed vectors: WTF
      }
    }

    // clean up
    //ann::ANNAlgo::cleanup();    
    
    return output;
    
  }


  // DBSCAN Output Functions
  int dbscanOutput::findMatchingCluster( const std::vector<double>& testpoint, const dbPoints& data, const double radius ) {
    // inputs
    // ------
    //  testpoint: test point in data space. will try to find closest cluster to this point
    //  data: list of all points in the data. the same data set used to form this cluster output
    //  radius: maximum radius the test point needs to be to some member data point in a cluster
    // output
    // ------
    //  return int: index to cluster that contains a point closest to the test point
    // this is an order N search

    // -------------------------------
    // list of exceptions
    class querypoint_mismatch_ex: public std::exception
    {
      virtual const char* what() const throw()
      {
	return "dbscanOutput::findMatchingCluster. Query Point and data point sizes do not match.";
      }
    } querypoint_mismatch;
    // -------------------------------

    int matching_cluster = -1;
    double r2 = radius*radius;
    double closest_r2 = 0.0;
    for (int ic=0; ic<clusters.size(); ic++) {
      const dbCluster &cluster = clusters.at(ic);
      int nhits = clusters.at(ic).size();
      for (int ihit=0; ihit<nhits; ihit++) {
	int hitidx = cluster.at(ihit);
	double dist = 0.;
	if ( testpoint.size()!=data.at(hitidx).size() ) {
	  throw querypoint_mismatch;
	}
	for (int i=0; i<testpoint.size();i++) {
	  dist += (data.at(hitidx).at(i)-testpoint.at(i))*(data.at(hitidx).at(i)-testpoint.at(i));
	}
	if (dist<r2 && (matching_cluster==-1 || dist<closest_r2) ) {
	  matching_cluster = ic;
	  closest_r2 = dist;
	}
      }
    }
    return matching_cluster;
  }
}
