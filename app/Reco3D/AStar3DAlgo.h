#ifndef __ASTAR_3D_ALGO__
#define __ASTAR_3D_ALGO__

/** 

AStar algorithm assuming 3D!! grid points. 

Uses Image2D to hold image.

 **/

#include <iostream>
#include <queue>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <array>

// larcv
#include "DataFormat/Image2D.h"
#include "Base/PSet.h"

#include "AStar3DAlgoConfig.h"

namespace larcv {

  // LATTICE POSITION OBJECT
  class A3DPixPos_t : public std::array<int,3> {
    public:
      A3DPixPos_t() {
        for (int i=0; i<3; i++) at(i) = -1;
      };
      A3DPixPos_t(int u, int v, int w) {
        this->at(0) = u;
        this->at(1) = v;
        this->at(2) = w;
      };
      virtual ~A3DPixPos_t() {};
  };

  // NODE DEFINITION
  class AStar3DNode  {
  public:
    AStar3DNode() {
      u = v = w = 0;
      fscore=gscore=kscore=0.0;
      closed = false;
      inopenset = false;
      dir3d.resize(3,0.0);
      tyz.resize(3,0.0);
      prev = NULL;
      row = 0;
      within_image=false;
      pixval.resize(3,0.0);
      badchnode = false;
    };
    AStar3DNode( int u_, int v_, int w_, std::vector<float> tyz_ ) { 
      u = u_; 
      v = v_;
      w = w_;
      nodeid[0] = u;
      nodeid[1] = v;
      nodeid[2] = w;
      fscore=0.0;
      gscore=0.0;
      kscore=0.0;
      within_image=false;
      dir3d.resize(3,0);
      tyz = tyz_;
      closed = false;
      inopenset = false;
      prev = NULL;
      row = 0;
      pixval.resize(3,0.0);
      badchnode = false;
    };
    AStar3DNode( const AStar3DNode& src ) {
      u = src.u;
      v = src.v;
      w = src.w;
      nodeid = src.nodeid;
      fscore = src.fscore;
      gscore = src.gscore;
      kscore = src.kscore;
      dir3d = src.dir3d;
      tyz = src.tyz;
      closed = src.closed;
      within_image = src.within_image;
      inopenset = src.inopenset;
      prev = src.prev;
      row = src.row;
      cols = src.cols;
      pixval = src.pixval;
      badchnode = src.badchnode;
    };
    virtual ~AStar3DNode() {};

    int u; // lattice position corresponding to time or x
    int v; // lattice position corresponding to y
    int w; // latrice position corresponding to z
    A3DPixPos_t nodeid; // latice position key
    float fscore; // score of path: past path + to goal heuristic
    float gscore; // past path score
    float kscore; // curvature penalty

    std::vector<float> dir3d;
    std::vector<float> tyz; // (tick, y, z) in detector coordinates
    int row;
    std::vector<int> cols;
    std::vector<float> pixval;
    bool within_image;
    bool inopenset;
    bool closed;
    bool badchnode; // tag if its a badch node
    AStar3DNode* prev;

    bool operator <(const AStar3DNode& rhs ) const {
      //std::cout << "astar< " << fscore << " < " << rhs.fscore << std::endl;
      if ( fscore > rhs.fscore ) // lower scores get priority
        return true;
      return false;
    };
    bool operator==(const AStar3DNode& rhs ) const {
      if ( u==rhs.u && v==rhs.v && w==rhs.w ) return true;
      return false;
    };

    bool operator!=(const AStar3DNode& rhs ) const {
      if ( u!=rhs.u || v!=rhs.v || w!=rhs.w ) return true;
      return false;
    };

    std::string str() const { 
      std::stringstream ss;
      ss << "[node=(" << u << "," << v << "," << w << ") "
      	 << "tyz=(" << tyz[0] << "," << tyz[1] << "," << tyz[2] << ") "
      	 << "f=" << fscore << " "
      	 << "g=" << gscore << " "
      	 << "k=" << kscore << " "
      	 << "]";
      return ss.str();
    }

  };

  typedef std::vector< std::vector<AStar3DNode> > PathList_t;      

  //LATTICE DEFINITION
  typedef std::pair< A3DPixPos_t, AStar3DNode* > a3dpos_pair_t;

  class Lattice : public std::map< A3DPixPos_t, AStar3DNode* > {
  public:
    Lattice( const std::vector<float>& origin, const std::vector<int>& widths, const std::vector<float>& pixel_lengths, 
    	const std::vector<const larcv::ImageMeta* >& meta_v ) {
      m_origin = origin;
      m_widths = widths;
      m_cm_per_pixel = pixel_lengths;
      m_meta_v = meta_v;
    }
    virtual ~Lattice() { cleanup(); };

    A3DPixPos_t getNodePos( const std::vector<float>& pos );
    std::vector<float> getPos( const int u, const int v, const int w);
    std::vector<float> getPos( const A3DPixPos_t& );
    AStar3DNode* getNode( const std::vector<float>& pos );
    AStar3DNode* getNode( const int u, const int v, const int w);
    AStar3DNode* getNode( const A3DPixPos_t& nodeid );
    void getImageCoordinates( const A3DPixPos_t& nodeid, int& row, std::vector<int>& cols, bool& within_image );
  
    const std::vector<int>& widths() const { return m_widths; }
    const std::vector<float>& origin() const { return m_origin; }
    const std::vector<float>& pixel_len() const { return m_cm_per_pixel; }    

    void setStartNodeWithPadding( AStar3DNode* start, const std::vector<int>& start_padding );

  protected:

    std::vector<float> m_origin;
    std::vector<int> m_widths;
    std::vector<float> m_cm_per_pixel;
    std::vector< const larcv::ImageMeta* > m_meta_v;

    // start padding
    AStar3DNode* m_start_node;
    A3DPixPos_t  m_start_nodeid;
    std::vector<int> m_start_padding;
    bool fCheckStartPadding;

    void cleanup(); // destroy nodes allocated on the heap


  };


  // NODE CONTAINER
  class AStar3DNodePtrList : public std::vector< AStar3DNode* > {

    public:
      AStar3DNodePtrList() {};
      virtual ~AStar3DNodePtrList() {};

    protected:

    // we want to sort from largest to smallest, because we will pop the nodes off the back of the various sets
    struct dirnodeptr_compare_t {
      bool operator() (const AStar3DNode* lhs, const AStar3DNode* rhs ) const {
        if ( lhs->fscore > rhs->fscore )
          return true;
        return false;
      };
    } m_thecomparator;

    struct dirnodeptr_compare_hscore_t {
      bool operator() (const AStar3DNode* lhs, const AStar3DNode* rhs ) const {
        if ( lhs->fscore-lhs->gscore > rhs->fscore-rhs->gscore )
          return true;
        return false;
      };
    } m_the_hscore_comparator;

    public:
    void sort() {
      std::sort(this->begin(),this->end(), m_thecomparator);
    }
    void sort_by_hscore() {
      std::sort(this->begin(),this->end(), m_the_hscore_comparator);
    }    

    void addnode( AStar3DNode* elem ) {
    	elem->inopenset = true;
      push_back( elem );
      //sort();
    }

  };

  // ALGO 
  class AStar3DAlgo {

    AStar3DAlgo() { verbose=2; };
  public:

    AStar3DAlgo( AStar3DAlgoConfig config ) { 
      _config = config; 
      setVerbose( _config.verbosity );
    };
    virtual ~AStar3DAlgo() {};
    
    void setVerbose( int v ) { verbose = v; };

    std::vector<AStar3DNode> findpath( const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v, const std::vector<larcv::Image2D>& tagged_v,
      const int start_row, const int goal_row, const std::vector<int>& start_cols, const std::vector<int>& goal_cols, int& goal_reached );

    std::vector<AStar3DNode> downsampleAndFindPath( const int downsampling_factor, const std::vector<larcv::Image2D>& img_v,
						    const std::vector<larcv::Image2D>& badch_v, const std::vector<larcv::Image2D>& tagged_v,
						    const int start_row, const int goal_row,
						    const std::vector<int>& start_cols, const std::vector<int>& goal_cols, int& goal_reached,
						    int compression_mode=-1 );
    
    const std::vector<larcv::Image2D>& getScoreImages() { return m_visualizedimgs; }
    
    // supporting 
    std::vector<AStar3DNode> makeRecoPath( AStar3DNode* start, AStar3DNode* goal, bool& path_completed );  

    void evaluateNeighborNodes( AStar3DNode* current, const AStar3DNode* start, const AStar3DNode* goal,
      const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v, const std::vector<larcv::Image2D>& tagged_v,
      AStar3DNodePtrList& openset, AStar3DNodePtrList& closedset, Lattice& lattice );

    bool evaluteLatticePoint( const A3DPixPos_t& latticept, AStar3DNode* current, const AStar3DNode* start, const AStar3DNode* goal,
      const std::vector<larcv::Image2D>& img_v, const std::vector<larcv::Image2D>& badch_v, const std::vector<larcv::Image2D>& tagged_v,
      AStar3DNodePtrList& openset, AStar3DNodePtrList& closedset, Lattice& lattice );

    float distanceFromCentralLine( const std::vector<float>& start_tyz, const std::vector<float>& end_tyz, const std::vector<float>& testpt_tyz );


  protected:

    AStar3DAlgoConfig _config;
    int verbose;

    std::vector<larcv::Image2D> visualizeScores( std::string score_name, const std::vector<larcv::Image2D>& orig_img_v, Lattice& lattice );    
    
    // void evaluateBadChNeighbors( AStar3DNode* current, const AStar3DNode* start, const AStar3DNode* goal,
   //    AStar3DNodePtrList& openset, AStar3DNodePtrList& closedset, 
   //    const int neighborhood_size, const int min_c, const int min_r, const int win_c, const int win_r, 
   //    const larcv::Image2D& img, const larcv::ImageMeta& meta, const bool use_bad_chs, std::map< A3DPixPos_t, AStar3DNode* >& position_lookup);

    std::vector< larcv::Image2D > m_visualizedimgs;
  };


}

#endif
