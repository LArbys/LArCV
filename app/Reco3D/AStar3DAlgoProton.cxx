#include "AStar3DAlgo.h"
#include "AStar3DAlgoProton.h"
#include <vector>
#include <cmath>

// larcv
#include "UBWireTool.h"

// larlite
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"

namespace larcv {

  std::vector<AStar3DNode> AStar3DAlgoProton::findpath( const std::vector<larcv::Image2D>& img_v,
                                                       const std::vector<larcv::Image2D>& badch_v,
                                                       const std::vector<larcv::Image2D>& tagged_v,
                                                       const int start_row, const int goal_row,
                                                       const std::vector<int>& start_cols,
                                                       const std::vector<int>& goal_cols,
                                                       int& goal_reached  ) {
    
    const larcv::ImageMeta& meta = img_v.front().meta();


    if ( verbose>0 )
      std::cout << "[[ASTAR 3D ALGO START]]" << std::endl;

    // turn image pixel coordinates into a 3D start and goal point
    std::vector<int> start_wids(start_cols.size());
    std::vector<int> goal_wids(goal_cols.size());    
    for (size_t p=0; p<goal_cols.size(); p++) {
      start_wids[p] = img_v.at(p).meta().pos_x( start_cols[p] );
      goal_wids[p]  = img_v.at(p).meta().pos_x( goal_cols[p] );      
    }
    double start_tri = 0.;
    int start_crosses = 0;
    std::vector< float > poszy_start(2,0.0);
    larcv::UBWireTool::wireIntersection( start_wids, poszy_start, start_tri, start_crosses );

    double goal_tri = 0.;
    int goal_crosses = 0;
    std::vector< float > poszy_goal(2,0.0);
    larcv::UBWireTool::wireIntersection( goal_wids, poszy_goal, goal_tri, goal_crosses );

    std::vector<float> startpos(3,0);
    startpos[0] = meta.pos_y( start_row );
    startpos[1] = poszy_start[1];
    startpos[2] = poszy_start[0];

    std::vector<float> goalpos(3,0);
    goalpos[0] = meta.pos_y( goal_row );
    goalpos[1] = poszy_goal[1];
    goalpos[2] = poszy_goal[0];

    //if ( start_tri>5 || goal_tri>5 ) {      
    //  std::cout << "start wires provided (" << start_wids[0] << "," << start_wids[1] << "," << start_wids[1] << ") tri=" << start_tri << std::endl;
    //  std::cout << "goal wires provided (" << goal_wids[0] << "," << goal_wids[1] << "," << goal_wids[1] << ") tri=" << goal_tri << std::endl;
    //  throw std::runtime_error("AStar3DAlgoProton::findpath[error] start or goal point not a good 3D space point.");
    //}

    // next, define the lattice
    float cm_per_tick = ::larutil::LArProperties::GetME()->DriftVelocity()*0.5; // [cm/usec]*[usec/tick]
    float cm_per_wire = 0.3;
    //float cm_per_row  = cm_per_tick*meta.pixel_height();
    float cm_per_col  = cm_per_wire*meta.pixel_width();

    std::vector<float> cm_per_pixel(3);
    cm_per_pixel[0] = meta.pixel_height(); // ticks per row
    cm_per_pixel[1] = cm_per_col;
    cm_per_pixel[2] = cm_per_col;

    // we set t0 to be 0 on the lattice
    float tick0 = ( meta.pos_y(start_row) < meta.pos_y(goal_row) ) ? meta.pos_y(start_row) : meta.pos_y(goal_row); 
    float y0    = ( startpos[1] < goalpos[1] )  ? startpos[1] : goalpos[1];
    float z0    = ( startpos[2] < goalpos[2] )  ? startpos[2] : goalpos[2];

    // now we define the lattice the search will go over
    std::vector<int> lattice_widths(3);
    lattice_widths[0] = abs( goal_row - start_row ); // rows
    lattice_widths[1] = (int)fabs( startpos[1] - goalpos[1] )/cm_per_col;
    lattice_widths[2] = (int)fabs( startpos[2] - goalpos[2] )/cm_per_col;

    // add padding
    for (int i=0; i<3; i++)
      lattice_widths[i] += 2*_config.lattice_padding;

    // define the origin of the lattice in detector space
    std::vector<float> origin_lattice(3);
    origin_lattice[0] = tick0 -_config.lattice_padding*cm_per_pixel[0]; // in ticks
    origin_lattice[1] = y0 - _config.lattice_padding*cm_per_col;   // in cm
    origin_lattice[2] = z0 - _config.lattice_padding*cm_per_col;   // in cm

    std::vector<float> max_lattice(3,0);
    for (int i=0; i<3; i++)
      max_lattice[i] = origin_lattice[i] + lattice_widths[i]*cm_per_pixel[i];

    // get the imagemetas for each plane so we can package them for the lattice
    std::vector< const larcv::ImageMeta* > meta_v;
    for ( size_t p=0; p<img_v.size(); p++) {
      const larcv::ImageMeta* ptr_meta = &(img_v.at(p).meta());
      meta_v.push_back( ptr_meta );
    }


    // reserve enough space for half full occupancy
    size_t max_nelements = (size_t)lattice_widths[0]*lattice_widths[1]*lattice_widths[2];

    // finally make the lattice
    Lattice lattice( origin_lattice, lattice_widths, cm_per_pixel, meta_v );
    if ( verbose>0 ) {
      std::cout << "Defining Lattice" << std::endl;
      std::cout << "origin: (" << origin_lattice[0] << "," << origin_lattice[1] << "," << origin_lattice[2] << ")" << std::endl;
      std::cout << "max-range: (" << max_lattice[0] << "," << max_lattice[1] << "," << max_lattice[2] << ")" << std::endl;
      std::cout << "number of nodes per dimension: (" << lattice_widths[0] << "," << lattice_widths[1] << "," << lattice_widths[2] << ")" << std::endl;
      std::cout << "max elements: " << max_nelements << std::endl;
    }
    //lattice.reserve(max_nelements);

    // we now make some definitions
    AStar3DNodePtrList openset;
    AStar3DNodePtrList closedset;
    openset.reserve(max_nelements);
    closedset.reserve(max_nelements);

    // make the target node (window coorindates)
    AStar3DNode* goal = lattice.getNode( goalpos );
    if ( goal==nullptr) {
      A3DPixPos_t goalnode = lattice.getNodePos( goalpos );
      if ( verbose>0 )
        std::cout << "goal=(" << goalpos[0] << "," << goalpos[1] << "," << goalpos[2] << ") "
                  << "node->(" << goalnode[0] << "," << goalnode[1] << "," << goalnode[2] << ")" << std::endl;
      throw std::runtime_error("Was not able to produce a valid goal node.");
    }
    else {
      if ( verbose>0 )
        std::cout << "Goal Node: " << goal->str() << " within-image=" << goal->within_image << std::endl;
    }

    // make starting node (window coordinates)
    AStar3DNode* start = lattice.getNode( startpos );
    if ( start==nullptr) {
      A3DPixPos_t startnode = lattice.getNodePos( startpos );
      if ( verbose>0 )      
        std::cout << "start=(" << startpos[0] << "," << startpos[1] << "," << startpos[2] << ") "
                  << "node->(" << startnode[0] << "," << startnode[1] << "," << startnode[2] << ")" << std::endl;
      throw std::runtime_error("Was not able to produce a valid start node.");
    }
    else {
      if ( verbose>0 )
        std::cout << "Start Node: " << start->str() << std::endl;      
    }

    // start fscore gets set to the heuristic
    for (int i=1; i<3; i++ ) {
      float di= goalpos[i]-startpos[i];
      start->fscore += di*di;
    }
    float dt = (goalpos[0]-startpos[0])*cm_per_tick;
    start->fscore += dt*dt;
    start->fscore = sqrt(start->fscore);

    // set the start direction to heads towards the goal
    start->dir3d.resize(3,0.0);
    for (int i=0; i<3; i++) {
      start->dir3d[i] = (goalpos[i]-startpos[i])/start->fscore;
    }

    // add the start node into the openset
    openset.addnode( start );
    
    if ( verbose>0 ) {
      std::cout << "start astar algo." << std::endl;
      std::cout << "neighborhood sizes: " << _config.astar_neighborhood[0] << "," << _config.astar_neighborhood[1] << "," << _config.astar_neighborhood[2] << std::endl;
    }

    int nsteps = -1;
    AStar3DNode* current = NULL;
    while ( openset.size()>0 ) {
      // get current
      do {
        current = openset.back();
        openset.pop_back();
      } while ( current->closed );
      nsteps++;
      if ( verbose>2 || (verbose>0 && nsteps%100==0) ) {
        std::cout << "step=" << nsteps << ": get current node " << current->str() << ". "
                  << "number of remaining nodes in openset=" << openset.size() << std::endl;
      }

      if ( *current==*goal ) {
        if ( verbose>0 ) {
          std::cout << "astar goal reached. finished." << std::endl;
        }
        break;        
      }

      // scan through neighors, and ID the candidate successor node
      evaluateNeighborNodes( current, start, goal, img_v, badch_v, tagged_v, openset, closedset, lattice );
      //evaluateBadChNeighbors( current, start, goal, openset, closedset, 1, 
      //  min_c, min_r, win_c, win_r, img, meta, use_bad_chs, position_lookup );

      // finished with current node, put it on the closed set
      current->closed = true;
      closedset.addnode( current );

      openset.sort();

      if ( verbose>1 ) {
        std::cout << "step=" << nsteps << ": get current node " << current->str() << ". " << std::endl;
        std::cout << " in open openset: " << openset.size() << std::endl;
        //for ( auto node : openset ) 
        //  std::cout << "  " << node->str() << " g=" << node->gscore << std::endl;
        std::cout << " nodes in closedset: " << closedset.size() << std::endl;
      }
          

      if ( verbose>3 || (verbose>0 && nsteps%100==0) ) {
        std::cout << "in openset: " << openset.size() << std::endl;
        std::cout << "in closedset: " << closedset.size() << std::endl;
        std::cout << "fracion of max visited: " << float(closedset.size())/float(max_nelements) << std::endl;
        std::cout << "step " << nsteps << ". [enter] to continue." << std::endl;
        if ( verbose>4 )
          std::cin.get();
      }
      //std::cin.get();      

    }//end of while loop

    //larcv::Image2D fscoreimg = visualizeScores( "fscore", img, min_c, min_r, win_c, win_r, position_lookup );
    //larcv::Image2D gscoreimg = visualizeScores( "gscore", img, min_c, min_r, win_c, win_r, position_lookup );    
    //m_visualizedimgs.clear();
    //m_visualizedimgs.emplace_back( std::move(fscoreimg) );
    //m_visualizedimgs.emplace_back( std::move(gscoreimg) );    

    bool path_completed = false;
    std::vector<AStar3DNode> path;
    if ( *current!=*goal ) {
      std::cout << "did not make it to the goal. get the lowest h-score" << std::endl;      
      closedset.sort_by_hscore();
      int nnodes = (int)closedset.size();      
      for ( int inode = nnodes-1; inode>=0; inode-- ) {
        AStar3DNode* closed_node = closedset.at(inode);
        if ( closed_node->fscore==0 ) continue;
        current = closed_node;
        break;
      }
      // if ( verbose>0)
      std::cout << "could not reach goal. best node: " 
                   << " (" << img_v.front().meta().pos_x( current->cols[0] ) << "," << img_v.front().meta().pos_y( current->row ) << ")"
                   << " fscore=" << current->fscore << " g-score=" << current->gscore 
                   << " hscore=" << current->fscore-current->gscore << std::endl;
      goal_reached = 0;
    }
    else {
      goal_reached = 1;
    }

    path = makeRecoPath( start, current, path_completed );

    std::cout << "nsteps: " << nsteps << std::endl;
    std::cout << "path length: " << path.size() << std::endl;
    if ( verbose>0 ) {
      for ( auto& node : path )
        std::cout << " " << node.str() << " pixval=(" << node.pixval[0] << "," << node.pixval[1] << "," << node.pixval[2] << ")" << std::endl;
    }


    return path;
  }


  void AStar3DAlgoProton::evaluateNeighborNodes( AStar3DNode* current,
                                                const AStar3DNode* start,
                                                const AStar3DNode* goal,
                                                const std::vector<larcv::Image2D>& img_v,
                                                const std::vector<larcv::Image2D>& badch_v,
                                                const std::vector<larcv::Image2D>& tagged_v,
                                                AStar3DNodePtrList& openset,
                                                AStar3DNodePtrList& closedset,
                                                Lattice& lattice ) {

    const A3DPixPos_t& center = current->nodeid;

    int number_updates=0;

    for (int du=-_config.astar_neighborhood[0]; du<=_config.astar_neighborhood[0]; du++) {
      for (int dv=-_config.astar_neighborhood[1]; dv<=_config.astar_neighborhood[1]; dv++) {
        for (int dw=-_config.astar_neighborhood[2]; dw<=_config.astar_neighborhood[2]; dw++) {
          int u = center[0]+du;
          int v = center[1]+dv;
          int w = center[2]+dw;
          if ( u<0 || u>=lattice.widths()[0] || v<0 || v>=lattice.widths()[1] || w<0 || w>=lattice.widths()[2] )
            continue;
          if (du==0 && dv==0 && dw==0 )
            continue;

          A3DPixPos_t evalme( u, v, w);
          bool updated = evaluteLatticePoint( evalme, current, start, goal, img_v, badch_v, tagged_v, openset, closedset, lattice );
          if ( updated )
            number_updates++;
        }
      }
    }

    if ( verbose>1 )
      std::cout << "number of node updates: " << number_updates << std::endl;
  }

  bool AStar3DAlgoProton::evaluteLatticePoint( const A3DPixPos_t& latticept,
                                              AStar3DNode* current, const AStar3DNode* start,
                                              const AStar3DNode* goal,
                                              const std::vector<larcv::Image2D>& img_v,
                                              const std::vector<larcv::Image2D>& badch_v,
                                              const std::vector<larcv::Image2D>& tagged_v,
                                              AStar3DNodePtrList& openset,
                                              AStar3DNodePtrList& closedset,
                                              Lattice& lattice ) {
    // returns true if updated, false if not

    const int nplanes = img_v.size();

    AStar3DNode* neighbor_node = lattice.getNode( latticept );
    if ( neighbor_node==nullptr ) {
      std::stringstream ss;
      ss << " lattice pt (" << latticept[0] << "," << latticept[1] << "," << latticept[2] << ") node is NULL" << std::endl;
      std::cout << ss.str() << std::endl;
      std::runtime_error( ss.str() );
      return false;
    }

    if ( verbose>3 ) {
      std::cout << " lattice pt (" << latticept[0] << "," << latticept[1] << "," << latticept[2] << ") "
        << "img pos: (r=" << neighbor_node->row << "," << neighbor_node->cols[0] << "," << neighbor_node->cols[1] << "," << neighbor_node->cols[2] << ") "
        << "within_image=" << neighbor_node->within_image 
        << " closed=" << neighbor_node->closed 
        << " badchnode=" << neighbor_node->badchnode;
    }

    if ( neighbor_node->closed ) {
      if (verbose>3) std::cout << std::endl;
      return false; // don't reevaluate a closed node
    }
 
    if ( !neighbor_node->within_image )  {
      if (verbose>3) std::cout << std::endl;
      neighbor_node->closed = true;
      closedset.addnode(neighbor_node);
      return false; // clearly not worth evaluating
    }

    bool within_pad = false; // allowed region to allow start point to get onto a track (in case of mistakes by endpoint tagger)
    // let's have the lattest provide this designation, so we don't have to waste time here
    if ( abs(neighbor_node->row - start->row)<_config.astar_start_padding ) {
      // within row pad
      int planes_within_pad = 0;
      for (int p=0; p<nplanes; p++) {
        if ( abs( neighbor_node->cols[p] - start->cols[p])<_config.astar_start_padding ) {
          planes_within_pad++;
        }
        else if ( abs( neighbor_node->cols[p] - goal->cols[p]) < _config.astar_end_padding ) {
          planes_within_pad++;
        }
        else {
          break;
        }
      }
      if ( planes_within_pad==nplanes )
        within_pad = true;
    }
    if ( verbose>3 ) {
      std::cout << " within_pad=" << within_pad;
    }

    // is this neighbor the goal? we need to know so we can always include it as a possible landing point, even if in badch region.
    bool isgoal = true;
    if ( neighbor_node->row!=goal->row ) {
      isgoal = false;
    }
    else {
      //have to check cols too
      for (int p=0; p<nplanes; p++) {
        if ( goal->cols[p]!=neighbor_node->cols[p] ) {
          isgoal = false;
          break;
        }
      }
    }

    int nplanes_abovethreshold_or_bad = 0;
    int nplanes_bad = 0;
    std::vector<float> pixval(nplanes,0.0);
    for (int p=0; p<nplanes; p++){
      int row = neighbor_node->row;
      //if ( p==2 && row-1>=0 ) row--; // anode wire plane offset

      if ( img_v.at(p).pixel( row, neighbor_node->cols[p] )>_config.astar_threshold[p] ) {
        nplanes_abovethreshold_or_bad++;
        pixval.at(p) = img_v.at(p).pixel( row, neighbor_node->cols[p] );
      }
      else if ( badch_v.at(p).pixel(row,neighbor_node->cols[p])>0 ) {
        nplanes_abovethreshold_or_bad++;
        nplanes_bad++;
        pixval.at(p) = -1.0;
      }
    }
    if ( verbose>3 ) {
      std::cout << " nplanes_abovethreshold_or_bad=" << nplanes_abovethreshold_or_bad;
    }    
          
    // the criteria for NOT evaluating this node
    if ( !_config.accept_badch_nodes && !within_pad && !isgoal && ( nplanes_abovethreshold_or_bad<3 || nplanes_bad>1 ) ) {
      // the most basic criteria
      neighbor_node->closed = true; // we close this node as its not allowed
      closedset.addnode( neighbor_node);
      if ( verbose>3 ) std::cout << " closed by criteria-1" << std::endl;
      return false; // skip if below threshold
    }
    if ( _config.accept_badch_nodes && !within_pad && !isgoal && nplanes_abovethreshold_or_bad<_config.min_nplanes_w_hitpixel ) {
      neighbor_node->closed = true; // we close this node as it won't be used
      closedset.addnode( neighbor_node );
      if ( verbose>3 ) std::cout << " closed by criteria-2" << std::endl;      
      return false;
    }
    if ( !isgoal && _config.restrict_path && _config.path_restriction_radius<distanceFromCentralLine( start->tyz, goal->tyz, neighbor_node->tyz ) ) {
      neighbor_node->closed = true;      
      closedset.addnode( neighbor_node );
      if ( verbose>3 ) std::cout << " closed by criteria-3" << std::endl;      
      return false;
    }

    // is the node in a previously thrumu-tagged track
    // int nplanes_wtagged = 0;
    // for (int p=0; p<nplanes; p++) {
    // }
    // bool istagged_pixel = false;

    // is this a badch node?
    if ( nplanes_bad>=1 || nplanes_abovethreshold_or_bad<3 )
      neighbor_node->badchnode = true;

    if ( !neighbor_node->inopenset )
      openset.addnode( neighbor_node ); // add to openset if not on closed set nor is already on openset
    if ( verbose>3 ) std::cout << " added to openset." << std::endl;

    // how many past nodes in a row are bad?
    int npast_badnodes = 0;
    AStar3DNode* anode = current;
    for (int ipast=0; ipast<5; ipast++) {
      if ( anode->badchnode )
        npast_badnodes++;
      else
        break;
      if ( anode->prev!=nullptr )
        anode = anode->prev;
      else
        break;
    }

    // define the jump cost for this node
    // first get the diff vector from here to current node
    float cm_per_tick = ::larutil::LArProperties::GetME()->DriftVelocity()*0.5; // [cm/usec]*[usec/tick]    
    float norm1 = 0.;
    float dir1[3];    
    for (int i=1; i<3; i++) {
      float dx = neighbor_node->tyz[i] - current->tyz[i];
      dir1[i] = dx;
      norm1 += dx*dx;
    }
    float dt = ( neighbor_node->tyz[0]-current->tyz[0] )*cm_per_tick;
    dir1[0] = dt;
    norm1 += dt*dt;
    norm1 = sqrt(norm1);
    for (int i=0; i<3; i++ )
      dir1[i] /= norm1; 

    float jump_cost = norm1;
    //if ( neighbor_node->badchnode ) jump_cost = norm1*10.0; // pay a cost for using a badch node

      float curvature_cost = 0.0;
      
      if(doPxValEstimate){
          curvature_cost = 1;
          double px_value_cost = 0;
          int row = neighbor_node->row;
          for(size_t p = 0; p<3;p++){
              px_value_cost+=100*exp(-1.*img_v.at(p).pixel( row, neighbor_node->cols[p] )/20);// penalizes low value pixels
          }
          jump_cost*=(1+px_value_cost);
      }
      else{
          // calculate local curvature cost
          curvature_cost = 0.0;
          float dir2[3] = {0};
          if ( current->prev!=NULL ) {
              AStar3DNode* prev = current->prev;
              float norm2 = 0.;
              for (int i=1; i<3; i++ ) {
                  dir2[i] = current->tyz[i]-prev->tyz[i];
                  norm2 += dir2[i]*dir2[i];
              }
              dir2[0] = (current->tyz[0]-prev->tyz[0])*cm_per_tick;
              norm2 += dir2[0]*dir2[0];
              norm2 = sqrt(norm2);
              for ( int i=0; i<3; i++)
              dir2[i] /= norm2;
              float dcos = 0;
              for (int i=0; i<3; i++) {
                  dcos += dir1[i]*dir2[i];
              }
              curvature_cost = 0.1*exp(10*fabs(0.5*(1-dcos)));
              jump_cost *= curvature_cost;
              //std::cout << "plop" << std::endl;
          }
      }


      
    // calculate heuristic score (distance of node to goal node)
    float hscore = 0.;
    for (int i=1; i<3; i++) {
      float dx = (neighbor_node->tyz[i] - goal->tyz[i]);
      hscore += dx*dx;
    }
    dt = ( neighbor_node->tyz[0] - goal->tyz[0] )*cm_per_tick;
    hscore += dt*dt;
    hscore = sqrt(hscore);

    float gscore = current->gscore + jump_cost;
    float fscore = gscore + hscore;
    float kscore = curvature_cost;

    if ( isgoal )
      std::cout << "ISGOAL!" << std::endl;

    // is this gscore better than the current one (or is it unassigned?) (or is the goal)
    if ( isgoal || 
         ((neighbor_node->fscore==0 || neighbor_node->fscore>fscore) && npast_badnodes<10) ) {
      // update the neighbor to follow this path 
      if ( verbose>2 )
        std::cout << "  updating neighbor to with better path: from f=" << neighbor_node->fscore << " g=" << neighbor_node->gscore 
                  << " to f=" << fscore << " g=" << gscore << std::endl;
      neighbor_node->prev = current;
      neighbor_node->fscore = fscore;
      neighbor_node->gscore = gscore;
      neighbor_node->kscore = kscore;
      neighbor_node->pixval = pixval;
      if ( isgoal )
        neighbor_node->fscore = 0;
      //neighbor_node->dir3d = dir3d; not implemented yet
      return true;
    }
    else {
      if ( verbose>3 )
        std::cout << "  this neighbor already on better path. current-f=" << neighbor_node->fscore << " < proposed-f=" << fscore << std::endl;
    }    

    return false;
  }

  std::vector<AStar3DNode> AStar3DAlgoProton::makeRecoPath( AStar3DNode* start, AStar3DNode* goal, bool& path_completed ) {
                                                                                                
    path_completed = true;
    std::vector<AStar3DNode> path;
    AStar3DNode* current = goal;
    while ( *current!=*start ) {
      AStar3DNode copy( *current );
      path.emplace_back(std::move(copy));
      current = current->prev;
      if ( current==NULL )
        break;
    }
    if ( current==NULL || *current!=*start ) {
      path_completed = false;
      return path;
    }
    AStar3DNode startout( *start );
    path.emplace_back( std::move(startout) );
    return path;

  }

  float AStar3DAlgoProton::distanceFromCentralLine( const std::vector<float>& start_tyz, const std::vector<float>& end_tyz, const std::vector<float>& testpt_tyz ) {
    // returns the shortest distance of the test point from the line segment between the start and end points
    // the coordinate system is in (tick, Y, Z)
    // will return a value in cm
    // stole formular from: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    float norm12 = 0;    
    std::vector<float> dir10( start_tyz.size(), 0.0 );
    std::vector<float> dir20( start_tyz.size(), 0.0 );
    std::vector<float> dir12( start_tyz.size(), 0.0 );
    for (size_t i=0; i<start_tyz.size(); i++) {
      dir10[i] = start_tyz.at(i) - testpt_tyz.at(i);
      dir20[i] = end_tyz.at(i)   - testpt_tyz.at(i);
      dir12[i] = end_tyz.at(i)   - start_tyz.at(i);
      if (i==0) {
        dir10[0] *= ::larutil::LArProperties::GetME()->DriftVelocity()*0.5; // [cm/usec]*[usec/tick]        
        dir20[0] *= ::larutil::LArProperties::GetME()->DriftVelocity()*0.5; // [cm/usec]*[usec/tick]        
        dir12[0] *= ::larutil::LArProperties::GetME()->DriftVelocity()*0.5; // [cm/usec]*[usec/tick]                        
      }
      norm12 += dir12[i]*dir12[i];      
    }
    norm12 = sqrt(norm12);
    if ( norm12==0 ) {
      throw std::runtime_error("AStar3DAlgoProton::distanceFromCentralLine[error] start and end point are the same. calculation undefined.");
    }


    // dir01 x dir02
    std::vector<float> dirX(dir10.size(),0.0);
    dirX[0] = dir10[1]*dir20[2] - dir10[2]*dir20[1];
    dirX[1] = dir10[2]*dir20[0] - dir10[0]*dir20[2];
    dirX[2] = dir10[0]*dir20[1] - dir10[1]*dir20[0];
    float normX = 0.;
    for (size_t i=0; i<dirX.size(); i++) {
      normX += dirX[i]*dirX[i];
    }
    normX = sqrt(normX);

    float dist = normX/norm12;
    return dist;
  }
}

