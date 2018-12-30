#ifndef __VIEW_ROOT__
#define __VIEW_ROOT__

/* *****************************************************
 * ViewROOT
 *
 * a utility to quickly look at contents of
 * the LArCV file, using the ROOT visualization
 * system.  Isn't meant to be a viewer, just 
 * a quick diagnostic tool. As a result,
 * it's not written to be efficient or to handle
 * complicated visualization.
 *
 * eventually replaced with Eve Viewer?
 *
 * *************************************************** */

#include <string>
#include <map>

class TObject;
class TCanvas;
class TH2D;

namespace larcv {

  class IOManager;

  class ViewROOT {
  protected:
    
    ViewROOT( std::string fmame );
    ~ViewROOT();

  public:

    static ViewROOT& getViewROOT( std::string fname="" );

    void setEntry( int entry );
    //static void listItems();
    void clearVisItems();
    void draw(   std::string datatype, std::string producer, int instance=-1, std::string opt="" );
    //static void remove( std::string datatype, std::string producer, int instance=-1, std::string opt="" );
		      
		     
  protected:

    bool _init;
    void initialize( std::string fname );
    
    // io objects
    IOManager* _io;
    void updateEntry() {};

    // vis objects
    int _canv_nplanes;
    struct VisInfo_t {
      std::string producer;
      std::string datatype;
      int instanceid;
      TObject* item;
      std::string _keyname;
    };
    std::map< std::string, VisInfo_t >    _items;

    TCanvas* _canv;
    TH2D* _default_image2d;
    std::string _current_image2d;
    int _current_image2d_id;
    
    static ViewROOT* _global_instance;

    // draw functions (ideally one for each object)

    // Image2D
    // PGraph
    // Pixel2D
    // ChStatus
    // ROI
    // Voxel3D (projected)

    // larlite draw objects (if compiler flag defined)
    // track
    // vertex
    // larflow
    
  };

};


#endif
