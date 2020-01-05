#ifndef LARLITE_AStarTracker_CXX
#define LARLITE_AStarTracker_CXX

#include "AStarTracker.h"
#include "DataFormat/track.h"
#include "DataFormat/vertex.h"
#include "DataFormat/hit.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/wire.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/TimeService.h"

#include "TCanvas.h"
#include "TF1.h"
#include "TFile.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TH2D.h"
#include "TLatex.h"
#include "TLine.h"
#include "TRandom3.h"
#include "TStyle.h"
#include "TVector3.h"

#include "AStarUtils.h"
#include "AStar3DAlgo.h"
#include "AStar3DAlgoProton.h"

#include <cassert>
#include <ctime>

namespace larcv {

    void AStarTracker::tellMe(std::string s, int verboseMin = 0){
        if(_verbose >= verboseMin) std::cout << s << std::endl;
    }
    //______________________________________________________
    void AStarTracker::SetTimeAndWireBoundsProtonsErez(){

        int itrack = 0;
        for(size_t tracknum = 0;tracknum<_SelectableTracks.size(); tracknum++){
            if(_run    != _SelectableTracks[tracknum][0])continue;
            if(_subrun != _SelectableTracks[tracknum][1])continue;
            if(_event  != _SelectableTracks[tracknum][2])continue;
            if(_track  != _SelectableTracks[tracknum][3])continue;
            itrack = tracknum;
        }

        time_bounds.clear();
        wire_bounds.clear();

        std::pair<double, double> time_bound;
        std::pair<double, double> wire_bound;
        double timeOffset = 2400;
        wire_bound.first  = _SelectableTracks[itrack][4];
        time_bound.first  = _SelectableTracks[itrack][5]+timeOffset;
        wire_bound.second = _SelectableTracks[itrack][6];
        time_bound.second = _SelectableTracks[itrack][7]+timeOffset;
        time_bounds.push_back(time_bound);
        wire_bounds.push_back(wire_bound);

        wire_bound.first  = _SelectableTracks[itrack][8];
        time_bound.first  = _SelectableTracks[itrack][9]+timeOffset;
        wire_bound.second = _SelectableTracks[itrack][10];
        time_bound.second = _SelectableTracks[itrack][11]+timeOffset;
        time_bounds.push_back(time_bound);
        wire_bounds.push_back(wire_bound);

        wire_bound.first  = _SelectableTracks[itrack][12];
        time_bound.first  = _SelectableTracks[itrack][13]+timeOffset;
        wire_bound.second = _SelectableTracks[itrack][14];
        time_bound.second = _SelectableTracks[itrack][15]+timeOffset;
        time_bounds.push_back(time_bound);
        wire_bounds.push_back(wire_bound);

        // equalize time ranges to intercept the same range on 3 views
        for(size_t iPlane=0;iPlane<3;iPlane++){
            for(size_t jPlane=0;jPlane<3;jPlane++){
                if(time_bounds[jPlane].first  <= time_bounds[iPlane].first) {time_bounds[iPlane].first  = time_bounds[jPlane].first;}
                if(time_bounds[jPlane].second >= time_bounds[iPlane].second){time_bounds[iPlane].second = time_bounds[jPlane].second;}
            }
        }

        // Update the range with margin
        double ImageMargin = 100.;
        double fractionImageMargin = 2;
        for(size_t iPlane=0;iPlane<3;iPlane++){
            time_bounds[iPlane].first  -= std::max(ImageMargin,fractionImageMargin*(time_bounds[iPlane].second-time_bounds[iPlane].first));
            time_bounds[iPlane].second += std::max(ImageMargin,fractionImageMargin*(time_bounds[iPlane].second-time_bounds[iPlane].first));
            wire_bounds[iPlane].first  -= std::max(ImageMargin,fractionImageMargin*(wire_bounds[iPlane].second-wire_bounds[iPlane].first));
            wire_bounds[iPlane].second += std::max(ImageMargin,fractionImageMargin*(wire_bounds[iPlane].second-wire_bounds[iPlane].first));

            if(time_bounds[iPlane].first < 0) time_bounds[iPlane].first = 0;
            if(wire_bounds[iPlane].first < 0) wire_bounds[iPlane].first = 0;

            wire_bounds[iPlane].first  = (size_t)(wire_bounds[iPlane].first + 0.5);
            wire_bounds[iPlane].second = (size_t)(wire_bounds[iPlane].second + 0.5);
        }

        // make sure the number of rows and cols are divisible by _compressionFactor
        for(size_t iPlane=0;iPlane<3;iPlane++){
            while(!( (size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first)%_compressionFactor_t == 0)){time_bounds[iPlane].second++;}
            tellMe(Form("%zu rows for %d compression factor",(size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first),_compressionFactor_t),1);
        }
        for(size_t iPlane = 0;iPlane<3;iPlane++){

            while(!( (size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first)%_compressionFactor_w == 0)){wire_bounds[iPlane].second++;}
            tellMe(Form("%zu cols for %d compression factor",(size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first),_compressionFactor_w),1);
        }

    }
    //______________________________________________________
    void AStarTracker::SetTimeAndWireBounds(){
        time_bounds.clear();
        wire_bounds.clear();
        time_bounds.reserve(3);
        wire_bounds.reserve(3);
        for(size_t iPlane=0; iPlane<3; ++iPlane) {
            time_bounds[iPlane].first  = 1.e9;
            time_bounds[iPlane].second = 0.;
            wire_bounds[iPlane].first  = 1.e9;
            wire_bounds[iPlane].second = 0.;
        }

        for(size_t iPlane=0;iPlane<3;iPlane++){
            // make sure starting point is projected back in the range
            if(X2Tick(start_pt.X(),iPlane) < time_bounds[iPlane].first ) time_bounds[iPlane].first  = X2Tick(start_pt.X(),iPlane);
            if(X2Tick(start_pt.X(),iPlane) > time_bounds[iPlane].second) time_bounds[iPlane].second = X2Tick(start_pt.X(),iPlane);
            double wireProjStartPt = larutil::GeometryHelper::GetME()->Point_3Dto2D(start_pt,iPlane).w / 0.3;
            if(wireProjStartPt < wire_bounds[iPlane].first ) wire_bounds[iPlane].first  = wireProjStartPt;
            if(wireProjStartPt > wire_bounds[iPlane].second) wire_bounds[iPlane].second = wireProjStartPt;

            // make sure ending point is projected back in the range
            if(X2Tick(end_pt.X(),iPlane) < time_bounds[iPlane].first ) time_bounds[iPlane].first  = X2Tick(end_pt.X(),iPlane);
            if(X2Tick(end_pt.X(),iPlane) > time_bounds[iPlane].second) time_bounds[iPlane].second = X2Tick(end_pt.X(),iPlane);
            double wireProjEndPt   = larutil::GeometryHelper::GetME()->Point_3Dto2D(end_pt,iPlane).w / 0.3;
            if((larutil::GeometryHelper::GetME()->Point_3Dto2D(end_pt,iPlane).w / 0.3) < wire_bounds[iPlane].first ) wire_bounds[iPlane].first  = wireProjEndPt;
            if((larutil::GeometryHelper::GetME()->Point_3Dto2D(end_pt,iPlane).w / 0.3) > wire_bounds[iPlane].second) wire_bounds[iPlane].second = wireProjEndPt;
        }
        // equalize time ranges to intercept the same range on 3 views
        for(size_t iPlane=0;iPlane<3;iPlane++){
            for(size_t jPlane=0;jPlane<3;jPlane++){
                if(time_bounds[jPlane].first  <= time_bounds[iPlane].first) {time_bounds[iPlane].first  = time_bounds[jPlane].first;}
                if(time_bounds[jPlane].second >= time_bounds[iPlane].second){time_bounds[iPlane].second = time_bounds[jPlane].second;}
            }
        }
        // Update the range with margin
        double ImageMargin = 100.;
        double fractionImageMargin = 2;
        for(size_t iPlane=0;iPlane<3;iPlane++){
            time_bounds[iPlane].first  -= std::max(ImageMargin,fractionImageMargin*(time_bounds[iPlane].second-time_bounds[iPlane].first));
            time_bounds[iPlane].second += std::max(ImageMargin,fractionImageMargin*(time_bounds[iPlane].second-time_bounds[iPlane].first));
            wire_bounds[iPlane].first  -= std::max(ImageMargin,fractionImageMargin*(wire_bounds[iPlane].second-wire_bounds[iPlane].first));
            wire_bounds[iPlane].second += std::max(ImageMargin,fractionImageMargin*(wire_bounds[iPlane].second-wire_bounds[iPlane].first));

            if(time_bounds[iPlane].first < 0) time_bounds[iPlane].first = 0;
            if(wire_bounds[iPlane].first < 0) wire_bounds[iPlane].first = 0;

            wire_bounds[iPlane].first  = (size_t)(wire_bounds[iPlane].first + 0.5);
            wire_bounds[iPlane].second = (size_t)(wire_bounds[iPlane].second + 0.5);
        }

        // make sure the number of rows and cols are divisible by _compressionFactor
        for(size_t iPlane=0;iPlane<3;iPlane++){
            while(!( (size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first)%_compressionFactor_t == 0)){
                time_bounds[iPlane].second++;//=(_compressionFactor_t-((size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first)%_compressionFactor_t));
            }
            tellMe(Form("%zu rows for %d compression factor",(size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first),_compressionFactor_t),1);
        }

        for(size_t iPlane = 0;iPlane<3;iPlane++){

            while(!( (size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first)%_compressionFactor_w == 0)){
                wire_bounds[iPlane].second++;//=(_compressionFactor_w-((size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first)%_compressionFactor_w));
            }
            tellMe(Form("%zu cols for %d compression factor",(size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first),_compressionFactor_w),1);
        }
    }
    //______________________________________________________
    void AStarTracker::SetTimeAndWireBounds(TVector3 pointStart, TVector3 pointEnd){

        time_bounds.clear();
        wire_bounds.clear();
        //time_bounds.reserve(3);
        //wire_bounds.reserve(3);
        for(size_t iPlane=0; iPlane<3; ++iPlane) {
            std::pair<double,double> timelimit;
            std::pair<double,double> wirelimit;
            timelimit.first  = 1.e9;
            timelimit.second = 0.;
            wirelimit.first  = 1.e9;
            wirelimit.second = 0.;
            time_bounds.push_back(timelimit);
            wire_bounds.push_back(wirelimit);
            //time_bounds[iPlane] = timelimit;
            //wire_bounds[iPlane] = wirelimit;
        }

        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);

        for(size_t iPlane=0;iPlane<3;iPlane++){
            // make sure starting point is projected back in the range
            if(X2Tick(pointStart.X(),iPlane) < time_bounds[iPlane].first ) time_bounds[iPlane].first  = X2Tick(pointStart.X(),iPlane);
            if(X2Tick(pointStart.X(),iPlane) > time_bounds[iPlane].second) time_bounds[iPlane].second = X2Tick(pointStart.X(),iPlane);

            double wireProjStartPt = larutil::GeometryHelper::GetME()->Point_3Dto2D(pointStart,iPlane).w / 0.3;
            if(wireProjStartPt < wire_bounds[iPlane].first ) wire_bounds[iPlane].first  = wireProjStartPt;
            if(wireProjStartPt > wire_bounds[iPlane].second) wire_bounds[iPlane].second = wireProjStartPt;

            // make sure ending point is projected back in the range
            if(X2Tick(pointEnd.X(),iPlane) < time_bounds[iPlane].first ) time_bounds[iPlane].first  = X2Tick(pointEnd.X(),iPlane);
            if(X2Tick(pointEnd.X(),iPlane) > time_bounds[iPlane].second) time_bounds[iPlane].second = X2Tick(pointEnd.X(),iPlane);

            double wireProjEndPt   = larutil::GeometryHelper::GetME()->Point_3Dto2D(pointEnd,iPlane).w / 0.3;
            if((wireProjEndPt) < wire_bounds[iPlane].first ) wire_bounds[iPlane].first  = wireProjEndPt;
            if((wireProjEndPt) > wire_bounds[iPlane].second) wire_bounds[iPlane].second = wireProjEndPt;
        }
        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);

        // equalize time ranges to intercept the same range on 3 views
        for(size_t iPlane=0;iPlane<3;iPlane++){
            for(size_t jPlane=0;jPlane<3;jPlane++){
                if(time_bounds[jPlane].first  <= time_bounds[iPlane].first) {time_bounds[iPlane].first  = time_bounds[jPlane].first;}
                if(time_bounds[jPlane].second >= time_bounds[iPlane].second){time_bounds[iPlane].second = time_bounds[jPlane].second;}
            }
        }
        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);

        // Update the range with margin
        double ImageMargin = 100.;
        double fractionImageMargin = 2;
        for(size_t iPlane=0;iPlane<3;iPlane++){
            time_bounds[iPlane].first  -= std::max(ImageMargin,fractionImageMargin*(time_bounds[iPlane].second-time_bounds[iPlane].first));
            time_bounds[iPlane].second += std::max(ImageMargin,fractionImageMargin*(time_bounds[iPlane].second-time_bounds[iPlane].first));
            wire_bounds[iPlane].first  -= std::max(ImageMargin,fractionImageMargin*(wire_bounds[iPlane].second-wire_bounds[iPlane].first));
            wire_bounds[iPlane].second += std::max(ImageMargin,fractionImageMargin*(wire_bounds[iPlane].second-wire_bounds[iPlane].first));

            if(time_bounds[iPlane].first < 0) time_bounds[iPlane].first = 0;
            if(wire_bounds[iPlane].first < 0) wire_bounds[iPlane].first = 0;

            wire_bounds[iPlane].first  = (size_t)(wire_bounds[iPlane].first + 0.5);
            wire_bounds[iPlane].second = (size_t)(wire_bounds[iPlane].second + 0.5);
        }
        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);

        // make sure the number of rows and cols are divisible by _compressionFactor
        for(size_t iPlane=0;iPlane<3;iPlane++){
            while(!( (size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first)%_compressionFactor_t == 0)){
                time_bounds[iPlane].second++;//=(_compressionFactor_t-((size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first)%_compressionFactor_t));
            }
            tellMe(Form("%zu rows for %d compression factor",(size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first),_compressionFactor_t),0);

            while(!( (size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first)%_compressionFactor_w == 0)){
                wire_bounds[iPlane].second++;//=(_compressionFactor_w-((size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first)%_compressionFactor_w));
            }
            tellMe(Form("%zu cols for %d compression factor",(size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first),_compressionFactor_w),0);
        }

        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);
    }
    //______________________________________________________
    void AStarTracker::SetTimeAndWireBounds(std::vector<TVector3> points){

        time_bounds.clear();
        wire_bounds.clear();
        //time_bounds.reserve(3);
        //wire_bounds.reserve(3);
        for(size_t iPlane=0; iPlane<3; ++iPlane) {
            std::pair<double,double> timelimit;
            std::pair<double,double> wirelimit;
            timelimit.first  = 1.e9;
            timelimit.second = 0.;
            wirelimit.first  = 1.e9;
            wirelimit.second = 0.;
            time_bounds.push_back(timelimit);
            wire_bounds.push_back(wirelimit);
            //time_bounds[iPlane] = timelimit;
            //wire_bounds[iPlane] = wirelimit;
        }

        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);

        for(size_t iPlane=0;iPlane<3;iPlane++){
            // make sure starting point is projected back in the range
            for(size_t iPoint = 0;iPoint<points.size();iPoint++){
                double Xpoint = X2Tick(points[iPoint].X(),iPlane);
                double wireProjStartPt = larutil::GeometryHelper::GetME()->Point_3Dto2D(points[iPoint],iPlane).w / 0.3 ;

                if( Xpoint < time_bounds[iPlane].first ) time_bounds[iPlane].first  = Xpoint;
                if( Xpoint > time_bounds[iPlane].second) time_bounds[iPlane].second = Xpoint;

                if(wireProjStartPt < wire_bounds[iPlane].first ) wire_bounds[iPlane].first  = wireProjStartPt;
                if(wireProjStartPt > wire_bounds[iPlane].second) wire_bounds[iPlane].second = wireProjStartPt;
            }
        }
        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);

        // equalize time ranges to intercept the same range on 3 views
        for(size_t iPlane=0;iPlane<3;iPlane++){
            for(size_t jPlane=0;jPlane<3;jPlane++){
                if(time_bounds[jPlane].first  <= time_bounds[iPlane].first) {time_bounds[iPlane].first  = time_bounds[jPlane].first;}
                if(time_bounds[jPlane].second >= time_bounds[iPlane].second){time_bounds[iPlane].second = time_bounds[jPlane].second;}
            }
        }
        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);

        // Update the range with margin
        double ImageMargin = 100.;
        double fractionImageMargin = 2;
        for(size_t iPlane=0;iPlane<3;iPlane++){
            time_bounds[iPlane].first  -= std::max(ImageMargin,fractionImageMargin*(time_bounds[iPlane].second-time_bounds[iPlane].first));
            time_bounds[iPlane].second += std::max(ImageMargin,fractionImageMargin*(time_bounds[iPlane].second-time_bounds[iPlane].first));
            wire_bounds[iPlane].first  -= std::max(ImageMargin,fractionImageMargin*(wire_bounds[iPlane].second-wire_bounds[iPlane].first));
            wire_bounds[iPlane].second += std::max(ImageMargin,fractionImageMargin*(wire_bounds[iPlane].second-wire_bounds[iPlane].first));

            if(time_bounds[iPlane].first < 0) time_bounds[iPlane].first = 0;
            if(wire_bounds[iPlane].first < 0) wire_bounds[iPlane].first = 0;

            wire_bounds[iPlane].first  = (size_t)(wire_bounds[iPlane].first + 0.5);
            wire_bounds[iPlane].second = (size_t)(wire_bounds[iPlane].second + 0.5);
        }
        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);

        // make sure the number of rows and cols are divisible by _compressionFactor
        for(size_t iPlane=0;iPlane<3;iPlane++){
            while(!( (size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first)%_compressionFactor_t == 0)){
                time_bounds[iPlane].second++;//=(_compressionFactor_t-((size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first)%_compressionFactor_t));
            }
            tellMe(Form("%zu rows for %d compression factor",(size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first),_compressionFactor_t),0);

            while(!( (size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first)%_compressionFactor_w == 0)){
                wire_bounds[iPlane].second++;//=(_compressionFactor_w-((size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first)%_compressionFactor_w));
            }
            tellMe(Form("%zu cols for %d compression factor",(size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first),_compressionFactor_w),0);
        }

        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);
    }
    //______________________________________________________
    void AStarTracker::SetTimeAndWireBounds(std::vector<TVector3> points, std::vector<larcv::ImageMeta> meta){
        time_bounds.clear();
        wire_bounds.clear();
        //time_bounds.reserve(3);
        //wire_bounds.reserve(3);
        for(size_t iPlane=0; iPlane<3; ++iPlane) {
            std::pair<double,double> timelimit;
            std::pair<double,double> wirelimit;
            timelimit.first  = 1.e9;
            timelimit.second = 0.;
            wirelimit.first  = 1.e9;
            wirelimit.second = 0.;
            time_bounds.push_back(timelimit);
            wire_bounds.push_back(wirelimit);
            //time_bounds[iPlane] = timelimit;
            //wire_bounds[iPlane] = wirelimit;
        }

        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);

        for(size_t iPlane=0;iPlane<3;iPlane++){
            // make sure starting point is projected back in the range
            for(size_t iPoint = 0;iPoint<points.size();iPoint++){

                double _parent_x = points[iPoint].X();
                double _parent_y = points[iPoint].Y();
                double _parent_z = points[iPoint].Z();
                double _parent_t = 0;
                double x_pixel, y_pixel;
                ProjectTo3D(meta[iPlane],_parent_x,_parent_y,_parent_z,_parent_t,iPlane,x_pixel,y_pixel); // y_pixel is time

                if( y_pixel < time_bounds[iPlane].first ) time_bounds[iPlane].first  = y_pixel;
                if( y_pixel > time_bounds[iPlane].second) time_bounds[iPlane].second = y_pixel;

                if(x_pixel < wire_bounds[iPlane].first ) wire_bounds[iPlane].first  = x_pixel;
                if(x_pixel > wire_bounds[iPlane].second) wire_bounds[iPlane].second = x_pixel;
            }
        }
        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);

        // equalize time ranges to intercept the same range on 3 views
        for(size_t iPlane=0;iPlane<3;iPlane++){
            for(size_t jPlane=0;jPlane<3;jPlane++){
                if(time_bounds[jPlane].first  <= time_bounds[iPlane].first) {time_bounds[iPlane].first  = time_bounds[jPlane].first;}
                if(time_bounds[jPlane].second >= time_bounds[iPlane].second){time_bounds[iPlane].second = time_bounds[jPlane].second;}
            }
        }
        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);

        // Update the range with margin
        double ImageMargin = 100.;
        double fractionImageMargin = 1;
        for(size_t iPlane=0;iPlane<3;iPlane++){
            time_bounds[iPlane].first  -= std::max(ImageMargin,fractionImageMargin*(time_bounds[iPlane].second-time_bounds[iPlane].first));
            time_bounds[iPlane].second += std::max(ImageMargin,fractionImageMargin*(time_bounds[iPlane].second-time_bounds[iPlane].first));
            wire_bounds[iPlane].first  -= std::max(ImageMargin,fractionImageMargin*(wire_bounds[iPlane].second-wire_bounds[iPlane].first));
            wire_bounds[iPlane].second += std::max(ImageMargin,fractionImageMargin*(wire_bounds[iPlane].second-wire_bounds[iPlane].first));

            if(time_bounds[iPlane].first < 0) time_bounds[iPlane].first = 0;
            if(wire_bounds[iPlane].first < 0) wire_bounds[iPlane].first = 0;

            wire_bounds[iPlane].first  = (size_t)(wire_bounds[iPlane].first + 0.5);
            wire_bounds[iPlane].second = (size_t)(wire_bounds[iPlane].second + 0.5);
        }
        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);

        // make sure the number of rows and cols are divisible by _compressionFactor
        /*for(size_t iPlane=0;iPlane<3;iPlane++){
         while(!( (size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first)%_compressionFactor_t == 0)){
         time_bounds[iPlane].second++;//=(_compressionFactor_t-((size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first)%_compressionFactor_t));
         }
         tellMe(Form("%zu rows for %d compression factor",(size_t)(time_bounds[iPlane].second - time_bounds[iPlane].first),_compressionFactor_t),0);

         while(!( (size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first)%_compressionFactor_w == 0)){
         wire_bounds[iPlane].second++;//=(_compressionFactor_w-((size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first)%_compressionFactor_w));
         }
         tellMe(Form("%zu cols for %d compression factor",(size_t)(wire_bounds[iPlane].second - wire_bounds[iPlane].first),_compressionFactor_w),0);
         }*/

        tellMe(Form("wire_bounds.size() = %zu",wire_bounds.size()),2);
        tellMe(Form("time_bounds.size() = %zu",time_bounds.size()),2);
    }
    //______________________________________________________
    void AStarTracker::Reconstruct(){
        _eventTreated++;
        tellMe("creating ch status and tag image",1);
        chstatus_image_v.clear();
        std::vector<larcv::Image2D> tag_image_v(3);
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            tag_image_v[iPlane] = hit_image_v[iPlane];
            tag_image_v[iPlane].paint(0);

            chstatus_image_v.push_back(hit_image_v[iPlane]);
            chstatus_image_v[iPlane].paint(0);
        }

        RecoedPath.clear();
        int goal_reached = 0;

        //_______________________________________________
        // Get Start and end points
        //-----------------------------------------------
        tellMe("getting start and end points",0);
        std::vector<int> start_cols(3);
        std::vector<int> end_cols(3);
        int start_row(0), end_row(0);

        //tellMe("ok here");

        for(size_t iPlane=0;iPlane<3;iPlane++){

            if(hit_image_v[iPlane].meta().width()==0)continue;

            double x_pixel, y_pixel;
            ProjectTo3D(hit_image_v[iPlane].meta(),start_pt.X(),start_pt.Y(),start_pt.Z(),0,iPlane,x_pixel,y_pixel);
            start_cols[iPlane] = (int)(x_pixel);
            tellMe(Form("start_colz[%zu] = %.1f",iPlane,x_pixel),1);
            if(x_pixel<=0){tellMe("ERROR, startPt outside of image range",0);return;}
            start_row = y_pixel;

            ProjectTo3D(hit_image_v[iPlane].meta(),end_pt.X(),end_pt.Y(),end_pt.Z(),0,iPlane,x_pixel,y_pixel);
            //double wireProjEndPt   = x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x;
            end_cols[iPlane]   = x_pixel;// hit_image_v[iPlane].meta().col(wireProjEndPt);
            end_row = y_pixel;
        }

        tellMe("........configuring A*", 0);


        //_______________________________________________
        // Configure A* algo
        //-----------------------------------------------
        larcv::AStar3DAlgoConfig config;
        config.accept_badch_nodes = true;
        config.astar_threshold.resize(3,1);    // min value for a pixel to be considered non 0
        config.astar_neighborhood.resize(3,3); //can jump over n empty pixels in all planes
        config.astar_start_padding = 3;        // allowed region around the start point
        config.astar_end_padding = 3;          // allowed region around the end point
        config.lattice_padding = 5;            // margin around the edges
        config.min_nplanes_w_hitpixel = 3;     // minimum allowed coincidences between non 0 pixels across planes
        config.restrict_path = false;          // do I want to restrict to a cylinder around the strainght line of radius defined bellow ?
        config.path_restriction_radius = 30.0;

        //_______________________________________________
        // Define A* algo
        //-----------------------------------------------
        larcv::AStar3DAlgoProton algo( config );
        algo.setVerbose(0);
        algo.setPixelValueEsitmation(true);

        tellMe("...........starting A*", 0);
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            tellMe(Form("plane %zu %zu rows and %zu cols before findpath", iPlane,hit_image_v[iPlane].meta().rows(),hit_image_v[iPlane].meta().cols()),1);
        }
        RecoedPath = algo.findpath( hit_image_v, chstatus_image_v, tag_image_v, start_row, end_row, start_cols, end_cols, goal_reached );
        tellMe("..........done with A*", 0);
        if(goal_reached == 1){
            _eventSuccess++;
        }

        //_______________________________________________
        // Make track out of 3D nodes
        //-----------------------------------------------
        tellMe("making newTrack",1);
        Make3DpointList();
    }
    //______________________________________________________
    void AStarTracker::EnhanceDerivative(){
        tellMe("EnhanceDerivative()",0);
        std::vector<larcv::Image2D> enhanced_v(3);
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            larcv::Image2D enhanced = hit_image_v[iPlane];
            for(size_t icol = 0;icol<hit_image_v[iPlane].meta().cols();icol++){
                for(size_t irow = 0; irow< hit_image_v[iPlane].meta().rows();irow++){
                    if(irow==0 || irow==hit_image_v[iPlane].meta().rows()-1)continue;
                    if(hit_image_v[iPlane].pixel(irow,icol) == _deadWireValue) continue;
                    double value = hit_image_v[iPlane].pixel(irow,icol)*0.5*(hit_image_v[iPlane].pixel(irow,icol)-hit_image_v[iPlane].pixel(irow-1,icol));
                    if(value > 0)enhanced.set_pixel(irow,icol,value);
                    else enhanced.set_pixel(irow,icol,0);
                }
            }
            enhanced_v[iPlane] = enhanced;
        }
        hit_image_v = enhanced_v;
    }
    //______________________________________________________
    void AStarTracker::MaskTrack(){
        tellMe("MaskTrack",0);
        if(_vertexTracks.size()!=0)MaskVertex();
        double shellMask = 2;
        double MaskedValue = 0;
        if(_3DTrack.size() < 3)return;
        for(size_t iNode = 2;iNode<_3DTrack.size()-2; iNode++){
            for(size_t iPlane = 0;iPlane<3;iPlane++){
                double x_proj,y_proj;
                ProjectTo3D(hit_image_v[iPlane].meta(),_3DTrack[iNode].X(),_3DTrack[iNode].Y(),_3DTrack[iNode].Z(),0,iPlane,x_proj,y_proj);
                TVector3 A(x_proj,y_proj,0);
                ProjectTo3D(hit_image_v[iPlane].meta(),_3DTrack[iNode+1].X(),_3DTrack[iNode+1].Y(),_3DTrack[iNode+1].Z(),0,iPlane,x_proj,y_proj);
                TVector3 B(x_proj,y_proj,0);
                for(size_t irow = 0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    if(irow < std::min(A.Y()-20, B.Y()-20))continue;
                    if(irow > std::max(A.Y()+20, B.Y()+20))continue;

                    for(size_t icol = 0;icol<hit_image_v[iPlane].meta().cols();icol++){
                        if(icol < std::min(A.X()-20, B.X()-20))continue;
                        if(icol > std::max(A.X()+20, B.X()+20))continue;
                        if(hit_image_v[iPlane].pixel(irow,icol) == 0)continue;
                        double alphaCol = ((icol+0.5)-A.X())/(B.X()-A.X());
                        double alphaRow = ((irow+0.5)-A.Y())/(B.Y()-A.Y());
                        TVector3 pC((icol),(irow),0);

                        if(alphaCol >= 0 && alphaCol <= 1 && alphaRow >= 0 && alphaRow <= 1){
                            if(GetDist2line(A,B,pC) < shellMask){
                                hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
				masked_hit_image_v[iPlane].set_pixel(irow,icol,10.0);
                            }
                        }
                        if((_3DTrack[iNode]-_3DTrack[0]).Mag() < 2){
                            if((pC-A).Mag() < 0.75*shellMask || (pC-B).Mag() < 0.5*shellMask){
                                hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
				masked_hit_image_v[iPlane].set_pixel(irow,icol,10.0);
                            }
                        }
                        else{
                            if((pC-A).Mag() < 1.5*shellMask || (pC-B).Mag() < shellMask){
                                hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
				masked_hit_image_v[iPlane].set_pixel(irow,icol,10.0);
                            }
                        }


                    }//icol
                }// irow
            }//iPlane
        }//iNode
    }
    //______________________________________________________
    void AStarTracker::MaskVertex(){
        tellMe("MaskVertex()",0);
        double shellMask = 4;
        double MaskedValue = 0;
        if(_vertexTracks.size() == 0)return;
        for(size_t itrack=0;itrack<_vertexTracks.size();itrack++){
            //tellMe(Form("itrack = %zu / %zu",itrack,_vertexTracks.size()),0);
            if(_vertexTracks[itrack].size() < 3)continue;
            for(size_t iNode = 1;iNode<_vertexTracks[itrack].size()-1; iNode++){
                //tellMe(Form("iNode = %zu / %zu",iNode,_vertexTracks[itrack].size()-1),0);
                for(size_t iPlane = 0; iPlane<3; iPlane++){
                    //tellMe(Form("iPlane = %zu / 3",iPlane),0);
                    double x_proj,y_proj;
                    ProjectTo3D(hit_image_v[iPlane].meta(),_vertexTracks[itrack][iNode].X(),_vertexTracks[itrack][iNode].Y(),_vertexTracks[itrack][iNode].Z(),0,iPlane,x_proj,y_proj);
                    TVector3 A(x_proj,y_proj,0);
                    //tellMe("A OK",0);
                    ProjectTo3D(hit_image_v[iPlane].meta(),_vertexTracks[itrack][iNode+1].X(),_vertexTracks[itrack][iNode+1].Y(),_vertexTracks[itrack][iNode+1].Z(),0,iPlane,x_proj,y_proj);
                    TVector3 B(x_proj,y_proj,0);
                    //tellMe("B OK",0);
                    for(size_t irow = 0;irow<hit_image_v[iPlane].meta().rows();irow++){
                        if(irow < std::min(A.Y()-20, B.Y()-20))continue;
                        if(irow > std::max(A.Y()+20, B.Y()+20))continue;

                        for(size_t icol = 0;icol<hit_image_v[iPlane].meta().cols();icol++){
                            if(icol < std::min(A.X()-20, B.X()-20))continue;
                            if(icol > std::max(A.X()+20, B.X()+20))continue;
                            if(hit_image_v[iPlane].pixel(irow,icol) == 0)continue;
                            double alphaCol = ((icol+0.5)-A.X())/(B.X()-A.X());
                            double alphaRow = ((irow+0.5)-A.Y())/(B.Y()-A.Y());
                            TVector3 pC((icol+0.5),(irow+0.5),0);
                            double alpha = ((B-A).Dot(pC-A))/(pow((B-A).Mag(),2));

                            if(alpha  >= -0.1 && alpha <= 1.1){
                                if(GetDist2line(A,B,pC) < shellMask){
                                    hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
				    masked_hit_image_v[iPlane].set_pixel(irow,icol,10.0);
                                }
                            }
                            if((_vertexTracks[itrack][iNode]-_vertexTracks[itrack][0]).Mag() < 2){
                                if((pC-A).Mag() < 0.75*shellMask || (pC-B).Mag() < 0.75*shellMask){
                                    hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
				    masked_hit_image_v[iPlane].set_pixel(irow,icol,10.0);
                                }
                            }
                            else{
                                if((pC-A).Mag() < 1.5*shellMask || (pC-B).Mag() < 1.5*shellMask){
                                    hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
				    masked_hit_image_v[iPlane].set_pixel(irow,icol,10.0);
                                }
                            }


                        }//icol
                    }// irow
                    //tellMe("Image scanned",0);
                }//iPlane
            }//iNode
        }//itrack
    }
    //______________________________________________________
    void AStarTracker::ReconstructVertex(){
        tellMe("ReconstructVertex()",0);
        //fEventOutput = TFile::Open(Form("%s/cVertex_%05d_%05d_%05d_%03d.root",_outdir.c_str(),_run,_subrun,_event,_track),"RECREATE");
        std::vector<TVector3> thisvertex;
        thisvertex.push_back(start_pt);
        MakeDeadWireList();
	hit_image_v.clear();
	masked_hit_image_v.clear();
        CropFullImage2boundsIntegrated(thisvertex);
        //DrawROI();
        //MakeDeadWireList();
        //ShaveTracks();
        SetTrackInfo(_run, _subrun, _event, _track);
        ConstructVertex();
        tellMe("Construct vertex done",0);
        if(_tooShortDeadWire)RecoverFromFail();
        std::cout << "Recovered if failed" << std::endl;

        //gDetector->Write();
        //gWorld->Write();
        //fEventOutput->Close();
    }
    //______________________________________________________
    void AStarTracker::ConstructTrack(){
        tellMe("ConstructTrack()",0);
        ImprovedCluster();
        end_pt = GetFurtherFromVertex();
        SortAndOrderPoints();
        RegularizeTrack();

//        for(size_t i = 1;i<_3DTrack.size();i++){
//            std::cout << i << "\t" << (_3DTrack[i]-_3DTrack[i-1]).Mag() << std::endl;
//        }

        ComputeLength();
        if(_Length3D > 0){
            std::cout << "add track" << std::endl;
            _3DTrack.push_back(end_pt);
            _vertexEndPoints.push_back(end_pt);
            //FillInTrack();
            _vertexTracks.push_back(_3DTrack);
            ComputeLength();
            //ComputeNewdQdX();
	    //if (_DrawOutputs){
	    //  hLength->Fill(_Length3D);
	   // }
            _track++;
            //DrawTrack();
        }
        else{std::cout << "xxxxx don't add track xxxxx" << std::endl;}
    }
    //______________________________________________________
    void AStarTracker::ConstructVertex(){
        tellMe("ConstructVertex()",0);
        size_t oldSize = 5;
        size_t newsize = 2;

        _missingTrack = false;
        _jumpingTracks = false;
        _branchingTracks = false;
        _tooShortDeadWire = false;
        _tooShortFaintTrack = false;
        _nothingReconstructed = false;

        if(_vertexTracks.size()    !=0)_vertexTracks.clear();
        if(_closestWall.size()     !=0)_closestWall.clear();
        if(_closestWall_SCE.size() !=0)_closestWall_SCE.clear();
        if(_vertexLength.size()    !=0)_vertexLength.clear();
        if(_vertexQDQX.size()      !=0)_vertexQDQX.clear();
        if(_vertexEndPoints.size() !=0)_vertexEndPoints.clear();
        if(_vertex_dQdX_v.size()   !=0)_vertex_dQdX_v.clear();
        if(_Track15cmAvgPt.size()  !=0)_Track15cmAvgPt.clear();
        _vertexEndPoints.push_back(start_pt);

        // look for tracks that start from the vertex
        while(newsize!=oldSize && newsize<9){
            oldSize = _vertexTracks.size();
            ConstructTrack();
            MaskVertex();
            if(_vertexTracks.size() == 0 && newsize!=1)newsize=1;
            else{newsize = _vertexTracks.size();}
            _track++;
        }
        // at that point I should ahve all tracks that start at the vertex point

        // look for tracks that could start from found end points
        TVector3 vertexPoint = start_pt;
        if(_vertexTracks.size()>1){
            for(size_t iEnd=1;iEnd<_vertexEndPoints.size();iEnd++){
                std::cout << "end point " << iEnd << " / " << _vertexEndPoints.size() << std::endl;
                start_pt = _vertexEndPoints[iEnd];
                MaskVertex();
                newsize = 1;
                while(newsize!=oldSize && newsize<9){// allows for several tracks per end point => branching tracks?, showers?...
                    oldSize = _vertexTracks.size();
                    ConstructTrack();
                    if(_vertexTracks.size() != oldSize)MaskVertex();
                    if(_vertexTracks.size() == 0 && newsize!=1)newsize=1;
                    else{newsize = _vertexTracks.size();}
                    _track++;
                }
            }
        }

        start_pt = vertexPoint;

        CleanUpVertex();
        _vertexLength = GetVertexLength();
        DiagnoseVertex();
        MakeVertexTrack();
        //if (_DrawOutputs) DrawVertex();
        //std::cout << "vertex drawn" << std::endl;
    }
    //______________________________________________________
    void AStarTracker::CleanUpVertex(){
        // loop through tracks, if they have one point in common, fuse them and re-do the sort and order and regularize
        std::vector<std::vector<TVector3> > newVertexTracks;// = _vertexTracks;
        std::vector<std::vector<TVector3> > Tracks2fuse;
        std::vector<int> trackFused;
        bool skip = false;
        _branchingTracks = false;
        for(size_t itrack = 0;itrack<_vertexTracks.size();itrack++){
            if(Tracks2fuse.size()!=0)Tracks2fuse.clear();
            for(size_t ptrack=0;ptrack<trackFused.size();ptrack++){if(trackFused[ptrack] == itrack)skip=true;}
            if(skip)continue;
            start_pt = _vertexTracks[itrack][0];
            Tracks2fuse.push_back(_vertexTracks[itrack]);
            trackFused.push_back(itrack);

            for(size_t jtrack = itrack+1;jtrack<_vertexTracks.size();jtrack++){
                skip=false;
                for(size_t ptrack=0;ptrack<trackFused.size();ptrack++){if(trackFused[ptrack] == jtrack)skip=true;}
                if(skip)continue;
                if(_vertexTracks[itrack].back() == _vertexTracks[jtrack][0] || _vertexTracks[jtrack].back() == _vertexTracks[itrack][0]){
                    Tracks2fuse.push_back(_vertexTracks[jtrack]);
                    trackFused.push_back(jtrack);
                }
            }

            std::vector<TVector3> fusedTrack;
            if(Tracks2fuse.size() > 2)_branchingTracks = true;
            for(size_t kTrack=0;kTrack<Tracks2fuse.size();kTrack++){
                for(size_t iNode=0;iNode<Tracks2fuse[kTrack].size();iNode++){
                    fusedTrack.push_back(Tracks2fuse[kTrack][iNode]);
                }
            }
            _3DTrack = fusedTrack;
            _track++;
            CropFullImage2boundsIntegrated(_vertexEndPoints);
            //ShaveTracks();
            //ComputeNewdQdX();
            //DrawTrack();
            ComputeLength();
            if(_Length3D > 2)newVertexTracks.push_back(_3DTrack);
        }

        _vertexTracks = newVertexTracks;
    }
    //______________________________________________________
    void AStarTracker::DiagnoseVertex(){
        tellMe("DiagnoseVertex()",0);

        _tooShortFaintTrack= false;
        _tooShortDeadWire= false;
        _tooManyTracksAtVertex= false;
        _possiblyCrossing= false;
        _possibleCosmic= false;
        _jumpingTracks = false;

        if(_vertexTracks.size() == 0){_nothingReconstructed = true; return;} // if no track reconstructed, probably wrong vertex
        CropFullImage2boundsIntegrated(_vertexEndPoints);
        //ShaveTracks();
        MaskVertex();

        ComputeClosestWall();
        //ComputeClosestWall_SCE();

        if(_jumpingTracks_v.size()      != 0) _jumpingTracks_v.clear();
        if(_possiblyCrossing_v.size()   != 0) _possiblyCrossing_v.clear();
        if(_tooShortDeadWire_v.size()   != 0) _tooShortDeadWire_v.clear();
        if(_tooShortFaintTrack_v.size() != 0) _tooShortFaintTrack_v.clear();
        if(_goodTrack_v.size()          != 0) _goodTrack_v.clear();

        for(size_t itrack=0;itrack<_vertexTracks.size();itrack++){
            _jumpingTracks      = false;
            _tooShortDeadWire   = false;
            _tooShortFaintTrack = false;
            _possiblyCrossing   = false;


            if(_vertexLength[itrack] > _MinLength) DiagnoseTrack(itrack);

            _jumpingTracks_v.push_back(     _jumpingTracks     );
            _possiblyCrossing_v.push_back(  _possiblyCrossing  );
            _tooShortDeadWire_v.push_back(  _tooShortDeadWire  );
            _tooShortFaintTrack_v.push_back(_tooShortFaintTrack);

            bool GoodTrack      = false;
            if(!(_tooShortDeadWire || _tooShortFaintTrack || _branchingTracks || _jumpingTracks))GoodTrack = true;
            _goodTrack_v.push_back(GoodTrack);

        }

        for(size_t itrack=0;itrack<_vertexTracks.size();itrack++){
            if(_jumpingTracks_v[itrack]      == true) _jumpingTracks      = true;
            if(_possiblyCrossing_v[itrack]   == true) _possiblyCrossing   = true;
            if(_tooShortDeadWire_v[itrack]   == true) _tooShortDeadWire   = true;
            if(_tooShortFaintTrack_v[itrack] == true) _tooShortFaintTrack = true;

        }

        // Find if tracks are back to back at the vertex point
        std::vector<std::vector<double> > Angle_v = GetVertexAngle(15,0);
        for(size_t itrack = 0;itrack<_vertexTracks.size();itrack++){
            for(size_t jtrack = 0;jtrack<_vertexTracks.size();jtrack++){
                if(Angle_v[itrack][jtrack] > 170)_possibleCosmic=true;
            }
        }



        // look for vertex with more than 2 tracks at the vertex point
        int NtracksAtVertex=0;
        _tooManyTracksAtVertex = false;
        for(size_t itrack = 0;itrack<_vertexTracks.size();itrack++){
            if(_vertexTracks[itrack][0] == start_pt && _vertexLength[itrack] > _MinLength)NtracksAtVertex++;
        }
        if(NtracksAtVertex > 2)_tooManyTracksAtVertex = true;
        if(NtracksAtVertex == 1)_missingTrack = true;
        if(NtracksAtVertex == 0)_nothingReconstructed=true;
        std::cout << NtracksAtVertex << "tracks at vertex" << std::endl;


        if(_jumpingTracks        ){tellMe("_jumpingTracks        ",0);}
        if(_tooShortFaintTrack   ){tellMe("_tooShortFaintTrack   ",0);}
        if(_tooShortDeadWire     ){tellMe("_tooShortDeadWire     ",0);}
        if(_tooManyTracksAtVertex){tellMe("_tooManyTracksAtVertex",0);}
        if(_possiblyCrossing     ){tellMe("_possiblyCrossing     ",0);}
        if(_possibleCosmic       ){tellMe("_possibleCosmic       ",0);}
        if(!_tooManyTracksAtVertex && !_tooShortDeadWire && !_tooShortFaintTrack && !_possibleCosmic && !_possiblyCrossing && !_jumpingTracks){tellMe("seems OK",0);}

    }
    //______________________________________________________
    void AStarTracker::DiagnoseTrack(size_t itrack){
        tellMe("DiagnoseTrack",0);

        // find events for which the track is too small because of dead wires || ends on dead wires
        double x,y,z;
        double radiusSphere = 3;
        double NrandomPts = 100*radiusSphere*radiusSphere;

        int NaveragePts = 0;
        int NpointsEvaluate=0;
        TVector3 AveragePoint;
        TVector3 newPoint;
        TVector3 oldEndPoint=_vertexTracks[itrack].back();
        double fractionSphereEmpty[3] = {0,0,0};
        double fractionSphereDead[3]  = {0,0,0};
        double fractionSphereTrack[3] = {0,0,0};

        bool PlaneDead[3]  = {0,0,0};
        bool PlaneTrack[3] = {0,0,0};

        for(size_t iNode=0;iNode<_vertexTracks[itrack].size();iNode++){
            if((_vertexTracks[itrack][iNode]-oldEndPoint).Mag() < 20){NaveragePts++;AveragePoint+=(_vertexTracks[itrack][iNode]-oldEndPoint);}
        }
        if(NaveragePts!=0){AveragePoint*=1./NaveragePts;}
        AveragePoint+=oldEndPoint;

        for(size_t i=0;i<NrandomPts;i++){
            ran.Sphere(x,y,z,radiusSphere);
            newPoint.SetXYZ(oldEndPoint.X()+x,oldEndPoint.Y()+y,oldEndPoint.Z()+z);
            if( (oldEndPoint-AveragePoint).Dot(newPoint-oldEndPoint)/( (oldEndPoint-AveragePoint).Mag()*(newPoint-oldEndPoint).Mag() ) < 0.8 )continue;
            NpointsEvaluate++;
            double x_pixel,y_pixel;
            double projValue[3];
            for(size_t iPlane=0;iPlane<3;iPlane++){
                ProjectTo3D(hit_image_v[iPlane].meta(),newPoint.X(),newPoint.Y(),newPoint.Z(),0,iPlane,x_pixel,y_pixel);
                projValue[iPlane] = hit_image_v[iPlane].pixel(y_pixel,x_pixel);
                if(projValue[iPlane] == 0)fractionSphereEmpty[iPlane]++;
                if(projValue[iPlane] == _deadWireValue) fractionSphereDead[iPlane]++;
                if(projValue[iPlane] > _deadWireValue)fractionSphereTrack[iPlane]++;
            }
        }
        int NplanesDead = 0;
        int NplanesTrack= 0;
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            fractionSphereTrack[iPlane]*=100./NpointsEvaluate;
            fractionSphereEmpty[iPlane]*=100./NpointsEvaluate;
            fractionSphereDead[iPlane ]*=100./NpointsEvaluate;
            /*std::cout << "track : " << itrack << ", plane : " << iPlane << " : " << fractionSphereTrack[iPlane] << "% on track" << std::endl;
            std::cout << "track : " << itrack << ", plane : " << iPlane << " : " << fractionSphereEmpty[iPlane] << "% on empty" << std::endl;
            std::cout << "track : " << itrack << ", plane : " << iPlane << " : " << fractionSphereDead[iPlane ]  << "% on dead" << std::endl;*/
            if(fractionSphereDead[iPlane]  > 40){NplanesDead++;PlaneDead[iPlane]=true;}
            else{PlaneDead[iPlane]=false;}

            if(fractionSphereTrack[iPlane] > 3){NplanesTrack++;PlaneTrack[iPlane]=true;}
            else{PlaneTrack[iPlane]=false;}
        }
        std::cout << NplanesDead << " dead wires planes" << std::endl;
        std::cout << NplanesTrack << " track planes" << std::endl;


        if(NplanesDead == 3 && NplanesTrack == 0) _tooShortDeadWire = true;
        if(NplanesDead == 2 && NplanesTrack == 1) _tooShortDeadWire = true;
        if(NplanesDead == 1 && NplanesTrack == 2) _tooShortDeadWire = true;

        if(NplanesDead == 0 && NplanesTrack == 2) _tooShortFaintTrack = true;
        if(NplanesDead == 0 && NplanesTrack == 1) _tooShortFaintTrack = true;
        if(NplanesDead == 1 && NplanesTrack == 1) _tooShortFaintTrack = true;

        for(size_t iPlane=0;iPlane<3;iPlane++){
            if(PlaneTrack[iPlane]==false && PlaneDead[iPlane] ==false && _tooShortFaintTrack){
                tellMe(Form("failure on plane %zu",iPlane),0);
                //if(iPlane==2){_tooShortFaintTrack = false;_jumpingTracks=true;}// plane 2 is not an induction plane, should not have failures
                if(iPlane!=2){failedPlane = iPlane;}
            }
        }

        //find tracks that approach the edge of the detector too much
        double FVshell = 2;
        if(_closestWall[itrack] < FVshell) _possiblyCrossing = true;

        if(_jumpingTracks)tellMe("_jumpingTracks",0);
        if(_tooShortDeadWire)tellMe("_tooShortDeadWire",0);
        if(_tooShortFaintTrack)tellMe("_tooShortFaintTrack",0);
        if(_possiblyCrossing)tellMe("Leaving Fiducial Volume",0);


    }
    //______________________________________________________
    void AStarTracker::ReconstructEvent(){
        tellMe("ReconstructEvent()",0);
        if(_eventVertices.size() == 0){
            tellMe("ERROR, no vertex found",0);
            return;
        }
        fEventOutput = TFile::Open(Form("root/cVertex_%05d_%05d_%05d.root",_run,_subrun,_event),"RECREATE");

        for(size_t iVertex = 0;iVertex<_eventVertices.size();iVertex++){
            start_pt = _eventVertices[iVertex];
            std::vector<TVector3> thisvertex;
            thisvertex.push_back(start_pt);
            CropFullImage2boundsIntegrated(thisvertex);
            //ShaveTracks();
            SetTrackInfo(_run, _subrun, _event, _track);
            ConstructVertex();
            std::cout << std::endl << std::endl;
        }

        gDetector->Write();
        gWorld->Write();
        fEventOutput->Close();
    }
    //______________________________________________________
    void AStarTracker::ImprovedCluster(){
        tellMe("ImprovedCluster()",0);

	std::clock_t begin = std::clock();
	
        bool foundNewPoint = true;
        bool terminate = false;
        std::vector<TVector3> list3D;
        double x,y,z;
        int Ncrop=0;
        double rmax = 6;
        if(_IsMCC9 == false)rmax=6;
        else rmax = 6;

        TVector3 lastNode = start_pt;
        list3D.push_back(lastNode);// make sure the vertex point is in the future track;
        int iter = 0;

        /*std::vector<TVector3> vectorList;
        for(size_t i=0;i<100000;i++){
            TVector3 newCandidate;
            ran.Sphere(x,y,z,ran.Uniform(0.5,3));
            newCandidate.SetXYZ(lastNode.X()+x,lastNode.Y()+y,lastNode.Z()+z);
            //if(!CheckEndPointsInVolume(newCandidate)) continue; // point out of detector volume
            vectorList.push_back(newCandidate);
        }
        TH2D *hThetaPhi = new TH2D("hThetaPhi","hThetaPhi",100,-3.15,3.15,100,-3.15,3.15);
        for(size_t i=0;i<vectorList.size();i++){
            double x_proj, y_proj;
            bool pointOK = true;
            for(size_t iPlane=0;iPlane<3;iPlane++){
                ProjectTo3D(hit_image_v[iPlane].meta(),vectorList[i].X(), vectorList[i].Y(), vectorList[i].Z(),0, iPlane, x_proj,y_proj);
                if(hit_image_v[iPlane].pixel(y_proj,x_proj) < _ADCthreshold){pointOK = false;}
            }
            if(pointOK == true)hThetaPhi->Fill((vectorList[i]-lastNode).Theta(),(vectorList[i]-lastNode).Phi());
        }
        TCanvas *cThetaPhiVertex = new TCanvas(Form("cThetaPhiVertex_%05d_%05d_%05d_%04d_%04d",_run,_subrun,_event,_vtxID,_track),Form("cThetaPhiVertex_%05d_%05d_%05d_%04d_%04d",_run,_subrun,_event,_vtxID,_track),800,600);
        hThetaPhi->Draw("colz");
        cThetaPhiVertex->SaveAs(Form("%s/%s.root",_outdir.c_str(),cThetaPhiVertex->GetName()));
        cThetaPhiVertex->SaveAs(Form("%s/%s.png",_outdir.c_str(),cThetaPhiVertex->GetName()));
        hThetaPhi->Delete();
        cThetaPhiVertex->Delete();*/



        while(foundNewPoint && terminate == false){
            //if(_IsMCC9)std::cout << "update search pattern for MCC9" << std::endl;
            rmax = 2+4*exp(-1.*((lastNode-start_pt).Mag())/5.);//reduces the search range as the track size increases
            foundNewPoint = false;
            iter++;
            int Ndead = 0;
            double MaxsummedADC = 0;
            TVector3 newCandidate;
            std::vector<TVector3> thisList;
            thisList.push_back(lastNode);

            //for this iteration, throw points and look for possible candidates
            // find points on a small sphere around the last found point
            for(size_t i = 0;i<10000;i++){
                double r = ran.Uniform(0.1,rmax);
                ran.Sphere(x,y,z,r);
                newCandidate.SetXYZ(lastNode.X()+x,lastNode.Y()+y,lastNode.Z()+z);
                if(!CheckEndPointsInVolume(newCandidate)) continue; // point out of detector volume
                thisList.push_back(newCandidate);
            }

            //if we have a nicely defined track, find point in a larger cone after the last found point
            if(list3D.size() > 2 && (lastNode-start_pt).Mag() > _MinLength){
                TVector3 AvPt(0,0,0);
                int NpointAveragedOn = 0;
                //try and find the track direction on the last few cm
                for(size_t iNode=0;iNode<list3D.size();iNode++){
                    if( (list3D[iNode]-lastNode).Mag() < 10 ){
                        AvPt+=(list3D[iNode]-lastNode);
                        NpointAveragedOn++;
                    }
                }
                if(NpointAveragedOn!=0){AvPt*=1./NpointAveragedOn;}
                AvPt+=lastNode;

                for(size_t i = 0;i<10000;i++){
                    double r = ran.Uniform(0,2*rmax);
                    ran.Sphere(x,y,z,r);
                    newCandidate.SetXYZ(lastNode.X()+x,lastNode.Y()+y,lastNode.Z()+z);
                    if(!CheckEndPointsInVolume(newCandidate)) continue; // point out of detector volume
                    if((lastNode-AvPt).Angle(newCandidate-lastNode) > 30*TMath::Pi()/180.)continue;
                    thisList.push_back(newCandidate);
                }
            }

            // decide which point we keep
            for(size_t i = 0;i<thisList.size();i++){
                newCandidate = thisList[i];

                double coord2D[3][3];// for each plane : x,y and ADC value;
                bool TooClose = false;

                if(list3D.size() >= 1){// point too close to already found point
                    for(size_t i = 0;i<list3D.size()-1;i++){
                        if((newCandidate-list3D[i]).Mag() < 0.3) TooClose=true;
                    }
                }

                if(TooClose)continue;

                int NplanesOK=0;
                double pointSummedADC = 0;
                for(size_t iPlane = 0;iPlane<3;iPlane++){
                    double x_proj, y_proj;
                    ProjectTo3D(hit_image_v[iPlane].meta(),newCandidate.X(), newCandidate.Y(), newCandidate.Z(),0, iPlane, x_proj,y_proj);
                    coord2D[iPlane][0] = x_proj;
                    coord2D[iPlane][1] = y_proj;
                    coord2D[iPlane][2] = 0;

                    if(x_proj > (int)hit_image_v[iPlane].meta().cols()-10
                       || y_proj > (int)hit_image_v[iPlane].meta().rows()-10
                       || x_proj < 10
                       || y_proj < 10) {terminate = true;break;}
                    coord2D[iPlane][2] = hit_image_v[iPlane].pixel(y_proj,x_proj);
                    if(coord2D[iPlane][2] != 0){NplanesOK++;}
                    if(coord2D[iPlane][2] != _deadWireValue)pointSummedADC+=coord2D[iPlane][2];
                }
                if(terminate){
                    bool ReallyTerminate = false;
                    for(size_t iPlane=0;iPlane<3;iPlane++){
                        double x_proj,y_proj;
                        ProjectTo3D(original_full_image_v[iPlane].meta(),newCandidate.X(), newCandidate.Y(), newCandidate.Z(),0, iPlane, x_proj,y_proj);
                        if((x_proj > (int)original_full_image_v[iPlane].meta().cols()-10
                            || y_proj > (int)original_full_image_v[iPlane].meta().rows()-10
                            || x_proj < 10
                            || y_proj < 10)){ReallyTerminate = true;}
                    }
                    if(!ReallyTerminate && Ncrop < 20){
                        CropFullImage2boundsIntegrated(list3D);
                        //ShaveTracks();
                        Ncrop++;
                        MaskVertex();
                        terminate=false;
                    }
                }
                Ndead = 0;
                for(size_t iPlane=0;iPlane<3;iPlane++){if(coord2D[iPlane][2] == _deadWireValue)Ndead++;}
                if(NplanesOK > 2 && Ndead<2){
                    if(pointSummedADC > MaxsummedADC){
                        MaxsummedADC = pointSummedADC;
                        list3D.push_back(newCandidate);
                        foundNewPoint = true;
                    }
                }
            }//end of thisList loop 

            //decide what is the starting point for the next iteration by finding point furthest from start
            double dist = 0;
            if(_3DTrack.size() > 0){
                for(size_t i = 0;i<list3D.size();i++){
                    if((_3DTrack[0]-list3D[i]).Mag() > dist){lastNode = list3D[i];dist = (_3DTrack[0]-list3D[i]).Mag();}
                }
            }
        }

        _3DTrack = list3D;

	clock_t end = clock();
	char msg[50];
	sprintf(msg,"ImprovedCluster() elapsed=%.3f secs",float(end - begin) / CLOCKS_PER_SEC);
	tellMe(msg,0);

    }
    //______________________________________________________
    void AStarTracker::PreSortAndOrderPoints(){
        TVector3 FurtherFromVertex = GetFurtherFromVertex();
        std::vector<TVector3> newTrack;
        newTrack.push_back(start_pt);
        for(size_t iNode=0;iNode<_3DTrack.size();iNode++){
            if((_3DTrack[iNode]-FurtherFromVertex).Mag() < (newTrack.back()-FurtherFromVertex).Mag()){newTrack.push_back(_3DTrack[iNode]);}
        }
        _3DTrack = newTrack;
    }
    //______________________________________________________
    void AStarTracker::SortAndOrderPoints(){
        //try and get a "track"
        //(1) find point further away from the vertex
        //(2) for each point, look for closest point, which stays the closest to the "current point -> end point line"
        //(3) add this point to the track
        //(4) start again
        PreSortAndOrderPoints();
        TVector3 FurtherFromVertex;
        double dist2vertex = 0;
        for(size_t iNode = 0;iNode<_3DTrack.size();iNode++){
            if( (_3DTrack[iNode]-start_pt).Mag() > dist2vertex){dist2vertex = (_3DTrack[iNode]-start_pt).Mag(); FurtherFromVertex = _3DTrack[iNode];}
        }
        end_pt = FurtherFromVertex;
        std::vector<TVector3> list3D;
        list3D.push_back(start_pt);
        TVector3 bestCandidate;
        double distScore = 1e9;
        double dirScore = 1e9;
        double remainDistScore = 1e9;
        double remainDirScore = 1e9;
        double globalScore = 1e9;
        double bestGlobalScore = 1e9;
        bool goOn = true;
        while(list3D.size() < _3DTrack.size() && goOn == true){
            bestGlobalScore = 1e9;
            for(size_t iNode=0;iNode<_3DTrack.size();iNode++){
                bool alreadySelected = false;
                if(ArePointsEqual(_3DTrack[iNode],list3D.back())){alreadySelected = true;}
                for(size_t ilist = 0;ilist<list3D.size();ilist++){
                    //if( ArePointsEqual(list3D[ilist],_3DTrack[iNode]) ){alreadySelected = true;}
                    if( list3D[ilist] == _3DTrack[iNode] ){alreadySelected = true;}
                }
                if(alreadySelected)continue;

                double dist       = (_3DTrack[iNode]-list3D.back()    ).Mag();
                double remainDist = (_3DTrack[iNode]-FurtherFromVertex).Mag();
                double dir;
                if(list3D.size()>=2){dir = (list3D.back()-list3D[list3D.size()-2]).Dot( _3DTrack[iNode]-list3D[list3D.size()-2] )/( (list3D.back()-list3D[list3D.size()-2]).Mag()*(_3DTrack[iNode]-list3D[list3D.size()-2]).Mag() );}
                else{dir = 1;}
                double remainDir  = (FurtherFromVertex-list3D.back()).Dot(_3DTrack[iNode]-list3D.back())/((FurtherFromVertex-list3D.back()).Mag()*(_3DTrack[iNode]-list3D.back()).Mag() );

                int NplanesDead=0;
                for(size_t iPlane = 0;iPlane<3;iPlane++){
                    double x_proj, y_proj;
                    ProjectTo3D(hit_image_v[iPlane].meta(),_3DTrack[iNode].X(), _3DTrack[iNode].Y(), _3DTrack[iNode].Z(),0, iPlane, x_proj,y_proj);
                    if(hit_image_v[iPlane].pixel(y_proj,x_proj) == 20) NplanesDead++;
                }
                double DeadPlaneScore = 10*NplanesDead;

                distScore       = 5*dist;
                remainDistScore = 0.1*remainDist;
                dirScore        = 2*(2-dir);
                remainDirScore  = 10*(2-remainDir);
                globalScore     = distScore + remainDistScore + dirScore + remainDirScore + DeadPlaneScore;

                if(globalScore < bestGlobalScore){bestGlobalScore = globalScore;bestCandidate = _3DTrack[iNode];}
            }
            if((bestCandidate-list3D.back()).Mag() > 20){goOn = false;}
            if(goOn)list3D.push_back(bestCandidate);
        }
        _3DTrack = list3D;
    }
    //______________________________________________________
    void AStarTracker::Make3DpointList(){
        double nodeX,nodeY,nodeZ;
        std::vector<TVector3> list3D;
        TVector3 lastNode;
        //for(size_t iNode=RecoedPath.size()-1;iNode < RecoedPath.size();iNode--){
        int Nnodes = RecoedPath.size()-1;
        for(size_t iNode=0;iNode < RecoedPath.size();iNode++){
            double time = hit_image_v[0].meta().br().y+(RecoedPath[Nnodes-iNode].row)*hit_image_v[0].meta().pixel_height();
            nodeX = Tick2X(time,0);
            nodeY = RecoedPath.at(Nnodes-iNode).tyz[1];
            nodeZ = RecoedPath.at(Nnodes-iNode).tyz[2];
            TVector3 node(nodeX,nodeY,nodeZ);
            list3D.push_back(node);
            lastNode = node;
        }

        _3DTrack = list3D;

    }
    //______________________________________________________
    void AStarTracker::ComputedQdX(){
        tellMe("ComputedQdX()",0);
        if(_3DTrack.size() == 0){tellMe("ERROR, run the 3D reconstruction first",0);return;}
        if(_dQdx.size() != 0)_dQdx.clear();
        double Npx = 10;
        double dist,xp,yp,alpha;
        for(size_t iNode = 0;iNode<_3DTrack.size()-1;iNode++){
            std::vector<double> node_dQdx(3);
            for(size_t iPlane = 0;iPlane<3;iPlane++){
                double localdQdx = 0;
                int NpxLocdQdX = 0;
                double X1,Y1,X2,Y2;
                ProjectTo3D(hit_image_v[iPlane].meta(),_3DTrack[iNode].X(), _3DTrack[iNode].Y(), _3DTrack[iNode].Z(),0, iPlane, X1,Y1);
                ProjectTo3D(hit_image_v[iPlane].meta(),_3DTrack[iNode+1].X(), _3DTrack[iNode+1].Y(), _3DTrack[iNode+1].Z(),0, iPlane, X2,Y2);
                if(X1 == X2){X1 -= 0.0001;}
                if(Y1 == Y2){Y1 -= 0.0001;}
                TVector3 A(X1,Y1,0); // projection of node iNode on plane iPlane
                TVector3 B(X2,Y2,0); // projection of node iNode+1 on plane iPlane

                for(size_t icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                    for(size_t irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                        if(hit_image_v[iPlane].pixel(irow,icol) == 0)continue;
                        double theta0; // angle with which the segment [iNode, iNode+1] is ssen by the current point
                        double theta1; // angle with which the segment [jNode, jNode+1] is seen by the current point
                        double X0 = icol;
                        double Y0 = irow;
                        if(X0 == X1 || X0 == X2){X0 += 0.0001;}
                        if(Y0 == Y1 || Y0 == Y2){Y0 += 0.0001;}
                        alpha = ( (X2-X1)*(X0-X1)+(Y2-Y1)*(Y0-Y1) )/( pow(X2-X1,2)+pow(Y2-Y1,2) ); //normalized scalar product of the 2 vectors
                        xp = X1+alpha*(X2-X1); // projects X0 onto the [AB] segment
                        yp = Y1+alpha*(Y2-Y1);
                        dist = sqrt(pow(xp-X0,2)+pow(yp-Y0,2));
                        double dist2point = std::min( sqrt(pow(X0-X1,2)+pow(Y0-Y1,2)), sqrt(pow(X0-X2,2)+pow(Y0-Y2,2)) );
                        if( alpha >= 0 && alpha <= 1 && dist > Npx ) continue;
                        if( (alpha < 0 || alpha > 1) && dist2point > Npx ) continue;
                        theta0 = std::abs(std::acos(((X1-X0)*(X2-X0)+(Y1-Y0)*(Y2-Y0))/(sqrt(pow(X1-X0,2)+pow(Y1-Y0,2))*sqrt(pow(X2-X0,2)+pow(Y2-Y0,2)))));
                        bool bestCandidate = true;
                        for(size_t jNode=0;jNode<_3DTrack.size()-1;jNode++){
                            if(jNode==iNode)continue;
                            double x1,y1,x2,y2, alpha2, xp2,yp2, dist2;
                            ProjectTo3D(hit_image_v[iPlane].meta(),_3DTrack[jNode].X(), _3DTrack[jNode].Y(), _3DTrack[jNode].Z(),0, iPlane, x1,y1);
                            ProjectTo3D(hit_image_v[iPlane].meta(),_3DTrack[jNode+1].X(), _3DTrack[jNode+1].Y(), _3DTrack[jNode+1].Z(),0, iPlane, x2,y2);

                            if(x1 == x2){x1 -= 0.0001;}
                            if(y1 == y2){y1 -= 0.0001;}

                            alpha2 = ( (x2-x1)*(X0-x1)+(y2-y1)*(Y0-y1) )/( pow(x2-x1,2)+pow(y2-y1,2) );
                            xp2 = x1+alpha2*(x2-x1);
                            yp2 = y1+alpha2*(y2-y1);
                            dist2 = sqrt(pow(xp2-X0,2)+pow(yp2-Y0,2));

                            double dist2point2 = std::min( sqrt(pow(X0-x1,2)+pow(Y0-y1,2)), sqrt(pow(X0-x2,2)+pow(Y0-y2,2)) );

                            if( alpha2 >= 0 && alpha2 <= 1 && dist2 > Npx ) continue;
                            if( (alpha2 < 0 || alpha2 > 1) && dist2point2 > Npx ) continue;

                            theta1 = std::abs(std::acos(((x1-X0)*(x2-X0)+(y1-Y0)*(y2-Y0))/(sqrt(pow(x1-X0,2)+pow(y1-Y0,2))*sqrt(pow(x2-X0,2)+pow(y2-Y0,2)))));
                            if(theta1 >= theta0){bestCandidate=false;}
                        }//jNode
                        if(!bestCandidate)continue;
                        //double nodeDirX =  _3DTrack[iNode+1].X()-_3DTrack[iNode].X();
                        //double nodeDirY =  _3DTrack[iNode+1].Y()-_3DTrack[iNode].Y();
                        //double nodeDirZ =  _3DTrack[iNode+1].Z()-_3DTrack[iNode].Z();
                        //double norm = sqrt(nodeDirX*nodeDirX+nodeDirY*nodeDirY+nodeDirZ*nodeDirZ);
                        localdQdx+=hit_image_v[iPlane].pixel(irow,icol)/*/norm*/;
                        NpxLocdQdX++;

                    }// irow
                }// icol
                if(NpxLocdQdX!=0)node_dQdx[iPlane] = localdQdx/NpxLocdQdX;
                else node_dQdx[iPlane] = localdQdx;
            }// iPlane
            _dQdx.push_back(node_dQdx);
        }// iNode

        std::vector<double> nodedQdxtot;
        for(size_t iNode=0;iNode<_dQdx.size();iNode++){// get total dQdx per noded
            int NplanesOK = 0;
            double dQdxtot = 0;
            for(size_t iPlane = 0;iPlane<3;iPlane++){
                if(_dQdx[iNode][iPlane] != 0){dQdxtot+=_dQdx[iNode][iPlane];NplanesOK++;}
            }
            if(NplanesOK!=0){dQdxtot*=3/NplanesOK;}
            nodedQdxtot.push_back(dQdxtot);
            //if (_DrawOutputs) {
            //    hdQdx->Fill(dQdxtot/_Length3D);
            //    hLengthdQdX->Fill(ComputeLength(iNode),dQdxtot);
           // }
        }
        _vertexQDQX.push_back(nodedQdxtot);
    }
    //______________________________________________________
    void AStarTracker::ComputeNewdQdX(){
        tellMe("ComputeNewdQdX()",0);
        if(_3DTrack.size() == 0){tellMe("ERROR no track found",0);return;}
        double shellPix=2;
        if(dQdXperPlane_v.size()!=0)dQdXperPlane_v.clear();
        if(track_dQdX_v.size()  !=0)track_dQdX_v.clear();

        CropFullImage2boundsIntegrated(_3DTrack);

        for(size_t iPlane = 0;iPlane<3;iPlane++){
            std::vector<double> dQdXOnePlane_v;
            for(size_t iNode=0;iNode<_3DTrack.size();iNode++){
                int Npx = 0;
                double dQdx_per_plane = 0;
                double x_proj,y_proj;

                ProjectTo3D(hit_image_v[iPlane].meta(),_3DTrack[iNode].X(), _3DTrack[iNode].Y(), _3DTrack[iNode].Z(),0, iPlane, x_proj,y_proj);
                for(size_t irow = 0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    if(irow < y_proj-20 || irow > y_proj+20)continue;
                    for(size_t icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                        if(icol<x_proj-20 || icol>x_proj+20)continue;
                        if(sqrt( pow(irow+0.5-y_proj,2)+pow(icol+0.5-x_proj,2) ) < shellPix){
                            if(hit_image_v[iPlane].pixel(irow,icol) != _deadWireValue
                               && hit_image_v[iPlane].pixel(irow,icol)!=0
                               && hit_image_v[iPlane].pixel(irow,icol) != recoveredvalue){
                                Npx++;
                                dQdx_per_plane+= hit_image_v[iPlane].pixel(irow,icol);

                            }
                        }
                    }
                }
                if(Npx!=0)dQdXOnePlane_v.push_back(dQdx_per_plane/Npx);
                else dQdXOnePlane_v.push_back(dQdx_per_plane);
            }
            //_thisLarliteTrack.add_dqdx(dQdXOnePlane_v);
            dQdXperPlane_v.push_back(dQdXOnePlane_v);
        }
        for(size_t iNode=0;iNode<_3DTrack.size();iNode++){
            double thisdqdx = 0;
            for(size_t iPlane=0;iPlane<3;iPlane++){
                thisdqdx+=dQdXperPlane_v[iPlane][iNode];
            }
            track_dQdX_v.push_back(thisdqdx);
        }
        _vertex_dQdX_v.push_back(track_dQdX_v);
    }
    //______________________________________________________
    std::vector<double> AStarTracker::ComputeTruncateddQdX(double nDevs=1){
        tellMe("ComputeTruncateddQdX()",0);
        if(_3DTrack.size() == 0){
            tellMe("ERROR no track found",0);
        }

        double circRad   = 2;
        double searchCut = 20;

        if(TruncateddQdXperPlane_v.size()!=0) TruncateddQdXperPlane_v.clear();
        if(RawdQdXperPlane_v.size()!=0)       RawdQdXperPlane_v.clear();

        CropFullImage2boundsIntegrated(_3DTrack);

        for(size_t iPlane = 0; iPlane<3; iPlane++){
            std::vector<double> RawdQdXOnePlane_v;

            for(size_t iNode=0; iNode<_3DTrack.size(); iNode++){

                double x_proj,y_proj;
                ProjectTo3D(hit_image_v[iPlane].meta(), _3DTrack[iNode].X(), _3DTrack[iNode].Y(), _3DTrack[iNode].Z(), 0 , iPlane, x_proj , y_proj);

                short  Npx = 0;
                double dQdx_per_plane_point = 0;

                for(size_t irow = 0; irow<hit_image_v[iPlane].meta().rows(); irow++){
                    for(size_t icol=0; icol<hit_image_v[iPlane].meta().cols(); icol++){
                        if( std::abs(irow-y_proj) > searchCut || std::abs(icol-x_proj) > searchCut ) continue;

                        if( sqrt( pow(irow+0.5-y_proj,2)+pow(icol+0.5-x_proj,2) ) < circRad && hit_image_v[iPlane].pixel(irow,icol) != _deadWireValue && hit_image_v[iPlane].pixel(irow,icol)!=0 ){
                            Npx++;
                            dQdx_per_plane_point += hit_image_v[iPlane].pixel(irow,icol);
                        }
                    }
                }

                if( Npx!=0 ) RawdQdXOnePlane_v.push_back(dQdx_per_plane_point/Npx);
                else       RawdQdXOnePlane_v.push_back(dQdx_per_plane_point);
            }

            RawdQdXperPlane_v.push_back(RawdQdXOnePlane_v);
        }

        for (size_t iPlane = 0; iPlane < 3; iPlane++){
            double RawdQdXMedian;
            double RawdQdXMean = 0;
            double RawdQdXRMS;
            double TruncateddQdX = 0;
            double NumTruncated  = 0;
            std::vector<double> sortDummy = RawdQdXperPlane_v[iPlane];
            size_t numPts = sortDummy.size();

            std::sort(sortDummy.begin(),sortDummy.end());

            if(numPts % 2 == 0) RawdQdXMedian = (sortDummy[numPts/2 -1] + sortDummy[numPts/2 + 1]) / 2.0 ;
            else RawdQdXMedian = sortDummy[numPts/2];

            for (size_t pt = 0; pt < sortDummy.size(); pt++) { RawdQdXMean += sortDummy[pt] / numPts; }

            for (size_t pt = 0; pt < sortDummy.size(); pt++) { RawdQdXRMS  += 1/numPts * pow( sortDummy[pt] - RawdQdXMean , 2); }
            RawdQdXRMS = sqrt(RawdQdXRMS);

            for (size_t pt = 0; pt < sortDummy.size(); pt++) {
                if (std::abs( sortDummy[pt] - RawdQdXMedian ) < nDevs*RawdQdXRMS ) {
                    TruncateddQdX += sortDummy[pt];
                    NumTruncated++;
                }
            }
            TruncateddQdXperPlane_v.push_back(TruncateddQdX / NumTruncated);

        }

        return TruncateddQdXperPlane_v;

    }
    //______________________________________________________
    bool AStarTracker::IsGoodVertex(){
        if(_tooShortDeadWire || _tooShortFaintTrack || _tooManyTracksAtVertex || _missingTrack || _nothingReconstructed || _branchingTracks || _jumpingTracks)return false;
        else return true;
    }
    //______________________________________________________
    void AStarTracker::CreateDataImage(std::vector<larlite::wire> wire_v){
        tellMe("Entering CreateImages(wire)",1);
        hit_image_v.clear();
        //hit_image_v.reserve(3);
        const size_t num_planes = 3;

        // Using the range, construct Image2D

        for(size_t iPlane=0; iPlane<num_planes; ++iPlane) {
            // Retrieve boundaries
            auto const& time_bound = time_bounds[iPlane];
            auto const& wire_bound = wire_bounds[iPlane];
            // If no hit on this plane, make an empty image
            if(wire_bound.second <= wire_bound.first ||
               time_bound.second <= time_bound.first ) {
                tellMe("Adding Empty Image",0);
                hit_image_v[iPlane] = larcv::Image2D();
                continue;
            }
            // Construct meta
            size_t image_cols = (size_t)(wire_bound.second - wire_bound.first);
            size_t image_rows = (size_t)(time_bound.second - time_bound.first);
            tellMe(Form("%zu cols and %zu rows",image_cols,image_rows),1);
            larcv::ImageMeta hit_meta((double)image_cols, (double)image_rows,
                                      image_rows, image_cols,
                                      (size_t) wire_bound.first,  // origin x = min wire
                                      (size_t) time_bound.second, // origin y = max time
                                      iPlane);


            // Prepare hit image data + fill

            std::vector<float> image_data(hit_meta.rows() * hit_meta.cols(), 0.);
            size_t row,col;
            for(auto wire : wire_v){
                unsigned int ch = wire.Channel();
                unsigned int detWire = larutil::Geometry::GetME()->ChannelToWire(ch);
                unsigned int detPlane = larutil::Geometry::GetME()->ChannelToPlane(ch);
                tellMe(Form("ch : %d, wire : %d, plane : %d", ch, detWire, detPlane),2);
                if(detPlane != iPlane) continue;
                if(detWire > wire_bound.first && detWire < wire_bound.second){
                    for (auto & iROI : wire.SignalROI().get_ranges()) {
                        int FirstTick = iROI.begin_index();
                        int time_tick = FirstTick+2400;
                        for (float ADC : iROI) {
                            if ( time_tick > time_bound.first && time_tick < time_bound.second ) {
                                row = hit_meta.rows()-(size_t)(time_bound.second - time_tick + 0.5)-1;
                                col = (size_t)(detWire - wire_bound.first + 0.5);
                                if(ADC >= _ADCthreshold){image_data[hit_meta.rows() * col + row] = ADC;}// inverts time axis to go with LArCV convention
                            }
                            time_tick++;
                        } // ADC
                    }// iROI
                } // detWire
            }
            tellMe("image_data OK",1);
            larcv::Image2D hit_image(std::move(hit_meta),std::move(image_data));
            tellMe("hit_image OK",1);
            tellMe(Form("hit_image_v.size() = %zu",hit_image_v.size()),1);
            // compress Images
            tellMe("compress Images",1);
            if(_compressionFactor_t > 1 || _compressionFactor_w > 1 ){
                tellMe(Form("plane %zu size : %zu rows, %zu cols for a %d x %d compression",iPlane,hit_image.meta().rows(),hit_image.meta().cols(),_compressionFactor_t,_compressionFactor_w),1);
                hit_image.compress(hit_image.meta().rows()/_compressionFactor_t, hit_image.meta().cols()/_compressionFactor_w);
                tellMe(Form("plane %zu %zu rows and %zu cols after compression", iPlane,hit_image.meta().rows(),hit_image.meta().cols()),1);
                tellMe("image compressed",1);
            }
            hit_image_v.push_back(hit_image);
        }
    }
    //______________________________________________________
    void AStarTracker::FeedTrack(std::vector<TVector3> newTrack){
        tellMe("FeedTrack",0);
        tellMe(Form("newTrack.size() = %zu",newTrack.size()),0);
        _3DTrack = newTrack;
        _vertexTracks.push_back(newTrack);
        start_pt = newTrack[0];
        _vertexEndPoints.push_back(start_pt);
        _vertexEndPoints.push_back(newTrack.back());
    }
    //______________________________________________________
    void AStarTracker::DrawTrack(){
        tellMe("DrawTrack",0);
        TH2D *hImage[3];
        TGraph *gTrack[3];
        TGraph *gStartNend[3];
        TGraph *gStart[3];
        TGraph *gSum[3];
        TGraph *gFurther[3];
        double x_pixel_st, y_pixel_st,x_pixel_nd, y_pixel_nd,x_pixel, y_pixel;


        TVector3 SumPoints;
        for(size_t iNode = 0;iNode<_3DTrack.size();iNode++){
            SumPoints+=(_3DTrack[iNode]-_3DTrack[0]);
        }
        SumPoints*=1./_3DTrack.size();
        SumPoints+=_3DTrack[0];

        TVector3 FurtherFromVertex;
        double dist2vertex = 0;
        for(size_t iNode = 0;iNode<_3DTrack.size();iNode++){
            if( (_3DTrack[iNode]-start_pt).Mag() > dist2vertex){dist2vertex = (_3DTrack[iNode]-start_pt).Mag(); FurtherFromVertex = _3DTrack[iNode];}
        }

        for(size_t iPlane=0;iPlane<3;iPlane++){
            gTrack[iPlane] = new TGraph();
            gStartNend[iPlane] = new TGraph();
            gStart[iPlane] = new TGraph();
            gSum[iPlane] = new TGraph();
            gFurther[iPlane] = new TGraph();
            ProjectTo3D(hit_image_v[iPlane].meta(),
                        start_pt.X(),
                        start_pt.Y(),
                        start_pt.Z(),
                        0,
                        iPlane,
                        x_pixel_st,y_pixel_st); // y_pixel is time
            gStartNend[iPlane]->SetPoint(0,x_pixel_st*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel_st*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
            gStart[iPlane]->SetPoint(0,x_pixel_st*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel_st*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
            ProjectTo3D(hit_image_v[iPlane].meta(),end_pt.X(),end_pt.Y(),end_pt.Z(),0,iPlane,x_pixel_nd,y_pixel_nd); // y_pixel is time
            gStartNend[iPlane]->SetPoint(1,x_pixel_nd*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel_nd*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);

            for(size_t iNode = 0;iNode<_3DTrack.size();iNode++){
                ProjectTo3D(hit_image_v[iPlane].meta(),_3DTrack[iNode].X(),_3DTrack[iNode].Y(),_3DTrack[iNode].Z(),0,iPlane,x_pixel,y_pixel); // y_pixel is time

                gTrack[iPlane]->SetPoint(iNode,x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
            }
            ProjectTo3D(hit_image_v[iPlane].meta(),SumPoints.X(),SumPoints.Y(),SumPoints.Z(),0,iPlane,x_pixel,y_pixel);
            gSum[iPlane]->SetPoint(0,x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
            ProjectTo3D(hit_image_v[iPlane].meta(),FurtherFromVertex.X(),FurtherFromVertex.Y(),FurtherFromVertex.Z(),0,iPlane,x_pixel,y_pixel);
            gFurther[iPlane]->SetPoint(0,x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);

            hImage[iPlane] = new TH2D(Form("hImage_%05d_%05d_%05d_%04d_%zu",_run,_subrun,_event,_track,iPlane),
                                      ";wire;time (sample)",//Form("hImage_%05d_%05d_%05d_%04d_%zu;wire;time",_run,_subrun,_event,_track,iPlane),
                                      hit_image_v[iPlane].meta().cols(),
                                      hit_image_v[iPlane].meta().tl().x,
                                      hit_image_v[iPlane].meta().tl().x+hit_image_v[iPlane].meta().width(),
                                      hit_image_v[iPlane].meta().rows(),
                                      hit_image_v[iPlane].meta().br().y,
                                      hit_image_v[iPlane].meta().br().y+hit_image_v[iPlane].meta().height());


            for(size_t icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                for(size_t irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    hImage[iPlane]->SetBinContent(icol+1,irow+1,hit_image_v[iPlane].pixel(irow,icol));
                }
            }
        }

        TCanvas *c = new TCanvas(Form("c_%05d_%05d_%05d_%04d",_run,_subrun,_event,_track),Form("c_%05d_%05d_%05d_%04d",_run,_subrun,_event,_track),1800,1800);
        c->Divide(1,3);
        c->cd(1)->Divide(3,1);
        for(size_t iPlane=0;iPlane<3;iPlane++){
            c->cd(1)->cd(iPlane+1);
            hImage[iPlane]->Draw("colz");
            gTrack[iPlane]->SetMarkerStyle(7);
            gTrack[iPlane]->SetLineColor(2);
            gTrack[iPlane]->Draw("same LP");
            gStartNend[iPlane]->SetMarkerStyle(20);
            gStartNend[iPlane]->SetMarkerSize();
            gStartNend[iPlane]->Draw("same P");
            gStart[iPlane]->SetMarkerStyle(7);
            gStart[iPlane]->SetMarkerColor(2);
            gStart[iPlane]->Draw("same P");
            gSum[iPlane]->SetMarkerStyle(20);
            gSum[iPlane]->SetMarkerColor(6);
            gSum[iPlane]->Draw("same P");
            gFurther[iPlane]->SetMarkerStyle(20);
            gFurther[iPlane]->SetMarkerColor(4);
            gFurther[iPlane]->Draw("same P");
            //gNewTrack[iPlane]->SetMarkerColor(2);
            //if(iter == iterMax)gNewTrack[iPlane]->SetMarkerColor(6);
            //gNewTrack[iPlane]->SetMarkerStyle(7);
            //if(list3D.size()>=1)gNewTrack[iPlane]->Draw("same P");
        }
        c->cd(2)->Divide(2,1);
        TH2D *hAngleLength = new TH2D(Form("hAngleLength%05d_%05d_%05d_%04d",_run,_subrun,_event,_track),Form("hAngleLength%05d_%05d_%05d_%04d;angle;length",_run,_subrun,_event,_track),220,-1.1,1.1,200,0,20);
        for(size_t iNode = 1;iNode < _3DTrack.size()-1;iNode++){
            double ilength = (_3DTrack[iNode]-_3DTrack[iNode-1]).Mag();
            double angle = (_3DTrack[iNode]-_3DTrack[iNode-1]).Dot( (_3DTrack[iNode+1]-_3DTrack[iNode-1]) )/( (_3DTrack[iNode]-_3DTrack[iNode-1]).Mag() * (_3DTrack[iNode+1]-_3DTrack[iNode-1]).Mag() );
            if(_DrawOutputs) {
                hAngleLength->Fill( angle, ilength );
                hAngleLengthGeneral->Fill(angle,ilength);
            }
        }
        c->cd(2)->cd(1);
        hAngleLengthGeneral->Draw("colz");
        hAngleLength->SetMarkerStyle(20);
        hAngleLength->Draw("same P");

        c->cd(2)->cd(2);
        TGraph *gTrackdQdX = new TGraph();
        for(size_t iNode=0;iNode<_3DTrack.size();iNode++){
            gTrackdQdX->SetPoint(iNode,ComputeLength(iNode),track_dQdX_v[iNode]);
        }
        gTrackdQdX->SetMarkerStyle(7);
        gTrackdQdX->Draw("ALP");

        //c->SaveAs(Form("%s.pdf",c->GetName()));
        c->SaveAs(Form("track/%s.png",c->GetName()));
        //c->SaveAs(Form("%s.jpg",c->GetName()));
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            hImage[iPlane]->Delete();
            gTrack[iPlane]->Delete();
            gStartNend[iPlane]->Delete();
            gStart[iPlane]->Delete();
            gSum[iPlane]->Delete();
            gFurther[iPlane]->Delete();
            //gNewTrack[iPlane]->Delete();
        }
        tellMe("histogram and gStartNend deleted",2);
    }
    //______________________________________________________
    void AStarTracker::DrawVertex(){
        if(_DrawVertical){DrawVertexVertical();}
        else DrawVertexHorizontal();
    }
    //______________________________________________________
    void AStarTracker::DrawVertexVertical(){
        tellMe("DrawVertexVertical",0);

        TH2D *hImage[3];
        double x_pixel_st, y_pixel_st;
        hit_image_v = GetFullImage();
        int NpadsVert = 4;
        if(NpadsVert<3)NpadsVert=3;
        TCanvas *c = new TCanvas(Form("cVertex_%05d_%05d_%05d_%04d_%04d",_run,_subrun,_event,_vtxID,_track),Form("cVertex_%05d_%05d_%05d_%04d_%04d",_run,_subrun,_event,_vtxID,_track),8000,1500*NpadsVert);
        c->SetFillColor(0);
        if(_DrawBlack)c->SetFillColor(1);
        c->SetFillStyle(0);
        std::vector<TPad*> pads(NpadsVert);
        for(int i=0;i<NpadsVert;i++){c->cd();pads[i] = new TPad(Form("pads_%d",i),Form("pads_%d",i),0,(((NpadsVert-i)-1)*1.)/NpadsVert,1,((NpadsVert-i)*1.)/NpadsVert);pads[i]->Draw();}
        //c->cd();pads[1] = new TPad("pads_1","pads_1",0,(((NpadsVert-1)-1)*1.)/NpadsVert,1,((NpadsVert-1)*1.)/NpadsVert);pads[1]->Draw();
        //c->cd();pads[2] = new TPad("pads_2","pads_2",0,(((NpadsVert-2)-1)*1.)/NpadsVert,1,((NpadsVert-2)*1.)/NpadsVert);pads[2]->Draw();
        //c->cd();pads[3] = new TPad("pads_3","pads_3",0,(((NpadsVert-3)-1)*1.)/NpadsVert,1,((NpadsVert-3)*1.)/NpadsVert);pads[3]->Draw();
        for(int iPlane=0;iPlane<3;iPlane++){
            pads[iPlane]->SetTopMargin(0.02);
            pads[iPlane]->SetBottomMargin(0.1);
            pads[iPlane]->SetRightMargin(0.08);
            pads[iPlane]->SetLeftMargin(0.08);
        }

        for(size_t iPlane=0;iPlane<3;iPlane++){
            hImage[iPlane] = new TH2D(Form("hImage_%05d_%05d_%05d_%04d_%04d_%zu",_run,_subrun,_event,_vtxID,_track,iPlane),
                                      ";wire # ;time (sample)",//Form("hImage_%05d_%05d_%05d_%04d_%04d_%zu;wire;time",_run,_subrun,_event,_vtxID,_track,iPlane),
                                      hit_image_v[iPlane].meta().cols(),
                                      hit_image_v[iPlane].meta().tl().x,
                                      hit_image_v[iPlane].meta().tl().x+hit_image_v[iPlane].meta().width(),
                                      hit_image_v[iPlane].meta().rows(),
                                      hit_image_v[iPlane].meta().br().y,
                                      hit_image_v[iPlane].meta().br().y+hit_image_v[iPlane].meta().height());

            

            for(size_t icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                for(size_t irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    if(hit_image_v[iPlane].pixel(irow,icol) != _deadWireValue) hImage[iPlane]->SetBinContent(icol+1,irow+1,hit_image_v[iPlane].pixel(irow,icol));
                }
            }
        }




        hImage[0]->GetXaxis()->SetTitle("U-plane wire #");
        hImage[1]->GetXaxis()->SetTitle("V-plane wire #");
        hImage[2]->GetXaxis()->SetTitle("Y-plane wire #");
        for(size_t iPlane=0;iPlane<3;iPlane++){
            ProjectTo3D(hit_image_v[iPlane].meta(),start_pt.X(),start_pt.Y(),start_pt.Z(),0,iPlane,x_pixel_st,y_pixel_st);
            pads[iPlane]->cd();
            if(!_DrawBlack){
                pads[iPlane]->SetFrameFillColor(0);
                pads[iPlane]->SetFrameLineColor(1);
                hImage[iPlane]->GetXaxis()->SetLabelColor(1);
                hImage[iPlane]->GetXaxis()->SetAxisColor(1);
                hImage[iPlane]->GetXaxis()->SetTitleColor(1);
                hImage[iPlane]->GetYaxis()->SetLabelColor(1);
                hImage[iPlane]->GetYaxis()->SetAxisColor(1);
                hImage[iPlane]->GetYaxis()->SetTitleColor(1);
            }
            else{
                pads[iPlane]->SetFrameFillColor(1);
                pads[iPlane]->SetFrameLineColor(0);
                hImage[iPlane]->GetXaxis()->SetLabelColor(0);
                hImage[iPlane]->GetXaxis()->SetAxisColor(0);
                hImage[iPlane]->GetXaxis()->SetTitleColor(0);
                hImage[iPlane]->GetYaxis()->SetLabelColor(0);
                hImage[iPlane]->GetYaxis()->SetAxisColor(0);
                hImage[iPlane]->GetYaxis()->SetTitleColor(0);
            }
            hImage[iPlane]->GetYaxis()->SetTitleOffset(0.8);
            hImage[iPlane]->GetXaxis()->SetTitleOffset(1);
            //hImage[iPlane]->GetXaxis()->SetRangeUser(0.25*(hImage[iPlane]->GetXaxis()->GetXmax()-hImage[iPlane]->GetXaxis()->GetXmin()),0.75*(hImage[iPlane]->GetXaxis()->GetXmax()-hImage[iPlane]->GetXaxis()->GetXmin()));
            //hImage[iPlane]->GetYaxis()->SetRangeUser(0.25*(hImage[iPlane]->GetYaxis()->GetXmax()-hImage[iPlane]->GetYaxis()->GetXmin()),0.75*(hImage[iPlane]->GetYaxis()->GetXmax()-hImage[iPlane]->GetYaxis()->GetXmin()));
            hImage[iPlane]->Draw("col");
        }

        ///////////////////////////////////////////////////////////////////////////
        //Drawing a search region at the end of the tracks for the diagnotsic tool
        /*
        std::vector< std::vector<TVector3> > trackEndPoints_v;
        std::vector<TVector3> thisTrackEndPoint;
        for(size_t itrack=0;itrack<_vertexTracks.size();itrack++){
            if(thisTrackEndPoint.size()!=0)thisTrackEndPoint.clear();
            TVector3 newPoint;
            int NaveragePts = 0;
            TVector3 AveragePoint;
            TVector3 oldEndPoint = _vertexTracks[itrack].back();
            for(size_t iNode=0;iNode<_vertexTracks[itrack].size();iNode++){
                if((_vertexTracks[itrack][iNode]-oldEndPoint).Mag() < 20){NaveragePts++;AveragePoint+=(_vertexTracks[itrack][iNode]-oldEndPoint);}
            }
            if(NaveragePts!=0){AveragePoint*=1./NaveragePts;}
            AveragePoint+=oldEndPoint;
            double x,y,z;
            for(size_t i=0;i<500;i++){
                ran.Sphere(x,y,z,3);
                newPoint.SetXYZ(oldEndPoint.X()+x,oldEndPoint.Y()+y,oldEndPoint.Z()+z);
                if( (oldEndPoint-AveragePoint).Dot(newPoint-oldEndPoint)/( (oldEndPoint-AveragePoint).Mag()*(newPoint-oldEndPoint).Mag() ) < 0.8 )continue;
                thisTrackEndPoint.push_back(newPoint);
            }

            int Niter = 100;
            for(size_t i=0;i<Niter;i++){
                double r = 4+(50./Niter)*i;
                for(size_t j=0;j<Niter;j++){
                    ran.Sphere(x,y,z,r);
                    newPoint.SetXYZ(oldEndPoint.X()+x,oldEndPoint.Y()+y,oldEndPoint.Z()+z);
                    if( (oldEndPoint-AveragePoint).Dot(newPoint-oldEndPoint)/( (oldEndPoint-AveragePoint).Mag()*(newPoint-oldEndPoint).Mag() ) < 0.99 )continue;
                    thisTrackEndPoint.push_back(newPoint);
                }
            }

            trackEndPoints_v.push_back(thisTrackEndPoint);
        }
        for(size_t iPlane=0;iPlane<3;iPlane++){
            pads[iPlane]->cd();
            if(trackEndPoints_v.size()!=0){
                for(size_t itrack = 0;itrack<trackEndPoints_v.size();itrack++){
                    TGraph *gTrackEndPoint = new TGraph();
                    for(size_t iPoint = 0;iPoint<trackEndPoints_v[itrack].size();iPoint++){
                        double x_pixel, y_pixel;
                        ProjectTo3D(hit_image_v[iPlane].meta(),trackEndPoints_v[itrack][iPoint].X(),trackEndPoints_v[itrack][iPoint].Y(),trackEndPoints_v[itrack][iPoint].Z(),0,iPlane,x_pixel,y_pixel);
                        gTrackEndPoint->SetPoint(iPoint,x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
                    }
                    c->cd(2)->cd(iPlane+1);
                    gTrackEndPoint->SetMarkerStyle(7);
                    gTrackEndPoint->Draw("same P");
                }
            }
        }*/
        // end of drawing the search region
        ///////////////////////////////////////////////////////////////////////////


        ///////////////////////////////////////////////////////////////////////////
        //Drawing a search region from the clustering tool
        /*
        std::vector< std::vector<TVector3> > SearchRegion_v;
        for(size_t itrack = 0;itrack<_vertexTracks.size();itrack++){
            double x,y,z;
            std::vector<TVector3> thisList;
            if(_vertexTracks[itrack].size()>3){
                size_t myPoint = (size_t)(0.5*_vertexTracks[itrack].size());
                double rmax = 2+4*exp(-1.*((_vertexTracks[itrack][myPoint]-_vertexTracks[itrack][0]).Mag())/5.);//reduces the search range as the track size increases
                TVector3 newCandidate;
                TVector3 lastNode = _vertexTracks[itrack][myPoint];
                start_pt = _vertexTracks[itrack][0];
                thisList.push_back(lastNode);

                //for this iteration, throw points and look for possible candidates
                // find points on a small sphere around the last found point
                for(size_t i = 0;i<10000;i++){
                    //double r = ran.Uniform(0,rmax);
                    double r = 2*rmax*(1-(1/(1+ran.Uniform(0,1))));
                    ran.Sphere(x,y,z,r);
                    newCandidate.SetXYZ(lastNode.X()+x,lastNode.Y()+y,lastNode.Z()+z);
                    if(!CheckEndPointsInVolume(newCandidate)) continue; // point out of detector volume
                    thisList.push_back(newCandidate);
                }

                //if we have a nicely defined track, find point in a larger cone after the last found point
                if(_vertexTracks[itrack].size() > 2 && (lastNode-start_pt).Mag() > _MinLength){
                    TVector3 AvPt(0,0,0);
                    int NpointAveragedOn = 0;
                    //try and find the track direction on the last few cm
                    for(size_t iNode=0;iNode<_vertexTracks[itrack].size();iNode++){
                        if(iNode > myPoint)continue;
                        if( (_vertexTracks[itrack][iNode]-lastNode).Mag() < 10 ){
                            AvPt+=(_vertexTracks[itrack][iNode]-lastNode);
                            NpointAveragedOn++;
                        }
                    }
                    if(NpointAveragedOn!=0){AvPt*=1./NpointAveragedOn;}
                    AvPt+=lastNode;

                    for(size_t i = 0;i<10000;i++){
                        //double r = ran.Uniform(0,2*rmax);
                        double r = 3*(6-rmax)*(1-(1/(1+ran.Uniform(0,1))));
                        ran.Sphere(x,y,z,r);
                        newCandidate.SetXYZ(lastNode.X()+x,lastNode.Y()+y,lastNode.Z()+z);
                        if(!CheckEndPointsInVolume(newCandidate)) continue; // point out of detector volume
                        if((lastNode-AvPt).Angle(newCandidate-lastNode) > 15*TMath::Pi()/180.)continue;
                        thisList.push_back(newCandidate);
                    }
                }
            }
            SearchRegion_v.push_back(thisList);
        }


        for(size_t iPlane=0;iPlane<3;iPlane++){
            pads[iPlane]->cd();
            if(SearchRegion_v.size()!=0){
                for(size_t itrack = 0;itrack<SearchRegion_v.size();itrack++){
                    TGraph *gSearchRegion = new TGraph();
                    for(size_t iPoint = 0;iPoint<SearchRegion_v[itrack].size();iPoint++){
                        double x_pixel, y_pixel;
                        ProjectTo3D(hit_image_v[iPlane].meta(),SearchRegion_v[itrack][iPoint].X(),SearchRegion_v[itrack][iPoint].Y(),SearchRegion_v[itrack][iPoint].Z(),0,iPlane,x_pixel,y_pixel);
                        gSearchRegion->SetPoint(iPoint,x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
                    }
                    c->cd(2)->cd(iPlane+1);
                    gSearchRegion->SetMarkerStyle(7);
                    gSearchRegion->Draw("same P");
                }
            }
        }*/
        // end of drawing the search region
        ///////////////////////////////////////////////////////////////////////////

        for(size_t i=0;i<_vertexTracks.size();i++){
            TGraph *gTrack[3];
            TGraph *gSelectionZone[3];
            double x_pixel, y_pixel;
            for(size_t iPlane=0;iPlane<3;iPlane++){gTrack[iPlane] = new TGraph();}
            for(size_t iPlane=0;iPlane<3;iPlane++){
                for(size_t iNode=0;iNode<_vertexTracks[i].size();iNode++){
                    ProjectTo3D(hit_image_v[iPlane].meta(),_vertexTracks[i][iNode].X(),_vertexTracks[i][iNode].Y(),_vertexTracks[i][iNode].Z(),0,iPlane,x_pixel,y_pixel); // y_pixel is time
                    gTrack[iPlane]->SetPoint(iNode,x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
                }
                gTrack[iPlane]->SetMarkerStyle(20);
                gTrack[iPlane]->SetLineWidth(3);
                if(i==0){
                    gTrack[iPlane]->SetLineColor(1);
                    gTrack[iPlane]->SetMarkerColor(1);
                }
                else{
                    gTrack[iPlane]->SetLineColor(i+1);
                    gTrack[iPlane]->SetMarkerColor(i+1);
                }
                pads[iPlane]->cd();
                if(gTrack[iPlane]->GetN() > 0)gTrack[iPlane]->Draw("same LP");
            }
        }

        ///////////////////////////////////////////////////////////////
        // Drawing "dQdx" of tracks found
        double Lmax = 0;
        double ionMax = 0;
        int itrackLmax = 0;

        if(NpadsVert>3 && _vertexTracks.size() != 0){
            pads[NpadsVert-1]->cd();
            TGraph *gTrackdQdX[_vertexTracks.size()][3];
            TGraph *gTrackdQdXrollAverage[_vertexTracks.size()][3];
            std::cout << "Make dEdx for " << _vertexTracks.size() << " tracks" << std::endl;
            if(_vertex_dQdX_v.size()!=_vertexTracks.size()){_vertex_dQdX_v.clear();ComputeNewdQdX();}
            for(size_t itrack=0;itrack<_vertexTracks.size();itrack++){
                for(size_t iPlane = 0;iPlane<3;iPlane++){
                    gTrackdQdX[itrack][iPlane] = new TGraph();
                    gTrackdQdXrollAverage[itrack][iPlane] = new TGraph();
                }
                double ion = 0;
                double ionAvg = 0;
                for(size_t iNode = 3;iNode < _vertexTracks[itrack].size();iNode++){
                    ion  = 0;
                    ionAvg=0;
                    for(size_t iPlane = 0;iPlane<3;iPlane++){
                        ion = _vertexLarliteTracks[itrack].DQdxAtPoint(iNode,_views[iPlane]);
                        if(iNode > 2 && iNode < _vertexTracks[itrack].size() -2){
                            ionAvg = ion;
                            ionAvg += _vertexLarliteTracks[itrack].DQdxAtPoint(iNode-2,_views[iPlane]);
                            ionAvg += _vertexLarliteTracks[itrack].DQdxAtPoint(iNode-1,_views[iPlane]);
                            ionAvg += _vertexLarliteTracks[itrack].DQdxAtPoint(iNode+1,_views[iPlane]);
                            ionAvg += _vertexLarliteTracks[itrack].DQdxAtPoint(iNode+2,_views[iPlane]);

                            ionAvg/=5;
                        }
                        else if(iNode<=2){ionAvg = ion;}
                        else {ionAvg = ion;}

                        if(ionAvg > ionMax)ionMax = ionAvg;
                        if(_vertexLarliteTracks[itrack].Length(iNode) > Lmax){Lmax = _vertexLarliteTracks[itrack].Length(iNode);itrackLmax = itrack;}

                        //if(_vertexLarliteTracks[itrack].Length(iNode) - _vertexLarliteTracks[itrack].Length(0) < 3) ionAvg = 0;
                        gTrackdQdX[itrack][iPlane]->SetPoint(iNode-3,_vertexLarliteTracks[itrack].Length(iNode),ion);
                        gTrackdQdXrollAverage[itrack][iPlane]->SetPoint(iNode-3,_vertexLarliteTracks[itrack].Length(iNode),ionAvg);

                    }
                }
            }


            for(size_t itrack = 0;itrack < _vertexTracks.size();itrack++){
                for(size_t iPlane = 0;iPlane<3;iPlane++){
                    gTrackdQdX[itrack][iPlane]->SetMarkerColor(itrack+1);
                    gTrackdQdX[itrack][iPlane]->SetMarkerStyle(iPlane+20);
                    gTrackdQdX[itrack][iPlane]->SetMarkerSize(2);
                    gTrackdQdX[itrack][iPlane]->SetLineColor(itrack+1);

                    //gTrackdQdXrollAverage[itrack][iPlane]->SetMarkerColor(itrack+1);
                    //gTrackdQdXrollAverage[itrack][iPlane]->SetMarkerStyle(iPlane+20);
                    //gTrackdQdXrollAverage[itrack][iPlane]->SetMarkerSize(4);
                    gTrackdQdXrollAverage[itrack][iPlane]->SetLineColor(itrack+1);
                    gTrackdQdXrollAverage[itrack][iPlane]->SetLineWidth(2*(iPlane+1));
                    if(itrack == itrackLmax){
                        gTrackdQdXrollAverage[itrack][iPlane]->Draw("AL");
                        gTrackdQdXrollAverage[itrack][iPlane]->GetYaxis()->SetRangeUser(0,std::max(100.,1.1*ionMax));
                    }
                }
            }

            for(size_t itrack = 0;itrack < _vertexTracks.size();itrack++){
                for(size_t iPlane = 0;iPlane<3;iPlane++){
                    //gTrackdQdX[itrack][iPlane]->Draw("same LP");
                    gTrackdQdXrollAverage[itrack][iPlane]->Draw("same L");
                }
            }
        }

        std::cout << "Printing the canvas" << std::endl;

        // End of drawing "dQdx" of tracks found
        ///////////////////////////////////////////////////////////////

        std::string label_tag = "";
        //c->SaveAs(Form("%s/%s.root",_outdir.c_str(),c->GetName()));
        if(_nothingReconstructed){label_tag+="_nothing_reconstructed";}
        if(_missingTrack){label_tag+="_onlyOneTrack";}
        if(_tooShortDeadWire){label_tag+="_EndPointInDeadRegion";}
        if(_tooManyTracksAtVertex){label_tag+="_TooManyTracksAtVertex";}
        if(_branchingTracks){label_tag+="_BranchingTracks";}
        if(_tooShortFaintTrack){label_tag+="_TooShortFaintTrack";}
        if(_jumpingTracks){label_tag+="_JumpTrack";}
        if(IsGoodVertex()){label_tag+="_OKtracks";}
        if(NumberRecoveries!=0){label_tag+=Form("_recovered%d",NumberRecoveries);}
        c->SaveAs(Form("%s/%s_%s.png",_outdir.c_str(),c->GetName(),label_tag.c_str()));
        c->SaveAs(Form("%s/%s_%s.root",_outdir.c_str(),c->GetName(),label_tag.c_str()));

        for(size_t iPlane = 0;iPlane<3;iPlane++){
            hImage[iPlane]->Delete();
        }
    }
    //______________________________________________________
    void AStarTracker::DrawVertexHorizontal(){
        tellMe("DrawVertexHorizontal",0);

        TH2D *hImage[3];
        TH2D *hImageMasked[3];
        TH2D *hTagImage[3];
        TGraph *gStartNend[3];
        TGraph *gStart[3];
        TGraph *gAverage[3];
        double x_pixel_st, y_pixel_st;
        //if(_tooShortFaintTrack_v.size() != _vertexTracks.size())DiagnoseVertex();


        hit_image_v = CropFullImage2bounds(_vertexTracks);
        //EnhanceDerivative();

        TCanvas *c = new TCanvas(Form("cVertex_%05d_%05d_%05d_%04d_%04d",_run,_subrun,_event,_vtxID,_track),Form("cVertex_%05d_%05d_%05d_%04d_%04d",_run,_subrun,_event,_vtxID,_track),1800,1200);
        c->SetFillColor(0);
        c->SetFillStyle(0);
        c->Divide(1,2);
        c->cd(1)->Divide(3,1);
        c->cd(2)->Divide(3,1);

        int NtracksAtVertex = 0;
        int NpointAveragedOn = 0;
        std::vector<TVector3> AvPt_v;
        for(size_t itrack = 0;itrack<_vertexTracks.size();itrack++){
            if(_vertexTracks[itrack][0] == start_pt){
                NtracksAtVertex++;
                TVector3 AvPt;
                NpointAveragedOn = 0;
                for(size_t iNode=1;iNode<_vertexTracks[itrack].size();iNode++){
                    if( (_vertexTracks[itrack][iNode]-start_pt).Mag() < 50 ){
                        AvPt+=(_vertexTracks[itrack][iNode]-start_pt);
                        NpointAveragedOn++;
                    }
                }
                if(NpointAveragedOn!=0){AvPt*=1./NpointAveragedOn;}
                AvPt_v.push_back(AvPt+start_pt);
            }
        }
        for(size_t iPlane = 0;iPlane<3;iPlane++){gAverage[iPlane]=new TGraph();}
        for(size_t iPt = 0;iPt<AvPt_v.size();iPt++){
            double x_proj,y_proj;
            for(size_t iPlane=0;iPlane<3;iPlane++){
                ProjectTo3D(hit_image_v[iPlane].meta(),AvPt_v[iPt].X(),AvPt_v[iPt].Y(),AvPt_v[iPt].Z(),0,iPlane,x_proj,y_proj);
                gAverage[iPlane]->SetPoint(iPt,x_proj*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_proj*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
            }
        }
        for(size_t iPlane=0;iPlane<3;iPlane++){
            hImage[iPlane] = new TH2D(Form("hImage_%05d_%05d_%05d_%04d_%04d_%zu",_run,_subrun,_event,_vtxID,_track,iPlane),
                                      ";wire;time (sample)",//Form("hImage_%05d_%05d_%05d_%04d_%04d_%zu;wire;time",_run,_subrun,_event,_vtxID,_track,iPlane),
                                      hit_image_v[iPlane].meta().cols(),
                                      hit_image_v[iPlane].meta().tl().x,
                                      hit_image_v[iPlane].meta().tl().x+hit_image_v[iPlane].meta().width(),
                                      hit_image_v[iPlane].meta().rows(),
                                      hit_image_v[iPlane].meta().br().y,
                                      hit_image_v[iPlane].meta().br().y+hit_image_v[iPlane].meta().height());
            for(size_t icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                for(size_t irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    hImage[iPlane]->SetBinContent(icol+1,irow+1,hit_image_v[iPlane].pixel(irow,icol));
                }
            }
        }
        std::vector< std::vector<TVector3> > trackEndPoints_v;
        std::vector<TVector3> thisTrackEndPoint;
        MaskVertex();
        //ShaveTracks();

        for(size_t itrack=0;itrack<_vertexTracks.size();itrack++){
            if(thisTrackEndPoint.size()!=0)thisTrackEndPoint.clear();
            TVector3 newPoint;
            int NaveragePts = 0;
            TVector3 AveragePoint;
            TVector3 oldEndPoint = _vertexTracks[itrack].back();
            for(size_t iNode=0;iNode<_vertexTracks[itrack].size();iNode++){
                if((_vertexTracks[itrack][iNode]-oldEndPoint).Mag() < 20){NaveragePts++;AveragePoint+=(_vertexTracks[itrack][iNode]-oldEndPoint);}
            }
            if(NaveragePts!=0){AveragePoint*=1./NaveragePts;}
            AveragePoint+=oldEndPoint;
            double x,y,z;

            for(size_t i=0;i<500;i++){
                ran.Sphere(x,y,z,3);
                newPoint.SetXYZ(oldEndPoint.X()+x,oldEndPoint.Y()+y,oldEndPoint.Z()+z);
                if( (oldEndPoint-AveragePoint).Dot(newPoint-oldEndPoint)/( (oldEndPoint-AveragePoint).Mag()*(newPoint-oldEndPoint).Mag() ) < 0.8 )continue;
                thisTrackEndPoint.push_back(newPoint);
            }

            if(_tooShortDeadWire_v.size()!=0){
                if(_tooShortDeadWire_v[itrack]){
                    int Niter = 100;
                    for(size_t i=0;i<Niter;i++){
                        double r = 4+(50./Niter)*i;
                        for(size_t j=0;j<Niter;j++){
                            ran.Sphere(x,y,z,r);
                            newPoint.SetXYZ(oldEndPoint.X()+x,oldEndPoint.Y()+y,oldEndPoint.Z()+z);
                            if( (oldEndPoint-AveragePoint).Dot(newPoint-oldEndPoint)/( (oldEndPoint-AveragePoint).Mag()*(newPoint-oldEndPoint).Mag() ) < 0.99 )continue;
                            thisTrackEndPoint.push_back(newPoint);
                        }
                    }
                }
            }

            trackEndPoints_v.push_back(thisTrackEndPoint);

        }
        for(size_t iPlane=0;iPlane<3;iPlane++){
            gStart[iPlane] = new TGraph();
            gStartNend[iPlane] = new TGraph();
            ProjectTo3D(hit_image_v[iPlane].meta(),start_pt.X(),start_pt.Y(),start_pt.Z(),0,iPlane,x_pixel_st,y_pixel_st);
            gStart[iPlane]->SetPoint(0,x_pixel_st*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel_st*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
            for(size_t iend = 0;iend<_vertexEndPoints.size();iend++){
                ProjectTo3D(hit_image_v[iPlane].meta(),_vertexEndPoints[iend].X(),_vertexEndPoints[iend].Y(),_vertexEndPoints[iend].Z(),0,iPlane,x_pixel_st,y_pixel_st);
                gStartNend[iPlane]->SetPoint(iend,x_pixel_st*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel_st*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
            }

            hImageMasked[iPlane] = new TH2D(Form("hImageMasked_%05d_%05d_%05d_%04d_%04d_%zu",_run,_subrun,_event,_vtxID,_track,iPlane),
                                            ";wire;time (sample)",//Form("hImageMasked_%05d_%05d_%05d_%04d_%04d_%zu;wire;time",_run,_subrun,_event,_vtxID,_track,iPlane),
                                            hit_image_v[iPlane].meta().cols(),
                                            hit_image_v[iPlane].meta().tl().x,
                                            hit_image_v[iPlane].meta().tl().x+hit_image_v[iPlane].meta().width(),
                                            hit_image_v[iPlane].meta().rows(),
                                            hit_image_v[iPlane].meta().br().y,
                                            hit_image_v[iPlane].meta().br().y+hit_image_v[iPlane].meta().height());
            hTagImage[iPlane] = new TH2D(Form("hTagImage_%05d_%05d_%05d_%04d_%04d_%zu",_run,_subrun,_event,_vtxID,_track,iPlane),
                                         ";wire;time (sample)",//Form("hTagImage_%05d_%05d_%05d_%04d_%04d_%zu;wire;time",_run,_subrun,_event,_vtxID,_track,iPlane),
                                         hit_image_v[iPlane].meta().cols(),
                                         hit_image_v[iPlane].meta().tl().x,
                                         hit_image_v[iPlane].meta().tl().x+hit_image_v[iPlane].meta().width(),
                                         hit_image_v[iPlane].meta().rows(),
                                         hit_image_v[iPlane].meta().br().y,
                                         hit_image_v[iPlane].meta().br().y+hit_image_v[iPlane].meta().height());


            for(size_t icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                for(size_t irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    hImageMasked[iPlane]->SetBinContent(icol+1,irow+1,hit_image_v[iPlane].pixel(irow,icol));
                }
            }
            for(size_t icol=0;icol<CroppedTaggedPix_v[iPlane].meta().cols();icol++){
                for(size_t irow=0;irow<CroppedTaggedPix_v[iPlane].meta().rows();irow++){
                    hTagImage[iPlane]->SetBinContent(icol+1,irow+1,CroppedTaggedPix_v[iPlane].pixel(irow,icol));

                }
            }
            c->cd(1)->cd(iPlane+1);
            c->cd(1)->cd(iPlane+1)->SetFrameFillColor(0);
            c->cd(1)->cd(iPlane+1)->SetFrameLineColor(1);
            hImage[iPlane]->GetXaxis()->SetLabelColor(1);
            hImage[iPlane]->GetXaxis()->SetAxisColor(1);
            hImage[iPlane]->GetXaxis()->SetTitleColor(1);
            hImage[iPlane]->GetYaxis()->SetTitleOffset(1.50);
            hImage[iPlane]->GetYaxis()->SetLabelColor(1);
            hImage[iPlane]->GetYaxis()->SetAxisColor(1);
            hImage[iPlane]->GetYaxis()->SetTitleColor(1);
            hImage[iPlane]->Draw("colz");

            c->cd(2)->cd(iPlane+1);
            c->cd(2)->cd(iPlane+1)->SetFrameFillColor(0);
            c->cd(2)->cd(iPlane+1)->SetFrameLineColor(1);
            hImageMasked[iPlane]->GetXaxis()->SetLabelColor(1);
            //hImageMasked[iPlane]->GetXaxis()->SetLabelSize(0.05);
            hImageMasked[iPlane]->GetXaxis()->SetAxisColor(1);
            hImageMasked[iPlane]->GetXaxis()->SetTitleColor(1);
            hImageMasked[iPlane]->GetYaxis()->SetTitleOffset(1.50);
            hImageMasked[iPlane]->GetYaxis()->SetLabelColor(1);
            hImageMasked[iPlane]->GetYaxis()->SetAxisColor(1);
            hImageMasked[iPlane]->GetYaxis()->SetTitleColor(1);
            hImageMasked[iPlane]->Draw("colz");
            if(hTagImage[iPlane]->Integral()>1)hTagImage[iPlane]->Draw("same colz");

            /*if(trackEndPoints_v.size()!=0){
                for(size_t itrack = 0;itrack<trackEndPoints_v.size();itrack++){
                    TGraph *gTrackEndPoint = new TGraph();
                    for(size_t iPoint = 0;iPoint<trackEndPoints_v[itrack].size();iPoint++){
                        double x_pixel, y_pixel;
                        ProjectTo3D(hit_image_v[iPlane].meta(),trackEndPoints_v[itrack][iPoint].X(),trackEndPoints_v[itrack][iPoint].Y(),trackEndPoints_v[itrack][iPoint].Z(),0,iPlane,x_pixel,y_pixel);
                        gTrackEndPoint->SetPoint(iPoint,x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
                    }
                    c->cd(2)->cd(iPlane+1);
                    gTrackEndPoint->SetMarkerStyle(7);
                    if(_possiblyCrossing_v.at(itrack) == true) gTrackEndPoint->SetMarkerColor(2);
                    if(_tooShortFaintTrack_v.at(itrack) == true) gTrackEndPoint->SetMarkerColor(3);
                    if(_tooShortDeadWire_v.at(itrack) == true) gTrackEndPoint->SetMarkerColor(4);
                    gTrackEndPoint->Draw("same P");
                }
            }*/
        }
        for(size_t i=0;i<_vertexTracks.size();i++){
            TGraph *gTrack[3];
            double x_pixel, y_pixel;
            for(size_t iPlane=0;iPlane<3;iPlane++){gTrack[iPlane] = new TGraph();}
            for(size_t iPlane=0;iPlane<3;iPlane++){
                for(size_t iNode=0;iNode<_vertexTracks[i].size();iNode++){
                    ProjectTo3D(hit_image_v[iPlane].meta(),_vertexTracks[i][iNode].X(),_vertexTracks[i][iNode].Y(),_vertexTracks[i][iNode].Z(),0,iPlane,x_pixel,y_pixel); // y_pixel is time
                    gTrack[iPlane]->SetPoint(iNode,x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
                }
                c->cd(2)->cd(iPlane+1);
                gTrack[iPlane]->SetMarkerStyle(7);
                if(i==0){
                    gTrack[iPlane]->SetLineColor(1);
                    gTrack[iPlane]->SetMarkerColor(1);
                }
                else{
                    gTrack[iPlane]->SetLineColor(i+1);
                    gTrack[iPlane]->SetMarkerColor(i+1);
                }
                if(gTrack[iPlane]->GetN() > 0)gTrack[iPlane]->Draw("same LP");
                c->cd(1)->cd(iPlane+1);
                if(gTrack[iPlane]->GetN() > 0)gTrack[iPlane]->Draw("same LP");
            }
        }


        for(size_t iPlane=0;iPlane<3;iPlane++){
            c->cd(2)->cd(iPlane+1);
            gStartNend[iPlane]->SetMarkerColor(1);
            gStartNend[iPlane]->SetMarkerStyle(20);
            if(gStartNend[iPlane]->GetN()>0)gStartNend[iPlane]->Draw("same P");
            gStart[iPlane]->SetMarkerStyle(20);
            gStart[iPlane]->SetMarkerColor(2);
            /*if(_possibleCosmic){
                gStart[iPlane]->SetMarkerColor(3);
                gStart[iPlane]->SetMarkerSize(2);
            }
            if(!_possibleCosmic && _possiblyCrossing){
                gStart[iPlane]->SetMarkerColor(4);
                gStart[iPlane]->SetMarkerSize(3);
            }*/
            if(gStart[iPlane]->GetN()>0)gStart[iPlane]->Draw("same P");
            gAverage[iPlane]->SetMarkerColor(3);
            gAverage[iPlane]->SetMarkerStyle(20);
            if(gAverage[iPlane]->GetN()>0)gAverage[iPlane]->Draw("same P");
        }



        std::string label_tag = "";
        //c->SaveAs(Form("%s/%s.root",_outdir.c_str(),c->GetName()));
        if(_nothingReconstructed){label_tag+="_nothing_reconstructed";}
        if(_missingTrack){label_tag+="_onlyOneTrack";}
        if(_tooShortDeadWire){label_tag+="_EndPointInDeadRegion";}
        if(_tooManyTracksAtVertex){label_tag+="_TooManyTracksAtVertex";}
        if(_branchingTracks){label_tag+="_BranchingTracks";}
        if(_tooShortFaintTrack){label_tag+="_TooShortFaintTrack";}
        if(_jumpingTracks){label_tag+="_JumpTrack";}
        if(IsGoodVertex()){label_tag+="_OKtracks";}
        if(NumberRecoveries!=0){label_tag+=Form("_recovered%d",NumberRecoveries);}
        c->SaveAs(Form("%s/%s_%s.png",_outdir.c_str(),c->GetName(),label_tag.c_str()));
        c->SaveAs(Form("%s/%s_%s.root",_outdir.c_str(),c->GetName(),label_tag.c_str()));

        for(size_t iPlane = 0;iPlane<3;iPlane++){
            hImage[iPlane]->Delete();
            hTagImage[iPlane]->Delete();
            gStart[iPlane]->Delete();
            gAverage[iPlane]->Delete();
            hImageMasked[iPlane]->Delete();
            gStartNend[iPlane]->Delete();
        }
        /*std::vector<double> ionPerTrack = GetAverageIonization();
        double ionMax = 0;
        double ionMin = 1e9;
        size_t trackIonMax = 0;
        size_t trackIonMin = 0;*/

        /*for(size_t itrack = 0;itrack<ionPerTrack.size();itrack++){
            if(ionPerTrack[itrack] > ionMax){ionMax=ionPerTrack[itrack];trackIonMax=itrack;}
            if(ionPerTrack[itrack] < ionMin){ionMin=ionPerTrack[itrack];trackIonMin=itrack;}
        }
        tellMe("I'm Also here 2",0);*/
        /*std::cout << "best guess : " << std::endl;
        std::cout << "track " << trackIonMax << " is the proton" << std::endl;
        std::cout << "track " << trackIonMin << " is the muon" << std::endl;
        for(size_t itrack = 0;itrack<_vertexTracks.size();itrack++){
            if(_vertexLength[itrack]!=0){
                tellMe(Form("track %zu",itrack),0);
                tellMe(Form("if muon....: %.1f MeV",sMuonRange2T->Eval(_vertexLength[itrack])));
                tellMe(Form("if proton..: %.1f MeV",sProtonRange2T->Eval(_vertexLength[itrack])));
            }
        }*/
    }
    //______________________________________________________
    void AStarTracker::DrawVertex3D(){
        std::cout << "DrawVertex3D()" << std::endl;
        TCanvas *c = new TCanvas(Form("cVertex_%05d_%05d_%05d_%04d_%04d",_run,_subrun,_event,_vtxID,_track),Form("cVertex_%05d_%05d_%05d_%04d_%04d",_run,_subrun,_event,_vtxID,_track),3000,3000);
        gWorld->SetTitle("");
        gWorld->Draw("AP");
        gDetector->Draw("same P");

        c->SetFillColor(0);
        c->SetFillStyle(0);
        if(_DrawBlack){
            c->SetFillColor(1);
            c->SetFillStyle(1);
        }

        std::vector<TGraph2D*> gTrack3D;
        for(int i = 0;i<_vertexTracks.size();i++){
            TGraph2D *gTrack3D_oneshot = new TGraph2D();
            gTrack3D.push_back(gTrack3D_oneshot);
        }

        for(int i = 0;i<_vertexTracks.size();i++){
            for(int j=0;j<_vertexTracks[i].size();j++){
                gTrack3D[i]->SetPoint(j,_vertexTracks[i][j].Z(),_vertexTracks[i][j].X(),_vertexTracks[i][j].Y());
            }
            gTrack3D[i]->SetMarkerColor(i+1);
            if(_DrawBlack){
                gTrack3D[i]->SetMarkerColor(i+2);
                gTrack3D[i]->GetXaxis()->SetLabelColor(0);
                gTrack3D[i]->GetXaxis()->SetAxisColor(0);
                gTrack3D[i]->GetXaxis()->SetTitleColor(0);
                gTrack3D[i]->GetYaxis()->SetLabelColor(0);
                gTrack3D[i]->GetYaxis()->SetAxisColor(0);
                gTrack3D[i]->GetYaxis()->SetTitleColor(0);
                gTrack3D[i]->GetZaxis()->SetLabelColor(0);
                gTrack3D[i]->GetZaxis()->SetAxisColor(0);
                gTrack3D[i]->GetZaxis()->SetTitleColor(0);
            }
            gTrack3D[i]->SetMarkerStyle(7);
            if(i==0){gTrack3D[i]->Draw("same AP");}
            else{gTrack3D[i]->Draw("same P");}
        }

        c->SaveAs(Form("%s/%s_3D.root",_outdir.c_str(),c->GetName()));
        /*int Npng = 72;
        for(int i=0;i<Npng;i++){
            c->cd(1)->SetPhi(10+i*360./Npng);
            //c->SaveAs(Form("%s/%s_3D_%03d.png",_outdir.c_str(),c->GetName(),i));
            std::cout << i*100./Npng << std::endl;
            c->Print(Form("%s/%s_3D.gif+5",_outdir.c_str(),c->GetName()));
        }
        std::cout << Form("%s/%s_3D.gif++",_outdir.c_str(),c->GetName()) << std::endl;
        c->Print(Form("%s/%s_3D.gif++",_outdir.c_str(),c->GetName()));*/
        //c->SaveAs(Form("%s/%s.png",_outdir.c_str(),c->GetName()));
    }
    //______________________________________________________
    void AStarTracker::ReadSplineFile(){
        std::cout << "GOT: " << _spline_file << std::endl;
        assert (!_spline_file.empty());
        TFile *fSplines = TFile::Open(_spline_file.c_str(),"READ");
        sMuonRange2T   = (TSpline3*)fSplines->Get("sMuonRange2T");
        sMuonT2dEdx    = (TSpline3*)fSplines->Get("sMuonT2dEdx");
        sProtonRange2T = (TSpline3*)fSplines->Get("sProtonRange2T");
        sProtonT2dEdx  = (TSpline3*)fSplines->Get("sProtonT2dEdx");
        fSplines->Close();
    }
    //______________________________________________________
    void AStarTracker::SetSplineFile(const std::string& fpath) {
        _spline_file = fpath;
        std::cout << "set spline file=" << _spline_file << std::endl;
    }
    //______________________________________________________
    void AStarTracker::DrawROI(){
        std::cout << "DrawROI()" << std::endl;
        TH2D *hImage[3];
        TGraph *gStartNend[3];
        TGraph *gStart[3];

        for(size_t iPlane=0;iPlane<3;iPlane++){
            hImage[iPlane] = new TH2D(Form("hImage_%d_%d_%d_%d_%zu",_run,_subrun,_event,_track,iPlane),
                                      ";wire;time (sample)",//Form("hImage_%d_%d_%d_%d_%zu;wire;time",_run,_subrun,_event,_track,iPlane),
                                      hit_image_v[iPlane].meta().cols(),
                                      hit_image_v[iPlane].meta().tl().x,
                                      hit_image_v[iPlane].meta().tl().x+hit_image_v[iPlane].meta().width(),
                                      hit_image_v[iPlane].meta().rows(),
                                      hit_image_v[iPlane].meta().br().y,
                                      hit_image_v[iPlane].meta().br().y+hit_image_v[iPlane].meta().height());

            for(size_t icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                for(size_t irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    hImage[iPlane]->SetBinContent(icol+1,irow+1,hit_image_v[iPlane].pixel(irow,icol));
                }
            }

            gStartNend[iPlane] = new TGraph();
            gStart[iPlane] = new TGraph();
            double x_pixel_st, y_pixel_st,x_pixel_end, y_pixel_end;
            double Xstart,Ystart,Xend,Yend;

            ProjectTo3D(hit_image_v[iPlane].meta(),
                        start_pt.X(),start_pt.Y(),start_pt.Z(),
                        0,iPlane,x_pixel_st,y_pixel_st);

            Xstart = x_pixel_st*hit_image_v[iPlane].meta().pixel_width() +hit_image_v[iPlane].meta().tl().x;
            Ystart = y_pixel_st*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y;
            tellMe(Form("[DrawROI] StartPt plane %zu : (%.1f, %.1f)",iPlane, Xstart, Ystart),1);
            gStartNend[iPlane]->SetPoint(0,Xstart,Ystart);
            gStart[iPlane]->SetPoint(0,Xstart,Ystart);

            ProjectTo3D(hit_image_v[iPlane].meta(),
                        end_pt.X(),end_pt.Y(),end_pt.Z(),
                        0,iPlane,x_pixel_end,y_pixel_end);

            Xend = x_pixel_end*hit_image_v[iPlane].meta().pixel_width() +hit_image_v[iPlane].meta().tl().x;
            Yend = y_pixel_end*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y;
            gStartNend[iPlane]->SetPoint(1,Xend,Yend);


        }

        TCanvas *c = new TCanvas(Form("ROI_%d_%d_%d_%d",_run,_subrun,_event,_track),Form("ROI_%d_%d_%d_%d",_run,_subrun,_event,_track),450,150);
        c->Divide(3,1);
        for(size_t iPlane=0;iPlane<3;iPlane++){
            c->cd(iPlane+1);
            hImage[iPlane]->Draw("colz");
            gStart[iPlane]->SetMarkerStyle(7);
            gStartNend[iPlane]->SetMarkerStyle(4);
            gStartNend[iPlane]->SetMarkerSize(0.25);
            gStartNend[iPlane]->Draw("same P");
            gStart[iPlane]->Draw("same P");
        }
        c->SaveAs(Form("%s.pdf",c->GetName()));

        for(size_t iPlane = 0;iPlane<3;iPlane++){
            hImage[iPlane]->Delete();
            gStartNend[iPlane]->Delete();
            gStart[iPlane]->Delete();
        }
        tellMe("histogram and gStartNend deleted",1);

    }
    //______________________________________________________
    void AStarTracker::RegularizeTrack(){
        tellMe("RegularizeTrack()",1);
        std::vector<TVector3> newTrack;
        TVector3 newPoint;

        if(_3DTrack.size() > 2){
            for(size_t iPoint = 0;iPoint<_3DTrack.size()-1;iPoint++){
                if((_3DTrack[iPoint+1]-_3DTrack[iPoint]).Mag()>5)continue;
                newPoint = (_3DTrack[iPoint+1]+_3DTrack[iPoint])*0.5;
                newTrack.push_back(newPoint);
            }
            _3DTrack = newTrack;
        }


        if(newTrack.size() != 0)newTrack.clear();
        double dist;
        double distMin;
        // select closest point as next one
        bool pointAlreadyChosen = false;
        newTrack.push_back(_3DTrack[0]);
        while(_3DTrack.size() != newTrack.size()){
            distMin=1e9;
            for(size_t iNode=0;iNode<_3DTrack.size();iNode++){
                pointAlreadyChosen = false;
                for(size_t jNode = 0;jNode<newTrack.size();jNode++){
                    if(_3DTrack[iNode] == newTrack[jNode])pointAlreadyChosen = true;
                }
                if(pointAlreadyChosen)continue;
                dist = (_3DTrack[iNode]-newTrack.back()).Mag();
                if(dist < distMin){distMin = dist;newPoint = _3DTrack[iNode];}
            }
            if(distMin > 20) break;

            if (_DrawOutputs) {
                hDist2point->Fill(distMin);
            }
            newTrack.push_back(newPoint);
        }
        _3DTrack=newTrack;
        // make sure I don't have a point that "goes back" w.r.t. the previously selected point and the next one in the _3Dtrack
        newTrack.clear();
        newTrack.push_back(_3DTrack[0]);
        for(size_t iNode = 1;iNode < _3DTrack.size()-1;iNode++){
            double angle = (_3DTrack[iNode]-newTrack.back()).Dot( (_3DTrack[iNode+1]-newTrack.back()) )/( (_3DTrack[iNode]-newTrack.back()).Mag() * (_3DTrack[iNode+1]-newTrack.back()).Mag() );
            if(angle > -0.5) newTrack.push_back(_3DTrack[iNode]);
        }
        _3DTrack = newTrack;

        // remove strange things arount the vertex point
        if((start_pt-_3DTrack[0]).Mag() >= (start_pt-_3DTrack[1]).Mag())_3DTrack[0] = _3DTrack[1];
        if((start_pt-_3DTrack[_3DTrack.size()-1]).Mag() >= (start_pt-_3DTrack[_3DTrack.size()-2]).Mag())_3DTrack[_3DTrack.size()-1] = _3DTrack[_3DTrack.size()-2];
        // remove "useless" points that are parts of straight portions and too close to previous points
        std::vector<TVector3> newPoints;
        int oldNumPoints = _3DTrack.size();
        bool removedPoint = true;
        int iteration = 0;
        double distMax = 0.5;
        double distLastPoint = 1;
        while(removedPoint == true && iteration<100){
            removedPoint =false;
            iteration++;
            newPoints.clear();
            newPoints.push_back(_3DTrack[0]);
            for(size_t iNode = 1;iNode<_3DTrack.size()-1;iNode++){
                double dist2line = GetDist2line(newPoints[newPoints.size()-1],_3DTrack[iNode+1],_3DTrack[iNode]);
                double dist2point = (_3DTrack[iNode]-newPoints[newPoints.size()-1]).Mag();
                if(dist2line < distMax && dist2point < distLastPoint){
                    removedPoint = true;
                    continue;
                }
                else newPoints.push_back(_3DTrack[iNode]);
            }
            newPoints.push_back(_3DTrack[_3DTrack.size()-1]);
            _3DTrack = newPoints;
            if(_3DTrack[0]!=start_pt){
                std::vector<TVector3>::iterator it;
                it = _3DTrack.begin();
                _3DTrack.insert (it,start_pt);
            }
            //if(_3DTrack.back() != end_pt){_3DTrack.push_back(end_pt);}
        }
        // add points every cm between start_pt and _3DTrack[1];

        //DumpTrack();
        /*TVector3 vertexNeighbour = start_pt;
        int iterNeighbour = 0;
        std::vector<TVector3> vertexNeighbour_v;
        while((vertexNeighbour-start_pt).Mag() < (_3DTrack[1]-start_pt).Mag() && iterNeighbour<10){
            iterNeighbour++;
            vertexNeighbour = start_pt+(_3DTrack[1]-start_pt)*(iterNeighbour/(_3DTrack[1]-start_pt).Mag());
            vertexNeighbour_v.push_back(vertexNeighbour);
        }
        if(vertexNeighbour_v.size()!=0){
            std::vector<TVector3>::iterator it;
            it = _3DTrack.begin();
            _3DTrack.insert(it+1,vertexNeighbour_v.begin(),vertexNeighbour_v.end());
        }*/
        //DumpTrack();

        // add points between points
        newTrack.clear();
        //std::cout << "_3DTrack.size() before adding points : " << _3DTrack.size() << std::endl;
        if(_3DTrack.size() > 1){
            double dR = 0.5;
            newTrack.push_back(_3DTrack[0]);

            for(size_t i=0;i<_3DTrack.size()-1;i++){
                int iter = 0;
                newPoint = _3DTrack[i];
                while((newPoint-_3DTrack[i]).Mag() < (_3DTrack[i+1]-_3DTrack[i]).Mag()){
                    iter++;
                    TVector3 DirVec = (_3DTrack[i+1]-_3DTrack[i]);
                    DirVec.SetMag(1);
                    newPoint = _3DTrack[i]+iter*dR*DirVec ;
                    if((newPoint-_3DTrack[i]).Mag() < (_3DTrack[i+1]-_3DTrack[i]).Mag()
                       && (newPoint-_3DTrack[i+1]).Mag() > dR
                       ){
                        newTrack.push_back(newPoint);
                    }
                }
                newTrack.push_back(_3DTrack[i+1]);
            }
            //_3DTrack = newTrack;
            _3DTrack.clear();
            for(size_t j=0;j<newTrack.size();j++){_3DTrack.push_back(newTrack[j]);}
        }
        //std::cout << "_3DTrack.size() after adding points : " << _3DTrack.size() << std::endl;

    }
    //______________________________________________________
    void AStarTracker::ComputeLength(){
        //tellMe("ComputeLength()",0);
        _Length3D = 0;
        if(_3DTrack.size()>2){
            for(size_t iNode = 0;iNode<_3DTrack.size()-1;iNode++){
                _Length3D+=(_3DTrack[iNode+1]-_3DTrack[iNode]).Mag();
            }
        }
        tellMe(Form("ComputeLength() : %.1f cm",_Length3D),0);

    }
    //______________________________________________________
    double AStarTracker::ComputeLength(int node){
        tellMe("ComputeLength()",1);
        double Length3D = 0;
        if(_3DTrack.size()>2){
            for(size_t iNode = node;iNode<_3DTrack.size()-1;iNode++){
                Length3D+=(_3DTrack[iNode+1]-_3DTrack[iNode]).Mag();
            }
        }
        return Length3D;
    }
    //______________________________________________________
    void AStarTracker::WorldInitialization(){
        gWorld = new TGraph2D();
        gWorld->SetNameTitle("gWorld","gWorld");

        gDetector = new TGraph2D();
        gDetector->SetNameTitle("gDetector","gDetector");

        double detectorLength = 1036.8;//cm
        double detectorheight = 233;//cm
        double detectorwidth  = 256.35;//cm
        double Ymindet = -0.5*detectorheight;
        double Ymaxdet = 0.5*detectorheight;
        double Xmindet = 0;
        double Xmaxdet = detectorwidth;
        double Zmindet = 0;
        double Zmaxdet = detectorLength;
        int imax;
        for(size_t i = 0;i<1000;i++){
            gDetector->SetPoint(4*i+0,Zmindet+i*(detectorLength/(1000-1)), Xmindet, Ymindet);
            gDetector->SetPoint(4*i+1,Zmindet+i*(detectorLength/(1000-1)), Xmindet, Ymaxdet);
            gDetector->SetPoint(4*i+2,Zmindet+i*(detectorLength/(1000-1)), Xmaxdet, Ymindet);
            gDetector->SetPoint(4*i+3,Zmindet+i*(detectorLength/(1000-1)), Xmaxdet, Ymaxdet);

            imax = 4*i+3;
        }
        int newimax;
        for(size_t i = 0;i<100;i++){
            int ipoint = imax+1+i;
            gDetector->SetPoint(4*ipoint+0,Zmindet, Xmindet+i*(detectorwidth/(100-1)), Ymindet);
            gDetector->SetPoint(4*ipoint+1,Zmindet, Xmindet+i*(detectorwidth/(100-1)), Ymaxdet);
            gDetector->SetPoint(4*ipoint+2,Zmaxdet, Xmindet+i*(detectorwidth/(100-1)), Ymindet);
            gDetector->SetPoint(4*ipoint+3,Zmaxdet, Xmindet+i*(detectorwidth/(100-1)), Ymaxdet);

            newimax=ipoint;
        }
        imax = newimax;
        for(size_t i = 0;i<100;i++){
            int ipoint = imax+1+i;
            gDetector->SetPoint(4*ipoint+0,Zmindet, Xmindet, Ymindet+i*(detectorheight/(100-1)));
            gDetector->SetPoint(4*ipoint+1,Zmindet, Xmaxdet, Ymindet+i*(detectorheight/(100-1)));
            gDetector->SetPoint(4*ipoint+2,Zmaxdet, Xmindet, Ymindet+i*(detectorheight/(100-1)));
            gDetector->SetPoint(4*ipoint+3,Zmaxdet, Xmaxdet, Ymindet+i*(detectorheight/(100-1)));
        }

        gWorld->SetPoint(0,-0.1*detectorLength,-0.6*detectorLength+0.5*detectorwidth,-0.6*detectorLength);
        gWorld->SetPoint(1,-0.1*detectorLength,-0.6*detectorLength+0.5*detectorwidth, 0.6*detectorLength);

        gWorld->SetPoint(2,-0.1*detectorLength, 0.6*detectorLength+0.5*detectorwidth,-0.6*detectorLength);
        gWorld->SetPoint(3,-0.1*detectorLength, 0.6*detectorLength+0.5*detectorwidth, 0.6*detectorLength);

        gWorld->SetPoint(4, 1.1*detectorLength,-0.6*detectorLength+0.5*detectorwidth,-0.6*detectorLength);
        gWorld->SetPoint(5, 1.1*detectorLength,-0.6*detectorLength+0.5*detectorwidth, 0.6*detectorLength);

        gWorld->SetPoint(6, 1.1*detectorLength, 0.6*detectorLength+0.5*detectorwidth,-0.6*detectorLength);
        gWorld->SetPoint(7, 1.1*detectorLength, 0.6*detectorLength+0.5*detectorwidth, 0.6*detectorLength);
    }
    //______________________________________________________
    void AStarTracker::ShaveTracks(){
        
        tellMe("ShaveTracks()",0);

	std::clock_t begin = std::clock();

        bool erasedPixel=true;
        while(erasedPixel == true){
            erasedPixel=false;
            for(size_t iPlane=0;iPlane<3;iPlane++){
                for(size_t icol = 3;icol<hit_image_v[iPlane].meta().cols()-3;icol++){
                    for(size_t irow = 3;irow<hit_image_v[iPlane].meta().rows()-3;irow++){
                        if(hit_image_v[iPlane].pixel(irow,icol) == 0 || hit_image_v[iPlane].pixel(irow,icol) == _deadWireValue || hit_image_v[iPlane].pixel(irow,icol) == _deadWireValue+0.01)continue;

                        if(iPlane!=2 && hit_image_v[iPlane].pixel(irow  ,icol  ) != _deadWireValue

                           && hit_image_v[iPlane].pixel(irow  ,icol-1) == 0
                           && hit_image_v[iPlane].pixel(irow  ,icol+1) == 0
                           && hit_image_v[iPlane].pixel(irow+1,icol  ) == 0
                           && hit_image_v[iPlane].pixel(irow+1,icol+1) == 0
                           && hit_image_v[iPlane].pixel(irow+1,icol-1) == 0
                           && hit_image_v[iPlane].pixel(irow-1,icol  ) != 0
                           && hit_image_v[iPlane].pixel(irow-2,icol  ) != 0){
                            hit_image_v[iPlane].set_pixel(irow,icol,0);erasedPixel = true;
                        }
                        /* if(hit_image_v[iPlane].pixel(irow  ,icol  ) != _deadWireValue
                         && hit_image_v[iPlane].pixel(irow+2,icol+1) == 0
                         && hit_image_v[iPlane].pixel(irow+1,icol+1) == 0
                         && hit_image_v[iPlane].pixel(irow  ,icol+1) == 0
                         && hit_image_v[iPlane].pixel(irow-1,icol+1) == 0
                         && hit_image_v[iPlane].pixel(irow-2,icol+1) == 0
                         ){
                         hit_image_v[iPlane].set_pixel(irow,icol,0);erasedPixel = true;
                         }*/
                        /*if(hit_image_v[iPlane].pixel(irow  ,icol  ) != _deadWireValue
                         && hit_image_v[iPlane].pixel(irow+2,icol-1) == 0
                         && hit_image_v[iPlane].pixel(irow+1,icol-1) == 0
                         && hit_image_v[iPlane].pixel(irow  ,icol-1) == 0
                         && hit_image_v[iPlane].pixel(irow-1,icol-1) == 0
                         && hit_image_v[iPlane].pixel(irow-2,icol-1) == 0
                         ){
                         hit_image_v[iPlane].set_pixel(irow,icol,0);erasedPixel = true;
                         }*/
                        if(iPlane!=2 && hit_image_v[iPlane].pixel(irow  ,icol  ) != _deadWireValue
                           && hit_image_v[iPlane].pixel(irow+1,icol-1) == 0
                           && hit_image_v[iPlane].pixel(irow+1,icol+1) == 0
                           && hit_image_v[iPlane].pixel(irow  ,icol-1) == 0
                           && hit_image_v[iPlane].pixel(irow  ,icol+1) == 0
                           && hit_image_v[iPlane].pixel(irow-1,icol-1) == 0
                           && hit_image_v[iPlane].pixel(irow-1,icol+1) == 0
                           ){
                            hit_image_v[iPlane].set_pixel(irow,icol,0);erasedPixel = true;
                        }
                        if(hit_image_v[iPlane].pixel(irow  ,icol  ) != _deadWireValue
                           && hit_image_v[iPlane].pixel(irow+1,icol-1) == 0
                           && hit_image_v[iPlane].pixel(irow+1,icol  ) == 0
                           && hit_image_v[iPlane].pixel(irow+1,icol+1) == 0

                           && hit_image_v[iPlane].pixel(irow  ,icol-1) == 0
                           && hit_image_v[iPlane].pixel(irow  ,icol+1) == 0

                           && hit_image_v[iPlane].pixel(irow-1,icol-1) == 0
                           && hit_image_v[iPlane].pixel(irow-1,icol  ) == 0
                           && hit_image_v[iPlane].pixel(irow-1,icol+1) == 0

                           ){
                            hit_image_v[iPlane].set_pixel(irow,icol,0);erasedPixel = true;
                        }

                        if(hit_image_v[iPlane].pixel(irow  ,icol  ) != _deadWireValue
                           && hit_image_v[iPlane].pixel(irow+2,icol-2) == 0
                           && hit_image_v[iPlane].pixel(irow+2,icol-1) == 0
                           && hit_image_v[iPlane].pixel(irow+2,icol-0) == 0
                           && hit_image_v[iPlane].pixel(irow+2,icol+1) == 0
                           && hit_image_v[iPlane].pixel(irow+2,icol+2) == 0

                           && hit_image_v[iPlane].pixel(irow+1,icol-2) == 0
                           && hit_image_v[iPlane].pixel(irow+1,icol+2) == 0

                           && hit_image_v[iPlane].pixel(irow  ,icol-2) == 0
                           && hit_image_v[iPlane].pixel(irow  ,icol+2) == 0

                           && hit_image_v[iPlane].pixel(irow-1,icol-2) == 0
                           && hit_image_v[iPlane].pixel(irow-1,icol+2) == 0

                           && hit_image_v[iPlane].pixel(irow-2,icol-2) == 0
                           && hit_image_v[iPlane].pixel(irow-2,icol-1) == 0
                           && hit_image_v[iPlane].pixel(irow-2,icol-0) == 0
                           && hit_image_v[iPlane].pixel(irow-2,icol+1) == 0
                           && hit_image_v[iPlane].pixel(irow-2,icol+2) == 0

                           ){
                            hit_image_v[iPlane].set_pixel(irow,icol,0);erasedPixel = true;
                        }

                        if(   hit_image_v[iPlane].pixel(irow  ,icol  ) != _deadWireValue

                           && hit_image_v[iPlane].pixel(irow-3  ,icol-3) == 0
                           && hit_image_v[iPlane].pixel(irow-2  ,icol-3) == 0
                           && hit_image_v[iPlane].pixel(irow-1  ,icol-3) == 0
                           && hit_image_v[iPlane].pixel(irow    ,icol-3) == 0
                           && hit_image_v[iPlane].pixel(irow+1  ,icol-3) == 0
                           && hit_image_v[iPlane].pixel(irow+2  ,icol-3) == 0

                           && hit_image_v[iPlane].pixel(irow-3  ,icol-2) == 0
                           && hit_image_v[iPlane].pixel(irow+3  ,icol-2) == 0

                           && hit_image_v[iPlane].pixel(irow-3  ,icol-1) == 0
                           && hit_image_v[iPlane].pixel(irow+3  ,icol-1) == 0

                           && hit_image_v[iPlane].pixel(irow-3  ,icol) == 0
                           && hit_image_v[iPlane].pixel(irow+3  ,icol) == 0

                           && hit_image_v[iPlane].pixel(irow-3  ,icol+1) == 0
                           && hit_image_v[iPlane].pixel(irow+3  ,icol+1) == 0

                           && hit_image_v[iPlane].pixel(irow-3  ,icol+2) == 0
                           && hit_image_v[iPlane].pixel(irow+3  ,icol+2) == 0

                           && hit_image_v[iPlane].pixel(irow-3  ,icol+3) == 0
                           && hit_image_v[iPlane].pixel(irow-2  ,icol+3) == 0
                           && hit_image_v[iPlane].pixel(irow-1  ,icol+3) == 0
                           && hit_image_v[iPlane].pixel(irow    ,icol+3) == 0
                           && hit_image_v[iPlane].pixel(irow+1  ,icol+3) == 0
                           && hit_image_v[iPlane].pixel(irow+2  ,icol+3) == 0
                           && hit_image_v[iPlane].pixel(irow+3  ,icol+3) == 0
                           ){
                            hit_image_v[iPlane].set_pixel(irow,icol,0);erasedPixel = true;
                        }

                    }
                }
            }
        }

	clock_t end = clock();
	char msg[50];
	sprintf(msg,"ShaveTracks() elapsed=%.3f secs",float(end - begin) / CLOCKS_PER_SEC);
	tellMe(msg,0);
    }
    //______________________________________________________
    bool AStarTracker::CheckEndPointsInVolume(TVector3 point){
        bool endpointInRange = true;
        if(point.X() <  0    ){endpointInRange = false;}
        if(point.X() >  270  ){endpointInRange = false;}

        if(point.Y() < -120  ){endpointInRange = false;}
        if(point.Y() >  120  ){endpointInRange = false;}

        if(point.Z() <  1    ){endpointInRange = false;}
        if(point.Z() >  1036 ){endpointInRange = false;}

        return endpointInRange;
    }
    //______________________________________________________
    bool AStarTracker::initialize() {
        tellMe("initialize()",0);
        ReadSplineFile();
        _eventTreated = 0;
        _eventSuccess = 0;
        _DrawVertical = false;
        _DrawBlack = false;
        _deadWireValue = _ADCthreshold+1;
        time_bounds.reserve(3);
        wire_bounds.reserve(3);
        hit_image_v.reserve(3);
        randomSeed = 1;
        ran.SetSeed(randomSeed);
        _MinLength = 3; // cm
        if(_DrawOutputs){
            hAngleLengthGeneral = new TH2D("hAngleLengthGeneral","hAngleLengthGeneral;angle;length",220,-1.1,1.1,200,0,20);
            hDist2point   = new TH1D("hDist2point",  "hDist2point",  300,0,100);
            hDistance2Hit = new TH1D("hDistance2Hit","hDistance2Hit",100,0,50);
            hdQdx   = new TH1D("hdQdx","hdQdx",500,0,100);
            hLength = new TH1D("hLength","hLength",100,0,1000);
            hLengthdQdX = new TH2D("hLengthdQdX","hLengthdQdX;L;dQdx",100,0,1000,500,0,500);
            hdQdxEntries = new TH1D("hdQdxEntries","hdQdxEntries",hdQdx->GetNbinsX(),hdQdx->GetXaxis()->GetXmin(),hdQdx->GetXaxis()->GetXmax());
            hdQdX2D = new TH2D("hdQdX2D","hdQdX2D",200,0,100,300,0,300);
        }
        WorldInitialization();
        _views[0] = larlite::geo::kU;
        _views[1] = larlite::geo::kV;
        _views[2] = larlite::geo::kZ;
        return true;
    }
    //______________________________________________________
    bool AStarTracker::finalize() {
        tellMe("Finalizing",0);
        return true;
    }
    //______________________________________________________
    bool AStarTracker::ArePointsEqual(TVector3 A, TVector3 B){
        if((A-B).Mag() < 2)return true;
        else return false;
    }
    //______________________________________________________
    bool AStarTracker::CheckEndPoints(std::vector< std::pair<int,int> > endPix){
        tellMe(Form("start point : (%f,%f,%f)",start_pt.X(),start_pt.Y(),start_pt.Z()),2);
        tellMe(Form("end   point : (%f,%f,%f)",end_pt.X(),end_pt.Y(),end_pt.Z()),2);
        TVector3 EndPoint[2] = {start_pt,end_pt};
        double dR = 0.1;
        double closeEnough = 4;

        for(size_t point = 0;point<2;point++){
            double minDist = EvalMinDist(EndPoint[point],endPix);
            double PreviousMinDist = 2*minDist;
            tellMe(Form("dist = %f.1", minDist),0);
            if(minDist <= closeEnough){tellMe("Point OK",0);continue;}
            tellMe("Updating point",0);
            TVector3 newPoint = EndPoint[point];
            int iter = 0;
            int iterMax = 10000;
            while (minDist > closeEnough && iter < iterMax) {
                iter++;
                std::vector<TVector3> openSet = GetOpenSet(newPoint,dR);
                double minDist = 1e9;
                for(auto neighbour : openSet){
                    double dist = EvalMinDist(neighbour,endPix);
                    if(dist < minDist){minDist=dist;newPoint = neighbour;}
                }
                tellMe(Form("iteration #%d",iter),1);
                if(minDist < closeEnough || iter >= iterMax || minDist >= PreviousMinDist) break;
                PreviousMinDist = minDist;
            }
            EndPoint[point] = newPoint;
            tellMe("Point updated",0);
        }
        return true;
    }
    //______________________________________________________
    bool AStarTracker::IsInSegment(TVector3 A, TVector3 B, TVector3 C){
        if((C-A).Dot(B-A)/(B-A).Mag() > 0 && (C-A).Dot(B-A)/(B-A).Mag() < 1) return true;
        else return false;
    }
    //______________________________________________________
    bool AStarTracker::IsTrackIn(std::vector<TVector3> trackA, std::vector<TVector3> trackB){
        // is track A a sub-sample of track B?
        double dist = 0;
        for(size_t iNode = 0;iNode<trackA.size();iNode++){
            dist += GetDist2track(trackA[iNode],trackB);
        }
        if(trackA.size()>1)dist/=trackA.size();
        std::cout << "average distance " << dist << " cm" << std::endl;
        if(dist > 2)return false;
        else return true;
    }
    //______________________________________________________
    double AStarTracker::EvalMinDist(TVector3 point){
        tellMe(Form("EvalMinDist : (%f,%f,%f)",point.X(),point.Y(),point.Z()),2);
        double minDist = 1e9;
        //double minDistplane[3] = {1e9,1e9,1e9};
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            tellMe(Form("plane %zu :",iPlane),2);
            int pointCol = hit_image_v[iPlane].meta().col(larutil::GeometryHelper::GetME()->Point_3Dto2D(point,iPlane).w/0.3);
            int pointRow = hit_image_v[iPlane].meta().row(X2Tick(point.X(),iPlane));
            tellMe(Form("\t (%d,%d)",pointRow,pointCol),2);
            double dist = 1e9;
            bool notEmpty = false;
            for(size_t icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                for(size_t irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    if(hit_image_v[iPlane].pixel(irow,icol) == 0)continue;
                    if(std::abs((int)(icol-pointCol)) > 20 || std::abs((int)(irow-pointRow)) > 20) continue;
                    dist = sqrt(pow(pointCol-icol,2)+pow(pointRow-irow,2));
                    notEmpty = true;
                    //if(dist < minDistplane[iPlane])minDistplane[iPlane]=dist;
                    if(dist < minDist)minDist=dist;
                }
            }
            //if(!notEmpty) minDistplane[iPlane] = 1;
        }
        //if(minDistplane[0] == 0 || minDistplane[1] == 0 || minDistplane[2] == 0) minDist = 0;
        //else minDist = minDistplane[0]+minDistplane[1]+minDistplane[2];
        return minDist;
    }
    //______________________________________________________
    double AStarTracker::EvalMinDist(TVector3 point, std::vector< std::pair<int,int> > endPix){
        tellMe(Form("EvalMinDist : (%f,%f,%f)",point.X(),point.Y(),point.Z()),2);
        double minDist = 1e9;
        //double minDistplane[3] = {1e9,1e9,1e9};
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            tellMe(Form("plane %zu :",iPlane),2);
            int pointCol = hit_image_v[iPlane].meta().col(larutil::GeometryHelper::GetME()->Point_3Dto2D(point,iPlane).w/0.3);
            int pointRow = hit_image_v[iPlane].meta().row(X2Tick(point.X(),iPlane));
            tellMe(Form("\t (%d,%d)",pointRow,pointCol),2);
            double dist = 1e9;
            bool notEmpty = false;
            for(size_t icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                for(size_t irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    if(icol != endPix[iPlane].first && irow != endPix[iPlane].second)continue;
                    if(hit_image_v[iPlane].pixel(irow,icol) == 0)continue;
                    if(std::abs((int)(icol-pointCol)) > 10 || std::abs((int)(irow-pointRow)) > 10) continue;
                    dist = sqrt(pow(pointCol-icol,2)+pow(pointRow-irow,2));
                    notEmpty = true;
                    //if(dist < minDistplane[iPlane])minDistplane[iPlane]=dist;
                    if(dist < minDist)minDist=dist;
                }
            }
            //if(!notEmpty) minDistplane[iPlane] = 1;
        }
        //if(minDistplane[0] == 0 || minDistplane[1] == 0 || minDistplane[2] == 0) minDist = 0;
        //else minDist = minDistplane[0]+minDistplane[1]+minDistplane[2];
        return minDist;
    }
    //______________________________________________________
    double AStarTracker::X2Tick(double x, size_t plane) const {

        //auto ts = larutil::TimeService::GetME();
        auto larp = larutil::LArProperties::GetME();
        // (X [cm] / Drift Velocity [cm/us] - TPC waveform tick 0 offset) ... [us]
        //double tick = (x / larp->DriftVelocity() - ts->TriggerOffsetTPC() - _speedOffset);
        double tick = (x/ larp->DriftVelocity())*2+3200;
        // 1st plane's tick
        /*if(plane==0) return tick * 2;// 2 ticks/us
         // 2nd plane needs more drift time for plane0=>plane1 gap (0.3cm) difference
         tick -= 0.3 / larp->DriftVelocity(larp->Efield(1));
         // 2nd plane's tick
         if(plane==1) return tick * 2;
         // 3rd plane needs more drift time for plane1=>plane2 gap (0.3cm) difference
         tick -= 0.3 / larp->DriftVelocity(larp->Efield(2));*/
        return tick;// * 2;
    }
    //______________________________________________________
    double AStarTracker::Tick2X(double tick, size_t plane) const{
        //auto ts = larutil::TimeService::GetME();
        auto larp = larutil::LArProperties::GetME();
        // remove tick offset due to plane
        /*if(plane >= 1) {tick += 0.3/larp->DriftVelocity(larp->Efield(1));}
         if(plane >= 2) {
         tick += 0.3/larp->DriftVelocity(larp->Efield(2));
         }*/
        //tick/=2.;
        //double x = (tick+ts->TriggerOffsetTPC()+_speedOffset)*larp->DriftVelocity();

        return ((tick-3200)*larp->DriftVelocity())*0.5;
    }
    //______________________________________________________
    double AStarTracker::GetEnergy(std::string partType, double Length = -1){
        if(Length == -1)ComputeLength();
        if(partType == "proton"){
            if(Length == -1){return sProtonRange2T->Eval(_Length3D);}
            else return sProtonRange2T->Eval(Length);
        }
        else if(partType == "muon"){
            if(Length == -1){return sMuonRange2T->Eval(_Length3D);}
            else return sMuonRange2T->Eval(Length);
        }
        else{tellMe("ERROR, must be proton or muon for now",0); return -1;}
    }
    //______________________________________________________
    std::vector< std::vector<double> > AStarTracker::GetEnergies(){
        tellMe("GetEnergies()",0);
        std::vector< std::vector<double> > Energies_v;
        for(size_t itrack = 0;itrack<_vertexTracks.size();itrack++){
            std::vector<double> singleTrackEnergy(2);
            singleTrackEnergy[0] = GetEnergy("proton",_vertexLength[itrack]);
            singleTrackEnergy[1] = GetEnergy("muon",_vertexLength[itrack]);
            Energies_v.push_back(singleTrackEnergy);
        }
        return Energies_v;
    }
    //______________________________________________________
    std::vector<double> AStarTracker::GetAverageIonization(double distAvg){
        tellMe("GetAverageIonization()",0);
        if(_vertex_dQdX_v.size()!=_vertexTracks.size()){_vertex_dQdX_v.clear();ComputeNewdQdX();}
        std::vector<double> averageIonizatioPerTrack(_vertexTracks.size(),0);
        for(size_t itrack = 0;itrack < _vertexLarliteTracks.size();itrack++){
            double ionTotal = 0;
            double ion = 0;
            int NnodesTot = 0;
            averageIonizatioPerTrack[itrack] = ionTotal;
            for(size_t iNode = 0;iNode < _vertexLarliteTracks[itrack].NumberTrajectoryPoints();iNode++){
                if(distAvg!= -1 && _vertexLarliteTracks[itrack].Length(0)-_vertexLarliteTracks[itrack].Length(iNode) > distAvg)continue;
                int NplanesOK = 0;
                ion  = 0;
                for(size_t iPlane = 0;iPlane<3;iPlane++){
                    ion += _vertexLarliteTracks[itrack].DQdxAtPoint(iNode,_views[iPlane]);
                    if(_vertexLarliteTracks[itrack].DQdxAtPoint(iNode,_views[iPlane])!=0)NplanesOK++;
                }
                if(ion!=0)NnodesTot++;
                if(NplanesOK!=0)ion*=3./NplanesOK;
                ionTotal+=ion;
            }
            if(NnodesTot == 0) NnodesTot = 1;
            averageIonizatioPerTrack[itrack] = ionTotal/NnodesTot;
        }
        return averageIonizatioPerTrack;
    }
    //______________________________________________________
    std::vector<double> AStarTracker::GetAverageIonization_Yplane(double distAvg){
        tellMe("GetAverageIonization_Yplane()",0);
        if(_vertex_dQdX_v.size()!=_vertexTracks.size()){_vertex_dQdX_v.clear();ComputeNewdQdX();}
        std::vector<double> averageIonizatioPerTrack(_vertexTracks.size(),0);
        for(size_t itrack = 0;itrack < _vertexLarliteTracks.size();itrack++){
            double ionTotal = 0;
            double ion = 0;
            int NnodesTot = 0;
            averageIonizatioPerTrack[itrack] = ionTotal;
            for(size_t iNode = 0;iNode < _vertexLarliteTracks[itrack].NumberTrajectoryPoints();iNode++){
                if(distAvg!= -1 && _vertexLarliteTracks[itrack].Length(0)-_vertexLarliteTracks[itrack].Length(iNode) > distAvg)continue;
                int NplanesOK = 0;
                ion  = 0;
                ion += _vertexLarliteTracks[itrack].DQdxAtPoint(iNode,_views[2]);
                if(ion!=0)NnodesTot++;
                ionTotal+=ion;
            }
            if(NnodesTot == 0) NnodesTot = 1;
            averageIonizatioPerTrack[itrack] = ionTotal/NnodesTot;
        }
        return averageIonizatioPerTrack;
    }
    //______________________________________________________
    std::vector<double> AStarTracker::GetTotalIonization(double distAvg){
        tellMe(Form("GetTotalIonization(%.1f)",distAvg),0);
        if(_vertex_dQdX_v.size()!=_vertexTracks.size()){_vertex_dQdX_v.clear();ComputeNewdQdX();}
        std::vector<double> IonizatioPerTrack(_vertexTracks.size(),0);
        for(size_t itrack = 0;itrack < _vertexLarliteTracks.size();itrack++){
            double ionTotal = 0;
            double ion = 0;
            IonizatioPerTrack[itrack] = ionTotal;
            for(size_t iNode = 0;iNode < _vertexLarliteTracks[itrack].NumberTrajectoryPoints();iNode++){
                if(distAvg!= -1 && _vertexLarliteTracks[itrack].Length(0)-_vertexLarliteTracks[itrack].Length(iNode) > distAvg)continue;
                int NplanesOK = 0;
                ion  = 0;
                for(size_t iPlane = 0;iPlane<3;iPlane++){
                    ion += _vertexLarliteTracks[itrack].DQdxAtPoint(iNode,_views[iPlane]);
                    if(_vertexLarliteTracks[itrack].DQdxAtPoint(iNode,_views[iPlane])!=0)NplanesOK++;
                }
                if(NplanesOK!=0)ion*=3./NplanesOK;
                ionTotal+=ion;
            }
            IonizatioPerTrack[itrack] = ionTotal;
        }
        return IonizatioPerTrack;
    }
    //______________________________________________________
    std::vector<double> AStarTracker::GetTotalIonization_Yplane(double distAvg){
        tellMe(Form("GetTotalIonization(%.1f)",distAvg),0);
        if(_vertex_dQdX_v.size()!=_vertexTracks.size()){_vertex_dQdX_v.clear();ComputeNewdQdX();}
        std::vector<double> IonizatioPerTrack(_vertexTracks.size(),0);
        for(size_t itrack = 0;itrack < _vertexLarliteTracks.size();itrack++){
            double ionTotal = 0;
            double ion = 0;
            IonizatioPerTrack[itrack] = ionTotal;
            for(size_t iNode = 0;iNode < _vertexLarliteTracks[itrack].NumberTrajectoryPoints();iNode++){
                if(distAvg!= -1 && _vertexLarliteTracks[itrack].Length(0)-_vertexLarliteTracks[itrack].Length(iNode) > distAvg)continue;
                int NplanesOK = 0;
                ion  = 0;
                ion += _vertexLarliteTracks[itrack].DQdxAtPoint(iNode,_views[2]);
                ionTotal+=ion;
            }
            IonizatioPerTrack[itrack] = ionTotal;
        }
        return IonizatioPerTrack;
    }
    //______________________________________________________
    std::vector< std::vector<double> >  AStarTracker::GetTotalPixADC(){
        tellMe("GetTotalPixADC()",0);

        double shellMask = 7;
        double MaskedValue = 0;
        std::vector< std::vector<double> > totalADC(_vertexTracks.size() );
        if(_vertexTracks.size() == 0) return totalADC;

        for(size_t itrack = 0;itrack < _vertexTracks.size();itrack++){
            std::vector<double> ADCvalue(3,0);
            hit_image_v = CropFullImage2bounds(_vertexTracks);

            for(size_t iPlane=0; iPlane<3;iPlane++){
                double x_proj,y_proj;
                ProjectTo3D(hit_image_v[iPlane].meta(),_vertexTracks[itrack][0].X(),_vertexTracks[itrack][0].Y(),_vertexTracks[itrack][0].Z(),0,iPlane,x_proj,y_proj);
                TVector3 strPt(x_proj,y_proj,0);
                ProjectTo3D(hit_image_v[iPlane].meta(),_vertexTracks[itrack].back().X(),_vertexTracks[itrack].back().Y(),_vertexTracks[itrack].back().Z(),0,iPlane,x_proj,y_proj);
                TVector3 ndPt(x_proj,y_proj,0);

                for(size_t irow = 0;irow<hit_image_v[iPlane].meta().rows();irow++){

                    if(!(irow > strPt.Y()-20 && irow<strPt.Y()+20) && !(irow > ndPt.Y()-20 && irow < ndPt.Y()+20))continue;

                    for(size_t icol = 0;icol<hit_image_v[iPlane].meta().cols();icol++){
                        if(!(icol > strPt.X()-20 && icol<strPt.X()+20) && !(icol > ndPt.X()-20 && icol < ndPt.X()+20))continue;
                        if(hit_image_v[iPlane].pixel(irow,icol) == _deadWireValue || hit_image_v[iPlane].pixel(irow,icol) == 0)continue;
                        TVector3 pC((icol+0.5),(irow+0.5),0);
                        if((pC-strPt).Mag() < shellMask || (pC-ndPt).Mag() < shellMask){
                            ADCvalue[iPlane]+=hit_image_v[iPlane].pixel(irow,icol);
                            hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
                        }
                    }
                }
                if(_vertexTracks[itrack].size() < 3)continue;
                for(size_t iNode = 1;iNode < _vertexTracks[itrack].size()-1;iNode++){
                    for(size_t iPlane = 0; iPlane<3; iPlane++){

                        ProjectTo3D(hit_image_v[iPlane].meta(),_vertexTracks[itrack][iNode].X(),_vertexTracks[itrack][iNode].Y(),_vertexTracks[itrack][iNode].Z(),0,iPlane,x_proj,y_proj);
                        TVector3 A(x_proj,y_proj,0);
                        ProjectTo3D(hit_image_v[iPlane].meta(),_vertexTracks[itrack][iNode+1].X(),_vertexTracks[itrack][iNode+1].Y(),_vertexTracks[itrack][iNode+1].Z(),0,iPlane,x_proj,y_proj);
                        TVector3 B(x_proj,y_proj,0);

                        for(size_t irow = 0;irow<hit_image_v[iPlane].meta().rows();irow++){
                            if(irow < std::min(A.Y()-20, B.Y()-20))continue;
                            if(irow > std::max(A.Y()+20, B.Y()+20))continue;
                            for(size_t icol = 0;icol<hit_image_v[iPlane].meta().cols();icol++){
                                if(icol < std::min(A.X()-20, B.X()-20))continue;
                                if(icol > std::max(A.X()+20, B.X()+20))continue;
                                if(hit_image_v[iPlane].pixel(irow,icol) == _deadWireValue || hit_image_v[iPlane].pixel(irow,icol) == 0)continue;
                                double alphaCol = ((icol+0.5)-A.X())/(B.X()-A.X());
                                double alphaRow = ((irow+0.5)-A.Y())/(B.Y()-A.Y());
                                TVector3 pC((icol+0.5),(irow+0.5),0);
                                double alpha = ((B-A).Dot(pC-A))/(pow((B-A).Mag(),2));

                                if(alpha  >= -0.1 && alpha <= 1.1){
                                    if(GetDist2line(A,B,pC) < shellMask){
                                        ADCvalue[iPlane]+=hit_image_v[iPlane].pixel(irow,icol);
                                        hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
                                    }
                                }
                                if((_vertexTracks[itrack][iNode]-_vertexTracks[itrack][0]).Mag() < 2){
                                    if((pC-A).Mag() < shellMask || (pC-B).Mag() < shellMask){
                                        ADCvalue[iPlane]+=hit_image_v[iPlane].pixel(irow,icol);
                                        hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
                                    }
                                }
                                else{
                                    if((pC-A).Mag() < shellMask || (pC-B).Mag() < shellMask){
                                        ADCvalue[iPlane]+=hit_image_v[iPlane].pixel(irow,icol);
                                        hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
                                    }
                                }

                            }//icol
                        }
                    }
                }
            }

            totalADC[itrack] = ADCvalue;
        }

        return totalADC;
    }
    //______________________________________________________
    std::vector< std::vector<double> >  AStarTracker::GetTotalPixADC(float tkLen){
        tellMe("GetTotalPixADC()",0);

        double shellMask = 5;
        double MaskedValue = 0;
        std::vector< std::vector<double> > totalADC(_vertexTracks.size() );
        if(_vertexTracks.size() == 0) return totalADC;


        for(size_t itrack= 0; itrack < _vertexTracks.size(); itrack++){
            std::vector<double> ADCvalue(3,0);
            for(size_t iNode = 0;iNode < _vertexTracks[itrack].size()-1;iNode++){

                if((_vertexTracks[itrack][iNode]-_vertexTracks[itrack][0]).Mag() > tkLen) continue;

                for(size_t iPlane = 0; iPlane<3; iPlane++){

                    double x_proj,y_proj;
                    ProjectTo3D(hit_image_v[iPlane].meta(),_vertexTracks[itrack][iNode].X(),_vertexTracks[itrack][iNode].Y(),_vertexTracks[itrack][iNode].Z(),0,iPlane,x_proj,y_proj);
                    TVector3 A(x_proj,y_proj,0);
                    ProjectTo3D(hit_image_v[iPlane].meta(),_vertexTracks[itrack][iNode+1].X(),_vertexTracks[itrack][iNode+1].Y(),_vertexTracks[itrack][iNode+1].Z(),0,iPlane,x_proj,y_proj);
                    TVector3 B(x_proj,y_proj,0);

                    for(size_t irow = 0;irow<hit_image_v[iPlane].meta().rows();irow++){
                        if(irow < std::min(A.Y()-20, B.Y()-20))continue;
                        if(irow > std::max(A.Y()+20, B.Y()+20))continue;
                        for(size_t icol = 0;icol<hit_image_v[iPlane].meta().cols();icol++){
                            if(icol < std::min(A.X()-20, B.X()-20))continue;
                            if(icol > std::max(A.X()+20, B.X()+20))continue;
                            if(hit_image_v[iPlane].pixel(irow,icol) == _deadWireValue || hit_image_v[iPlane].pixel(irow,icol) == 0)continue;
                            double alphaCol = ((icol+0.5)-A.X())/(B.X()-A.X());
                            double alphaRow = ((irow+0.5)-A.Y())/(B.Y()-A.Y());
                            TVector3 pC((icol+0.5),(irow+0.5),0);
                            double alpha = ((B-A).Dot(pC-A))/(pow((B-A).Mag(),2));

                            if(alpha  >= -0.1 && alpha <= 1.1){
                                if(GetDist2line(A,B,pC) < shellMask){
                                    ADCvalue[iPlane]+=hit_image_v[iPlane].pixel(irow,icol);
                                    hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
                                }
                            }
                            if((_vertexTracks[itrack][iNode]-_vertexTracks[itrack][0]).Mag() < 2){
                                if((pC-A).Mag() < shellMask || (pC-B).Mag() < shellMask){
                                    ADCvalue[iPlane]+=hit_image_v[iPlane].pixel(irow,icol);
                                    hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
                                }
                            }
                            else{
                                if((pC-A).Mag() < shellMask || (pC-B).Mag() < shellMask){
                                    ADCvalue[iPlane]+=hit_image_v[iPlane].pixel(irow,icol);
                                    hit_image_v[iPlane].set_pixel(irow,icol,MaskedValue);
                                }
                            }
                        }//icol
                    }
                }
            }
            totalADC[itrack] = ADCvalue;
        }
        return totalADC;
    }
    //______________________________________________________
    std::vector<double> AStarTracker::GetVertexLength(){
        tellMe("GetVertexLength()",0);
        if(_vertexLength.size()==_vertexTracks.size())return _vertexLength;

        std::vector<double> VertexLengths(_vertexTracks.size(),0);
        for(size_t itrack = 0;itrack< _vertexTracks.size();itrack++){
            _3DTrack = _vertexTracks[itrack];
            ComputeLength();
            VertexLengths[itrack] = _Length3D;
        }
        return VertexLengths;
    }
    //______________________________________________________
    std::vector<double> AStarTracker::GetClosestWall(){
        if(_closestWall.size()     != _vertexTracks.size()) ComputeClosestWall();
        return _closestWall;
    }
    //______________________________________________________
    std::vector<double> AStarTracker::GetClosestWall_SCE(){
        if(_closestWall_SCE.size() != _vertexTracks.size()) ComputeClosestWall_SCE();
        return _closestWall_SCE;
    }
    //______________________________________________________
    void AStarTracker::ComputeClosestWall(){
        if(_closestWall.size() ==_vertexTracks.size()) return;
        if(_closestWall.size()!=0)_closestWall.clear();

        //find tracks that approach the edge of the detector too much
        double detectorLength = 1036.8;//cm
        double detectorheight = 233;//cm
        double detectorwidth  = 256.35;//cm
        double Ymindet = -0.5*detectorheight;
        double Ymaxdet = 0.5*detectorheight;
        double Xmindet = 0;
        double Xmaxdet = detectorwidth;
        double Zmindet = 0;
        double Zmaxdet = detectorLength;

        double minApproach = 1e9;
        for(size_t itrack=0;itrack<_vertexTracks.size();itrack++){
            minApproach = 1e9;
            for(size_t iNode=0;iNode<_vertexTracks[itrack].size();iNode++){
                if( std::abs(_vertexTracks[itrack][iNode].X()-Xmindet) < minApproach) minApproach = std::abs(_vertexTracks[itrack][iNode].X()-Xmindet);
                if( std::abs(_vertexTracks[itrack][iNode].X()-Xmaxdet) < minApproach) minApproach = std::abs(_vertexTracks[itrack][iNode].X()-Xmaxdet);
                if( std::abs(_vertexTracks[itrack][iNode].Y()-Ymindet) < minApproach) minApproach = std::abs(_vertexTracks[itrack][iNode].Y()-Ymindet);
                if( std::abs(_vertexTracks[itrack][iNode].Y()-Ymaxdet) < minApproach) minApproach = std::abs(_vertexTracks[itrack][iNode].Y()-Ymaxdet);
                if( std::abs(_vertexTracks[itrack][iNode].Z()-Zmindet) < minApproach) minApproach = std::abs(_vertexTracks[itrack][iNode].Z()-Zmindet);
                if( std::abs(_vertexTracks[itrack][iNode].Z()-Zmaxdet) < minApproach) minApproach = std::abs(_vertexTracks[itrack][iNode].Z()-Zmaxdet);
            }
            _closestWall.push_back(minApproach);
        }
    }
    //______________________________________________________
    void AStarTracker::Add_SCE_to_Tracks(){
        tellMe("Add_SCE_to_Tracks()",0);
        if(_vertexTracks_SCE.size()!=0)_vertexTracks_SCE.clear();
        larutil::SpaceChargeMicroBooNE sce(larutil::SpaceChargeMicroBooNE::kMCC9_Backward);

        for(size_t itrack = 0;itrack<_vertexTracks.size();itrack++){
            std::vector<TVector3> CorrectedTrack;
            for(size_t iNode = 0;iNode<_vertexTracks[itrack].size();iNode++){
                double pt_X = _vertexTracks[itrack][iNode].X();
                double pt_Y = _vertexTracks[itrack][iNode].Y();
                double pt_Z = _vertexTracks[itrack][iNode].Z();

                auto const sceOffset = sce.GetPosOffsets(pt_X,pt_Y,pt_Z);

                double sceptX = pt_X - sceOffset[0] + 0.7;
                double sceptY = pt_Y + sceOffset[1];
                double sceptZ = pt_Z + sceOffset[2];

                TVector3 CorrectedNode(sceptX,sceptY,sceptZ);
                CorrectedTrack.push_back(CorrectedNode);
            }
            _vertexTracks_SCE.push_back(CorrectedTrack);
        }
        _vertexTracks = _vertexTracks_SCE;

        // corrected the vector of 3D points, I now need to push that into a new larlite::track
        
        if(_vertexLarliteTracks_sceadded.size()!=0)_vertexLarliteTracks_sceadded.clear();

        for(size_t itrack=0;itrack<_vertexTracks.size();itrack++){
            _3DTrack=_vertexTracks[itrack];
            larlite::track _thisLarliteTrack_sceadded;

            for(size_t iNode = 0;iNode<_3DTrack.size();iNode++){
                _thisLarliteTrack_sceadded.add_vertex(_3DTrack[iNode]);
            }
            for(size_t iNode = 0;iNode<_3DTrack.size();iNode++){
                if(iNode<_3DTrack.size()-1){
                    _thisLarliteTrack_sceadded.add_direction(_3DTrack[iNode+1]-_3DTrack[iNode]);
                }
                else{
                    _thisLarliteTrack_sceadded.add_direction(_3DTrack[iNode]-_3DTrack[iNode-1]);
                }
            }

            for(size_t iPlane = 0;iPlane<3;iPlane++){
                std::vector<double> dQdXOnePlane_v;
                for(size_t iNode = 0;iNode<_vertexLarliteTracks[itrack].NumberTrajectoryPoints();iNode++){
                    dQdXOnePlane_v.push_back(_vertexLarliteTracks[itrack].DQdxAtPoint(iNode,_views[iPlane]));
                }
                _thisLarliteTrack_sceadded.add_dqdx(dQdXOnePlane_v);
            }
            _vertexLarliteTracks_sceadded.push_back(_thisLarliteTrack_sceadded);

        }

        _vertexLarliteTracks = _vertexLarliteTracks_sceadded;

    }
    //______________________________________________________
    void AStarTracker::ComputeClosestWall_SCE(){
        tellMe("ComputeClosestWall_SCE()",0);
        if(_closestWall_SCE.size() ==_vertexTracks.size()) return;
        if(_closestWall_SCE.size()!=0)_closestWall_SCE.clear();

        larutil::SpaceChargeMicroBooNE sce(larutil::SpaceChargeMicroBooNE::kMCC9_Backward);
        //std::cout << "sce defined" << std::endl;

        //find tracks that approach the edge of the detector too much
        double detectorLength = 1036.8;//cm
        double detectorheight = 233;//cm
        double detectorwidth  = 256.35;//cm
        double Ymindet = -0.5*detectorheight;
        double Ymaxdet = 0.5*detectorheight;
        double Xmindet = 0;
        double Xmaxdet = detectorwidth;
        double Zmindet = 0;
        double Zmaxdet = detectorLength;

        //std::cout << "XYZ boundaires defined" << std::endl;

        double minApproach = 1e9;
        for(size_t itrack=0;itrack<_vertexTracks.size();itrack++){
            for(size_t iNode=0;iNode<_vertexTracks[itrack].size();iNode++){

                double pt_X = _vertexTracks[itrack][iNode].X();
                double pt_Y = _vertexTracks[itrack][iNode].Y();
                double pt_Z = _vertexTracks[itrack][iNode].Z();

                //std::cout << "about to get sceOffset" << std::endl;
                auto const sceOffset = sce.GetPosOffsets(pt_X,pt_Y,pt_Z);
                //std::cout << "sceOffset gotten" << std::endl;

                // from discussion with Chris on 3/14/2019:
                // put sce on truth info : subtract the offset for X and add it for Y and Z (from true track to where it will be with sce)
                // for reco (from where the track is recoed to where it would be without space charge) :

                double sceptX = pt_X - sceOffset[0] + 0.7;
                double sceptY = pt_Y + sceOffset[1];
                double sceptZ = pt_Z + sceOffset[2];

                //std::cout << "point corrected for SCE" << std::endl;


                if( std::abs(sceptX-Xmindet) < minApproach) minApproach = std::abs(sceptX-Xmindet);
                if( std::abs(sceptX-Xmaxdet) < minApproach) minApproach = std::abs(sceptX-Xmaxdet);
                if( std::abs(sceptY-Ymindet) < minApproach) minApproach = std::abs(sceptY-Ymindet);
                if( std::abs(sceptY-Ymaxdet) < minApproach) minApproach = std::abs(sceptY-Ymaxdet);
                if( std::abs(sceptZ-Zmindet) < minApproach) minApproach = std::abs(sceptZ-Zmindet);
                if( std::abs(sceptZ-Zmaxdet) < minApproach) minApproach = std::abs(sceptZ-Zmaxdet);
            }
            _closestWall_SCE.push_back(minApproach);
        }
    }
    //______________________________________________________
    std::vector<std::vector<double> > AStarTracker::GetVertexAngle(double dAverage = 15, double dMin = 0){
        //tellMe(Form("GetVertexAngle(%.1f,%.1f)",dAverage,dMin),0);
        int NtracksAtVertex = 0;
        int NpointAveragedOn = 0;
        std::vector<TVector3> AvPt_v;
        std::vector<double> thisTrackAngles;
        std::vector<std::vector<double> > vertexAngles_v;

        double phi,theta;// angles w.r.t. the detector phi is in the (X,Y) plane => azimuthal angle, theta is w.r.t the Z axis => polar angle.

        if(_Track15cmAvgPt.size()!=0)_Track15cmAvgPt.clear();
        for(size_t itrack = 0;itrack<_vertexTracks.size();itrack++){
            if(_vertexTracks[itrack][0] == start_pt){
                NtracksAtVertex++;
                TVector3 AvPt;
                NpointAveragedOn = 0;
                for(size_t iNode=1;iNode<_vertexTracks[itrack].size();iNode++){
                    if( (_vertexTracks[itrack][iNode]-start_pt).Mag() < dAverage  && (_vertexTracks[itrack][iNode]-start_pt).Mag() > dMin ){
                        AvPt+=(_vertexTracks[itrack][iNode]-start_pt);
                        NpointAveragedOn++;
                    }
                }
                if(NpointAveragedOn!=0){AvPt*=1./NpointAveragedOn;}
                AvPt_v.push_back(AvPt+start_pt);
                if(dAverage == 15)_Track15cmAvgPt.push_back(AvPt+start_pt);
            }
        }


        if(vertexAngles_v.size()!=0){vertexAngles_v.clear();}
        if(_vertexPhi.size()!=0){_vertexPhi.clear();}
        if(_vertexTheta.size()!=0){_vertexTheta.clear();}

        if(AvPt_v.size() == 0){
            if(thisTrackAngles.size()!=0)thisTrackAngles.clear();
            if(vertexAngles_v.size() !=0)vertexAngles_v.clear();
            thisTrackAngles.push_back(0);
            vertexAngles_v.push_back(thisTrackAngles);
            return vertexAngles_v;
        }
        if(AvPt_v.size() == 1){
            theta = (AvPt_v[0]-start_pt).Theta();
            phi   = (AvPt_v[0]-start_pt).Phi();
            _vertexPhi.push_back(phi);
            _vertexTheta.push_back(theta);
            if(thisTrackAngles.size()!=0)thisTrackAngles.clear();
            if(vertexAngles_v.size() !=0)vertexAngles_v.clear();
            thisTrackAngles.push_back(0);
            vertexAngles_v.push_back(thisTrackAngles);
            return vertexAngles_v;
        }

        for(size_t iPt = 0;iPt<AvPt_v.size();iPt++){
            if(thisTrackAngles.size()!=0)thisTrackAngles.clear();
            theta = (AvPt_v[iPt]-start_pt).Theta();
            phi   = (AvPt_v[iPt]-start_pt).Phi();
            _vertexPhi.push_back(phi);
            _vertexTheta.push_back(theta);
            for(size_t jPt = 0;jPt<AvPt_v.size();jPt++){
                double thisAngle;
                //if(iPt==jPt){thisAngle=0;thisTrackAngles.push_back(thisAngle);}
                thisAngle = (AvPt_v[iPt]-start_pt).Angle(AvPt_v[jPt]-start_pt)*180/3.1415;
                thisTrackAngles.push_back(thisAngle);
                //tellMe(Form("%zu : %zu => %.1f",iPt,jPt,thisAngle),0);
            }
            vertexAngles_v.push_back(thisTrackAngles);
        }
        return vertexAngles_v;
    }
    //______________________________________________________
    std::vector<double> AStarTracker::GetOldVertexAngle(double dAverage = 5){
        // Find if tracks are back to back at the vertex point
        // angle computed by averaging over the first x cm around the vertex
        int NtracksAtVertex = 0;
        int NpointAveragedOn = 0;
        std::vector<TVector3> AvPt_v;
        std::vector<double> vertexAngles_v;

        for(size_t itrack = 0;itrack<_vertexTracks.size();itrack++){
            if(_vertexTracks[itrack][0] == start_pt){
                NtracksAtVertex++;
                TVector3 AvPt;
                NpointAveragedOn = 0;
                for(size_t iNode=1;iNode<_vertexTracks[itrack].size();iNode++){
                    if( (_vertexTracks[itrack][iNode]-start_pt).Mag() < dAverage ){
                        AvPt+=(_vertexTracks[itrack][iNode]-start_pt);
                        NpointAveragedOn++;
                    }
                }
                if(NpointAveragedOn!=0){AvPt*=1./NpointAveragedOn;}
                AvPt_v.push_back(AvPt+start_pt);
            }
        }


        if(AvPt_v.size() >= 2){
            for(size_t iPt = 0;iPt<AvPt_v.size();iPt++){
                for(size_t jPt = 0;jPt<AvPt_v.size();jPt++){
                    if(jPt==iPt)continue;
                    double thisAngle = (AvPt_v[iPt]-start_pt).Angle(AvPt_v[jPt]-start_pt)*180/3.1415;
                    vertexAngles_v.push_back(thisAngle);
                }
            }
        }
        else{vertexAngles_v.push_back(-1);}

        return vertexAngles_v;
    }
    //______________________________________________________
    double AStarTracker::GetTotalDepositedCharge(){
        if(_3DTrack.size() == 0){tellMe("ERROR, run the 3D reconstruction first",0);return -1;}
        if(_dQdx.size() == 0){tellMe("ERROR, run the dqdx reco first",0);return -1;}
        double Qtot = 0;
        for(size_t iNode = 0;iNode<_dQdx.size();iNode++){
            double dqdx = 0;
            int Nplanes = 0;
            for(size_t iPlane = 0;iPlane<3;iPlane++){
                dqdx+=_dQdx[iNode][iPlane];
                if(_dQdx[iNode][iPlane] != 0)Nplanes++;
            }
            if(Nplanes!=0)dqdx*=3./Nplanes;
            Qtot+=dqdx;
        }
        return Qtot;
    }
    //______________________________________________________
    double AStarTracker::GetDist2line(TVector3 A, TVector3 B, TVector3 C){
        TVector3 CrossProd = (B-A).Cross(C-A);
        TVector3 U = B-A;
        return CrossProd.Mag()/U.Mag();
    }
    //______________________________________________________
    double AStarTracker::GetDist2track(TVector3 thisPoint, std::vector<TVector3> thisTrack){
        double dist = 15000;
        if(thisTrack.size() > 2){
            for(size_t iNode = 0;iNode<thisTrack.size()-1;iNode++){
                if(thisTrack[iNode] == thisPoint)continue;
                double alpha = ( (thisPoint-thisTrack[iNode]).Dot(thisTrack[iNode+1]-thisTrack[iNode]) )/( (thisPoint-thisTrack[iNode]).Mag()*(thisTrack[iNode+1]-thisTrack[iNode]).Mag() );
                if(alpha > 0 && alpha<1){
                    dist = std::min(dist, GetDist2line(thisTrack[iNode],thisTrack[iNode+1],thisPoint) );
                }
                dist = std::min( dist, (thisPoint-thisTrack[iNode]  ).Mag());
                dist = std::min( dist, (thisPoint-thisTrack[iNode+1]).Mag());
            }
        }
        return dist;
    }
    //______________________________________________________
    TVector3 AStarTracker::CheckEndPoints(TVector3 point){
        tellMe(Form("start point : (%f,%f,%f)",point.X(),point.Y(),point.Z()),2);
        double dR = 0.1;
        double closeEnough = 4;

        double minDist = EvalMinDist(point);
        double PreviousMinDist = 2*minDist;
        tellMe(Form("dist = %f.1", minDist),1);
        if(minDist <= closeEnough){tellMe("Point OK",1);return point;}
        tellMe("Updating point",1);
        TVector3 newPoint = point;
        int iter = 0;
        int iterMax = 10000;
        while (minDist > closeEnough && iter < iterMax) {
            iter++;
            std::vector<TVector3> openSet = GetOpenSet(newPoint,dR);
            double minDist = 1e9;
            for(auto neighbour : openSet){
                double dist = EvalMinDist(neighbour);
                if(dist < minDist){minDist=dist;newPoint = neighbour;}
            }
            tellMe(Form("iteration #%d",iter),1);
            if(minDist < closeEnough || iter >= iterMax || minDist >= PreviousMinDist) break;
            PreviousMinDist = minDist;
        }
        tellMe("Point updated",1);
        return newPoint;
    }
    //______________________________________________________
    TVector3 AStarTracker::GetFurtherFromVertex(){
        TVector3 FurtherFromVertex(1,1,1);
        double dist2vertex = 0;
        for(size_t iNode = 0;iNode<_3DTrack.size();iNode++){
            if( (_3DTrack[iNode]-start_pt).Mag() > dist2vertex){dist2vertex = (_3DTrack[iNode]-start_pt).Mag(); FurtherFromVertex = _3DTrack[iNode];}
        }
        return FurtherFromVertex;
    }
    //______________________________________________________
    TVector3 AStarTracker::CheckEndPoints(TVector3 point, std::vector< std::pair<int,int> > endPix){
        tellMe(Form("start point : (%f,%f,%f)",point.X(),point.Y(),point.Z()),2);
        double dR = 0.1;
        double closeEnough = 4;

        double minDist = EvalMinDist(point,endPix);
        double PreviousMinDist = 2*minDist;
        tellMe(Form("dist = %f.1", minDist),0);
        if(minDist <= closeEnough){tellMe("Point OK",0);return point;}
        tellMe("Updating point",0);
        TVector3 newPoint = point;
        int iter = 0;
        int iterMax = 10000;
        while (minDist > closeEnough && iter < iterMax) {
            iter++;
            std::vector<TVector3> openSet = GetOpenSet(newPoint,dR);
            double minDist = 1e9;
            for(auto neighbour : openSet){
                double dist = EvalMinDist(neighbour,endPix);
                if(dist < minDist){minDist=dist;newPoint = neighbour;}
            }
            tellMe(Form("iteration #%d",iter),1);
            if(minDist < closeEnough || iter >= iterMax || minDist >= PreviousMinDist) break;
            PreviousMinDist = minDist;
        }
        tellMe("Point updated",0);
        return newPoint;
    }
    //______________________________________________________
    std::vector<TVector3> AStarTracker::FitBrokenLine(){
        std::vector<TVector3> Skeleton;

        Skeleton.push_back(start_pt);
        Skeleton.push_back(GetFurtherFromVertex());

        std::cout << "not implemented yet" << std::endl;
        // (1) compute summed distance of all 3D points to the current line
        // (2) add 3D points if needed
        // (3) vary 3D point position to get best fit
        // (4) as long as not best possible fit, add and fit 3D points
        return _3DTrack;
    }
    //______________________________________________________
    std::vector<TVector3> AStarTracker::OrderList(std::vector<TVector3> list){
        std::vector<TVector3> newList;
        newList.push_back(list[0]);
        double dist = 1e9;
        double distMin = 1e9;
        int newCandidate;
        bool goOn = true;
        int iter = 0;
        while(goOn == true && list.size()>0){
            iter++;
            for(size_t iNode=0;iNode<list.size();iNode++){
                if(list[iNode] == newList.back())continue;
                dist = (list[iNode]-newList.back()).Mag();
                if(dist<distMin){distMin=dist;newCandidate = iNode;}
            }


            if(distMin > 10 || list.size() <= 1){
                goOn = false;
                break;
            }
            if(distMin <= 10){
                newList.push_back(list[newCandidate]);
                list.erase(list.begin()+newCandidate);
                distMin = 100000;
            }
        }

        return newList;
    }
    //______________________________________________________
    std::vector<TVector3> AStarTracker::GetOpenSet(TVector3 newPoint, double dR){
        std::vector<TVector3> openSet;

        for(int ix = -1;ix<2;ix++){
            for(int iy = -1;iy<2;iy++){
                for(int iz = -1;iz<2;iz++){
                    //if(ix == 0 && iy == 0 && iz == 0)continue;
                    TVector3 neighbour = newPoint;
                    neighbour.SetX(newPoint.X()+dR*ix);
                    neighbour.SetY(newPoint.Y()+dR*iy);
                    neighbour.SetZ(newPoint.Z()+dR*iz);
                    openSet.push_back(neighbour);
                }
            }
        }
        return openSet;
    }
    //______________________________________________________
    std::vector<TVector3> AStarTracker::GetOpenSet(TVector3 newPoint, int BoxSize, double dR){
        std::vector<TVector3> openSet;

        for(int ix = (int)(-0.5*BoxSize);ix<(int)(0.5*BoxSize);ix++){
            for(int iy = (int)(-0.5*BoxSize);iy<(int)(0.5*BoxSize);iy++){
                for(int iz = (int)(-0.5*BoxSize);iz<(0.5*BoxSize);iz++){
                    //if(ix == 0 && iy == 0 && iz == 0)continue;
                    TVector3 neighbour = newPoint;
                    neighbour.SetX(newPoint.X()+dR*ix);
                    neighbour.SetY(newPoint.Y()+dR*iy);
                    neighbour.SetZ(newPoint.Z()+dR*iz);
                    openSet.push_back(neighbour);
                }
            }
        }
        return openSet;
    }
    //______________________________________________________
    std::vector<larcv::Image2D> AStarTracker::CropFullImage2bounds(std::vector< std::vector<TVector3> > _vertex_v){
        if(_3DTrack.size()!=0)_3DTrack.clear();
        for(size_t i=0;i<_vertex_v.size();i++){
            for(size_t iNode=0;iNode<_vertex_v[i].size();iNode++){
                _3DTrack.push_back(_vertex_v[i][iNode]);
            }
        }
        if(_3DTrack.size()==0) _3DTrack.push_back(start_pt);
        return CropFullImage2bounds(_3DTrack);
    }

    //______________________________________________________

    /** 
     * crop image and shave tracks in one call
     *
     * note: TMW modified version makes no crop
     * 
     * @param[in] EndPoints collection of space points around which to crop
     * 
     */
    void AStarTracker::CropFullImage2boundsIntegrated(std::vector<TVector3> EndPoints) {
      // TMW: we try to avoid the cropping. huge computation time waste, a lot of reallocation of memory and scanning across image. 
      // causes exception to be thrown in production.
      if ( hit_image_v.size()==0 ) {
	std::cout << "AStarTracker::CropFullImage2boundsIntegrated() make hit_image_v" << std::endl;
	// we use the image where dead wires are painted with values 
	// and the image is inverted for whatever reason
	hit_image_v = GetFullImage();
	// I suspect I need to create a blank erased pixel image
	// then call a new version of shavetracks() that uses this to mask
	masked_hit_image_v.clear();
	for ( auto const& hitimg : hit_image_v ) {
	  larcv::Image2D maskimg(hitimg.meta());
	  maskimg.paint(0.0);
	  masked_hit_image_v.emplace_back( std::move(maskimg) );
	}
      }

      // The old routine
      //hit_image_v = CropFullImage2bounds(EndPoints);
      /*EnhanceDerivative();*/
      //ShaveTracks();
    }

    //______________________________________________________


    //______________________________________________________
    void AStarTracker::ClearDeadWireList(){
        if(_deadWires_v.size()!=0)_deadWires_v.clear();
    }
    //______________________________________________________
    void AStarTracker::MakeDeadWireList(){
        std::cout << "MakeDeadWireList()" << std::endl;
        if(_deadWires_v.size()!=0){std::cout << "dead wires already acquired for this image" << std::endl;return;}
        // find dead/bad wires and paint them with 20
        for(size_t iPlane=0;iPlane<3;iPlane++){
            std::vector<double> deadWires_oneplane;
            for(size_t icol = 0;icol<original_full_image_v[iPlane].meta().cols();icol++){
                int summedVal = 0;
                for(size_t irow = 0;irow<original_full_image_v[iPlane].meta().rows();irow++){
                    //std::cout << icol << "\t" << irow << "\t" << original_full_image_v[iPlane].pixel(irow,icol) << std::endl;
                    if(original_full_image_v[iPlane].pixel(irow,icol) > _ADCthreshold)summedVal+=original_full_image_v[iPlane].pixel(irow,icol);
                }
                if(summedVal == 0){
                    deadWires_oneplane.push_back(icol);
                }
                //else std::cout << "Active Wire!" << std::endl;
            }
            _deadWires_v.push_back(deadWires_oneplane);
        }
        std::cout << "_deadWires_v.size() = " << _deadWires_v.size() << std::endl;
        for(size_t i=0;i<_deadWires_v.size();i++){
            std::cout << "_deadWires_v[" << i << "].size() = " << _deadWires_v[i].size() << std::endl;
        }
    }
    //______________________________________________________
    std::vector<larcv::Image2D> AStarTracker::CropFullImage2bounds(std::vector<TVector3> EndPoints){
        std::vector<larcv::ImageMeta> Full_meta_v(3);
        for(size_t iPlane = 0;iPlane<3;iPlane++){Full_meta_v[iPlane] = original_full_image_v[iPlane].meta();}

        SetTimeAndWireBounds(EndPoints, Full_meta_v);
        std::vector<std::pair<double,double> > time_bounds = GetTimeBounds();
        std::vector<std::pair<double,double> > wire_bounds = GetWireBounds();
        std::vector<larcv::Image2D> data_images;
        std::vector<larcv::Image2D> tagged_images;
        //if(_deadWires_v.size()!=0){_deadWires_v.clear();}
        // find dead/bad wires and paint them with 20
        for(size_t iPlane=0;iPlane<3;iPlane++){
            //std::vector<int> deadWires_oneplane;
            for(size_t icol = 0;icol<original_full_image_v[iPlane].meta().cols();icol++){
                int summedVal = 0;
                for(size_t irow = 0;irow<original_full_image_v[iPlane].meta().rows();irow++){
                    if(original_full_image_v[iPlane].pixel(irow,icol) > _ADCthreshold)summedVal+=original_full_image_v[iPlane].pixel(irow,icol);
                }
                if(summedVal == 0){
                    //deadWires_oneplane.push_back(icol);
                    original_full_image_v[iPlane].paint_col(icol,_deadWireValue);
                }
            }
            //_deadWires_v.push_back(deadWires_oneplane);
        }
        //std::cout << "_deadWires_v.size() = " << _deadWires_v.size() << std::endl;
        //for(uint i=0;i<_deadWires_v.size();i++){
        //    std::cout << "_deadWires_v[" << i << "].size() = " << _deadWires_v[i].size() << std::endl;
        //}
        // make smaller image by inverting and cropping the full one
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            double image_width  = 1.*(size_t)((wire_bounds[iPlane].second - wire_bounds[iPlane].first)*original_full_image_v[iPlane].meta().pixel_width());
            double image_height = 1.*(size_t)((time_bounds[iPlane].second - time_bounds[iPlane].first)*original_full_image_v[iPlane].meta().pixel_height());
            size_t col_count    = (size_t)(image_width / original_full_image_v[iPlane].meta().pixel_width());
            size_t row_count    = (size_t)(image_height/ original_full_image_v[iPlane].meta().pixel_height());
            double origin_x     = original_full_image_v[iPlane].meta().tl().x+wire_bounds[iPlane].first *original_full_image_v[iPlane].meta().pixel_width();
            double origin_y     = original_full_image_v[iPlane].meta().br().y+time_bounds[iPlane].second*original_full_image_v[iPlane].meta().pixel_height();
            larcv::ImageMeta newMeta(image_width, 6.*row_count,row_count, col_count,origin_x,origin_y,iPlane);
            larcv::ImageMeta newTagMeta(image_width, 6.*row_count,row_count, col_count,origin_x,origin_y,iPlane);

            newMeta = original_full_image_v[iPlane].meta().overlap(newMeta);
            newTagMeta = taggedPix_v[iPlane].meta().overlap(newTagMeta);

            larcv::Image2D newImage = original_full_image_v[iPlane].crop(newMeta);
            larcv::Image2D newTaggedImage = taggedPix_v[iPlane].crop(newTagMeta);

            larcv::Image2D invertedImage = newImage;
            larcv::Image2D invertedTagImage = newTaggedImage;

            for(size_t icol = 0;icol<newImage.meta().cols();icol++){
                for(size_t irow = 0;irow<newImage.meta().rows();irow++){
                    if(newImage.pixel(irow, icol) > _ADCthreshold){invertedImage.set_pixel(newImage.meta().rows()-irow-1, icol,newImage.pixel(irow, icol));}
                    else{invertedImage.set_pixel(newImage.meta().rows()-irow-1, icol,0);}
                }
            }
            data_images.push_back(invertedImage);

            for(size_t icol = 0;icol<newTaggedImage.meta().cols();icol++){
                for(size_t irow = 0;irow<newTaggedImage.meta().rows();irow++){
                    if(newTaggedImage.pixel(irow, icol) > _ADCthreshold){invertedTagImage.set_pixel(newTaggedImage.meta().rows()-irow-1, icol,newTaggedImage.pixel(irow, icol));}
                    else{invertedTagImage.set_pixel(newTaggedImage.meta().rows()-irow-1, icol,0);}
                }
            }
            tagged_images.push_back(invertedTagImage);
        }
        CroppedTaggedPix_v = tagged_images;

        for(size_t iPlane = 0;iPlane<3;iPlane++){
            for(size_t icol = 0;icol<data_images[iPlane].meta().cols();icol++){
                for(size_t irow = 0;irow<data_images[iPlane].meta().rows();irow++){
                    if(CroppedTaggedPix_v[iPlane].pixel(irow, icol) > _ADCthreshold){data_images[iPlane].set_pixel(irow,icol,0);}
                }
            }
        }

        return data_images;
    }
    //______________________________________________________
    std::vector<larcv::Image2D> AStarTracker::GetFullImage(){
        std::vector<larcv::ImageMeta> Full_meta_v(3);
        for(size_t iPlane = 0;iPlane<3;iPlane++){Full_meta_v[iPlane] = original_full_image_v[iPlane].meta();}

        //SetTimeAndWireBounds(EndPoints, Full_meta_v);
        //std::vector<std::pair<double,double> > time_bounds = GetTimeBounds();
        //std::vector<std::pair<double,double> > wire_bounds = GetWireBounds();
        std::vector<larcv::Image2D> data_images;
        std::vector<larcv::Image2D> tagged_images;

        // find dead/bad wires and paint them with 20
        for(size_t iPlane=0;iPlane<3;iPlane++){
            for(size_t icol = 0;icol<original_full_image_v[iPlane].meta().cols();icol++){
                int summedVal = 0;
                for(size_t irow = 0;irow<original_full_image_v[iPlane].meta().rows();irow++){
                    if(original_full_image_v[iPlane].pixel(irow,icol) > _ADCthreshold)summedVal+=original_full_image_v[iPlane].pixel(irow,icol);
                }
                if(summedVal == 0){
                    original_full_image_v[iPlane].paint_col(icol,_deadWireValue);
                }
            }
        }
        // make smaller image by inverting and cropping the full one
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            //double image_width  = 1.*(size_t)((wire_bounds[iPlane].second - wire_bounds[iPlane].first)*original_full_image_v[iPlane].meta().pixel_width());
            //double image_height = 1.*(size_t)((time_bounds[iPlane].second - time_bounds[iPlane].first)*original_full_image_v[iPlane].meta().pixel_height());
            //size_t col_count    = (size_t)(image_width / original_full_image_v[iPlane].meta().pixel_width());
            //size_t row_count    = (size_t)(image_height/ original_full_image_v[iPlane].meta().pixel_height());
            //double origin_x     = original_full_image_v[iPlane].meta().tl().x+wire_bounds[iPlane].first *original_full_image_v[iPlane].meta().pixel_width();
            //double origin_y     = original_full_image_v[iPlane].meta().br().y+time_bounds[iPlane].second*original_full_image_v[iPlane].meta().pixel_height();
            larcv::ImageMeta newMeta = original_full_image_v[iPlane].meta();//(image_width, 6.*row_count,row_count, col_count,origin_x,origin_y,iPlane);
            larcv::ImageMeta newTagMeta = original_full_image_v[iPlane].meta();//(image_width, 6.*row_count,row_count, col_count,origin_x,origin_y,iPlane);

            newMeta = original_full_image_v[iPlane].meta().overlap(newMeta);
            newTagMeta = taggedPix_v[iPlane].meta().overlap(newTagMeta);

            larcv::Image2D newImage = original_full_image_v[iPlane].crop(newMeta);
            larcv::Image2D newTaggedImage = taggedPix_v[iPlane].crop(newTagMeta);

            larcv::Image2D invertedImage = newImage;
            larcv::Image2D invertedTagImage = newTaggedImage;

            for(size_t icol = 0;icol<newImage.meta().cols();icol++){
                for(size_t irow = 0;irow<newImage.meta().rows();irow++){
                    if(newImage.pixel(irow, icol) > _ADCthreshold){invertedImage.set_pixel(newImage.meta().rows()-irow-1, icol,newImage.pixel(irow, icol));}
                    else{invertedImage.set_pixel(newImage.meta().rows()-irow-1, icol,0);}
                }
            }
            data_images.push_back(invertedImage);

            for(size_t icol = 0;icol<newTaggedImage.meta().cols();icol++){
                for(size_t irow = 0;irow<newTaggedImage.meta().rows();irow++){
                    if(newTaggedImage.pixel(irow, icol) > _ADCthreshold){invertedTagImage.set_pixel(newTaggedImage.meta().rows()-irow-1, icol,newTaggedImage.pixel(irow, icol));}
                    else{invertedTagImage.set_pixel(newTaggedImage.meta().rows()-irow-1, icol,0);}
                }
            }
            tagged_images.push_back(invertedTagImage);
        }
        CroppedTaggedPix_v = tagged_images;

        for(size_t iPlane = 0;iPlane<3;iPlane++){
            for(size_t icol = 0;icol<data_images[iPlane].meta().cols();icol++){
                for(size_t irow = 0;irow<data_images[iPlane].meta().rows();irow++){
                    if(CroppedTaggedPix_v[iPlane].pixel(irow, icol) > _ADCthreshold){data_images[iPlane].set_pixel(irow,icol,0);}
                }
            }
        }
        
        return data_images;
    }
    //______________________________________________________
    std::vector<std::pair<int, int> > AStarTracker::GetWireTimeProjection(TVector3 point){
        std::vector<std::pair<int, int> > projection(3);
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            std::pair<int, int> planeProj;
            planeProj.first = (int)(larutil::GeometryHelper::GetME()->Point_3Dto2D(point,iPlane).w / 0.3);
            planeProj.second = (int)(X2Tick(point.X(),iPlane));
            projection[iPlane] = planeProj;
        }
        return projection;
    }
    //-------------------------------------------------------
    std::vector<int> AStarTracker::GetRecoGoodness(){
        std::vector<int> recoGoodness_v(9);

        recoGoodness_v[0] = (int)_missingTrack;
        recoGoodness_v[1] = (int)_nothingReconstructed;
        recoGoodness_v[2] = (int)_tooShortDeadWire;
        recoGoodness_v[3] = (int)_tooShortFaintTrack;
        recoGoodness_v[4] = (int)_tooManyTracksAtVertex;
        recoGoodness_v[5] = (int)_possibleCosmic;
        recoGoodness_v[6] = (int)_possiblyCrossing;
        recoGoodness_v[7] = (int)_branchingTracks;
        recoGoodness_v[8] = (int)_jumpingTracks;

        return recoGoodness_v;
    }
    //-------------------------------------------------------
    std::vector<int> AStarTracker::GetVtxQuality(){
        if(_goodTrack_v.size()!=_vertexTracks.size())DiagnoseVertex();
        std::vector<int> _goodTrack_v_int;
        for(size_t i=0;i<_goodTrack_v.size();i++){
            _goodTrack_v_int.push_back((int)_goodTrack_v[i]);
        }
        return _goodTrack_v_int;
    }
    //-------------------------------------------------------
    void AStarTracker::RecoverFromFail(){
        tellMe("RecoverFromFail()",0);
        double x,y,z;
        recoveredvalue = _deadWireValue+100;
        bool recoverFromFaint = false;
        bool recoverFromDead = true;
        int NpointAveragedOn = 0;

        //if(( recoverFromFaint && _tooShortFaintTrack) || (recoverFromDead && _tooShortDeadWire)){DrawVertex();}

        for(size_t itrack=0;itrack<_vertexTracks.size();itrack++){
            TVector3 newPoint;
            int NaveragePts = 0;
            TVector3 AveragePoint;
            TVector3 oldEndPoint = _vertexTracks.at(itrack).back();

            // faint tracks
            if(recoverFromFaint && _tooShortFaintTrack_v.at(itrack)){
                TVector3 oldEndPoint;
                std::cout << "plane failed : " << failedPlane << std::endl;
                double x_pixel_old,y_pixel_old;
                oldEndPoint = _vertexTracks.at(itrack).back();

                ProjectTo3D(original_full_image_v.at(failedPlane).meta(),oldEndPoint.X(),oldEndPoint.Y(),oldEndPoint.Z(),0,failedPlane,x_pixel_old,y_pixel_old);
                //for(size_t irow = 0;irow<original_full_image_v.at(failedPlane).meta().rows();irow++){
                for(auto irow = (y_pixel_old-20);irow < (y_pixel_old+20);irow++){
                    for(int imargin=0;imargin<5;imargin++){
                        //if( original_full_image_v.at(failedPlane).pixel(    irow,x_pixel_old-imargin) <= recoveredvalue)
                        try {
                            original_full_image_v.at(failedPlane).set_pixel(original_full_image_v.at(failedPlane).meta().rows()-irow,x_pixel_old-imargin,recoveredvalue);
                        } catch (...) {
                            std::cout << "WARNING" << std::endl;
                            std::cout << std::endl;
                            std::cout << "BUG IDENTIFIED" << std::endl;
                            std::cout << std::endl;
                            std::cout << "WARNING" << std::endl;
                            continue;
                        }


                        //if( original_full_image_v.at(failedPlane).pixel(    irow,x_pixel_old+imargin) <= recoveredvalue)
                        try {
                            original_full_image_v.at(failedPlane).set_pixel(original_full_image_v.at(failedPlane).meta().rows()-irow,x_pixel_old+imargin,recoveredvalue);
                        } catch (...) {
                            std::cout << "WARNING" << std::endl;
                            std::cout << std::endl;
                            std::cout << "BUG IDENTIFIED" << std::endl;
                            std::cout << std::endl;
                            std::cout << "WARNING" << std::endl;
                            continue;
                        }
                    }
                }
            }
            // end of faint track recovery

            // dead wires

            if(recoverFromDead && _tooShortDeadWire_v.at(itrack)){//adding non-zero pixels for dead region failures
                //start by finding the direction towards the end of the track for each individual plane
                TVector3 AvPt;
                NpointAveragedOn = 0;
                for(size_t iNode=1;iNode<_vertexTracks.at(itrack).size();iNode++){
                    if( (_vertexTracks.at(itrack).at(iNode)-oldEndPoint).Mag() < 20 ){
                        AvPt+=(_vertexTracks.at(itrack).at(iNode)-oldEndPoint);
                        NpointAveragedOn++;
                    }
                }
                if(NpointAveragedOn!=0){AvPt*=1./NpointAveragedOn;}
                AvPt+=oldEndPoint;

                //add 1 to dead wire pixels that are along the line
                for(int iPlane=0;iPlane<3;iPlane++){
                    std::cout << "plane " << iPlane << std::endl;
                    double x_px_avg(0),y_px_avg(0);
                    double x_px_end(0),y_px_end(0);
                    ProjectTo3D(original_full_image_v[iPlane].meta(),AvPt.X()       ,AvPt.Y()       ,AvPt.Z()       ,0,iPlane,x_px_avg,y_px_avg);
                    ProjectTo3D(original_full_image_v[iPlane].meta(),oldEndPoint.X(),oldEndPoint.Y(),oldEndPoint.Z(),0,iPlane,x_px_end,y_px_end);

                    TVector3 A(x_px_avg,y_px_avg,0);
                    TVector3 B(x_px_end,y_px_end,0);

                    for(size_t icol=0;icol<original_full_image_v[iPlane].meta().cols()-1;icol++){//==x_pixel
                        for(size_t irow=0;irow<original_full_image_v[iPlane].meta().rows()-1;irow++){//==y_pixel

                            if(original_full_image_v[iPlane].pixel(irow,icol) != _deadWireValue)continue;
                            TVector3 C(icol,irow,0);

                            if((C-B).Mag()>10000)continue;
                            double cosangle = cos((B-A).Angle(C-B));
                            if( cosangle < 1-exp(-((C-B).Mag())/5.))continue;
                            bool notContiguous = false;
                            for(int j=0;j<std::abs(icol-x_px_end)-1;j++){
                                if(original_full_image_v[iPlane].pixel(irow,icol-((icol-x_px_end)/(std::abs(icol-x_px_end)))*j) !=_deadWireValue
                                   && original_full_image_v[iPlane].pixel(irow,icol-((icol-x_px_end)/(std::abs(icol-x_px_end)))*j) !=recoveredvalue){
                                    notContiguous = true;
                                    continue;
                                }
                            }
                            if(notContiguous)continue;
                            //hImage[iPlane]->SetBinContent(icol+1,irow+1,_deadWireValue+100);

                            try {
                                original_full_image_v[iPlane].set_pixel(original_full_image_v.at(failedPlane).meta().rows()-irow,icol,recoveredvalue);
                            } catch (...) {
                                std::cout << "WARNING" << std::endl;
                                std::cout << std::endl;
                                std::cout << "BUG IDENTIFIED" << std::endl;
                                std::cout << std::endl;
                                std::cout << "WARNING" << std::endl;
                                continue;
                            }
                            
                        }
                    }
                }//end of the loop adding non-zero pixels for dead region failures
            }
            // end of dead wire recovery
        }
        
        
        
        if(((_tooShortFaintTrack && recoverFromFaint) || (recoverFromDead && _tooShortDeadWire)) && NumberRecoveries < 10){
            NumberRecoveries++;
            std::cout << "Recovered " << NumberRecoveries << " times" << std::endl;
            ReconstructVertex();
        }
    }
    //-------------------------------------------------------
    void AStarTracker::FillInTrack(){
        std::vector<TVector3> newTrack;
        if(newTrack.size() != 0)newTrack.clear();
        //if(_3DTrack.back()!= end_pt)_3DTrack.push_back(end_pt);
        for(size_t ipoint=0;ipoint<_3DTrack.size()-1;ipoint++){
            newTrack.push_back(_3DTrack[ipoint]);
            TVector3 vertexNeighbour = _3DTrack[ipoint];
            int iterNeighbour = 1;
            while((vertexNeighbour-_3DTrack[ipoint]).Mag() < (_3DTrack[ipoint+1]-_3DTrack[ipoint]).Mag() && iterNeighbour<1e6){
                vertexNeighbour = _3DTrack[ipoint]+0.3*(_3DTrack[ipoint+1]-_3DTrack[ipoint])*(iterNeighbour/(_3DTrack[ipoint+1]-_3DTrack[ipoint]).Mag());
                if((vertexNeighbour-_3DTrack[ipoint]).Mag() < (_3DTrack[ipoint+1]-_3DTrack[ipoint]).Mag()){
                    iterNeighbour++;
                    newTrack.push_back(vertexNeighbour);
                }
                else iterNeighbour = 1e7;
            }
        }
        newTrack.push_back(_3DTrack.back());
        
        _3DTrack = newTrack;
    }
    //-------------------------------------------------------
    std::vector<TVector3>   AStarTracker::GetEndPoints(){
        tellMe("GetEndPoints()",0);

        std::vector<TVector3> vertexEndPoints(_vertexTracks.size());
        for(size_t itrack = 0;itrack< _vertexTracks.size();itrack++){
            TVector3 trackEndPoint = _vertexTracks[itrack].back();
            vertexEndPoints[itrack] = trackEndPoint;
        }
        return vertexEndPoints;
    }
    //_______________________________________________________
    void AStarTracker::Get3DtracksFromLarlite(){
        std::cout << "Get3DtracksFromLarlite()" << std::endl;
        if(_vertexTracks.size()!=0)_vertexTracks.clear();
        for(size_t itrack = 0; itrack<_vertexLarliteTracks.size();itrack++){
            std::vector<TVector3> thisTrack(_vertexLarliteTracks[itrack].NumberTrajectoryPoints());
            for(size_t iNode=0;iNode<_vertexLarliteTracks[itrack].NumberTrajectoryPoints();iNode++){
                thisTrack[iNode]=_vertexLarliteTracks[itrack].LocationAtPoint(iNode);
            }
            _vertexTracks.push_back(thisTrack);
        }
    }
    //______________________________________________________
    bool AStarTracker::IsGoodTrack(){
        for(auto trackinfo:_SelectableTracks){
            if(_run    != trackinfo[0])continue;
            if(_subrun != trackinfo[1])continue;
            if(_event  != trackinfo[2])continue;
            if(_track  != trackinfo[3])continue;
            return true;
        }
        return false;
    }
    //______________________________________________________
    void AStarTracker::FeedVtxGoodness(std::vector<bool> goodTracks_v){
        if(goodTracks_v.size() != 9)return;
        _missingTrack          = goodTracks_v[0];
        _nothingReconstructed  = goodTracks_v[1];
        _tooShortDeadWire      = goodTracks_v[2];
        _tooShortFaintTrack    = goodTracks_v[3];
        _tooManyTracksAtVertex = goodTracks_v[4];
        _possibleCosmic        = goodTracks_v[5];
        _possiblyCrossing      = goodTracks_v[6];
        _branchingTracks       = goodTracks_v[7];
        _jumpingTracks         = goodTracks_v[8];
    }
    //______________________________________________________
    void AStarTracker::MakeVertexTrack(){
        tellMe("MakeVertexTracks()",0);
        if(_vertexLarliteTracks.size()!=0)_vertexLarliteTracks.clear();
        for(size_t itrack=0;itrack<_vertexTracks.size();itrack++){
            _3DTrack=_vertexTracks[itrack];
            MakeTrack();
            _vertexLarliteTracks.push_back(_thisLarliteTrack);
        }
    }
    //______________________________________________________
    void AStarTracker::MakeTrack(){
        tellMe("MakeTrack()",0);
        _thisLarliteTrack.clear_data();
        ComputeNewdQdX();
        std::vector<std::vector<double> > dQdXperPlaneCorr;// dQdXperPlaneCorr[iPlane][iNode]
        double ion[3] = {0,0,0};
        for(size_t iNode = 0;iNode < _3DTrack.size();iNode++){
            int Nnearby[3] = {0,0,0};
            double x_proj_inode[3], y_proj_inode[3];
            for(size_t iPlane=0;iPlane<3;iPlane++){
                ProjectTo3D(hit_image_v[iPlane].meta(),
                            _3DTrack[iNode].X(),
                            _3DTrack[iNode].Y(),
                            _3DTrack[iNode].Z(),
                            0,
                            iPlane,
                            x_proj_inode[iPlane],
                            y_proj_inode[iPlane]);
            }
            for(size_t jNode = 0;jNode<_3DTrack.size();jNode++){
                double drpoint[3];
                double x_proj_jnode[3], y_proj_jnode[3];
                for(size_t iPlane=0;iPlane<3;iPlane++){
                    ProjectTo3D(hit_image_v[iPlane].meta(),
                                _3DTrack[jNode].X(),
                                _3DTrack[jNode].Y(),
                                _3DTrack[jNode].Z(),
                                0,
                                iPlane,
                                x_proj_jnode[iPlane],
                                y_proj_jnode[iPlane]);
                    drpoint[iPlane] = sqrt( pow( x_proj_inode[iPlane]-x_proj_jnode[iPlane] ,2)+pow(y_proj_inode[iPlane]-y_proj_jnode[iPlane],2) );
                    if(drpoint[iPlane] < 2)Nnearby[iPlane]++;
                }
            }
            TVector3 LocalVector;
            if(iNode!=0 && iNode!= _3DTrack.size()-1) LocalVector = _3DTrack[iNode+1]-_3DTrack[iNode-1];
            else if(iNode==0) LocalVector = _3DTrack[iNode+1]-_3DTrack[iNode];
            else LocalVector = _3DTrack[iNode]-_3DTrack[iNode-1];
            for(size_t iPlane = 0;iPlane<3;iPlane++){
                ion[iPlane] = dQdXperPlane_v[iPlane][iNode]/(Nnearby[iPlane]*LocalVector.Mag());
                dQdXperPlane_v[iPlane][iNode] = ion[iPlane];
            }
        }

        for(size_t iPlane = 0;iPlane<3;iPlane++){
            _thisLarliteTrack.add_dqdx(dQdXperPlane_v[iPlane]);
        }
        for(size_t iNode = 0;iNode<_3DTrack.size();iNode++){
            _thisLarliteTrack.add_vertex(_3DTrack[iNode]);
        }
        for(size_t iNode = 0;iNode<_3DTrack.size();iNode++){
            if(iNode<_3DTrack.size()-1){
                _thisLarliteTrack.add_direction(_3DTrack[iNode+1]-_3DTrack[iNode]);
            }
            else{
                _thisLarliteTrack.add_direction(_3DTrack[iNode]-_3DTrack[iNode-1]);
            }
        }
    }
    //______________________________________________________
    void AStarTracker::DumpTrack(){
        tellMe("_________",0);
        tellMe("DumpTrack",0);
        tellMe("_________",0);
        for(auto point:_3DTrack){
            tellMe(Form("(%.1f,%.1f,%.1f)",point.X(),point.Y(),point.Z()),0);
        }
        tellMe("_________",0);
    }
    //______________________________________________________
    void AStarTracker::TellMeRecoedPath(){
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            for(size_t i = 0;i<RecoedPath.size();i++){
                tellMe(Form("node %02zu \t plane[%zu] : %d, %d, %.1f ||| tyz : %.1f, %.1f, %.1f",i,iPlane,RecoedPath[i].row,RecoedPath[i].cols[iPlane],RecoedPath[i].pixval[iPlane],RecoedPath[i].tyz[0],RecoedPath[i].tyz[1],RecoedPath[i].tyz[2]),0);
            }
        }
    }
    
}
#endif
