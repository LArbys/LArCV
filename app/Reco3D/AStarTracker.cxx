#ifndef LARLITE_AStarTracker_CXX
#define LARLITE_AStarTracker_CXX

#include "AStarTracker.h"
#include "DataFormat/track.h"
#include "DataFormat/hit.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/wire.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/TimeService.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TLine.h"
#include "TF1.h"
#include "TVector3.h"

//#include "LArCV/core/DataFormat/ChStatus.h"
#include "AStarUtils.h"

#include "AStar3DAlgo.h"
#include "AStar3DAlgoProton.h"

namespace larcv {
    
    double AStarTracker::X2Tick(double x, size_t plane) const {

        //auto ts = larutil::TimeService::GetME();
        auto larp = larutil::LArProperties::GetME();
        //std::cout << "ts->TriggerOffsetTPC() = " << ts->TriggerOffsetTPC() << std::endl;
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
        //std::cout << "tick : " << tick;
        //auto ts = larutil::TimeService::GetME();
        auto larp = larutil::LArProperties::GetME();
        // remove tick offset due to plane
        /*if(plane >= 1) {tick += 0.3/larp->DriftVelocity(larp->Efield(1));}
         if(plane >= 2) {
         tick += 0.3/larp->DriftVelocity(larp->Efield(2));
         }*/
        //tick/=2.;
        //double x = (tick+ts->TriggerOffsetTPC()+_speedOffset)*larp->DriftVelocity();
        //std::cout << "  =>  x : " << x << "  => new tick : " << X2Tick(x,plane) << std::endl;

        return ((tick-3200)*larp->DriftVelocity())*0.5;
    }
    //______________________________________________________
    void AStarTracker::tellMe(std::string s, int verboseMin = 0){
        if(_verbose >= verboseMin) std::cout << s << std::endl;
    }
    //______________________________________________________
    void AStarTracker::SetTimeAndWireBoundsProtonsErez(){

        int itrack = 0;
        for(int tracknum = 0;tracknum<_SelectableTracks.size(); tracknum++){
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
        double ImageMargin = 50.;
        double fractionImageMargin = 0.25;
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
        for(int iPlane = 0;iPlane<3;iPlane++){

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
        double ImageMargin = 50.;
        double fractionImageMargin = 0.25;
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

        for(int iPlane = 0;iPlane<3;iPlane++){

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
        double ImageMargin = 50.;
        double fractionImageMargin = 0.25;
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
            for(int iPoint = 0;iPoint<points.size();iPoint++){
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
        double ImageMargin = 50.;
        double fractionImageMargin = 0.25;
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
            for(int iPoint = 0;iPoint<points.size();iPoint++){

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
        double ImageMargin = 50.;
        double fractionImageMargin = 0.25;
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
    bool AStarTracker::CheckEndPointsInVolume(TVector3 point){
        bool endpointInRange = true;
        if(point.X() <  0    ){endpointInRange = false;}
        if(point.X() >  260  ){endpointInRange = false;}

        if(point.Y() < -115  ){endpointInRange = false;}
        if(point.Y() >  115  ){endpointInRange = false;}

        if(point.Z() <  1    ){endpointInRange = false;}
        if(point.Z() >  1036 ){endpointInRange = false;}

        return endpointInRange;
    }
    //______________________________________________________
    bool AStarTracker::initialize() {
        ReadSplineFile();
        ReadProtonTrackFile();
        _eventTreated = 0;
        _eventSuccess = 0;
        time_bounds.reserve(3);
        wire_bounds.reserve(3);
        hit_image_v.reserve(3);
        hDistance2Hit = new TH1D("hDistance2Hit","hDistance2Hit",100,0,50);
        if(_DrawOutputs){
            hdQdx = new TH1D("hdQdx","hdQdx",100,0,100);
            hdQdxEntries = new TH1D("hdQdxEntries","hdQdxEntries",hdQdx->GetNbinsX(),hdQdx->GetXaxis()->GetXmin(),hdQdx->GetXaxis()->GetXmax());
            hdQdX2D = new TH2D("hdQdX2D","hdQdX2D",200,0,100,300,0,300);
        }
        return true;
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
        tellMe("getting start and end points",1);
        std::vector<int> start_cols(3);
        std::vector<int> end_cols(3);
        int start_row, end_row;

        //tellMe("ok here");

        for(size_t iPlane=0;iPlane<3;iPlane++){

            if(hit_image_v[iPlane].meta().width()==0)continue;

            double x_pixel, y_pixel;
            ProjectTo3D(hit_image_v[iPlane].meta(),start_pt.X(),start_pt.Y(),start_pt.Z(),0,iPlane,x_pixel,y_pixel);
            double wireProjStartPt = x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x;
            start_cols[iPlane] = hit_image_v[iPlane].meta().col(wireProjStartPt);
            start_row = y_pixel;

            ProjectTo3D(hit_image_v[iPlane].meta(),end_pt.X(),end_pt.Y(),end_pt.Z(),0,iPlane,x_pixel,y_pixel);
            double wireProjEndPt   = x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x;
            end_cols[iPlane]   = hit_image_v[iPlane].meta().col(wireProjEndPt);
            end_row = y_pixel;
        }

        //_______________________________________________
        // Configure A* algo
        //-----------------------------------------------
        larcv::AStar3DAlgoConfig config;
        config.accept_badch_nodes = true;
        config.astar_threshold.resize(3,1);    // min value for a pixel to be considered non 0
        config.astar_neighborhood.resize(3,3); //can jump over n empty pixels in all planes
        config.astar_start_padding = 3;       // allowed region around the start point
        config.astar_end_padding = 3;         // allowed region around the end point
        config.lattice_padding = 5;            // margin around the edges
        config.min_nplanes_w_hitpixel = 1;     // minimum allowed coincidences between non 0 pixels across planes
        config.restrict_path = false; // do I want to restrict to a cylinder around the strainght line of radius defined bellow ?
        config.path_restriction_radius = 30.0;

        //_______________________________________________
        // Define A* algo
        //-----------------------------------------------
        larcv::AStar3DAlgoProton algo( config );
        algo.setVerbose(0);
        algo.setPixelValueEsitmation(true);

        tellMe("starting A*",0);
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            tellMe(Form("plane %zu %zu rows and %zu cols before findpath", iPlane,hit_image_v[iPlane].meta().rows(),hit_image_v[iPlane].meta().cols()),1);
        }
        RecoedPath = algo.findpath( hit_image_v, chstatus_image_v, tag_image_v, start_row, end_row, start_cols, end_cols, goal_reached );
        tellMe("A* done",1);
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
    bool AStarTracker::finalize() {
        tellMe("Finalizing",0);
        tellMe(Form("efficiency : %d / %d = %.2f",_eventSuccess,_eventTreated,_eventSuccess*100./_eventTreated),0);

        if(hDistance2Hit->GetEntries() != 0){
            TCanvas *cDist = new TCanvas();
            hDistance2Hit->Draw();
            cDist->SetLogy();
            //hDistance2Hit->Write();
            tellMe(Form("hDistance2Hit mean: %.2f , RMS : %.2f",hDistance2Hit->GetMean(),hDistance2Hit->GetRMS()),0);
            tellMe(Form("entries : %.1f",hDistance2Hit->GetEntries()),0);
            cDist->SaveAs("DistanceReco2Hit.pdf");
        }

        if(_DrawOutputs){

            if(hdQdx->GetEntries() > 1){
                TCanvas *cdQdX = new TCanvas("cdQdX","cdQdX",800,600);
                for(int i = 0;i<hdQdx->GetNbinsX(); i++){
                    if(hdQdxEntries->GetBinContent(i+1) == 0) continue;
                    if(hdQdx->GetBinContent(i+1) == 0) continue;
                    double a = hdQdx->GetBinContent(i+1);
                    hdQdx->SetBinContent(i+1,a*1./hdQdxEntries->GetBinContent(i+1));
                    hdQdx->SetBinError(i+1,0.001*hdQdx->GetBinContent(i+1));

                }
                hdQdx->Draw();
                cdQdX->SaveAs("cdQdX.pdf");
            }
            if(hdQdX2D->GetEntries() > 1){
                TCanvas *cdQdX2D = new TCanvas("cdQdX2D","cdQdX2D",800,600);
                hdQdX2D->Draw("colz");
                cdQdX2D->SaveAs("cdQdX2D.pdf");
                cdQdX2D->SaveAs("cdQdX2D.root");
            }
        }
        return true;
    }
    //______________________________________________________
    TVector3 AStarTracker::CheckEndPoints(TVector3 point){
        tellMe(Form("start point : (%f,%f,%f)",point.X(),point.Y(),point.Z()),2);
        double dR = 0.1;
        double closeEnough = 4;

        double minDist = EvalMinDist(point);
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
                double dist = EvalMinDist(neighbour);
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
            for(int icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                for(int irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    if(hit_image_v[iPlane].pixel(irow,icol) == 0)continue;
                    if(std::abs(icol-pointCol) > 10 || std::abs(irow-pointRow) > 10) continue;
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
            for(int icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                for(int irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    if(icol != endPix[iPlane].first && irow != endPix[iPlane].second)continue;
                    if(hit_image_v[iPlane].pixel(irow,icol) == 0)continue;
                    if(std::abs(icol-pointCol) > 10 || std::abs(irow-pointRow) > 10) continue;
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
    void AStarTracker::Make3DpointList(){
        double nodeX,nodeY,nodeZ;
        std::vector<TVector3> list3D;
        //for(size_t iNode=RecoedPath.size()-1;iNode < RecoedPath.size();iNode--){
        for(size_t iNode=0;iNode < RecoedPath.size();iNode++){
            double time = hit_image_v[0].meta().br().y+(RecoedPath[iNode].row)*hit_image_v[0].meta().pixel_height();
            nodeX = Tick2X(time,0);
            nodeY = RecoedPath.at(iNode).tyz[1];
            nodeZ = RecoedPath.at(iNode).tyz[2];
            TVector3 node(nodeX,nodeY,nodeZ);
            list3D.push_back(node);
        }
        _3DTrack = list3D;
    }
    //______________________________________________________
    void AStarTracker::ComputedQdX(){
        if(_3DTrack.size() == 0){tellMe("ERROR, run the 3D reconstruction first",0);return;}
        if(_dQdx.size() != 0)_dQdx.clear();
        double Npx = 5;
        double dist,xp,yp,alpha;
        for(int iNode = 0;iNode<_3DTrack.size()-1;iNode++){
            std::vector<double> node_dQdx(3);
            for(int iPlane = 0;iPlane<3;iPlane++){
                double localdQdx = 0;
                int NpxLocdQdX = 0;
                double X1,Y1,X2,Y2;
                ProjectTo3D(hit_image_v[iPlane].meta(),_3DTrack[iNode].X(), _3DTrack[iNode].Y(), _3DTrack[iNode].Z(),0, iPlane, X1,Y1);
                ProjectTo3D(hit_image_v[iPlane].meta(),_3DTrack[iNode+1].X(), _3DTrack[iNode+1].Y(), _3DTrack[iNode+1].Z(),0, iPlane, X2,Y2);
                if(X1 == X2){X1 -= 0.0001;}
                if(Y1 == Y2){Y1 -= 0.0001;}
                TVector3 A(X1,Y1,0); // projection of node iNode on plane iPlane
                TVector3 B(X2,Y2,0); // projection of node iNode+1 on plane iPlane

                for(int icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                    for(int irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
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
            //std::cout << "construct meta" << std::endl;
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
    void AStarTracker::DrawTrack(){
        TH2D *hImage[3];
        //TH2D *hTrack[3];
        //TH2D *hIntegral[3];
        TGraph *gTrack[3];
        TGraph *gStartNend[3];

        for(size_t iPlane=0;iPlane<3;iPlane++){
            gTrack[iPlane] = new TGraph();
            hImage[iPlane] = new TH2D(Form("hImage_%d_%d_%d_%d_%zu",_run,_subrun,_event,_track,iPlane),
                                      Form("hImage_%d_%d_%d_%d_%zu;wire;time",_run,_subrun,_event,_track,iPlane),
                                      hit_image_v[iPlane].meta().cols(),
                                      hit_image_v[iPlane].meta().tl().x,
                                      hit_image_v[iPlane].meta().tl().x+hit_image_v[iPlane].meta().width(),
                                      hit_image_v[iPlane].meta().rows(),
                                      hit_image_v[iPlane].meta().br().y,
                                      hit_image_v[iPlane].meta().br().y+hit_image_v[iPlane].meta().height());

            //hImage[iPlane] = new TH2D(Form("hImage_%d_%d_%d_%d_%zu",_run,_subrun,_event,_track,iPlane),Form("hImage_%d_%d_%d_%d_%zu;wire;time",_run,_subrun,_event,_track,iPlane),hit_image_v[iPlane].meta().cols(),0,hit_image_v[iPlane].meta().cols(),hit_image_v[iPlane].meta().rows(),0,hit_image_v[iPlane].meta().rows());
            //hIntegral[iPlane] = new TH2D(Form("hIntegral_%d_%d_%d_%d_%zu",_run,_subrun,_event,_track,iPlane),Form("hIntegral_%d_%d_%d_%d_%zu",_run,_subrun,_event,_track,iPlane),hit_image_v[iPlane].meta().cols(),0,hit_image_v[iPlane].meta().cols(),hit_image_v[iPlane].meta().rows(),0,hit_image_v[iPlane].meta().rows());

            for(int icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                for(int irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    hImage[iPlane]->SetBinContent(icol+1,irow+1,hit_image_v[iPlane].pixel(irow,icol));
                }
            }

            gStartNend[iPlane] = new TGraph();
            double x_pixel, y_pixel;
            ProjectTo3D(hit_image_v[iPlane].meta(),
                      start_pt.X(),
                      start_pt.Y(),
                      start_pt.Z(),
                      0,
                      iPlane,
                      x_pixel,y_pixel); // y_pixel is time
            gStartNend[iPlane]->SetPoint(0,x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
            ProjectTo3D(hit_image_v[iPlane].meta(),end_pt.X(),end_pt.Y(),end_pt.Z(),0,iPlane,x_pixel,y_pixel); // y_pixel is time
            gStartNend[iPlane]->SetPoint(1,x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);




            /*for(size_t iNode=0;iNode<recoTrack.NumberTrajectoryPoints();iNode++){
                ProjectTo3D(hit_image_v[iPlane].meta(),
                          recoTrack.LocationAtPoint(iNode).X(),
                          recoTrack.LocationAtPoint(iNode).Y(),
                          recoTrack.LocationAtPoint(iNode).Z(),
                          0,
                          iPlane,
                          x_pixel,y_pixel); // y_pixel is time
                gTrack[iPlane]->SetPoint(iNode,x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
            }*/

            /*hTrack[iPlane] = new TH2D(Form("hTrack_%d_%d_%d_%d_%zu",_run,_subrun,_event,_track,iPlane),
                                      Form("hTrack_%d_%d_%d_%d_%zu;wire;time",_run,_subrun,_event,_track,iPlane),
                                      hit_image_v[iPlane].meta().cols(),
                                      hit_image_v[iPlane].meta().tl().x,
                                      hit_image_v[iPlane].meta().tl().x+hit_image_v[iPlane].meta().width(),
                                      hit_image_v[iPlane].meta().rows(),
                                      hit_image_v[iPlane].meta().br().y,
                                      hit_image_v[iPlane].meta().br().y+hit_image_v[iPlane].meta().height());*/

            for(size_t iNode = 0;iNode<_3DTrack.size();iNode++){
                ProjectTo3D(hit_image_v[iPlane].meta(),_3DTrack[iNode].X(),_3DTrack[iNode].Y(),_3DTrack[iNode].Z(),0,iPlane,x_pixel,y_pixel); // y_pixel is time

                //gTrack[iPlane]->SetPoint(iNode,hit_image_v[iPlane].meta().tl().x+(RecoedPath[iNode].cols[iPlane]+0.5)*hit_image_v[iPlane].meta().pixel_width(),hit_image_v[iPlane].meta().br().y+(RecoedPath[iNode].row+0.5)*hit_image_v[iPlane].meta().pixel_height());
                gTrack[iPlane]->SetPoint(iNode,x_pixel*hit_image_v[iPlane].meta().pixel_width()+hit_image_v[iPlane].meta().tl().x, y_pixel*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y);
                //hTrack[iPlane]->SetBinContent(RecoedPath[iNode].cols[iPlane]+1,RecoedPath[iNode].row+1,1);
            }
        }

        tellMe("computing dqdx",1);
        /*
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            //std::cout << "\t" << iPlane << std::endl;

            for(size_t iNode=0;iNode<recoTrack.NumberTrajectoryPoints()-1;iNode++){
                X1 = hit_image_v[iPlane].meta().col(larutil::GeometryHelper::GetME()->Point_3Dto2D(recoTrack.LocationAtPoint(iNode  ),iPlane).w/0.3);
                Y1 = hit_image_v[iPlane].meta().row(X2Tick(recoTrack.LocationAtPoint(iNode  ).X(),iPlane));

                X2 = hit_image_v[iPlane].meta().col(larutil::GeometryHelper::GetME()->Point_3Dto2D(recoTrack.LocationAtPoint(iNode+1),iPlane).w/0.3);
                Y2 = hit_image_v[iPlane].meta().row(X2Tick(recoTrack.LocationAtPoint(iNode+1).X(),iPlane));

                if(X1 == X2){X1 -= 0.0001;}
                if(Y1 == Y2){Y1 -= 0.0001;}

                for(int icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                    for(int irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                        //if(hit_image_v[iPlane].pixel(irow,icol) == 0)continue;
                        //----------------------------
                        // first, check if the point can belong to this segment
                        // measure the distance to the line
                        // check the angles
                        //----------------------------
                        double theta0; // angle with which the segment [iNode, iNode+1] is ssen by the current point
                        double theta1; // angle with which the segment [jNode, jNode+1] is seen by the current point
                        X0 = icol;//+0.5;
                        Y0 = irow;//+0.5;
                        if(X0 == X1 || X0 == X2){X0 += 0.0001;}
                        if(Y0 == Y1 || Y0 == Y2){Y0 += 0.0001;}
                        alpha = ( (X2-X1)*(X0-X1)+(Y2-Y1)*(Y0-Y1) )/( pow(X2-X1,2)+pow(Y2-Y1,2) ); // normalized scalar product of the 2 vectors
                        //if(alpha <= -0.25 || alpha >= 1.25) continue;
                        xp = X1+alpha*(X2-X1);
                        yp = Y1+alpha*(Y2-Y1);
                        dist = sqrt(pow(xp-X0,2)+pow(yp-Y0,2));
                        double dist2point = std::min( sqrt(pow(X0-X1,2)+pow(Y0-Y1,2)), sqrt(pow(X0-X2,2)+pow(Y0-Y2,2)) );

                        if( alpha >= 0 && alpha <= 1 && dist > Npx ) continue;
                        if( (alpha < 0 || alpha > 1) && dist2point > Npx ) continue;

                        theta0 = std::abs(std::acos(((X1-X0)*(X2-X0)+(Y1-Y0)*(Y2-Y0))/(sqrt(pow(X1-X0,2)+pow(Y1-Y0,2))*sqrt(pow(X2-X0,2)+pow(Y2-Y0,2)))));
                        // check all other nodes, if I find an angle bigger with a distance lower than Npx, reject the point for the current segment

                        bool bestCandidate = true;
                        for(size_t jNode=0;jNode<recoTrack.NumberTrajectoryPoints()-1;jNode++){
                            if(jNode==iNode)continue;
                            double x1,y1,x2,y2, alpha2, xp2,yp2, dist2;
                            x1 = hit_image_v[iPlane].meta().col(larutil::GeometryHelper::GetME()->Point_3Dto2D(recoTrack.LocationAtPoint(jNode  ),iPlane).w/0.3);
                            x2 = hit_image_v[iPlane].meta().col(larutil::GeometryHelper::GetME()->Point_3Dto2D(recoTrack.LocationAtPoint(jNode+1),iPlane).w/0.3);

                            y1 = hit_image_v[iPlane].meta().row(X2Tick(recoTrack.LocationAtPoint(jNode  ).X(),iPlane));
                            y2 = hit_image_v[iPlane].meta().row(X2Tick(recoTrack.LocationAtPoint(jNode+1).X(),iPlane));

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
                        }
                        if(!bestCandidate)continue;


                        hIntegral[iPlane]->Fill(X0,Y0);
                    }
                }
            }
        }
         */

        TCanvas *c = new TCanvas(Form("c_%d_%d_%d_%d",_run,_subrun,_event,_track),Form("c_%d_%d_%d_%d",_run,_subrun,_event,_track),450,150);
        c->Divide(3,1);
        for(int iPlane=0;iPlane<3;iPlane++){
            c->cd(iPlane+1);
            //hIntegral[iPlane]->Draw();
            hImage[iPlane]->Draw("colz");
            //hTrack[iPlane]->Draw("same colz");
            gTrack[iPlane]->SetLineColor(2);
            gTrack[iPlane]->Draw("same LP");
            gStartNend[iPlane]->SetMarkerStyle(20);
            gStartNend[iPlane]->SetMarkerSize(0.25);
            gStartNend[iPlane]->Draw("same P");
        }
        c->SaveAs(Form("%s.pdf",c->GetName()));
        for(int iPlane = 0;iPlane<3;iPlane++){
            hImage[iPlane]->Delete();
            //hTrack[iPlane]->Delete();
            //hIntegral[iPlane]->Delete();
            gTrack[iPlane]->Delete();
            gStartNend[iPlane]->Delete();
        }
        tellMe("histogram and gStartNend deleted",2);
    }
    //______________________________________________________
    void AStarTracker::DrawROI(){
        TH2D *hImage[3];
        TGraph *gStartNend[3];

        for(size_t iPlane=0;iPlane<3;iPlane++){
            hImage[iPlane] = new TH2D(Form("hImage_%d_%d_%d_%d_%zu",_run,_subrun,_event,_track,iPlane),
                                      Form("hImage_%d_%d_%d_%d_%zu;wire;time",_run,_subrun,_event,_track,iPlane),
                                      hit_image_v[iPlane].meta().cols(),
                                      hit_image_v[iPlane].meta().tl().x,
                                      hit_image_v[iPlane].meta().tl().x+hit_image_v[iPlane].meta().width(),
                                      hit_image_v[iPlane].meta().rows(),
                                      hit_image_v[iPlane].meta().br().y,
                                      hit_image_v[iPlane].meta().br().y+hit_image_v[iPlane].meta().height());

            for(int icol=0;icol<hit_image_v[iPlane].meta().cols();icol++){
                for(int irow=0;irow<hit_image_v[iPlane].meta().rows();irow++){
                    hImage[iPlane]->SetBinContent(icol+1,irow+1,hit_image_v[iPlane].pixel(irow,icol));
                }
            }

            gStartNend[iPlane] = new TGraph();
            double x_pixel_st, y_pixel_st,x_pixel_end, y_pixel_end;
            double Xstart,Ystart,Xend,Yend;

            ProjectTo3D(hit_image_v[iPlane].meta(),
                      start_pt.X(),start_pt.Y(),start_pt.Z(),
                      0,iPlane,x_pixel_st,y_pixel_st);

            Xstart = x_pixel_st*hit_image_v[iPlane].meta().pixel_width() +hit_image_v[iPlane].meta().tl().x;
            Ystart = y_pixel_st*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y;
            gStartNend[iPlane]->SetPoint(0,Xstart,Ystart);

            ProjectTo3D(hit_image_v[iPlane].meta(),
                      end_pt.X(),end_pt.Y(),end_pt.Z(),
                      0,iPlane,x_pixel_end,y_pixel_end);

            Xend = x_pixel_end*hit_image_v[iPlane].meta().pixel_width() +hit_image_v[iPlane].meta().tl().x;
            Yend = y_pixel_end*hit_image_v[iPlane].meta().pixel_height()+hit_image_v[iPlane].meta().br().y;
            gStartNend[iPlane]->SetPoint(1,Xend,Yend);


        }

        TCanvas *c = new TCanvas(Form("ROI_%d_%d_%d_%d",_run,_subrun,_event,_track),Form("ROI_%d_%d_%d_%d",_run,_subrun,_event,_track),450,150);
        c->Divide(3,1);
        for(int iPlane=0;iPlane<3;iPlane++){
            c->cd(iPlane+1);
            hImage[iPlane]->Draw("colz");
            //gStartNend[iPlane]->SetMarkerStyle(4);
            //gStartNend[iPlane]->SetMarkerSize(0.25);
            gStartNend[iPlane]->Draw("same P");
        }
        c->SaveAs(Form("%s.pdf",c->GetName()));

        for(int iPlane = 0;iPlane<3;iPlane++){
            hImage[iPlane]->Delete();
            gStartNend[iPlane]->Delete();
        }
        tellMe("histogram and gStartNend deleted",0);

    }
    //______________________________________________________
    std::vector<std::pair<int, int> > AStarTracker::GetWireTimeProjection(TVector3 point){
        std::vector<std::pair<int, int> > projection(3);
        for(int iPlane = 0;iPlane<3;iPlane++){
            std::pair<int, int> planeProj;
            planeProj.first = (int)(larutil::GeometryHelper::GetME()->Point_3Dto2D(point,iPlane).w / 0.3);
            planeProj.second = (int)(X2Tick(point.X(),iPlane));
            projection[iPlane] = planeProj;
        }
        return projection;
    }
    //______________________________________________________
    void AStarTracker::ReadSplineFile(){
        TFile *fSplines = TFile::Open("Proton_Muon_Range_dEdx_LAr_TSplines.root","READ");
        sMuonRange2T   = (TSpline3*)fSplines->Get("sMuonRange2T");
        sMuonT2dEdx    = (TSpline3*)fSplines->Get("sMuonT2dEdx");
        sProtonRange2T = (TSpline3*)fSplines->Get("sProtonRange2T");
        sProtonT2dEdx  = (TSpline3*)fSplines->Get("sProtonT2dEdx");
        fSplines->Close();
    }
    //______________________________________________________
    double AStarTracker::GetEnergy(std::string partType, double Length = -1){
        if(partType == "proton"){
            if(Length == -1){return sProtonRange2T->Eval(_Length3D);}
            return sProtonRange2T->Eval(Length);
        }
        else if(partType == "muon"){
            if(Length == -1){return sMuonRange2T->Eval(_Length3D);}
            return sMuonRange2T->Eval(Length);
        }
        else{tellMe("ERROR, must be proton or muon",0); return -1;}
    }
    //______________________________________________________
    double AStarTracker::GetTotalDepositedCharge(){
        if(_3DTrack.size() == 0){tellMe("ERROR, run the 3D reconstruction first",0);return -1;}
        if(_dQdx.size() == 0){tellMe("ERROR, run the dqdx reco first",0);return -1;}
        double Qtot = 0;
        for(int iNode = 0;iNode<_dQdx.size();iNode++){
            double dqdx = 0;
            int Nplanes = 0;
            for(int iPlane = 0;iPlane<3;iPlane++){
                dqdx+=_dQdx[iNode][iPlane];
                if(_dQdx[iNode][iPlane] != 0)Nplanes++;
            }
            if(Nplanes!=0)dqdx*=3./Nplanes;
            Qtot+=dqdx;
        }
        return Qtot;
    }
    //______________________________________________________
    bool IsInSegment(TVector3 A, TVector3 B, TVector3 C){
        if((C-A).Dot(B-A)/(B-A).Mag() > 0 && (C-A).Dot(B-A)/(B-A).Mag() < 1) return true;
        else return false;
    }
    //______________________________________________________
    void AStarTracker::RegularizeTrack(){
        std::vector<TVector3> newPoints;
        int oldNumPoints = _3DTrack.size();
        bool removedPoint = true;
        int iteration = 0;
        double distMax = 0.5;
        double distLastPoint = 1;
        while(removedPoint == true && iteration<100){
            removedPoint =false;
            iteration++;
            std::cout << "iteration #" << iteration << std::endl;
            newPoints.clear();
            newPoints.push_back(_3DTrack[0]);
            for(int iNode = 1;iNode<_3DTrack.size()-1;iNode++){
                double dist2line = GetDist2line(newPoints[newPoints.size()-1],_3DTrack[iNode+1],_3DTrack[iNode]);
                double dist2point = (_3DTrack[iNode]-newPoints[newPoints.size()-1]).Mag();
                if(dist2line < distMax && dist2point < distLastPoint){
                    //std::cout << "removing point, iteration #" << iteration << std::endl;
                    removedPoint = true;
                    continue;
                }
                else newPoints.push_back(_3DTrack[iNode]);
            }
            newPoints.push_back(_3DTrack[_3DTrack.size()-1]);
            _3DTrack = newPoints;
        }
        std::cout << "old number of points : " << oldNumPoints << " and new one : " << _3DTrack.size() << std::endl;
    }
    //______________________________________________________
    double AStarTracker::GetDist2line(TVector3 A, TVector3 B, TVector3 C){
        TVector3 CrossProd = (B-A).Cross(C-A);
        TVector3 U = B-A;
        return CrossProd.Mag()/U.Mag();
    }
    //______________________________________________________
    void AStarTracker::ComputeLength(){
        _Length3D = 0;
        for(int iNode = 0;iNode<_3DTrack.size()-1;iNode++){
            _Length3D+=(_3DTrack[iNode+1]-_3DTrack[iNode]).Mag();
        }
    }
    
    //
    //
    // When reading proton file
    //
    void AStarTracker::ReadProtonTrackFile(){
        _SelectableTracks.clear();
        std::vector<int> trackinfo(16);
        std::ifstream file("passedGBDT_extBNB_AnalysisTrees_cosmic_trained_only_on_mc_score_0.99.csv");
        if(!file){std::cout << "ERROR, could not open file of tracks to sort through" << std::endl;return;}
        std::string firstline;
        getline(file, firstline);
        bool goOn = true;
        int Run,SubRun,Event,TrackID,WireUMin,TimeUMin,WireUMax,TimeUMax,WireVMin,TimeVMin,WireVMax,TimeVMax,WireYMin,TimeYMin,WireYMax,TimeYMax;
        double BDTScore;
        while(goOn){
            file >> Run >> SubRun >> Event >> TrackID >> WireUMin >> TimeUMin >> WireUMax >> TimeUMax >> WireVMin >> TimeVMin >> WireVMax >> TimeVMax >> WireYMin >> TimeYMin >> WireYMax >> TimeYMax >> BDTScore;

            trackinfo[0] = Run;
            trackinfo[1] = SubRun;
            trackinfo[2] = Event;
            trackinfo[3] = TrackID;
            trackinfo[4] = WireUMin;
            trackinfo[5] = TimeUMin;
            trackinfo[6] = WireUMax;
            trackinfo[7] = TimeUMax;
            trackinfo[8] = WireVMin;
            trackinfo[9] = TimeVMin;
            trackinfo[10] = WireVMax;
            trackinfo[11] = TimeVMax;
            trackinfo[12] = WireYMin;
            trackinfo[13] = TimeYMin;
            trackinfo[14] = WireYMax;
            trackinfo[15] = TimeYMax;

            _SelectableTracks.push_back(trackinfo);
            if(file.eof()){goOn=false;break;}
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
    void AStarTracker::TellMeRecoedPath(){
        for(size_t iPlane = 0;iPlane<3;iPlane++){
            for(size_t i = 0;i<RecoedPath.size();i++){
                tellMe(Form("node %02zu \t plane[%zu] : %d, %d, %.1f ||| tyz : %.1f, %.1f, %.1f",i,iPlane,RecoedPath[i].row,RecoedPath[i].cols[iPlane],RecoedPath[i].pixval[iPlane],RecoedPath[i].tyz[0],RecoedPath[i].tyz[1],RecoedPath[i].tyz[2]),0);
            }
        }
    }

}
#endif
