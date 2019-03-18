void runDisplay(){
    std::string test21 = "/Users/hourlier/Documents/PostDocMIT/Research/MicroBooNE/FinalFiles/test21";
    std::string DataPATH = test21+"/overlays/OverlayNumuData";
    std::string outputPATH  = DataPATH+"/displays";
    std::string superaPATH  = DataPATH+"/supera";
    std::string larlitePATH = DataPATH+"/tracker_reco";
    std::string anaoutPATH  = DataPATH+"/tracker_anaout";
    std::string selectedVtx = test21+"/eventList.txt";

    std::cout << "Load target files" << std::endl;
    ifstream rseFile("/Users/hourlier/Documents/PostDocMIT/Research/MicroBooNE/FinalFiles/test21/eventList.txt");
    bool GoOn=true;
    int run, subrun, event;
    int vtx;
    std::vector<int> rse(3);
    char coma;
    std::vector< std::vector<int> > targetList;
    while(!(rseFile.eof()) && GoOn==true){
        rseFile >> run >> coma >> subrun >> coma >> event >> coma >> vtx;
        rse[0] = run;
        rse[1] = subrun;
        rse[2] = event;
        if(rseFile.eof()){GoOn=false;break;}
        std::cout << run << "\t" << subrun << "\t" << event << std::endl;
        targetList.push_back(rse);
    }
    std::cout << "targetList.size() = " << targetList.size() << std::endl;

    for(int i=0;i<targetList.size();i++){
        run    = targetList[i][0];
        subrun = targetList[i][1];
        event  = targetList[i][2];

        //if(run != 7004 || subrun != 465 || event!= 23256)continue;
        if(run != 7004 || subrun != 1134 || event!= 56703)continue;

        std::cout << run << "\t" << subrun << "\t" << event << std::endl;
        std::string larliteFile  = Form("%s/tracker_reco_*%d_%d_%d*.root",larlitePATH.c_str(),run,subrun,event);
        std::string analysisFile = Form("%s/tracker_anaout_*%d_%d_%d*.root",anaoutPATH.c_str(),run,subrun,event);
        std::string image2DFile  = Form("%s/supera*%d_%d_%d*.root",superaPATH.c_str(),run,subrun,event);
        std::string command = "time python mac/run_display.py "+image2DFile+" "+larliteFile+" "+analysisFile+" "+outputPATH;
        //std::string command = "time python mac/run_display.py "+image2DFile+" "+larliteFile+" "+analysisFile+" "+outputPATH+" "+selectedVtx;
        std::cout << command << std::endl;
        system(Form("%s",command.c_str()));
    }


}
