#include <iostream>
#include <fstream>

#include <set>

#include "TFile.h"
#include "TTree.h"
#include "TChain.h"

#include <stdexcept>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <iterator>

int main(int argc, const char** argv) {

  assert(argc == 4);

  std::string infile    = argv[1];
  std::string outfile   = argv[2];
  std::string entryfile = argv[3];

  std::ifstream ifs;
  ifs.open(entryfile, std::ifstream::in);

  std::string line;
  std::getline(ifs, line);
  std::istringstream iss(line);
  std::cout << "GOT line=" << line << std::endl;

  std::vector<std::string> sentry_v;
  
  std::copy(std::istream_iterator<std::string>(iss),
	    std::istream_iterator<std::string>(),
	    std::back_inserter(sentry_v));

  std::vector<int> entry_v;
  entry_v.reserve(sentry_v.size());
  for (const auto& sentry : sentry_v)
    entry_v.emplace_back(std::stoi(sentry));

  auto tf = TFile::Open(infile.c_str(),"READ");
  std::set<std::string> key_v;

  for (const auto& key : *tf->GetListOfKeys())
    key_v.insert(std::string(key->GetName()));
  
  tf->Close();
  
  std::vector<TChain*> tc_v;
  tc_v.reserve(key_v.size());

  for (auto& key : key_v) {
    TChain *tc = new TChain(key.c_str());
    tc->Add(infile.c_str());
    tc->SetBranchStatus("*",1);
    tc_v.emplace_back(tc);
  }

  auto tfo = TFile::Open(outfile.c_str(),"RECREATE");
  tfo->cd();

  std::vector<TTree*> tree_v;
  tree_v.reserve(key_v.size());

  for(auto tc : tc_v) {
    auto tree = tc->CloneTree(0);
    tree_v.emplace_back(tree);
  }
  
  for(auto entry : entry_v) {
    std::cout << "@entry=" << entry << std::endl;

    for(auto tc : tc_v)
      tc->GetEntry(entry);

    for(auto tree : tree_v)
      tree->Fill();
  }
  
  for(auto tree : tree_v)
    tree->Write();

  tfo->Close();

  return 0;
}
