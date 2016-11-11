#ifndef MCPARTICLETREE_INL
#define MCPARTICLETREE_INL

#include "MCParticleTree.h"

namespace larcv {
	namespace supera {

		template <class T, class U, class V, class W>
		void MCParticleTree<T, U, V, W>::configure(const Config_t& cfg)
		{
			set_verbosity((::larcv::msg::Level_t)(cfg.get<unsigned short>("Verbosity")));
			auto const& cropper_cfg = cfg.get<larcv::supera::Config_t>("Cropper");

			_min_energy_init_mcshower = cfg.get<double>("MCShowerMinEnergyInit");
			_min_energy_deposit_mcshower = cfg.get<double>("MCShowerMinEnergyDeposit");

			_min_energy_init_mctrack = cfg.get<double>("MCTrackMinEnergyInit");
			_min_energy_deposit_mctrack = cfg.get<double>("MCTrackMinEnergyDeposit");

			_min_nplanes = cfg.get<size_t>("MinNPlanes");

			auto pdg_list = cfg.get<std::vector<int> >("SpecialPDGList");
			auto pdg_init_energy = cfg.get<std::vector<double> >("SpecialPDGMinEnergyInit");
			auto pdg_deposit_energy = cfg.get<std::vector<double> >("SpecialPDGMinEnergyDeposit");

			if (pdg_list.size() != pdg_init_energy.size()) {
				LARCV_ERROR() << "SpecialPDGList and SpecialPDGMinEnergyInit must carry the same length vector!" << std::endl;
				throw ::larcv::larbys();
			}

			if (pdg_list.size() != pdg_deposit_energy.size()) {
				LARCV_ERROR() << "SpecialPDGList and SpecialPDGMinEnergyDeposit must carry the same length vector!" << std::endl;
				throw ::larcv::larbys();
			}

			_min_energy_init_pdg.clear();
			_min_energy_deposit_pdg.clear();
			for (size_t i = 0; i < pdg_list.size(); ++i) _min_energy_init_pdg.emplace(pdg_list[i], pdg_init_energy[i]);
			for (size_t i = 0; i < pdg_list.size(); ++i) _min_energy_deposit_pdg.emplace(pdg_list[i], pdg_deposit_energy[i]);

			_cropper.configure(cropper_cfg);
		}

		template <class T, class U, class V, class W>
		void MCParticleTree<T, U, V, W>::DefinePrimary(const std::vector<T>& mctruth_v)
		{
			LARCV_DEBUG() << "start" << std::endl;
			std::map<larcv::Vertex, larcv::ROI> roi_m;
			for (size_t mctruth_index = 0; mctruth_index < mctruth_v.size(); ++mctruth_index) {

				auto const& mctruth = mctruth_v[mctruth_index];

				for (size_t i = 0; i < (size_t)(mctruth.NParticles()); ++i) {

					auto const& mcp = mctruth.GetParticle(i);

					if ( !(mcp.StatusCode() == 0 && (mcp.PdgCode() == 12 || mcp.PdgCode() == -12 ||
					                                 mcp.PdgCode() == 14 || mcp.PdgCode() == -14 ||
					                                 mcp.PdgCode() == 16 || mcp.PdgCode() == -16) ) &&
					        mcp.StatusCode() != 1) continue;
					//if(mcp.StatusCode() != 1) continue;

					auto const& pos = mcp.Position(0);
					auto const& mom = mcp.Momentum(0);
					::larcv::Vertex vtx(pos.X(), pos.Y(), pos.Z(), pos.T());

					if (roi_m.find(vtx) == roi_m.end()) {
						::larcv::ROI roi;
						roi.MCTIndex(mctruth_index);
						if (mctruth.Origin() == 1) roi.Type(::larcv::kROIBNB);
						if (mctruth.Origin() == 2) roi.Type(::larcv::kROICosmic);
						roi.EnergyDeposit(0);
						roi.EnergyInit(0);
						roi.Position(vtx);
						roi_m.insert(std::make_pair(vtx, roi));
					}

					auto& roi = (*(roi_m.find(vtx))).second;

					if (roi.Type() == larcv::kROIBNB) {
						// Neutrino: fill momentum/energy_init info w/ nus
						if (mcp.PdgCode() == 12 || mcp.PdgCode() == -12 ||
						        mcp.PdgCode() == 14 || mcp.PdgCode() == -14 ||
						        mcp.PdgCode() == 16 || mcp.PdgCode() == -16 ) {
							roi.PdgCode(mcp.PdgCode());
							roi.Momentum(mom.X() * 1.e3, mom.Y() * 1.e3, mom.Z() * 1.e3);
							roi.EnergyInit(mom.E() * 1.e3);
							if (mctruth.NeutrinoSet()) {
								roi.NuCurrentType(mctruth.GetNeutrino().CCNC());
								roi.NuInteractionType(mctruth.GetNeutrino().InteractionType());
							}

						}
					} else {
						// Non neutrino: take larger energy particle as representative
						if (roi.EnergyInit() < mom.E() * 1.e3) {
							roi.PdgCode(mcp.PdgCode());
							roi.Momentum(mom.X() * 1.e3, mom.Y() * 1.e3, mom.Z() * 1.e3);
							roi.EnergyInit(mom.E() * 1.e3);
						}
					}
					
					// Define MCPNode
					MCPNode* pnode = new MCPNode( -1, -1, mcp.PdgCode(), vtx.T(), -1, MCPNode::kTruth, MCPNode::kNeutrino );
					// insert into tree
					nodelist.push_back( pnode );
					int idx = nodelist.size()-1;
					idx_primaries.push_back( idx );
					//idx_secondary2primary.insert( std::make_pair<idx,-1> ); // designate as primary
					//idx_vertexmap.insert(std::make_pair<vtx,idx>); // vertex mapped to primary node
				}
			}
			for (auto const& vtx_roi : roi_m) DefinePrimary(vtx_roi.first, vtx_roi.second);
		}


		template <class T, class U, class V, class W>
		void MCParticleTree<T, U, V, W>::DefinePrimaries(const std::vector<U>& mctrack_v, const int time_offset)
		{
		  // we define primaries from MCTrack information
		  LARCV_DEBUG() << "start" << std::endl;

		  for (size_t i = 0; i < mctrack_v.size(); ++i) {
		    auto const& mctrack = mctrack_v[i];
		 
		    if  ( mctrack.TrackID()!=mctrack.MotherTrackID() ) {
		      // this is a secondary
		      continue;
		    }

		    if (mctrack.size() < 2) {
		      LARCV_INFO() << "Ignoring MCTrack G4TrackID " << mctrack.TrackID()
				   << " PdgCode " << mctrack.PdgCode()
				   << " as it has < 2 steps in the detector" << std::endl;
		      continue;
		    }

		    double min_energy_init    = _min_energy_init_mctrack;
		    double min_energy_deposit = _min_energy_deposit_mctrack;

		    // Check if PDG is registered for a special handling
		    if (_min_energy_init_pdg.find(mctrack.PdgCode()) != _min_energy_init_pdg.end()) {
		      min_energy_init    = _min_energy_init_pdg[mctrack.PdgCode()];
		      min_energy_deposit = _min_energy_deposit_pdg[mctrack.PdgCode()];
		    }

		    if ((mctrack.Start().E() < min_energy_init) ) {
		      LARCV_INFO() << "Ignoring MCTrack G4TrackID " << mctrack.TrackID()
				   << " PdgCode " << mctrack.PdgCode()
				   << " as it has too small initial energy " << mctrack.Start().E()
				   << " MeV < " << min_energy_init << " MeV" << std::endl;
		      continue;
		    }
		    if ((mctrack.front().E() - mctrack.back().E()) < min_energy_deposit) {
		      LARCV_INFO() << "Ignoring MCTrack G4TrackID " << mctrack.TrackID()
				   << " PdgCode " << mctrack.PdgCode()
				   << " as it has too small deposit energy " << (mctrack.front().E() - mctrack.back().E())
				   << " MeV < " << min_energy_deposit << " MeV" << std::endl;
		      continue;
		    }

		    ::larcv::Vertex pri_vtx( mctrack.Start().X(), mctrack.Start().Y(), mctrack.Start().Z(), mctrack.Start().T() );

		    auto roi = _cropper.ParticleROI(mctrack,time_offset);
		    roi.MCSTIndex(i);
		    
		    if (roi.BB().size() < _min_nplanes) {
		      LARCV_INFO() << "Skipping Primary ROI as # planes (" << roi.BB().size() << ") < requirement (" << _min_nplanes << std::endl
				   << roi.dump() << std::endl;
		      continue;
		    }

		    // insert roi into vertex map
		    DefinePrimary( pri_vtx, roi );
		  }
		}

		template <class T, class U, class V, class W>
		void MCParticleTree<T, U, V, W>::DefinePrimaries(const std::vector<V>& mcshower_v, const int time_offset)
		{
		  // we define primaries from MCTrack information
		  LARCV_DEBUG() << "start" << std::endl;

		  for (size_t i = 0; i < mcshower_v.size(); ++i) {
		    auto const& mcshower = mcshower_v[i];
		 
		    if  ( mcshower.TrackID()!=mcshower.MotherTrackID() ) {
		      // this is a secondary
		      continue;
		    }

		    if (mcshower.size() < 2) {
		      LARCV_INFO() << "Ignoring Mcshower G4TrackID " << mcshower.TrackID()
				   << " PdgCode " << mcshower.PdgCode()
				   << " as it has < 2 steps in the detector" << std::endl;
		      continue;
		    }

		    double min_energy_init    = _min_energy_init_mcshower;
		    double min_energy_deposit = _min_energy_deposit_mcshower;

		    // Check if PDG is registered for a special handling
		    if (_min_energy_init_pdg.find(mcshower.PdgCode()) != _min_energy_init_pdg.end()) {
		      min_energy_init    = _min_energy_init_pdg[mcshower.PdgCode()];
		      min_energy_deposit = _min_energy_deposit_pdg[mcshower.PdgCode()];
		    }

		    if ((mcshower.Start().E() < min_energy_init) ) {
		      LARCV_INFO() << "Ignoring Mcshower G4TrackID " << mcshower.TrackID()
				   << " PdgCode " << mcshower.PdgCode()
				   << " as it has too small initial energy " << mcshower.Start().E()
				   << " MeV < " << min_energy_init << " MeV" << std::endl;
		      continue;
		    }
		    if ((mcshower.front().E() - mcshower.back().E()) < min_energy_deposit) {
		      LARCV_INFO() << "Ignoring Mcshower G4TrackID " << mcshower.TrackID()
				   << " PdgCode " << mcshower.PdgCode()
				   << " as it has too small deposit energy " << (mcshower.front().E() - mcshower.back().E())
				   << " MeV < " << min_energy_deposit << " MeV" << std::endl;
		      continue;
		    }

		    ::larcv::Vertex pri_vtx( mcshower.Start().X(), mcshower.Start().Y(), mcshower.Start().Z(), mcshower.Start().T() );

		    auto roi = _cropper.ParticleROI(mcshower,time_offset);
		    roi.MCSTIndex(i);
		    
		    if (roi.BB().size() < _min_nplanes) {
		      LARCV_INFO() << "Skipping Primary ROI as # planes (" << roi.BB().size() << ") < requirement (" << _min_nplanes << std::endl
				   << roi.dump() << std::endl;
		      continue;
		    }

		    // insert roi into vertex map
		    DefinePrimary( pri_vtx, roi );
		  }
		}

		template <class T, class U, class V, class W>
		void MCParticleTree<T, U, V, W>::RegisterSecondary(const std::vector<U>& mctrack_v, const int time_offset)
		{
			LARCV_DEBUG() << "start" << std::endl;
			::larcv::Vertex pri_vtx;
			for (size_t i = 0; i < mctrack_v.size(); ++i) {
				auto const& mctrack = mctrack_v[i];

				if  ( mctrack.TrackID()==mctrack.MotherTrackID() ) {
				  // this is a primary
				  continue;
				}

				if (mctrack.size() < 2) {
					LARCV_INFO() << "Ignoring MCTrack G4TrackID " << mctrack.TrackID()
					             << " PdgCode " << mctrack.PdgCode()
					             << " as it has < 2 steps in the detector" << std::endl;
					continue;
				}

				double min_energy_init    = _min_energy_init_mctrack;
				double min_energy_deposit = _min_energy_deposit_mctrack;

				// Check if PDG is registered for a special handling
				if (_min_energy_init_pdg.find(mctrack.PdgCode()) != _min_energy_init_pdg.end()) {
					min_energy_init    = _min_energy_init_pdg[mctrack.PdgCode()];
					min_energy_deposit = _min_energy_deposit_pdg[mctrack.PdgCode()];
				}

				if ((mctrack.Start().E() < min_energy_init) ) {
					LARCV_INFO() << "Ignoring MCTrack G4TrackID " << mctrack.TrackID()
					             << " PdgCode " << mctrack.PdgCode()
					             << " as it has too small initial energy " << mctrack.Start().E()
					             << " MeV < " << min_energy_init << " MeV" << std::endl;
					continue;
				}
				if ((mctrack.front().E() - mctrack.back().E()) < min_energy_deposit) {
					LARCV_INFO() << "Ignoring MCTrack G4TrackID " << mctrack.TrackID()
					             << " PdgCode " << mctrack.PdgCode()
					             << " as it has too small deposit energy " << (mctrack.front().E() - mctrack.back().E())
					             << " MeV < " << min_energy_deposit << " MeV" << std::endl;
					continue;
				}

				if (mctrack.AncestorStart().E() < 1.e6) {
					auto const& start = mctrack.AncestorStart();
					pri_vtx.Reset(start.X(), start.Y(), start.Z(), start.T());
				}
				else if (mctrack.MotherStart().E() < 1.e6) {
					auto const& start = mctrack.MotherStart();
					pri_vtx.Reset(start.X(), start.Y(), start.Z(), start.T());
				} else {
					auto const& start = mctrack.Start();
					pri_vtx.Reset(start.X(), start.Y(), start.Z(), start.T());
				}

				auto roi = _cropper.ParticleROI(mctrack,time_offset);
				roi.MCSTIndex(i);

				if (roi.BB().size() < _min_nplanes) {
					LARCV_INFO() << "Skipping ROI as # planes (" << roi.BB().size() << ") < requirement (" << _min_nplanes << std::endl
					             << roi.dump() << std::endl;
				}
				else RegisterSecondary(pri_vtx, roi);
			}
		}

		template <class T, class U, class V, class W>
		void MCParticleTree<T, U, V, W>::RegisterSecondary(const std::vector<U>& mctrack_v,
		        const std::vector<W>& simch_v, const int time_offset)
		{
			LARCV_DEBUG() << "start" << std::endl;
			::larcv::Vertex pri_vtx;
			for (size_t i = 0; i < mctrack_v.size(); ++i) {
				auto const& mctrack = mctrack_v[i];

				if (mctrack.size() < 2) {
					LARCV_INFO() << "Ignoring MCTrack G4TrackID " << mctrack.TrackID()
					             << " PdgCode " << mctrack.PdgCode()
					             << " as it has < 2 steps in the detector" << std::endl;
					continue;
				}

				double min_energy_init    = _min_energy_init_mctrack;
				double min_energy_deposit = _min_energy_deposit_mctrack;

				// Check if PDG is registered for a special handling
				if (_min_energy_init_pdg.find(mctrack.PdgCode()) != _min_energy_init_pdg.end()) {
					min_energy_init    = _min_energy_init_pdg[mctrack.PdgCode()];
					min_energy_deposit = _min_energy_deposit_pdg[mctrack.PdgCode()];
				}

				if ((mctrack.Start().E() < min_energy_init) ) {
					LARCV_INFO() << "Ignoring MCTrack G4TrackID " << mctrack.TrackID()
					             << " PdgCode " << mctrack.PdgCode()
					             << " as it has too small initial energy " << mctrack.Start().E()
					             << " MeV < " << min_energy_init << " MeV" << std::endl;
					continue;
				}
				if ((mctrack.front().E() - mctrack.back().E()) < min_energy_deposit) {
					LARCV_INFO() << "Ignoring MCTrack G4TrackID " << mctrack.TrackID()
					             << " PdgCode " << mctrack.PdgCode()
					             << " as it has too small deposit energy " << (mctrack.front().E() - mctrack.back().E())
					             << " MeV < " << min_energy_deposit << " MeV" << std::endl;
					continue;
				}

				if (mctrack.AncestorStart().E() < 1.e6) {
					auto const& start = mctrack.AncestorStart();
					pri_vtx.Reset(start.X(), start.Y(), start.Z(), start.T());
				}
				else if (mctrack.MotherStart().E() < 1.e6) {
					auto const& start = mctrack.MotherStart();
					pri_vtx.Reset(start.X(), start.Y(), start.Z(), start.T());
				} else {
					auto const& start = mctrack.Start();
					pri_vtx.Reset(start.X(), start.Y(), start.Z(), start.T());
				}

				auto roi = _cropper.ParticleROI(mctrack, simch_v, time_offset);
				roi.MCSTIndex(i);

				if (roi.BB().size() < _min_nplanes) {
					LARCV_INFO() << "Skipping ROI as # planes (" << roi.BB().size() << ") < requirement (" << _min_nplanes << std::endl
					             << roi.dump() << std::endl;
				}
				else RegisterSecondary(pri_vtx, roi);
			}
		}

		template <class T, class U, class V, class W>
		void MCParticleTree<T, U, V, W>::RegisterSecondary(const std::vector<V>& mcshower_v, const int time_offset)
		{
			LARCV_DEBUG() << "start" << std::endl;
			::larcv::Vertex pri_vtx;
			for (size_t i = 0; i < mcshower_v.size(); ++i) {
				auto const& mcshower = mcshower_v[i];

				double min_energy_init    = _min_energy_init_mcshower;
				double min_energy_deposit = _min_energy_deposit_mcshower;

				// Check if PDG is registered for a special handling
				if (_min_energy_init_pdg.find(mcshower.PdgCode()) != _min_energy_init_pdg.end()) {
					min_energy_init    = _min_energy_init_pdg[mcshower.PdgCode()];
					min_energy_deposit = _min_energy_deposit_pdg[mcshower.PdgCode()];
				}

				if ((mcshower.Start().E() < min_energy_init) ) {
					LARCV_INFO() << "Ignoring MCShower G4TrackID " << mcshower.TrackID()
					             << " PdgCode " << mcshower.PdgCode()
					             << " as it has too small initial energy " << mcshower.Start().E()
					             << " MeV < " << min_energy_init << " MeV" << std::endl;
					continue;
				}
				if (mcshower.DetProfile().E() < min_energy_deposit) {
					LARCV_INFO() << "Ignoring MCShower G4TrackID " << mcshower.TrackID()
					             << " PdgCode " << mcshower.PdgCode()
					             << " as it has too small deposited energy " << mcshower.DetProfile().E()
					             << " MeV < " << min_energy_deposit << " MeV" << std::endl;
					continue;
				}


				if (mcshower.AncestorStart().E() < 1.e6) {
					auto const& start = mcshower.AncestorStart();
					pri_vtx.Reset(start.X(), start.Y(), start.Z(), start.T());
				}
				else if (mcshower.MotherStart().E() < 1.e6) {
					auto const& start = mcshower.MotherStart();
					pri_vtx.Reset(start.X(), start.Y(), start.Z(), start.T());
				} else {
					auto const& start = mcshower.Start();
					pri_vtx.Reset(start.X(), start.Y(), start.Z(), start.T());
				}

				auto roi = _cropper.ParticleROI(mcshower,time_offset);
				roi.MCSTIndex(i);

				if (roi.BB().size() < _min_nplanes) {
					LARCV_INFO() << "Skipping ROI as # planes (" << roi.BB().size() << ") < requirement (" << _min_nplanes << std::endl
					             << roi.dump() << std::endl;
				}
				else RegisterSecondary(pri_vtx, roi);
			}
		}

		template <class T, class U, class V, class W>
		void MCParticleTree<T, U, V, W>::RegisterSecondary(const std::vector<V>& mcshower_v,
		        const std::vector<W>& simch_v, const int time_offset)
		{
			LARCV_DEBUG() << "start" << std::endl;
			::larcv::Vertex pri_vtx;
			for (size_t i = 0; i < mcshower_v.size(); ++i) {
				auto const& mcshower = mcshower_v[i];

				double min_energy_init    = _min_energy_init_mcshower;
				double min_energy_deposit = _min_energy_deposit_mcshower;

				// Check if PDG is registered for a special handling
				if (_min_energy_init_pdg.find(mcshower.PdgCode()) != _min_energy_init_pdg.end()) {
					min_energy_init    = _min_energy_init_pdg[mcshower.PdgCode()];
					min_energy_deposit = _min_energy_deposit_pdg[mcshower.PdgCode()];
				}

				if ((mcshower.Start().E() < min_energy_init) ) {
					LARCV_INFO() << "Ignoring MCShower G4TrackID " << mcshower.TrackID()
					             << " PdgCode " << mcshower.PdgCode()
					             << " as it has too small initial energy " << mcshower.Start().E()
					             << " MeV < " << min_energy_init << " MeV" << std::endl;
					continue;
				}
				if (mcshower.DetProfile().E() < min_energy_deposit) {
					LARCV_INFO() << "Ignoring MCShower G4TrackID " << mcshower.TrackID()
					             << " PdgCode " << mcshower.PdgCode()
					             << " as it has too small deposited energy " << mcshower.DetProfile().E()
					             << " MeV < " << min_energy_deposit << " MeV" << std::endl;
					continue;
				}


				if (mcshower.AncestorStart().E() < 1.e6) {
					auto const& start = mcshower.AncestorStart();
					pri_vtx.Reset(start.X(), start.Y(), start.Z(), start.T());
				}
				else if (mcshower.MotherStart().E() < 1.e6) {
					auto const& start = mcshower.MotherStart();
					pri_vtx.Reset(start.X(), start.Y(), start.Z(), start.T());
				} else {
					auto const& start = mcshower.Start();
					pri_vtx.Reset(start.X(), start.Y(), start.Z(), start.T());
				}

				auto roi = _cropper.ParticleROI(mcshower, simch_v, time_offset);
				roi.MCSTIndex(i);

				if (roi.BB().size() < _min_nplanes) {
					LARCV_INFO() << "Skipping ROI as # planes (" << roi.BB().size() << ") < requirement (" << _min_nplanes << std::endl
					             << roi.dump() << std::endl;
				}
				else RegisterSecondary(pri_vtx, roi);
			}
		}

		template <class T, class U, class V, class W>
		void MCParticleTree<T, U, V, W>::DefinePrimary(const larcv::Vertex& vtx, const larcv::ROI& interaction)
		{
			LARCV_DEBUG() << "start" << std::endl;
			auto iter = _roi_m.find(vtx);
			if (iter != _roi_m.end()) {
				LARCV_CRITICAL() << "Duplicate interaction definition @ (x,y,z,t) = "
				                 << "(" << vtx.X() << "," << vtx.Y() << "," << vtx.Z() << "," << vtx.T() << std::endl;
				throw larbys();
			}
			LARCV_INFO() << "Defining a new primary @ (x,y,z,t) = ("
			             << vtx.X() << "," << vtx.Y() << "," << vtx.Z() << "," << vtx.T() << ")" << std::endl;
			_roi_m.insert(std::make_pair(vtx, std::make_pair(interaction, ParticleROIArray_t())));
		}

		template <class T, class U, class V, class W>
		void MCParticleTree<T, U, V, W>::RegisterSecondary(const larcv::Vertex& vtx, const larcv::ROI& secondary)
		{
			LARCV_DEBUG() << "start" << std::endl;
			if (_roi_m.empty()) {
				LARCV_CRITICAL() << "No primary registered. Cannot register secondaries..." << std::endl;
				throw larbys();
			}

			LARCV_INFO() << "Requested to register secondary's ROI..." << std::endl
			             << secondary.dump().c_str();

			bool filter = true;
			for (auto const& meta : secondary.BB()) if (meta.height() > 1 && meta.width() > 1) { filter = false; break; }
			if (filter) {
				LARCV_INFO() << "All bounding boxes are invalid. Ignoring this request..." << std::endl;
				return;
			}

			auto iter = _roi_m.find(vtx);
			if (iter != _roi_m.end()) { 
			  (*iter).second.second.push_back(secondary);
			  // matches to a vertex
			}
			else {
				LARCV_INFO() << "Vertex does not exactly match with registered primary: (x,y,z,t) = ("
				             << vtx.X() << "," << vtx.Y() << "," << vtx.Z() << "," << vtx.T() << ")" << std::endl;
				
				// is the particle its own parent?
				if ( secondary.TrackID()==secondary.ParentTrackID() ) {
				  LARCV_INFO() << "Secondary is its own parent. Cosmic ray track." << std::endl;
				  return;
				}

				// Search closest in time but AFTER
				std::map<double, larcv::Vertex> time_order;
				for (auto const& key_value : _roi_m) time_order.emplace(std::make_pair(key_value.first.T(), key_value.first));

				::larcv::Vertex target;

				auto iter = time_order.find(vtx.T());
				if (iter != time_order.end()) target = (*iter).second;

				else {

					if (vtx.T() < (*(time_order.begin())).second.T()) {
						LARCV_CRITICAL() << "No primary found before this secondary!" << std::endl;
						throw larbys();
					}

					auto iter = time_order.upper_bound(vtx.T());

					if (iter == time_order.end())
						// Take the last one
						target = (*(time_order.rbegin())).second;
					else {
						--iter;
						target = (*iter).second;
					}
				}
				_roi_m[target].second.push_back(secondary);
			}

		}

		template <class T, class U, class V, class W>
		void MCParticleTree<T, U, V, W>::UpdatePrimaryROI()
		{
			LARCV_DEBUG() << "start" << std::endl;
			for (auto& roi_key_value : _roi_m) {

				//auto& roi_key = roi_key_value.first;
				auto& int_roi = roi_key_value.second;

				auto& pri_roi = int_roi.first;  // primary's ROI (to be updated)
				auto& sec_roi_v = int_roi.second; // secondarys ROI

				std::map<larcv::PlaneID_t, larcv::ImageMeta> sum_roi_m;
				double energy_deposit = 0;
				for (auto& sec_roi : sec_roi_v) {
					energy_deposit += sec_roi.EnergyDeposit();
					// loop over plane-by-plane ImageMeta
					for (auto const& bb : sec_roi.BB()) {
						if (!(bb.height() > 1 && bb.width() > 1)) continue;
						auto iter = sum_roi_m.find(bb.plane());
						if (iter == sum_roi_m.end())
							sum_roi_m[bb.plane()] = bb;
						else
							(*iter).second = (*iter).second.inclusive(bb);
					}
					sec_roi.MCTIndex(pri_roi.MCTIndex());
				}

				std::vector<larcv::ImageMeta> bb_v;
				bb_v.reserve(sum_roi_m.size());

				for (auto const& plane_roi : sum_roi_m)
					bb_v.push_back(plane_roi.second);

				pri_roi.EnergyDeposit(energy_deposit);

				pri_roi.SetBB(bb_v);
			}
		}

		template <class T, class U, class V, class W>
		std::vector<larcv::supera::InteractionROI_t> MCParticleTree<T, U, V, W>::GetPrimaryROI() const
		{
			LARCV_DEBUG() << "start" << std::endl;
			std::vector<larcv::supera::InteractionROI_t> res;
			res.reserve(_roi_m.size());
			for (auto const& vtx_roi : _roi_m) res.push_back(vtx_roi.second);
			return res;
		}
	}
}

#endif
