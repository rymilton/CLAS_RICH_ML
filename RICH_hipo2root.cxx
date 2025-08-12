#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cstdint>

#include "reader.h"
#include "TFile.h"
#include "TTree.h"

struct rec_particles_holder {
    std::vector<int> pid;
    std::vector<float> px, py, pz;
    std::vector<float> vx, vy, vz, vt;
    std::vector<int> charge;
    std::vector<float> beta, chi2pid;
    std::vector<short> status;

    void clear() {
        pid.clear();
        px.clear(); py.clear(); pz.clear();
        vx.clear(); vy.clear(); vz.clear(); vt.clear();
        charge.clear();
        beta.clear();
        chi2pid.clear();
        status.clear();
    }
};

struct rec_traj_holder {
    std::vector<short> pindex, index;
    std::vector<int> detector, layer;
    std::vector<float> x, y, z;
    std::vector<float> cx, cy, cz;
    std::vector<float> path, edge;

    void clear() {
        pindex.clear(); index.clear();
        detector.clear(); layer.clear();
        x.clear(); y.clear(); z.clear();
        cx.clear(); cy.clear(); cz.clear();
        path.clear(); edge.clear();
    }
};

// New struct for RICH::Hit bank
struct rich_hit_holder {
    std::vector<short> id, sector, tile, pmt, anode, cluster, xtalk, status, duration;
    std::vector<float> x, y, z, time, rawtime;

    void clear() {
        id.clear(); sector.clear(); tile.clear(); pmt.clear(); anode.clear();
        cluster.clear(); xtalk.clear(); status.clear(); duration.clear();
        x.clear(); y.clear(); z.clear(); time.clear(); rawtime.clear();
    }
};

struct rich_ring_holder {
    std::vector<short> id, hindex, pmt;
    std::vector<int> pindex, sector, anode, use, layers, compos;
    std::vector<int> hypo;
    std::vector<float> dtime, etaC, prob, dangle;

    void clear() {
        id.clear(); hindex.clear(); pmt.clear();
        pindex.clear(); sector.clear(); anode.clear();
        use.clear(); layers.clear(); compos.clear();
        hypo.clear();
        dtime.clear(); etaC.clear(); prob.clear(); dangle.clear();
    }
};

struct mc_particle_holder {
    std::vector<int> pid;
    std::vector<float> px, py, pz;
    std::vector<float> vx, vy, vz, vt;

    void clear() {
        pid.clear();
        px.clear(); py.clear(); pz.clear();
        vx.clear(); vy.clear(); vz.clear(); vt.clear();
    }
};

int main(int argc, char** argv) {

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << "input_directory input.hipo output_directory" << std::endl;
        return 1;
    }

    TString inputDir = argv[1];
    TString inputFile = argv[2];
    TString outputDir = argv[3];

    hipo::reader reader;
    reader.open(inputDir+inputFile);

    hipo::dictionary factory;
    reader.readDictionary(factory);

    hipo::bank rec_particles_bank(factory.getSchema("REC::Particle"));
    hipo::bank rec_traj_bank(factory.getSchema("REC::Traj"));
    hipo::bank rich_hit_bank(factory.getSchema("RICH::Hit"));
    hipo::bank rich_ring_bank(factory.getSchema("RICH::Ring"));
    hipo::bank mc_particle_bank(factory.getSchema("MC::Particle"));

    TFile outfile(outputDir+inputFile + TString(".root"), "RECREATE");
    TTree tree("events", "");

    rec_particles_holder recParticles;
    rec_traj_holder recTraj;
    rich_hit_holder richHits;
    rich_ring_holder richRing;
    mc_particle_holder mcParticles;

    // REC::Particles branches
    tree.Branch("REC::Particles.pid", &recParticles.pid);
    tree.Branch("REC::Particles.px", &recParticles.px);
    tree.Branch("REC::Particles.py", &recParticles.py);
    tree.Branch("REC::Particles.pz", &recParticles.pz);
    tree.Branch("REC::Particles.vx", &recParticles.vx);
    tree.Branch("REC::Particles.vy", &recParticles.vy);
    tree.Branch("REC::Particles.vz", &recParticles.vz);
    tree.Branch("REC::Particles.vt", &recParticles.vt);
    tree.Branch("REC::Particles.charge", &recParticles.charge);
    tree.Branch("REC::Particles.beta", &recParticles.beta);
    tree.Branch("REC::Particles.chi2pid", &recParticles.chi2pid);
    tree.Branch("REC::Particles.status", &recParticles.status);

    // REC::Traj branches
    tree.Branch("REC::Traj.pindex", &recTraj.pindex);
    tree.Branch("REC::Traj.index", &recTraj.index);
    tree.Branch("REC::Traj.detector", &recTraj.detector);
    tree.Branch("REC::Traj.layer", &recTraj.layer);
    tree.Branch("REC::Traj.x", &recTraj.x);
    tree.Branch("REC::Traj.y", &recTraj.y);
    tree.Branch("REC::Traj.z", &recTraj.z);
    tree.Branch("REC::Traj.cx", &recTraj.cx);
    tree.Branch("REC::Traj.cy", &recTraj.cy);
    tree.Branch("REC::Traj.cz", &recTraj.cz);
    tree.Branch("REC::Traj.path", &recTraj.path);
    tree.Branch("REC::Traj.edge", &recTraj.edge);

    // RICH::Hit branches
    tree.Branch("RICH::Hit.id", &richHits.id);
    tree.Branch("RICH::Hit.sector", &richHits.sector);
    tree.Branch("RICH::Hit.tile", &richHits.tile);
    tree.Branch("RICH::Hit.pmt", &richHits.pmt);
    tree.Branch("RICH::Hit.anode", &richHits.anode);
    tree.Branch("RICH::Hit.cluster", &richHits.cluster);
    tree.Branch("RICH::Hit.xtalk", &richHits.xtalk);
    tree.Branch("RICH::Hit.status", &richHits.status);
    tree.Branch("RICH::Hit.duration", &richHits.duration);
    tree.Branch("RICH::Hit.x", &richHits.x);
    tree.Branch("RICH::Hit.y", &richHits.y);
    tree.Branch("RICH::Hit.z", &richHits.z);
    tree.Branch("RICH::Hit.time", &richHits.time);
    tree.Branch("RICH::Hit.rawtime", &richHits.rawtime);

    tree.Branch("RICH::Ring.id", &richRing.id);
    tree.Branch("RICH::Ring.hindex", &richRing.hindex);
    tree.Branch("RICH::Ring.pindex", &richRing.pindex);
    tree.Branch("RICH::Ring.sector", &richRing.sector);
    tree.Branch("RICH::Ring.pmt", &richRing.pmt);
    tree.Branch("RICH::Ring.anode", &richRing.anode);
    tree.Branch("RICH::Ring.hypo", &richRing.hypo);
    tree.Branch("RICH::Ring.dtime", &richRing.dtime);
    tree.Branch("RICH::Ring.etaC", &richRing.etaC);
    tree.Branch("RICH::Ring.prob", &richRing.prob);
    tree.Branch("RICH::Ring.use", &richRing.use);
    tree.Branch("RICH::Ring.dangle", &richRing.dangle);
    tree.Branch("RICH::Ring.layers", &richRing.layers);
    tree.Branch("RICH::Ring.compos", &richRing.compos);

    tree.Branch("MC::Particle.pid", &mcParticles.pid);
    tree.Branch("MC::Particle.px", &mcParticles.px);
    tree.Branch("MC::Particle.py", &mcParticles.py);
    tree.Branch("MC::Particle.pz", &mcParticles.pz);
    tree.Branch("MC::Particle.vx", &mcParticles.vx);
    tree.Branch("MC::Particle.vy", &mcParticles.vy);
    tree.Branch("MC::Particle.vz", &mcParticles.vz);
    tree.Branch("MC::Particle.vt", &mcParticles.vt);

    hipo::event event;
    int counter = 0;

    while (reader.next() == true) {
        reader.read(event);

        recParticles.clear();
        recTraj.clear();
        richHits.clear();
        richRing.clear();
        mcParticles.clear();

        event.getStructure(rec_particles_bank);
        int nrows = rec_particles_bank.getRows();
        for (int row = 0; row < nrows; row++) {
            recParticles.pid.push_back(rec_particles_bank.getInt("pid", row));
            recParticles.px.push_back(rec_particles_bank.getFloat("px", row));
            recParticles.py.push_back(rec_particles_bank.getFloat("py", row));
            recParticles.pz.push_back(rec_particles_bank.getFloat("pz", row));
            recParticles.vx.push_back(rec_particles_bank.getFloat("vx", row));
            recParticles.vy.push_back(rec_particles_bank.getFloat("vy", row));
            recParticles.vz.push_back(rec_particles_bank.getFloat("vz", row));
            recParticles.vt.push_back(rec_particles_bank.getFloat("vt", row));
            recParticles.charge.push_back(rec_particles_bank.getByte("charge", row));
            recParticles.beta.push_back(rec_particles_bank.getFloat("beta", row));
            recParticles.chi2pid.push_back(rec_particles_bank.getFloat("chi2pid", row));
            recParticles.status.push_back(rec_particles_bank.getShort("status", row));
        }

        event.getStructure(rec_traj_bank);
        nrows = rec_traj_bank.getRows();
        for (int row = 0; row < nrows; row++) {
            recTraj.pindex.push_back(rec_traj_bank.getShort("pindex", row));
            recTraj.index.push_back(rec_traj_bank.getShort("index", row));
            recTraj.detector.push_back(rec_traj_bank.getByte("detector", row));
            recTraj.layer.push_back(rec_traj_bank.getByte("layer", row));
            recTraj.x.push_back(rec_traj_bank.getFloat("x", row));
            recTraj.y.push_back(rec_traj_bank.getFloat("y", row));
            recTraj.z.push_back(rec_traj_bank.getFloat("z", row));
            recTraj.cx.push_back(rec_traj_bank.getFloat("cx", row));
            recTraj.cy.push_back(rec_traj_bank.getFloat("cy", row));
            recTraj.cz.push_back(rec_traj_bank.getFloat("cz", row));
            recTraj.path.push_back(rec_traj_bank.getFloat("path", row));
            recTraj.edge.push_back(rec_traj_bank.getFloat("edge", row));
        }

        event.getStructure(rich_hit_bank);
        nrows = rich_hit_bank.getRows();
        for (int row = 0; row < nrows; row++) {
            richHits.id.push_back(rich_hit_bank.getShort("id", row));
            richHits.sector.push_back(rich_hit_bank.getShort("sector", row));
            richHits.tile.push_back(rich_hit_bank.getShort("tile", row));
            richHits.pmt.push_back(rich_hit_bank.getShort("pmt", row));
            richHits.anode.push_back(rich_hit_bank.getShort("anode", row));
            richHits.cluster.push_back(rich_hit_bank.getShort("cluster", row));
            richHits.xtalk.push_back(rich_hit_bank.getShort("xtalk", row));
            richHits.status.push_back(rich_hit_bank.getShort("status", row));
            richHits.duration.push_back(rich_hit_bank.getShort("duration", row));
            richHits.x.push_back(rich_hit_bank.getFloat("x", row));
            richHits.y.push_back(rich_hit_bank.getFloat("y", row));
            richHits.z.push_back(rich_hit_bank.getFloat("z", row));
            richHits.time.push_back(rich_hit_bank.getFloat("time", row));
            richHits.rawtime.push_back(rich_hit_bank.getFloat("rawtime", row));
        }

        event.getStructure(rich_ring_bank);
        nrows = rich_ring_bank.getRows();
        for (int row = 0; row < nrows; row++) {
            richRing.id.push_back(rich_ring_bank.getShort("id", row));
            richRing.hindex.push_back(rich_ring_bank.getShort("hindex", row));
            richRing.pmt.push_back(rich_ring_bank.getShort("pmt", row));

            richRing.pindex.push_back(rich_ring_bank.getInt("pindex", row));
            richRing.sector.push_back(rich_ring_bank.getInt("sector", row));
            richRing.anode.push_back(rich_ring_bank.getInt("anode", row));
            richRing.use.push_back(rich_ring_bank.getInt("use", row));
            richRing.layers.push_back(rich_ring_bank.getInt("layers", row));
            richRing.compos.push_back(rich_ring_bank.getInt("compos", row));
            richRing.hypo.push_back(rich_ring_bank.getInt("hypo", row));

            richRing.dtime.push_back(rich_ring_bank.getFloat("dtime", row));
            richRing.etaC.push_back(rich_ring_bank.getFloat("etaC", row));
            richRing.prob.push_back(rich_ring_bank.getFloat("prob", row));
            richRing.dangle.push_back(rich_ring_bank.getFloat("dangle", row));
        }

        event.getStructure(mc_particle_bank);
        nrows = mc_particle_bank.getRows();
        for (int row = 0; row < nrows; row++) {
            mcParticles.pid.push_back(mc_particle_bank.getInt("pid", row));
            mcParticles.px.push_back(mc_particle_bank.getFloat("px", row));
            mcParticles.py.push_back(mc_particle_bank.getFloat("py", row));
            mcParticles.pz.push_back(mc_particle_bank.getFloat("pz", row));
            mcParticles.vx.push_back(mc_particle_bank.getFloat("vx", row));
            mcParticles.vy.push_back(mc_particle_bank.getFloat("vy", row));
            mcParticles.vz.push_back(mc_particle_bank.getFloat("vz", row));
            mcParticles.vt.push_back(mc_particle_bank.getFloat("vt", row));
        }

        tree.Fill();
        counter++;
    }

    tree.Write();
    outfile.Close();

    std::cout << "Processed events = " << counter << std::endl;
    return 0;
}
