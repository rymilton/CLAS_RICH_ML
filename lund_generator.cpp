#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <TFile.h>
#include <TH1F.h>
#include <TF1.h>
#include <TMath.h>

using namespace std;

const double deg2rad = TMath::Pi() / 180.0;

struct Particle {
  int pdg;
  double mass;
  TLorentzVector vec;
};

Particle generateParticle(int pdg, double mass, int sector = 1) {
  double p = gRandom->Uniform(1.5, 8); // momentum in GeV

  double theta = gRandom->Uniform(5, 40) * deg2rad;
  double phi_low, phi_high;
  switch (sector) {
    case 1:
        phi_low = -30;
        phi_high = 30;
        break;
    case 2:
        phi_low = 30;
        phi_high = 90;
        break;
    case 3:
        phi_low = 90;
        phi_high = 150;
        break;
    case 4:
        phi_low = 150;
        phi_high = 210;
        break;
    case 5:
        phi_low = 210;
        phi_high = 270;
        break;
    case 6:
        phi_low = 270;
        phi_high = 330;
        break;
    default:
        phi_low = 50;
        phi_high = 50;
        break;
  }
  

  double phi = gRandom->Uniform(phi_low, phi_high) * deg2rad;
  if (phi > TMath::Pi())  phi -= 2 * TMath::Pi();
  if (phi < -TMath::Pi()) phi += 2 * TMath::Pi();
//   std::cout << "PDG "<< pdg<<": Sector " << sector << " min phi: "<< phi_low <<", max phi: "<< phi_high << std::endl;


  double px = p * sin(theta) * cos(phi);
  double py = p * sin(theta) * sin(phi);
  double pz = p * cos(theta);
  double E = sqrt(p*p + mass*mass);

  return { pdg, mass, TLorentzVector(px, py, pz, E) };
}


void lund_generator() {
  gRandom->SetSeed(time(0));

  // Particle info: PDG and mass (GeV)
  vector<pair<int, double>> particles = {
    {211,  0.1395}, // pi+
    {321,  0.4937}, // K+
    {2212, 0.9383}, // proton
    {-211,  0.1395}, // pi-
    {-321,  0.4937}, // K-
    {-2212, 0.9383}, // antiproton
  };
  const int nEv_pFile = 5000;
  const int nFiles = 2000;

  const double Mel = 0.000511;
  const double beamE = 10.6;

  TString dir = "/w/work/clas12/rmilton/RICH_data_generation/lund/positive_kaon_pion/";
  TString filename = dir + "file_";

  // Histograms
  TH1F hPel("Pel", "Electron P", 100, 0, 10);
  TH1F hThetael("Thel", "Electron Theta", 100, 0, 3.5);
  TH1F hPhiel("Phel", "Electron Phi", 100, -3.5, 3.5);
  TH1F hPhi1("Phi1", "phi(p1)", 100, -3.5, 3.5);
  TH1F hPhi2("Phi2", "phi(p2)", 100, -3.5, 3.5);
  TH1F hTheta1("Theta1", "theta(p1)", 100, 0, 3.5);
  TH1F hTheta2("Theta2", "theta(p2)",   100, 0, 3.5);

  map<int, TH1F*> hP, hTheta, hPhi;
  for (auto& [pdg, mass] : particles) {
    TString name = Form("P%d", pdg);
    hP[pdg] = new TH1F(name, name + " P", 100, 0, 10);
    name = Form("Th%d", pdg);
    hTheta[pdg] = new TH1F(name, name + " Theta", 100, 0, 3.5);
    name = Form("Ph%d", pdg);
    hPhi[pdg] = new TH1F(name, name + " Phi", 100, -3.5, 3.5);
  }

  for (int j = 0; j < nFiles; j++) {
    cout << "FileNb: " << j << endl;
    FILE* f;
    TString name = filename + Form("%.4d.dat", j);
    f = fopen(name.Data(), "w");

    Long64_t evInFile = 0;

    while (evInFile < nEv_pFile) {
      TLorentzVector beam(0, 0, beamE, sqrt(beamE*beamE + Mel*Mel));
      TLorentzVector target(0, 0, 0, 0.9383);

      // Electron
      auto electron = generateParticle(11, Mel, 7);
      double phi_el = electron.vec.Phi();

      // Two random particles
      int idx1 = gRandom->Integer(particles.size());
      int idx2;
      do {
        idx2 = gRandom->Integer(particles.size());
      } while (idx2 == idx1);
      // p1 gets sent to sector 1, p2 gets sent to sector 4
      auto p1 = generateParticle(particles[idx1].first, particles[idx1].second, 1);
      auto p2 = generateParticle(particles[idx2].first, particles[idx2].second, 4);
      fprintf(f, "%d %d %d %d %d %d %.6f %d %d %.6f\n", 3, 1, 1, 0, 0, 11, beamE, 2212, 0, 1.0);
      auto writeParticle = [&](int idx, const Particle& p) {
        fprintf(f, "%d %d %d %d %d %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                idx, 1, 1, p.pdg, 0, 0,
                p.vec.Px(), p.vec.Py(), p.vec.Pz(), p.vec.E(),
                p.mass, 0.0, 0.0, 0.0);
      };

      writeParticle(1, electron);
      writeParticle(2, p1);
      writeParticle(3, p2);
      
      double phi_p1 = p1.vec.Phi();
      double phi_p2 = p2.vec.Phi();
      double theta_p1 = p1.vec.Theta();
      double theta_p2 = p2.vec.Theta();
  
      hTheta1.Fill(theta_p1);
      hTheta2.Fill(theta_p2);
      hPel.Fill(electron.vec.P());
      hThetael.Fill(electron.vec.Theta());
      hPhiel.Fill(electron.vec.Phi());
  
      hP[p1.pdg]->Fill(p1.vec.P());
      hTheta[p1.pdg]->Fill(p1.vec.Theta());
      hPhi[p1.pdg]->Fill(p1.vec.Phi());
  
      hP[p2.pdg]->Fill(p2.vec.P());
      hTheta[p2.pdg]->Fill(p2.vec.Theta());
      hPhi[p2.pdg]->Fill(p2.vec.Phi());

      evInFile++;
    }

    fclose(f);
  }

  TFile *fout = TFile::Open("evegenplots.root", "recreate");
  hPel.Write(); hThetael.Write(); hPhiel.Write();
  hPhi1.Write();
  hPhi2.Write();
  hTheta1.Write();
  hTheta2.Write();

  for (auto& [pdg, hist] : hP) hist->Write();
  for (auto& [pdg, hist] : hTheta) hist->Write();
  for (auto& [pdg, hist] : hPhi) hist->Write();

  fout->Close();
}