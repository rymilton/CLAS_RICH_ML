import h5py as h5
import torch
from torch.utils.data import Dataset
import awkward as ak
import time
import numpy as np
class H5Dataset(Dataset):
    def __init__(self, data_file):
        start_time = time.time()
        self.data_file = data_file
        self.data = h5.File(self.data_file, 'r')

        # Storing our data in Awkward arrays
        self.RICH_hits, self.rec_traj, self.rec_particles, self.RICH_particles = {}, {}, {}, {}
        for entry in list(self.data["RICH_Hits"].keys()):
            self.RICH_hits[entry] = self.data[f"RICH_Hits/{entry}"][:].tolist()
        self.RICH_hits = ak.Array(self.RICH_hits)
        for entry in list(self.data["trajectories"].keys()):
            self.rec_traj[entry] = self.data[f"trajectories/{entry}"][:].tolist()
        self.rec_traj = ak.Array(self.rec_traj)
        for entry in list(self.data["reconstructed_particles"].keys()):
            self.rec_particles[entry] = self.data[f"reconstructed_particles/{entry}"][:].tolist()
        self.rec_particles = ak.Array(self.rec_particles)
        for entry in list(self.data["RICH_particles"].keys()):
            self.RICH_particles[entry] = self.data[f"RICH_particles/{entry}"][:].tolist()
        self.RICH_particles = ak.Array(self.RICH_particles)

        self.labels = ak.Array(self.data[f"truth_particles/MC::Particle.pid"][:].tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        hits_feature_names = ['RICH::Hit.rawtime', 'RICH::Hit.x', 'RICH::Hit.y']
        RICH_hits_event = np.stack([self.RICH_hits[idx][feature] for feature in hits_feature_names], axis=1)

        particle_feature_names = ["REC::Particles.p"]
        particle_momenta_event = np.stack([self.rec_particles[idx][feature] for feature in particle_feature_names], axis=1)
        trajectory_feature_names = ['REC::Traj.x', 'REC::Traj.y', 'REC::Traj.cx', 'REC::Traj.cy', 'REC::Traj.cz']
        aerogel_event = np.stack([self.rec_traj[idx][feature] for feature in trajectory_feature_names], axis=1)

        globals_event = ak.flatten(np.hstack([particle_momenta_event, aerogel_event]))

        reconstructed_pid = np.stack([self.rec_particles[idx][feature] for feature in ["REC::Particles.pid"]], axis=1)
        reconstructed_pid = ak.flatten(reconstructed_pid)
        reconstructed_pid = torch.tensor(reconstructed_pid, dtype=torch.long)

        RICH_particle_features = np.stack([self.RICH_particles[idx][feature] for feature in ["RICH::Particle.best_PID", "RICH::Particle.RQ"]], axis=1)
        RICH_pid = torch.tensor(RICH_particle_features[:,0], dtype=torch.float32)
        RICH_RQ = torch.tensor(RICH_particle_features[:,1], dtype=torch.float32)
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        sample = torch.tensor(RICH_hits_event, dtype=torch.float32)
        globals_event = torch.tensor(globals_event, dtype=torch.float32)
        return sample, label, globals_event, reconstructed_pid, RICH_pid, RICH_RQ

    def __del__(self):
        # Close the file when dataset is destroyed
        self.data.close()