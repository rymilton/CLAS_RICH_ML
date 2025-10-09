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
        self.RICH_hits, self.rec_traj, self.rec_particles = {}, {}, {}
        for entry in list(self.data["RICH_Hits"].keys()):
            self.RICH_hits[entry] = self.data[f"RICH_Hits/{entry}"][:].tolist()
        self.RICH_hits = ak.Array(self.RICH_hits)
        for entry in list(self.data["trajectories"].keys()):
            self.rec_traj[entry] = self.data[f"trajectories/{entry}"][:].tolist()
        self.rec_traj = ak.Array(self.rec_traj)
        for entry in list(self.data["reconstructed_particles"].keys()):
            self.rec_particles[entry] = self.data[f"reconstructed_particles/{entry}"][:].tolist()
        self.rec_particles = ak.Array(self.rec_particles)
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
        
        globals_event = np.hstack([particle_momenta_event, aerogel_b1_event, aerogel_b2_event, aerogel_b3_event])
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        sample = torch.tensor(RICH_hits_event, dtype=torch.float32)
        globals_event = torch.tensor(globals_event, dtype=torch.float32)
        return sample, label, globals_event

    def __del__(self):
        # Close the file when dataset is destroyed
        self.data.close()