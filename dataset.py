import h5py as h5
import torch
from torch.utils.data import Dataset
import numpy as np

class H5Dataset(Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = h5.File(self.data_file, 'r')

        # Store dataset groups for lazy access
        self.RICH_hits_ds = self.data["RICH_Hits"]
        self.rec_traj_ds = self.data["trajectories"]
        self.rec_particles_ds = self.data["reconstructed_particles"]
        self.RICH_particles_ds = self.data["RICH_particles"]
        self.labels_ds = self.data["truth_particles/MC::Particle.pid"]

    def __len__(self):
        return len(self.labels_ds)

    def __getitem__(self, idx):
        # --- RICH hits ---
        hits_feature_names = ['RICH::Hit.rawtime', 'RICH::Hit.x', 'RICH::Hit.y']
        RICH_hits_event = np.stack([self.RICH_hits_ds[feature][idx] for feature in hits_feature_names], axis=1)

        # --- Reconstructed particle momenta ---
        particle_feature_names = ["REC::Particles.p"]
        particle_momenta_event = np.stack([self.rec_particles_ds[feature][idx] for feature in particle_feature_names], axis=1)

        # --- Trajectories ---
        trajectory_feature_names = ['REC::Traj.x', 'REC::Traj.y', 'REC::Traj.cx', 'REC::Traj.cy', 'REC::Traj.cz']
        aerogel_event = np.stack([self.rec_traj_ds[feature][idx] for feature in trajectory_feature_names], axis=1)

        # --- Globals flattened ---
        globals_event = np.hstack([particle_momenta_event.flatten(), aerogel_event.flatten()])
        print(globals_event)
        # --- Reconstructed PID ---
        reconstructed_pid = np.array(self.rec_particles_ds["REC::Particles.pid"][idx], dtype=np.int64)
        reconstructed_pid = torch.tensor(reconstructed_pid, dtype=torch.long)


        # --- Reconstructed particle angles ---
        rec_theta = torch.tensor(self.rec_particles_ds["REC::Particles.theta"][idx], dtype=torch.float32)
        rec_phi = torch.tensor(self.rec_particles_ds["REC::Particles.phi"][idx], dtype=torch.float32)

        # --- RICH particle features ---
        RICH_pid = torch.tensor(self.RICH_particles_ds["RICH::Particle.best_PID"][idx], dtype=torch.float32)
        RICH_RQ = torch.tensor(self.RICH_particles_ds["RICH::Particle.RQ"][idx], dtype=torch.float32)
        cherenkov_angle = torch.tensor(self.RICH_particles_ds["RICH::Particle.best_ch"][idx], dtype=torch.float32)

        # --- Label ---
        label = torch.tensor(self.labels_ds[idx], dtype=torch.float32)

        # --- Sample ---
        sample = torch.tensor(RICH_hits_event, dtype=torch.float32)
        globals_event = torch.tensor(globals_event, dtype=torch.float32)

        return sample, label, globals_event, reconstructed_pid, RICH_pid, RICH_RQ, rec_theta, rec_phi, cherenkov_angle

    def __del__(self):
        self.data.close()
