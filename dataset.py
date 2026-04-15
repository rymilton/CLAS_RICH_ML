import h5py as h5
import torch
from torch.utils.data import Dataset
import numpy as np
from itertools import accumulate
import bisect


class H5Dataset(Dataset):
    def __init__(self, data_file, mode="train"):
        """
        mode: "train" or "inference"
        """
        self.mode = mode
        # Allow single file or list of files
        if isinstance(data_file, (list, tuple)):
            self.data_files = list(data_file)
        else:
            self.data_files = [data_file]

        # Open all files
        self.files = None
        self._init_dataset_handles()

        # Cache dataset handles per file
        self.RICH_hits_ds = []
        self.rec_traj_ds = []
        self.rec_particles_ds = []
        self.RICH_particles_ds = []
        self.labels_ds = []
        self.lengths = []

        for f in self.files:
            self.RICH_hits_ds.append(f["RICH_Hits"])
            self.rec_traj_ds.append(f["trajectories"])
            self.rec_particles_ds.append(f["reconstructed_particles"])
            self.RICH_particles_ds.append(f["RICH_particles"])
            self.labels_ds.append(f["truth_particles/MC::Particle.pid"])
            self.lengths.append(len(f["truth_particles/MC::Particle.pid"]))

        # Cumulative lengths for global → local index mapping
        self.cumulative_lengths = list(accumulate(self.lengths))

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _get_file_index(self, idx):
        """
        Map global index -> (file_idx, local_idx)
        """
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        prev_len = 0 if file_idx == 0 else self.cumulative_lengths[file_idx - 1]
        local_idx = idx - prev_len
        return file_idx, local_idx

    def _init_dataset_handles(self):
        if self.files is not None:
            return

        # Open HDF5 files in this worker
        self.files = [h5.File(f, "r") for f in self.data_files]

        self.RICH_hits_ds = []
        self.rec_traj_ds = []
        self.rec_particles_ds = []
        self.RICH_particles_ds = []
        self.labels_ds = []
        self.lengths = []

        for f in self.files:
            self.RICH_hits_ds.append(f["RICH_Hits"])
            self.rec_traj_ds.append(f["trajectories"])
            self.rec_particles_ds.append(f["reconstructed_particles"])
            self.RICH_particles_ds.append(f["RICH_particles"])
            self.labels_ds.append(f["truth_particles/MC::Particle.pid"])
            self.lengths.append(len(f["truth_particles/MC::Particle.pid"]))

        self.cumulative_lengths = list(accumulate(self.lengths))

    def __getitem__(self, idx):
        file_idx, idx = self._get_file_index(idx)

        RICH_hits_ds = self.RICH_hits_ds[file_idx]
        rec_traj_ds = self.rec_traj_ds[file_idx]
        rec_particles_ds = self.rec_particles_ds[file_idx]
        RICH_particles_ds = self.RICH_particles_ds[file_idx]
        labels_ds = self.labels_ds[file_idx]

        # --- RICH hits ---
        hits_feature_names = ["RICH::Hit.rawtime", "RICH::Hit.x", "RICH::Hit.y"]
        RICH_hits_event = np.stack(
            [RICH_hits_ds[feature][idx] for feature in hits_feature_names], axis=1
        )

        # --- Reconstructed particle momenta ---
        particle_feature_names = [
            "REC::Particles.p",
            # "REC::Particles.theta",
            # "REC::Particles.phi",
        ]
        particle_momenta_event = np.stack(
            [rec_particles_ds[feature][idx] for feature in particle_feature_names],
            axis=1,
        )

        # --- Trajectories ---
        trajectory_feature_names = [
            "REC::Traj.x",
            "REC::Traj.y",
            "REC::Traj.cx",
            "REC::Traj.cy",
            "REC::Traj.cz",
        ]
        aerogel_event = np.stack(
            [rec_traj_ds[feature][idx] for feature in trajectory_feature_names], axis=1
        )

        aerogel_layer_np = np.array(rec_traj_ds["REC::Traj.layer"][idx], dtype=int)
        num_layers = 3
        layer_onehot = np.eye(num_layers)[aerogel_layer_np - 1]  # shape (N, 3)

        aerogel_event = np.concatenate([aerogel_event, layer_onehot], axis=1)

        # --- Globals flattened ---
        globals_event = np.hstack(
            [particle_momenta_event.flatten(), aerogel_event.flatten()]
        )

        # --- Reconstructed PID ---
        reconstructed_pid = torch.from_numpy(
            rec_particles_ds["REC::Particles.pid"][idx],
        ).long()

        # --- Reconstructed particle angles ---
        rec_theta = torch.from_numpy(
            rec_particles_ds["REC::Particles.theta"][idx],
        ).float()
        rec_phi = torch.from_numpy(
            rec_particles_ds["REC::Particles.phi"][idx],
        ).float()
        # --- RICH particle features ---
        RICH_pid_np = RICH_particles_ds["RICH::Particle.best_PID"][idx]
        RICH_RQ_np = RICH_particles_ds["RICH::Particle.RQ"][idx]
        cherenkov_angle_np = RICH_particles_ds["RICH::Particle.best_ch"][idx]
        if len(RICH_pid_np) == 0:
            RICH_pid_np = np.array([-1.0], dtype=np.float32)
        if len(RICH_RQ_np) == 0:
            RICH_RQ_np = np.array([-1.0], dtype=np.float32)
        if len(cherenkov_angle_np) == 0:
            cherenkov_angle_np = np.array([-1.0], dtype=np.float32)
        RICH_pid = torch.from_numpy(
            RICH_pid_np,
        ).float()
        RICH_RQ = torch.from_numpy(
            RICH_RQ_np,
        ).float()
        cherenkov_angle = torch.from_numpy(
            cherenkov_angle_np,
        ).float()

        aerogel_layer = torch.from_numpy(
            aerogel_layer_np,
        ).float()

        # --- Label ---
        label = torch.from_numpy(labels_ds[idx]).float()

        # --- Sample ---
        sample = torch.from_numpy(RICH_hits_event).float()
        globals_event = torch.from_numpy(globals_event).float()

        if self.mode == "train":
            # Fast path for training
            return sample, label, globals_event
        else:
            # Full path for inference

            return (
                sample,
                label,
                globals_event,
                reconstructed_pid,
                RICH_pid,
                RICH_RQ,
                rec_theta,
                rec_phi,
                cherenkov_angle,
                aerogel_layer,
            )

    def __del__(self):
        for f in self.files:
            try:
                f.close()
            except Exception:
                pass
