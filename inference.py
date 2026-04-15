from torch.utils.data import DataLoader
from dataset import H5Dataset
import torch
import argparse
import os
from utils import LoadYaml
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import glob
import time


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="data.yaml",
        help="Basic config file containing general options",
        type=str,
    )
    parser.add_argument(
        "--training_config",
        default="training.yaml",
        help="Basic config file containing training options",
        type=str,
    )
    parser.add_argument(
        "--config_directory",
        default="./configs/",
        help="Directory containing the config files",
        type=str,
    )

    return parser.parse_args()


def collate_inference(batch):
    (
        hits,
        labels,
        globals_event,
        reconstructed_pid,
        RICH_pid,
        RICH_RQ,
        rec_theta,
        rec_phi,
        cherenkov_angle,
        aerogel_layer,
    ) = zip(*batch)
    lengths = torch.tensor([h.size(0) for h in hits])
    hits_padded = pad_sequence(hits, batch_first=True)
    max_len = hits_padded.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    return (
        hits_padded,
        torch.stack(labels),
        torch.stack(globals_event),
        mask,
        torch.stack(reconstructed_pid),
        torch.stack(RICH_pid),
        torch.stack(RICH_RQ),
        torch.stack(rec_theta),
        torch.stack(rec_phi),
        torch.stack(cherenkov_angle),
        torch.stack(aerogel_layer),
    )


def main():
    flags = parse_arguments()
    data_parameters = LoadYaml(flags.config, flags.config_directory)
    training_parameters = LoadYaml(flags.training_config, flags.config_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    # --- Load test files ---
    data_path = os.path.join(data_parameters["SAVE_DIRECTORY"], "*_test.h5")
    test_files = sorted(glob.glob(data_path))
    print("Found test files:", test_files)

    full_dataset = H5Dataset(test_files, mode="inference")

    dataloader = DataLoader(
        full_dataset,
        batch_size=training_parameters.get("BATCH_SIZE", 64),
        shuffle=False,
        collate_fn=collate_inference,
        num_workers=4 if use_cuda else 0,  # safer fallback
        pin_memory=use_cuda,
        persistent_workers=use_cuda,
    )

    # --- Load model ---
    from model import GravNetModel

    print("Loading model")
    model = GravNetModel(
        hit_dim=training_parameters.get("HIT_DIMENSIONS", 3),
        global_dim=training_parameters.get("GLOBAL_DIMENSIONS", 10),
        hidden_dim=training_parameters.get("HIDDEN_DIMENSIONS", 64),
        num_classes=training_parameters.get("NUMBER_CLASSES", 2),
        k=training_parameters.get("k", 16),
        dropout_rate=training_parameters.get("DROPOUT_RATE", 0),
    ).to(device)

    model.load_state_dict(
        torch.load(
            os.path.join(
                training_parameters["MODEL_SAVE_DIRECTORY"],
                "checkpoints/best_model.pth",
            ),
            map_location=device,
        )
    )
    model.eval()

    # --- Prepare storage for outputs ---
    all_probabilities = []
    all_labels = []
    all_reco_pid = []
    all_RICH_pid, all_RICH_RQ = [], []
    all_momentum = []
    all_rec_theta, all_rec_phi = [], []
    all_RICH_hits_x, all_RICH_hits_y, all_RICH_hits_time = [], [], []
    all_mask = []
    (
        all_rec_traj_x,
        all_rec_traj_y,
        all_rec_traj_cx,
        all_rec_traj_cy,
        all_rec_traj_cz,
    ) = ([], [], [], [], [])
    all_cherenkov_angles = []
    all_aerogel_layers = []

    print("Making predictions")
    load_times, compute_times = [], []
    end_of_compute = time.time()

    with torch.no_grad():
        for batch in dataloader:
            # --- measure load time ---
            load_start = end_of_compute
            (
                hits_padded,
                labels,
                globals_event,
                mask,
                reco_pid,
                RICH_PID,
                RICH_RQ,
                rec_theta,
                rec_phi,
                cherenkov_angle,
                aerogel_layer,
            ) = batch
            load_end = time.time()
            load_times.append(load_end - load_start)

            # --- Transfer to GPU ---
            hits_padded = hits_padded.to(device, non_blocking=use_cuda)
            globals_event = globals_event.to(device, non_blocking=use_cuda)
            mask = mask.to(device, non_blocking=use_cuda)

            # --- Compute ---
            compute_start = time.time()
            outputs = model(hits_padded, globals_event, mask)
            probs = torch.softmax(outputs, dim=1)
            if use_cuda:
                torch.cuda.synchronize()
            end_of_compute = time.time()
            compute_times.append(end_of_compute - compute_start)

            # --- Store results ---
            momentum = globals_event[:, 0]
            momentum_unscaled = momentum * (12 - 1) + 1
            all_momentum.append(momentum_unscaled.cpu())

            traj_x = globals_event[:, 1]
            all_rec_traj_x.append(traj_x.cpu())
            traj_y = globals_event[:, 2]
            all_rec_traj_y.append(traj_y.cpu())
            traj_cx = globals_event[:, 3]
            all_rec_traj_cx.append(traj_cx.cpu())
            traj_cy = globals_event[:, 4]
            all_rec_traj_cy.append(traj_cy.cpu())
            traj_cz = globals_event[:, 5]
            all_rec_traj_cz.append(traj_cz.cpu())

            all_probabilities.append(probs.cpu())
            all_labels.append(labels)
            all_reco_pid.append(reco_pid)
            all_RICH_pid.append(RICH_PID)
            all_RICH_RQ.append(RICH_RQ)
            all_rec_theta.append(rec_theta)
            all_rec_phi.append(rec_phi)
            all_cherenkov_angles.append(cherenkov_angle)
            all_aerogel_layers.append(aerogel_layer)

            rich_hit_time = hits_padded[:, :, 0]
            rich_hit_time_unscaled = rich_hit_time * (19500 - 125) + (125)
            all_RICH_hits_time.append(rich_hit_time_unscaled.cpu())

            rich_hit_x = hits_padded[:, :, 1]
            rich_hit_x_unscaled = rich_hit_x * (-37 - (-166)) + (-166)
            all_RICH_hits_x.append(rich_hit_x_unscaled.cpu())

            rich_hit_y = hits_padded[:, :, 2]
            rich_hit_y_unscaled = rich_hit_y * (78 - (-81)) + (-81)
            all_RICH_hits_y.append(rich_hit_y_unscaled.cpu())

            all_mask.append(mask.cpu())

    # --- Concatenate results ---
    all_probabilities = torch.cat(all_probabilities)
    all_labels = torch.cat(all_labels)
    all_reco_pid = torch.cat(all_reco_pid)
    all_momentum = torch.cat(all_momentum)
    all_RICH_pid = torch.cat(all_RICH_pid)
    all_RICH_RQ = torch.cat(all_RICH_RQ)
    all_rec_theta = torch.cat(all_rec_theta)
    all_rec_phi = torch.cat(all_rec_phi)
    all_cherenkov_angles = torch.cat(all_cherenkov_angles)
    all_aerogel_layers = torch.cat(all_aerogel_layers)
    all_rec_traj_x = torch.cat(all_rec_traj_x)
    all_rec_traj_y = torch.cat(all_rec_traj_y)
    all_rec_traj_cx = torch.cat(all_rec_traj_cx)
    all_rec_traj_cy = torch.cat(all_rec_traj_cy)
    all_rec_traj_cz = torch.cat(all_rec_traj_cz)

    # --- Save predictions ---
    output_file = os.path.join(
        training_parameters["MODEL_SAVE_DIRECTORY"], "test_predictions.pt"
    )
    torch.save(
        {
            "probabilities": all_probabilities,
            "labels": all_labels,
            "reco_pid": all_reco_pid,
            "RICH_PID": all_RICH_pid,
            "RICH_RQ": all_RICH_RQ,
            "reco_momentum": all_momentum,
            "reco_theta": all_rec_theta,
            "reco_phi": all_rec_phi,
            "reco_traj_x": all_rec_traj_x,
            "reco_traj_y": all_rec_traj_y,
            "reco_traj_cx": all_rec_traj_cx,
            "reco_traj_cy": all_rec_traj_cy,
            "reco_traj_cz": all_rec_traj_cz,
            "RICH_hits_x": all_RICH_hits_x,
            "RICH_hits_y": all_RICH_hits_y,
            "RICH_hits_time": all_RICH_hits_time,
            "RICH_hits_mask": all_mask,
            "RICH_cherenkov_angle": all_cherenkov_angles,
            "RICH_aerogel_layer": all_aerogel_layers,
        },
        output_file,
    )

    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
