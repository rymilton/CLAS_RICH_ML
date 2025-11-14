from torch.utils.data import DataLoader
from dataset import H5Dataset
import torch
import argparse
import os
from utils import LoadYaml
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

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

def collate_fn(batch):
    # batch is a list of (hits, label, globals_event) tuples
    hits, labels, globals_event, reco_pid, RICH_PID, RICH_RQ, rec_theta, rec_phi, cherenkov_angle = zip(*batch)

    # hits is a list of tensors with shape [num_hits, 3]
    lengths = [h.size(0) for h in hits]

    # pad hits to the longest event
    hits_padded = pad_sequence(hits, batch_first=True)  # (B, max_len, 3)

    # make mask: 1 for real hit, 0 for pad
    max_len = hits_padded.size(1)
    mask = torch.zeros(len(hits), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1

    # stack labels and globals_event
    labels = torch.stack(labels)              # (B, ...)
    globals_event = torch.stack(globals_event)  # (B, ...)
    reco_pid = torch.stack(reco_pid)
    RICH_PID = torch.stack(RICH_PID)
    RICH_RQ = torch.stack(RICH_RQ)
    rec_theta = torch.stack(rec_theta)
    rec_phi = torch.stack(rec_phi)
    cherenkov_angle = torch.stack(cherenkov_angle)

    return hits_padded, labels, globals_event, mask, reco_pid, RICH_PID, RICH_RQ, rec_theta, rec_phi, cherenkov_angle

def main():
    flags = parse_arguments()
    data_parameters = LoadYaml(flags.config, flags.config_directory)
    training_parameters = LoadYaml(flags.training_config, flags.config_directory)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # --- Load test dataset ---
    data_path = os.path.join(
        data_parameters["SAVE_DIRECTORY"],
        data_parameters["SAVE_FILE_NAME"]+"_test.h5"
    )
    test_dataset = H5Dataset(data_path)
    dataloader = DataLoader(
        test_dataset,
        batch_size=training_parameters.get("BATCH_SIZE", 64),
        shuffle=False,
        collate_fn=collate_fn
    )
    print("Size of test dataset:", len(test_dataset))

    # --- Load model ---
    from model import GravNetModel
    print("Loading model")
    model = GravNetModel(
        hit_dim=training_parameters.get("HIT_DIMENSIONS", 3),
        global_dim=training_parameters.get("GLOBAL_DIMENSIONS", 10),
        hidden_dim=training_parameters.get("HIDDEN_DIMENSIONS", 64),
        num_classes=training_parameters.get("NUMBER_CLASSES", 2),
        k=training_parameters.get("k", 16),
        dropout_rate=training_parameters.get("DROPOUT_RATE", 0)
    ).to(device)

    model.load_state_dict(torch.load(training_parameters["MODEL_SAVE_DIRECTORY"]+"/final_model.pth", map_location=device))
    model.eval()  # Important for inference

    all_probabilities = []
    all_labels = []
    all_reco_pid = []
    all_RICH_pid, all_RICH_RQ = [], []
    all_momentum = []
    all_rec_theta, all_rec_phi = [], []
    all_RICH_hits_x, all_RICH_hits_y, all_RICH_hits_time = [], [], []
    all_mask = []
    all_rec_traj_x, all_rec_traj_y, all_rec_traj_cx, all_rec_traj_cy, all_rec_traj_cz = [], [], [], [], []
    all_cherenkov_angles = []
    print("Making predictions")
    with torch.no_grad():
        for i, (hits_padded, labels, globals_event, mask, reco_pid, RICH_PID, RICH_RQ, rec_theta, rec_phi, cherenkov_angle) in enumerate(dataloader):
            hits_padded = hits_padded.to(device)
            globals_event = globals_event.to(device)
            mask = mask.to(device)
            
            outputs = model(hits_padded, globals_event, mask)
            probs = F.sigmoid(outputs)

            # print(globals_event)
            momentum = globals_event[:, 0]
            momentum_unscaled = momentum*(12 - 1) + 1
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
            all_labels.append(labels)  # optional, if you want to compare
            all_reco_pid.append(reco_pid)
            all_RICH_pid.append(RICH_PID)
            all_RICH_RQ.append(RICH_RQ)
            all_rec_theta.append(rec_theta)
            all_rec_phi.append(rec_phi)
            all_cherenkov_angles.append(cherenkov_angle)

            rich_hit_time = hits_padded[:, :, 0]
            rich_hit_time_unscaled = rich_hit_time*(19500 - 125) + (125)
            all_RICH_hits_time.append(rich_hit_time_unscaled.cpu())

            rich_hit_x = hits_padded[:, :, 1]
            rich_hit_x_unscaled = rich_hit_x*(-37 - (-166)) + (-166)
            all_RICH_hits_x.append(rich_hit_x_unscaled.cpu())

            rich_hit_y = hits_padded[:, :, 2]
            rich_hit_y_unscaled = rich_hit_y*(78 - (-81)) + (-81)
            all_RICH_hits_y.append(rich_hit_y_unscaled.cpu())

            all_mask.append(mask.cpu())

            

    print("Done with predictions. Saving.")
    # Concatenate results
    all_probabilities = torch.cat(all_probabilities)
    all_labels = torch.cat(all_labels)
    all_reco_pid = torch.cat(all_reco_pid)
    all_momentum = torch.cat(all_momentum)
    all_RICH_pid = torch.cat(all_RICH_pid)
    all_RICH_RQ = torch.cat(all_RICH_RQ)
    all_rec_theta = torch.cat(all_rec_theta)
    all_rec_phi = torch.cat(all_rec_phi)
    all_cherenkov_angles = torch.cat(all_cherenkov_angles)
    all_rec_traj_x = torch.cat(all_rec_traj_x)
    all_rec_traj_y = torch.cat(all_rec_traj_y)
    all_rec_traj_cx = torch.cat(all_rec_traj_cx)
    all_rec_traj_cy = torch.cat(all_rec_traj_cy)
    all_rec_traj_cz = torch.cat(all_rec_traj_cz)
    # --- Save predictions ---
    output_file = os.path.join(training_parameters["MODEL_SAVE_DIRECTORY"], "test_predictions.pt")
    torch.save({
        "probabilities": all_probabilities,
        "labels": all_labels,
        "reco_pid": all_reco_pid,
        "RICH_PID": all_RICH_pid,
        "RICH_RQ": all_RICH_RQ,
        "reco_momentum":all_momentum,
        "reco_theta":all_rec_theta,
        "reco_phi":all_rec_phi,
        "reco_traj_x":all_rec_traj_x,
        "reco_traj_y":all_rec_traj_y,
        "reco_traj_cx":all_rec_traj_cx,
        "reco_traj_cy":all_rec_traj_cy,
        "reco_traj_cz":all_rec_traj_cz,
        "RICH_hits_x":all_RICH_hits_x,
        "RICH_hits_y":all_RICH_hits_y,
        "RICH_hits_time":all_RICH_hits_time,
        "RICH_hits_mask":all_mask,
        "RICH_cherenkov_angle": all_cherenkov_angles,
    }, output_file)

    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
