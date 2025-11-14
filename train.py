from torch.utils.data import DataLoader, random_split
from dataset import H5Dataset
import torch
import argparse
import os 
from utils import LoadYaml
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
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
        help="Basic config file containing general options",
        type=str,
    )
    parser.add_argument(
        "--config_directory",
        default="./configs/",
        help="Directory containing the config files",
        type=str,
    )

    flags = parser.parse_args()

    return flags

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
    print("Starting up")
    flags = parse_arguments()
    data_parameters = LoadYaml(flags.config, flags.config_directory)
    training_parameters = LoadYaml(flags.training_config, flags.config_directory)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = os.path.join(
        data_parameters["SAVE_DIRECTORY"],
        data_parameters["SAVE_FILE_NAME"]+"_train.h5"
        )
    print("Opening training data")
    full_dataset = H5Dataset(data_path)
    val_fraction = training_parameters.get("VALIDATION_SPLIT", 0.2)
    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print("TRAINING SIZE:", train_size)
    print("VALIDATION SIZE:", val_size)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_parameters.get("BATCH_SIZE", 64), 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_parameters.get("BATCH_SIZE", 64), 
        shuffle=False, 
        collate_fn=collate_fn
    )

    print("Setting up model")
    from model import GravNetModel   # wherever your model is defined
    model = GravNetModel(
        hit_dim=training_parameters.get("HIT_DIMENSIONS",3),
        global_dim=training_parameters.get("GLOBAL_DIMENSIONS",10),
        hidden_dim=training_parameters.get("HIDDEN_DIMENSIONS",64),
        num_classes=training_parameters.get("NUMBER_CLASSES",2),
        k=training_parameters.get("k",16),
        dropout_rate=training_parameters.get("DROPOUT_RATE", 0)
    ).to(device)
    # --- Optimizer + Loss ---
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_parameters.get("LEARNING_RATE",1e-3)))
    checkpoint_dir = os.path.join(training_parameters["MODEL_SAVE_DIRECTORY"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Starting training")

    all_epoch_metrics = []
    training_time = time.time()
    for epoch in range(training_parameters.get("EPOCHS", 10)):
        print(f"Starting epoch {epoch}")
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0.0
        for hits_padded, labels, globals_event, mask, _, _, _, _, _, _ in train_loader:
            hits_padded = hits_padded.to(device)
            labels = labels.to(device)
            globals_event = globals_event.to(device)
            mask = mask.to(device)

            model.zero_grad()
            outputs = model(hits_padded, globals_event, mask)
            loss = criterion(outputs, labels)
            loss.backward() # Calculating gradients
            optimizer.step() # Updating parameters

            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hits_padded, labels, globals_event, mask, _, _, _, _, _, _ in val_loader:
                hits_padded = hits_padded.to(device)
                labels = labels.to(device)
                globals_event = globals_event.to(device)
                mask = mask.to(device)

                outputs = model(hits_padded, globals_event, mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch time: {time.time() - epoch_start_time:.2f}s")

        all_epoch_metrics.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        # --- Save checkpoint every epoch ---
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # --- Save final model ---
    final_model_path = os.path.join(training_parameters["MODEL_SAVE_DIRECTORY"], "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

    # --- Save loss history ---
    loss_file = os.path.join(training_parameters["MODEL_SAVE_DIRECTORY"], "training_losses.pt")
    torch.save(all_epoch_metrics, loss_file)
    print(f"Training losses saved to {loss_file}")
    print(f"Training took {time.time() - training_time} s")

if __name__ == "__main__":
    main()