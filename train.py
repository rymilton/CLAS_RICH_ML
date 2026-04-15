from torch.utils.data import DataLoader, random_split, Dataset
from dataset_thetaphicut import H5Dataset
import torch
import argparse
import os
from utils import LoadYaml
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import time
import glob


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


# ---------------- Collate Function ----------------
def collate_train(batch):
    hits, labels, globals_event = zip(*batch)
    lengths = torch.tensor([h.size(0) for h in hits])
    hits_padded = pad_sequence(hits, batch_first=True)  # (B, max_len, hit_dim)
    max_len = hits_padded.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]  # (B, max_len)

    # Convert labels from [1,0] or [0,1] to 0 or 1
    labels_stacked = torch.stack(labels)  # (B, 2)
    labels_binary = labels_stacked.argmax(dim=1).long()  # (B,)
    return hits_padded, labels_binary, torch.stack(globals_event), mask


# ---------------- Preloaded Dataset ----------------
class PreloadedDataset(Dataset):
    def __init__(self, h5_dataset=None, file_path=None):
        """
        Either provide h5_dataset to preload from HDF5, or load from a saved .pt file
        """
        if file_path:
            print(f"Loading preloaded dataset from {file_path}...")
            data = torch.load(file_path)
            self.samples = data["samples"]
            self.labels = data["labels"]
            self.globals = data["globals"]
            print(f"Loaded {len(self.samples)} events from disk.")
        elif h5_dataset:
            print("Preloading dataset into RAM from HDF5...")
            self.samples, self.labels, self.globals = [], [], []
            for i in range(len(h5_dataset)):
                sample, label, globals_event, *_ = h5_dataset[i]
                self.samples.append(sample)
                self.labels.append(label)
                self.globals.append(globals_event)
            print(f"Preloading complete. {len(self.samples)} events loaded.")
        else:
            raise ValueError("Provide either h5_dataset or file_path")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.globals[idx]


# ---------------- Training ----------------
def main():
    print("Starting up")
    flags = parse_arguments()
    data_parameters = LoadYaml(flags.config, flags.config_directory)
    training_parameters = LoadYaml(flags.training_config, flags.config_directory)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = os.path.join(data_parameters["SAVE_DIRECTORY"], "*_train.h5")
    train_files = sorted(glob.glob(data_path))
    print("Opening training data")
    print(train_files)
    # Load H5Dataset
    full_dataset = H5Dataset(train_files)

    # Path to save preloaded dataset
    preloaded_file = os.path.join(
        data_parameters["SAVE_DIRECTORY"], "preloaded_dataset.pt"
    )

    # Preload dataset if file doesn't exist
    if os.path.exists(preloaded_file):
        preloaded_dataset = PreloadedDataset(file_path=preloaded_file)
    else:
        preloaded_dataset = PreloadedDataset(h5_dataset=full_dataset)
        torch.save(
            {
                "samples": preloaded_dataset.samples,
                "labels": preloaded_dataset.labels,
                "globals": preloaded_dataset.globals,
            },
            preloaded_file,
        )
        print(f"Preloaded dataset saved to {preloaded_file}")

    # Split train/val
    val_fraction = training_parameters.get("VALIDATION_SPLIT", 0.2)
    val_size = int(len(preloaded_dataset) * val_fraction)
    train_size = len(preloaded_dataset) - val_size
    train_dataset, val_dataset = random_split(preloaded_dataset, [train_size, val_size])

    print("TRAINING SIZE:", train_size)
    print("VALIDATION SIZE:", val_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_parameters.get("BATCH_SIZE", 64),
        shuffle=True,
        collate_fn=collate_train,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_parameters.get("BATCH_SIZE", 64),
        shuffle=False,
        collate_fn=collate_train,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    print("Setting up model")
    from model import GravNetModel

    model = GravNetModel(
        hit_dim=training_parameters.get("HIT_DIMENSIONS", 3),
        global_dim=training_parameters.get("GLOBAL_DIMENSIONS", 10),
        hidden_dim=training_parameters.get("HIDDEN_DIMENSIONS", 64),
        num_classes=training_parameters.get("NUMBER_CLASSES", 2),
        k=training_parameters.get("k", 16),
        dropout_rate=training_parameters.get("DROPOUT_RATE", 0),
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training_parameters.get("LEARNING_RATE", 1e-3))
    )

    # Learning rate scheduler (configurable in training.yaml)
    scheduler_type = training_parameters.get("SCHEDULER", "StepLR")
    if scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(training_parameters.get("SCHEDULER_STEP_SIZE", 50)),
            gamma=float(training_parameters.get("SCHEDULER_GAMMA", 0.5)),
        )
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(training_parameters.get("SCHEDULER_GAMMA", 0.5)),
            patience=int(training_parameters.get("SCHEDULER_PATIENCE", 10)),
            threshold=float(training_parameters.get("SCHEDULER_THRESHOLD", 1e-3)),
            cooldown=int(training_parameters.get("SCHEDULER_COOLDOWN", 5)),
            min_lr=float(training_parameters.get("SCHEDULER_MIN_LR", 1e-6)),
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    checkpoint_dir = os.path.join(
        training_parameters["MODEL_SAVE_DIRECTORY"], "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Early stopping config
    early_stopping = bool(training_parameters.get("EARLY_STOPPING", True))
    early_stopping_patience = int(
        training_parameters.get("EARLY_STOPPING_PATIENCE", 30)
    )
    early_stopping_min_delta = float(
        training_parameters.get("EARLY_STOPPING_MIN_DELTA", 1e-4)
    )
    epochs_since_improvement = 0
    best_val_loss = float("inf")

    print("Starting training")
    all_epoch_metrics = []
    training_time = time.time()

    for epoch in range(training_parameters.get("EPOCHS", 10)):
        print(f"Starting epoch {epoch}")
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        load_times, compute_times = [], []
        end_of_compute = time.time()

        for i, (hits_padded, labels, globals_event, mask) in enumerate(train_loader):
            # -------- Data loading time --------
            load_start = end_of_compute
            load_end = time.time()
            load_times.append(load_end - load_start)

            # -------- Transfer to GPU --------
            hits_padded = hits_padded.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            globals_event = globals_event.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # -------- Compute time --------
            compute_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(hits_padded, globals_event, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            end_of_compute = time.time()
            compute_times.append(end_of_compute - compute_start)
            train_loss += loss.item()

        avg_train_loss = train_loss / len(load_times)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hits_padded, labels, globals_event, mask in val_loader:
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

        # Step scheduler
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr:.6g}")

        # Early stopping update
        if avg_val_loss + early_stopping_min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0

            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved BEST model to {best_model_path}")
        else:
            epochs_since_improvement += 1
            print(
                f"No improvement in val loss for {epochs_since_improvement}/{early_stopping_patience} epochs"
            )

        all_epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
            }
        )

        # Early stopping check
        if early_stopping and epochs_since_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered (no improvement for {early_stopping_patience} epochs)."
            )
            break

        # --- Save checkpoint ---
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # --- Save final model ---
    final_model_path = os.path.join(
        training_parameters["MODEL_SAVE_DIRECTORY"], "final_model.pth"
    )
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

    # --- Save loss history ---
    loss_file = os.path.join(
        training_parameters["MODEL_SAVE_DIRECTORY"], "training_losses.pt"
    )
    torch.save(all_epoch_metrics, loss_file)
    print(f"Training losses saved to {loss_file}")
    print(f"Training took {time.time() - training_time} s")


if __name__ == "__main__":
    main()
