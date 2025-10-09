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
    hits, labels, globals_event = zip(*batch)

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

    return hits_padded, labels, globals_event, mask

def main():
    flags = parse_arguments()
    data_parameters = LoadYaml(flags.config, flags.config_directory)
    training_parameters = LoadYaml(flags.training_config, flags.config_directory)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = os.path.join(
        data_parameters["SAVE_DIRECTORY"],
        data_parameters["SAVE_FILE_NAME"]+"_train.h5"
        )

    training_dataset = H5Dataset(data_path)
    dataloader = DataLoader(training_dataset, batch_size=training_parameters.get("BATCH_SIZE",64), shuffle=True, collate_fn=collate_fn)

    print("Setting up model")
    from model import GravNetModel   # wherever your model is defined
    model = GravNetModel(
        hit_dim=training_parameters.get("HIT_DIMENSIONS",3),
        global_dim=training_parameters.get("GLOBAL_DIMENSIONS",10),
        hidden_dim=training_parameters.get("HIDDEN_DIMENSIONS",64),
        num_classes=training_parameters.get("NUMBER_CLASSES",2),
        k=training_parameters.get("k",16)
    ).to(device)
    # --- Optimizer + Loss ---
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_dir = os.path.join(training_parameters["MODEL_SAVE_DIRECTORY"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Starting training")

    for epoch in range(training_parameters.get("EPOCHS", 10)):
        print(f"Starting epoch {epoch}")
        running_loss = 0.0
        for hits_padded, labels, globals_event, mask in dataloader:
            hits_padded = hits_padded.to(device)
            labels = labels.to(device)
            globals_event = globals_event.to(device)
            mask = mask.to(device)

            model.zero_grad()
            outputs = model(hits_padded, globals_event, mask)
            loss = criterion(outputs, labels)
            loss.backward() # Calculating gradients
            optimizer.step() # Updating parameters

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{training_parameters.get('EPOCHS', 10)} - Loss: {avg_loss:.4f}")

        # --- Save checkpoint every epoch ---
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # --- Save final model ---
    final_model_path = os.path.join(training_parameters["MODEL_SAVE_DIRECTORY"], "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()