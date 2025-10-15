import torch
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

model_dir = "//volatile/clas12/rmilton/RICH_models/RICH_sector4_positives_kaons_pions_100epochs/"

losses = torch.load(model_dir+"/training_losses.pt")
loss_epochs = [losses[i]["epoch"] for i in range(len(losses))]
train_loss_per_epoch = [losses[i]["train_loss"] for i in range(len(losses))]
validation_loss_per_epoch = [losses[i]["val_loss"] for i in range(len(losses))]

predictions = torch.load(model_dir+"/test_predictions.pt")
model_probabilities = predictions["probabilities"].cpu().detach().numpy()
true_labels = predictions["labels"].cpu().detach().numpy()
reconstructed_pid = predictions["reco_pid"].cpu().detach().numpy()[:, 0]
reconstructed_momentum = predictions["reco_momentum"].cpu().detach().numpy()

reco_accuracy = 0
pion_events_mask = (true_labels==[0, 1])[:, 0]
kaon_events_mask = (true_labels==[1, 0])[:, 0]

model_probabilities_for_kaons = model_probabilities[:, 0]
model_probabilities_for_pions = model_probabilities[:, 1]


# Loss curve
figure_loss = plt.figure(figsize=(12,8))
plt.plot(loss_epochs, train_loss_per_epoch, label="Training")
plt.plot(loss_epochs, validation_loss_per_epoch, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss (BCEWithLogits)")
plt.title("RICH Sector 4 positives, 100 Epochs")
plt.legend()
plt.savefig('losses.png')

# Momentum distribution
figure_loss = plt.figure(figsize=(12,8))
plt.hist(reconstructed_momentum, bins=50)
plt.xlabel("p (GeV)")
plt.title("RICH Sector 4 positives, 100 Epochs")
plt.yscale('log')
plt.ylabel("Counts (log)")
plt.savefig('momentum.png')


# Model probabilties for kaons
momentum_bins = np.linspace(2.3, 7.3, 11)
momentum_bin_centers = (momentum_bins[1:] + momentum_bins[:-1])/2

fig_kaons, axs_kaons = plt.subplots(nrows=5, ncols=2, figsize=(12,16))
axs_kaons = axs_kaons.flatten()
for i in range(len(momentum_bins)-1):
    lower_bin, upper_bin = momentum_bins[i], momentum_bins[i+1]
    momentum_mask = (reconstructed_momentum>=lower_bin) & (reconstructed_momentum < upper_bin)

    model_probabilities_for_kaons_momentum_bin = model_probabilities_for_kaons[momentum_mask]
    total_counts = len(model_probabilities_for_kaons_momentum_bin)
    kaon_events_mask_momentum_bin = kaon_events_mask[momentum_mask]
    pion_events_mask_momentum_bin = pion_events_mask[momentum_mask]

    axs_kaons[i].hist(
        model_probabilities_for_kaons_momentum_bin[kaon_events_mask_momentum_bin],
        range=(0,1),
        bins=20,
        label="True label: $K^+$",
        alpha=0.5,
    )
    axs_kaons[i].hist(
        model_probabilities_for_kaons_momentum_bin[pion_events_mask_momentum_bin],
        range=(0,1),
        bins=20,
        label="True label: $\pi^+$",
        alpha=0.5,
    )
    axs_kaons[i].legend(fontsize=12)
    axs_kaons[i].set_title(f"${round(lower_bin,3)}~GeV~ \leq p < {round(upper_bin,3)}~GeV$\n Total counts={total_counts}", fontsize=12)
plt.xlabel("Probability event contains $K^+$")
plt.ylabel("Entries")
plt.suptitle("RICH Sector 4 positives, 100 Epochs")
plt.tight_layout()
plt.savefig('kaon_probabilties.png')

# Model probabilties for pions
fig_pions, axs_pions = plt.subplots(nrows=5, ncols=2, figsize=(10,16))
axs_pions = axs_pions.flatten()

pion_event_probabilities = []
pion_mask_per_momentum = []
for i in range(len(momentum_bins)-1):
    lower_bin, upper_bin = momentum_bins[i], momentum_bins[i+1]
    momentum_mask = (reconstructed_momentum>=lower_bin) & (reconstructed_momentum < upper_bin)

    model_probabilities_for_pions_momentum_bin = model_probabilities_for_pions[momentum_mask]
    pion_event_probabilities.append(model_probabilities_for_pions_momentum_bin)
    total_counts = len(model_probabilities_for_pions_momentum_bin)
    kaon_events_mask_momentum_bin = kaon_events_mask[momentum_mask]
    pion_events_mask_momentum_bin = pion_events_mask[momentum_mask]
    pion_mask_per_momentum.append(pion_events_mask[momentum_mask])
    axs_pions[i].hist(
        model_probabilities_for_pions_momentum_bin[kaon_events_mask_momentum_bin],
        range=(0,1),
        bins=20,
        label="True label: $K^+$",
        alpha=0.5,
    )
    axs_pions[i].hist(
        model_probabilities_for_pions_momentum_bin[pion_events_mask_momentum_bin],
        range=(0,1),
        bins=20,
        label="True label: $\pi^+$",
        alpha=0.5,
    )
    axs_pions[i].legend(fontsize=12)
    axs_pions[i].set_title(f"${round(lower_bin,3)}~GeV~ \leq p < {round(upper_bin,3)}~GeV$\n Total counts={total_counts}", fontsize=12)
    axs_pions[i].set_xlabel("Probability event contains $\pi^+$", fontsize=12)
    axs_pions[i].set_ylabel("Counts", fontsize=12)
plt.suptitle("RICH Sector 4 positives, 100 Epochs")
plt.tight_layout()
plt.savefig('pion_probabilties.png')

# Model efficiencies for different probability thresholds
probability_thresholds = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
fig_thresholds, axs_threshold = plt.subplots(nrows=2, ncols=3, figsize=(16,10))
axs_threshold = axs_threshold.flatten()
for i, threshold in enumerate(probability_thresholds):
    pion_accuracies = []
    kaon_misidentifications = []
    for probabilities_per_momentum, pion_events_per_momentum in zip(pion_event_probabilities, pion_mask_per_momentum):
        threshold_mask = probabilities_per_momentum > threshold
        probabilities_passing_threshold = probabilities_per_momentum[threshold_mask]
        pion_events = pion_events_per_momentum[threshold_mask]

        num_correct_pion_events = len(probabilities_passing_threshold[pion_events])
        num_incorrect_kaon_events = len(probabilities_passing_threshold[~pion_events])

        pion_accuracies.append(num_correct_pion_events/len(probabilities_per_momentum[pion_events_per_momentum]))
        kaon_misidentifications.append(num_incorrect_kaon_events/len(probabilities_per_momentum[~pion_events_per_momentum]))

    axs_threshold[i].scatter(momentum_bin_centers, pion_accuracies, color='r', label="$\pi^+$")
    axs_threshold[i].scatter(momentum_bin_centers, kaon_misidentifications, color='b', label="$K^+$")
    axs_threshold[i].set_xlabel("p (GeV)", fontsize=12)
    axs_threshold[i].set_ylabel("Efficiency of passing $\pi^+$ cut", fontsize=12)
    axs_threshold[i].legend(fontsize=12)
    axs_threshold[i].set_title(f"Probability Threshold = {threshold}", fontsize=12)
    axs_threshold[i].set_ylim(-.1,1.1)
    axs_threshold[i].grid()
plt.suptitle(f"RICH Sector 4 positives, 100 Epochs")
plt.tight_layout()
plt.savefig(f'thresholds.png')
