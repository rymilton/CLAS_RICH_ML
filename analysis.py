import torch
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from sklearn.metrics import roc_curve, auc
import os
import matplotlib.colors as colors
model_dir = "/volatile/clas12/rmilton/RICH_models_LR1E-3_128hiddendimensions_200epochs_0.1validationsplit_3.15GeVmomentumcut_withbestch/"
print(model_dir)
# plot_title = "RICH Sector 4 positives p>3.15 GeV, 200 Epochs\nLR=1E-3, 128 hidden dimensions, 10% validation"
plot_title = "RICH Sector 4 positives p>3.15 GeV, 200 Epochs\nLR=1E-3, 128 hidden dimensions, 10% validation"
plot_directory = "./plots/"
os.makedirs(plot_directory, exist_ok=True)

# Opening the files
losses = torch.load(model_dir+"/training_losses.pt")
loss_epochs = [losses[i]["epoch"] for i in range(len(losses))]
train_loss_per_epoch = [losses[i]["train_loss"] for i in range(len(losses))]
validation_loss_per_epoch = [losses[i]["val_loss"] for i in range(len(losses))]

predictions = torch.load(model_dir+"/test_predictions.pt")
cherenkov_angles = predictions["RICH_cherenkov_angle"].cpu().detach().numpy()[:, 0]

model_probabilities = predictions["probabilities"].cpu().detach().numpy()
true_labels = predictions["labels"].cpu().detach().numpy()

reconstructed_pid = predictions["reco_pid"].cpu().detach().numpy()[:, 0]
reconstructed_momentum = predictions["reco_momentum"].cpu().detach().numpy()
RICH_PID = predictions["RICH_PID"].cpu().detach().numpy()
RICH_RQ = predictions["RICH_RQ"].cpu().detach().numpy()

rec_traj_x = predictions["reco_traj_x"].cpu().detach().numpy()
rec_traj_y = predictions["reco_traj_y"].cpu().detach().numpy()
rec_traj_cx = predictions["reco_traj_cx"].cpu().detach().numpy()
rec_traj_cy = predictions["reco_traj_cy"].cpu().detach().numpy()
rec_traj_cz = predictions["reco_traj_cz"].cpu().detach().numpy()

rec_theta = ak.flatten(predictions["reco_theta"].cpu().detach().numpy())
rec_phi = predictions["reco_phi"].cpu().detach().numpy()

RICH_hits_x_padded = predictions["RICH_hits_x"]
RICH_hits_y_padded = predictions["RICH_hits_y"]
RICH_hits_time_padded = predictions["RICH_hits_time"]
RICH_hits_mask = predictions["RICH_hits_mask"]
RICH_hits_x, RICH_hits_y, RICH_hits_time = [], [], []
# Loop over batches
for batch_x, batch_y, batch_time, batch_mask in zip(
    RICH_hits_x_padded, RICH_hits_y_padded, RICH_hits_time_padded, RICH_hits_mask
):
    # Loop over events in each batch
    for event_x, event_y, event_time, event_mask in zip(batch_x, batch_y, batch_time, batch_mask):
        valid = event_mask.cpu().numpy().astype(bool)

        # Apply mask and move to numpy (fast)
        x = event_x.cpu().numpy()[valid]
        y = event_y.cpu().numpy()[valid]
        t = event_time.cpu().numpy()[valid]

        RICH_hits_x.append(x)
        RICH_hits_y.append(y)
        RICH_hits_time.append(t)

# Convert all lists of numpy arrays into Awkward Arrays
RICH_hits_x = ak.Array(RICH_hits_x)
RICH_hits_y = ak.Array(RICH_hits_y)
RICH_hits_time = ak.Array(RICH_hits_time)


pion_events_mask = (true_labels==[1, 0])[:, 0]
kaon_events_mask = (true_labels==[0, 1])[:, 0]

model_probabilities_for_kaons = model_probabilities[:, 1]
model_probabilities_for_pions = model_probabilities[:, 0]

# Loss curve
figure_loss = plt.figure(figsize=(12,8))
plt.plot(loss_epochs, train_loss_per_epoch, label="Training")
plt.plot(loss_epochs, validation_loss_per_epoch, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss (BCEWithLogits)")
plt.title(plot_title)
plt.xlim(0, 200)
plt.legend()
plt.tight_layout()
plt.savefig(plot_directory+'losses.png')

# Momentum distribution
figure_loss = plt.figure(figsize=(12,8))
plt.hist(reconstructed_momentum, bins=50)
plt.xlabel("p (GeV)")
plt.title(plot_title)
plt.yscale('log')
plt.ylabel("Counts (log)")
plt.savefig(plot_directory+'momentum.png')


# Model probabilties for kaons
print(min(reconstructed_momentum), max(reconstructed_momentum))
momentum_bins = np.linspace(1.5, 7, 11)
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
plt.suptitle(plot_title)
plt.tight_layout()
plt.savefig(plot_directory+'kaon_probabilties.png')

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
plt.suptitle(plot_title)
plt.tight_layout()
plt.savefig(plot_directory+'pion_probabilties.png')

# Model efficiencies for different probability thresholds
probability_thresholds = np.linspace(0,1,20)
pion_probabilities = model_probabilities_for_pions[pion_events_mask]
kaon_probabilities = model_probabilities_for_pions[kaon_events_mask]

# Combine into one array of scores
y_scores = np.concatenate([pion_probabilities, kaon_probabilities])

# True labels: 1 for pion, 0 for kaon
y_true = np.concatenate([
    np.ones_like(pion_probabilities),
    np.zeros_like(kaon_probabilities)
])

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Compute AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Kaon misidentification')
plt.ylabel('Pion efficiency')
plt.title(plot_title)
plt.legend(loc='lower right')
plt.grid(True)
for i in range(0, len(thresholds), max(1, len(thresholds)//15)):
    plt.text(fpr[i], tpr[i],
             f"{thresholds[i]:.2f}",
             fontsize=8, color='black',
             ha='right', va='bottom', rotation=45)
plt.show()

print(f"AUC = {roc_auc:.4f}")
plt.savefig(plot_directory+"ROC.png")

probability_thresholds = [0.2, 0.4, 0.5, 0.7, 0.8, 0.9]
fig_thresholds, axs_threshold = plt.subplots(nrows=2, ncols=3, figsize=(16,10))
axs_threshold = axs_threshold.flatten()
for i, threshold in enumerate(probability_thresholds):
    pion_accuracies = []
    kaon_misidentifications = []
    empty_momentum_mask = np.ones(len(momentum_bin_centers), dtype=bool)
    for j, (probabilities_per_momentum, pion_events_per_momentum) in enumerate(zip(pion_event_probabilities, pion_mask_per_momentum)):
        if len(probabilities_per_momentum) ==0:
            empty_momentum_mask[j] = 0
            continue
        threshold_mask = probabilities_per_momentum > threshold
        probabilities_passing_threshold = probabilities_per_momentum[threshold_mask]
        pion_events = pion_events_per_momentum[threshold_mask]

        num_correct_pion_events = len(probabilities_passing_threshold[pion_events])
        num_incorrect_kaon_events = len(probabilities_passing_threshold[~pion_events]) # num kaons that pass the pion threshold

        # probabilities_per_momentum[pion_events_per_momentum] is the number of pion events in this momentum bin
        # probabilities_per_momentum[~pion_events_per_momentum] is the number of kaon events in this momentum bin
        if len(probabilities_per_momentum[pion_events_per_momentum]) > 0:
            pion_accuracies.append(num_correct_pion_events/len(probabilities_per_momentum[pion_events_per_momentum]))
        else:
            pion_accuracies.append(-1)
        if len(probabilities_per_momentum[~pion_events_per_momentum]) > 0:
            kaon_misidentifications.append(num_incorrect_kaon_events/len(probabilities_per_momentum[~pion_events_per_momentum]))
        else:
            kaon_misidentifications.append(-1)
    axs_threshold[i].scatter(momentum_bin_centers[empty_momentum_mask], pion_accuracies, color='r', label="$\pi^+$")
    axs_threshold[i].scatter(momentum_bin_centers[empty_momentum_mask], kaon_misidentifications, color='b', label="$K^+$")
    axs_threshold[i].set_xlabel("p (GeV)", fontsize=12)
    axs_threshold[i].set_ylabel("Efficiency of passing $\pi^+$ cut", fontsize=12)
    axs_threshold[i].legend(fontsize=12)
    axs_threshold[i].set_title(f"Probability Threshold = {threshold}", fontsize=12)
    axs_threshold[i].set_ylim(-.1,1.1)
    axs_threshold[i].grid()
    if threshold in [.5,.8]:
        print(f"Average pion accuracy at threshold={threshold}:", np.mean(pion_accuracies))
        print(f"Average kaon misidentifactionat threshold={threshold}:", np.mean(kaon_misidentifications))
        # print(f"Pion accuracies at threshold={threshold}:", pion_accuracies)
        # print(f"Kaon misidentifactions at threshold={threshold}:", kaon_misidentifications)
        ratios = []
        for p, k in zip(pion_accuracies, kaon_misidentifications):
            if k != 0:
                ratios.append(p / k)
            else:
                ratios.append(np.nan)  # or 0, or None, whatever makes sense

        print(f"Pion/kaon ratios at threshold={threshold}: {ratios}")
        print(f"Avg. Pion/kaon ratios at threshold={threshold}: {np.mean(ratios)}")

plt.suptitle(plot_title)
plt.tight_layout()
plt.savefig(plot_directory+f'thresholds.png')

# --- Comparison: Model vs Reconstructed PID (styled like thresholds.png) ---

fig_compare, axs_compare = plt.subplots(nrows=2, ncols=3, figsize=(16,10))
axs_compare = axs_compare.flatten()

for i, threshold in enumerate(probability_thresholds):
    pion_accuracies_model = []
    kaon_misidentifications_model = []

    pion_accuracies_reco = []
    kaon_misidentifications_reco = []

    empty_momentum_mask = np.ones(len(momentum_bin_centers), dtype=bool)
    
    for j, (probabilities_per_momentum, pion_events_per_momentum) in enumerate(zip(pion_event_probabilities, pion_mask_per_momentum)):
        if len(probabilities_per_momentum) == 0:
            empty_momentum_mask[j] = 0
            continue

        # ------------------------------
        # Model performance
        # ------------------------------
        threshold_mask = probabilities_per_momentum > threshold
        probs_pass = probabilities_per_momentum[threshold_mask]
        pions_pass = pion_events_per_momentum[threshold_mask]

        # Pion accuracy (model)
        if np.sum(pion_events_per_momentum) > 0:
            pion_accuracies_model.append(np.sum(pions_pass) / np.sum(pion_events_per_momentum))
        else:
            pion_accuracies_model.append(-1)

        # Kaon misidentification (model)
        if np.sum(~pion_events_per_momentum) > 0:
            kaon_misidentifications_model.append(np.sum(~pions_pass) / np.sum(~pion_events_per_momentum))
        else:
            kaon_misidentifications_model.append(-1)

        # ------------------------------
        # Reconstructed PID performance
        # ------------------------------
        momentum_bin_mask = (reconstructed_momentum >= momentum_bins[j]) & (reconstructed_momentum < momentum_bins[j+1])
        reco_pion_mask_bin = (reconstructed_pid[momentum_bin_mask] == 211)

        # Pion accuracy (reco)
        true_pions_bin = pion_events_per_momentum
        if np.sum(true_pions_bin) > 0:
            pion_accuracies_reco.append(np.sum(reco_pion_mask_bin[true_pions_bin]) / np.sum(true_pions_bin))
        else:
            pion_accuracies_reco.append(-1)

        # Kaon misidentification (reco)
        true_kaons_bin = ~pion_events_per_momentum
        if np.sum(true_kaons_bin) > 0:
            kaon_misidentifications_reco.append(np.sum(reco_pion_mask_bin[true_kaons_bin]) / np.sum(true_kaons_bin))
        else:
            kaon_misidentifications_reco.append(-1)

    # ------------------------------
    # Plot both on same axes
    # ------------------------------
    axs_compare[i].scatter(momentum_bin_centers[empty_momentum_mask], pion_accuracies_model, color='r', marker='o', label="GravNet $\pi^+$ ")
    axs_compare[i].scatter(momentum_bin_centers[empty_momentum_mask], kaon_misidentifications_model, color='b', marker='o', label="GravNet $K^+$")
    
    axs_compare[i].scatter(momentum_bin_centers[empty_momentum_mask], pion_accuracies_reco, color='r', marker='s', label="Reco $\pi^+$", facecolor='none')
    axs_compare[i].scatter(momentum_bin_centers[empty_momentum_mask], kaon_misidentifications_reco, color='b', marker='s', label="Reco $K^+$", facecolor='none')

    axs_compare[i].set_xlabel("p (GeV)", fontsize=12)
    axs_compare[i].set_ylabel("Efficiency of passing $\pi^+$ cut", fontsize=12)
    axs_compare[i].set_title(f"Probability Threshold = {threshold}", fontsize=12)
    axs_compare[i].set_ylim(-0.1, 1.1)
    axs_compare[i].grid()
    axs_compare[i].legend(fontsize=12)

plt.suptitle(plot_title)
plt.tight_layout()
plt.savefig(plot_directory+'model_vs_reco_pid_comparison.png')


# --- Comparison: Model vs RICH_PID (styled like thresholds.png) ---

fig_compare_rich, axs_compare_rich = plt.subplots(nrows=2, ncols=3, figsize=(16,10))
axs_compare_rich = axs_compare_rich.flatten()

# Assume RICH_PID uses the same convention: 211 = pion, 321 = kaon
for i, threshold in enumerate(probability_thresholds):
    pion_accuracies_model = []
    kaon_misidentifications_model = []

    pion_accuracies_rich = []
    kaon_misidentifications_rich = []

    empty_momentum_mask = np.ones(len(momentum_bin_centers), dtype=bool)
    
    for j, (probabilities_per_momentum, pion_events_per_momentum) in enumerate(zip(pion_event_probabilities, pion_mask_per_momentum)):
        if len(probabilities_per_momentum) == 0:
            empty_momentum_mask[j] = 0
            continue

        # ------------------------------
        # Model performance
        # ------------------------------
        threshold_mask = probabilities_per_momentum > threshold
        probs_pass = probabilities_per_momentum[threshold_mask]
        pions_pass = pion_events_per_momentum[threshold_mask]

        if np.sum(pion_events_per_momentum) > 0:
            pion_accuracies_model.append(np.sum(pions_pass) / np.sum(pion_events_per_momentum))
        else:
            pion_accuracies_model.append(-1)

        if np.sum(~pion_events_per_momentum) > 0:
            kaon_misidentifications_model.append(np.sum(~pions_pass) / np.sum(~pion_events_per_momentum))
        else:
            kaon_misidentifications_model.append(-1)

        # ------------------------------
        # RICH PID performance
        # ------------------------------
        momentum_bin_mask = (reconstructed_momentum >= momentum_bins[j]) & (reconstructed_momentum < momentum_bins[j+1])
        rich_pion_mask_bin = (RICH_PID[momentum_bin_mask] == 211)

        true_pions_bin = pion_events_per_momentum
        true_kaons_bin = ~pion_events_per_momentum

        # Pion accuracy (RICH)
        if np.sum(true_pions_bin) > 0:
            pion_accuracies_rich.append(np.sum(rich_pion_mask_bin[true_pions_bin]) / np.sum(true_pions_bin))
        else:
            pion_accuracies_rich.append(-1)

        # Kaon misidentification (RICH)
        if np.sum(true_kaons_bin) > 0:
            kaon_misidentifications_rich.append(np.sum(rich_pion_mask_bin[true_kaons_bin]) / np.sum(true_kaons_bin))
        else:
            kaon_misidentifications_rich.append(-1)

    # ------------------------------
    # Plot both on same axes
    # ------------------------------
    axs_compare_rich[i].scatter(momentum_bin_centers[empty_momentum_mask], pion_accuracies_model, color='r', marker='o', label="GravNet $\pi^+$")
    axs_compare_rich[i].scatter(momentum_bin_centers[empty_momentum_mask], kaon_misidentifications_model, color='b', marker='o', label="GravNet $K^+$")
    
    axs_compare_rich[i].scatter(momentum_bin_centers[empty_momentum_mask], pion_accuracies_rich, color='r', marker='s', label="RICH $\pi^+$", facecolor='none')
    axs_compare_rich[i].scatter(momentum_bin_centers[empty_momentum_mask], kaon_misidentifications_rich, color='b', marker='s', label="RICH $K^+$", facecolor='none')

    axs_compare_rich[i].set_xlabel("p (GeV)", fontsize=12)
    axs_compare_rich[i].set_ylabel("Efficiency of passing $\pi^+$ cut", fontsize=12)
    axs_compare_rich[i].set_title(f"Probability Threshold = {threshold}", fontsize=12)
    axs_compare_rich[i].set_ylim(-0.1, 1.1)
    axs_compare_rich[i].grid()
    axs_compare_rich[i].legend(fontsize=12)

plt.suptitle(plot_title)
plt.tight_layout()
plt.savefig(plot_directory+'model_vs_RICH_PID_comparison.png')
plt.show()

plt.figure()
plt.hist2d(np.array(reconstructed_momentum)[pion_events_mask], np.array(cherenkov_angles)[pion_events_mask], bins=100, range=((0,10), (.05,.4)), norm=colors.LogNorm())
plt.colorbar()
plt.xlabel('p (GeV/c)')
plt.ylabel('$\\theta_{C}$ (rad)')
plt.title(plot_title+ "\n Pion events")
plt.grid(True)
plt.show()
plt.savefig(plot_directory+"cherenkov_angles_pions.png")

plt.figure()
plt.hist2d(np.array(reconstructed_momentum)[kaon_events_mask], np.array(cherenkov_angles)[kaon_events_mask], bins=100, range=((0,10), (.05,.4)), norm=colors.LogNorm())
plt.colorbar()
plt.xlabel('p (GeV/c)')
plt.ylabel('$\\theta_{C}$ (rad)')
plt.title(plot_title+ "\n Kaon events")
plt.grid(True)
plt.show()
plt.savefig(plot_directory+"cherenkov_angles_kaons.png")

model_pion_mask = (model_probabilities_for_pions > 0.5)
plt.figure()
plt.hist2d(np.array(reconstructed_momentum)[model_pion_mask], np.array(cherenkov_angles)[model_pion_mask], bins=100, range=((0,7), (-.05,.4)), norm=colors.LogNorm())
plt.colorbar()
plt.xlabel('p (GeV/c)')
plt.ylabel('$\\theta_{C}$ (rad)')
plt.title(plot_title)
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(plot_directory+f"cherenkov_angles_pionmask.png")

model_kaon_mask = (model_probabilities_for_pions < 0.5)
plt.figure()
plt.hist2d(np.array(reconstructed_momentum)[model_kaon_mask], np.array(cherenkov_angles)[model_kaon_mask], bins=100, range=((0,7), (-.05,.4)), norm=colors.LogNorm())
plt.colorbar()
plt.xlabel('p (GeV/c)')
plt.ylabel('$\\theta_{C}$ (rad)')
plt.title(plot_title)
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(plot_directory+f"cherenkov_angles_kaonmask.png")

print(len(reconstructed_momentum), len(rec_theta))

# mask_names = [all_events, pions_as_pions, pions_as_kaons, kaons_as_pions, kaons_as_kaons]
mask_names = ["all_events", "pions_as_pions", "pions_as_kaons", "kaons_as_pions", "kaons_as_kaons"]
kaon_pion_masks = [
    (np.full_like(reconstructed_momentum, True, dtype=bool))   & (reconstructed_momentum   >3.15) & (reconstructed_momentum   < 3.7),
    (pion_events_mask) & (model_probabilities_for_pions > 0.8) & (reconstructed_momentum >3.15) & (reconstructed_momentum < 3.7),
    (pion_events_mask) & (model_probabilities_for_pions < 0.8) & (reconstructed_momentum >3.15) & (reconstructed_momentum < 3.7),
    (kaon_events_mask) & (model_probabilities_for_pions > 0.8) & (reconstructed_momentum >3.15) & (reconstructed_momentum < 3.7),
    (kaon_events_mask) & (model_probabilities_for_pions < 0.8) & (reconstructed_momentum >3.15) & (reconstructed_momentum < 3.7),
]

# kaon_pion_masks = [
#     (np.full_like(reconstructed_momentum, True, dtype=bool))  ,
#     (pion_events_mask) & (model_probabilities_for_pions > 0.8),
#     (pion_events_mask) & (model_probabilities_for_pions < 0.8),
#     (kaon_events_mask) & (model_probabilities_for_pions > 0.8),
#     (kaon_events_mask) & (model_probabilities_for_pions < 0.8),
# ]
title_dict = {
    "all_events": "All events",
    "pions_as_pions": "Pions classified as pions",
    "pions_as_kaons": "Pions classified as kaons",
    "kaons_as_pions": "Kaons classified as pions",
    "kaons_as_kaons": "Kaons classified as kaons",
}
for mask, mask_name in zip(kaon_pion_masks, mask_names):

    fig = plt.figure()
    plt.hist2d(
        reconstructed_momentum[mask],
        np.array(rec_theta[mask]),
        bins=100,
        range=((1,10), (0,25)),
        norm=colors.LogNorm()
    )
    plt.xlabel("p (GeV)")
    plt.ylabel("$\\theta$ (deg.)")
    plt.colorbar()
    plt.title(plot_title+f"\n {title_dict[mask_name]}, threshold=0.8", fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_directory+f"theta_momentum_{mask_name}.png")
    plt.close()

    fig = plt.figure()
    plt.hist2d(
        np.array(rec_traj_x[mask]),
        np.array(rec_traj_y[mask]),
        bins=50,
        range=((0,1), (0,1)),
        norm=colors.LogNorm()
    )
    plt.xlabel("Traj.x")
    plt.ylabel("Traj.y")
    plt.colorbar()
    plt.title(plot_title+f"\n {title_dict[mask_name]}, threshold=0.8", fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_directory+f"traj_xy_{mask_name}.png")
    plt.close()


    fig = plt.figure()
    plt.hist(ak.flatten(RICH_hits_time[mask]), bins=100, range=(0, 8000))
    plt.xlabel("RICH Raw time (ns)")
    plt.title(plot_title+f"\n {title_dict[mask_name]}, threshold=0.8", fontsize=12)
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(plot_directory+f"RICH_hits_time_{mask_name}.png")
    plt.close()

    fig = plt.figure()
    plt.hist(ak.num(RICH_hits_time[mask], axis=1), bins=100, range=(-0.5, 99.5))
    plt.xlabel("RICH Hit multiplicity")
    plt.title(plot_title+f"\n {title_dict[mask_name]}, threshold=0.8", fontsize=12)
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(plot_directory+f"RICH_hits_multiplicity_{mask_name}.png")
    plt.close()


    if mask_name in ["kaons_as_pions", "pions_as_pions"]:

        # Extract valid indices from the mask
        masked_indices = np.where(mask)[0]

        # Sort them by model probability (descending)
        sorted_indices = masked_indices[
            np.argsort(model_probabilities_for_pions[masked_indices])[::-1]
        ]

        # Take the top 20 events
        top_events = sorted_indices[:20]

        for j, event_idx in enumerate(top_events):
            
            prob = model_probabilities_for_pions[event_idx]

            fig = plt.figure()
            plt.scatter(
                np.array(RICH_hits_x[event_idx]),
                np.array(RICH_hits_y[event_idx]),
            )

            plt.xlabel("RICH x (cm)")
            plt.ylabel("RICH y (cm)")
            plt.xlim(-150, -30)
            plt.ylim(-75, 75)
            plt.title(plot_title + f"\n {title_dict[mask_name]}, pion probability = {prob}",
                    fontsize=12)

            plt.tight_layout()
            plt.savefig(plot_directory + f"RICH_hits_xy_event{j}_{mask_name}.png")
            plt.close()

        plt.figure()
        plt.hist2d(np.array(reconstructed_momentum)[mask], np.array(cherenkov_angles)[mask], bins=100, range=((0,7), (-.05,.4)), norm=colors.LogNorm())
        plt.colorbar()
        plt.xlabel('p (GeV/c)')
        plt.ylabel('$\\theta_{C}$ (rad)')
        plt.title(plot_title+f"\n {title_dict[mask_name]}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(plot_directory+f"cherenkov_angles_{mask_name}_3GeV.png")
        plt.close()

    kaon_pion_masks = [
        (np.full_like(reconstructed_momentum, True, dtype=bool)),
        (pion_events_mask) & (model_probabilities_for_pions > 0.8),
        (pion_events_mask) & (model_probabilities_for_pions < 0.8),
        (kaon_events_mask) & (model_probabilities_for_pions > 0.8),
        (kaon_events_mask) & (model_probabilities_for_pions < 0.8),
    ]
    
    for mask, mask_name in zip(kaon_pion_masks, mask_names):
        plt.figure()
        plt.hist2d(np.array(reconstructed_momentum)[mask], np.array(cherenkov_angles)[mask], bins=100, range=((0,7), (-.05,.4)), norm=colors.LogNorm())
        plt.colorbar()
        plt.xlabel('p (GeV/c)')
        plt.ylabel('$\\theta_{C}$ (rad)')
        plt.title(plot_title+f"\n {title_dict[mask_name]}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(plot_directory+f"cherenkov_angles_{mask_name}.png")
        plt.close()