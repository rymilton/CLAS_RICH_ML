import torch
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
from sklearn.metrics import roc_curve, auc
import os

model_dir = "/volatile/clas12/rmilton/RICH_models_LR1E-5_128hiddendimensions_200epochs_0.1validationsplit_3.15GeVmomentumcut/"
print(model_dir)
plot_title = "RICH Sector 4 positives p>3.15 GeV, 200 Epochs\nLR=1E-5, 128 hidden dimensions, 10% validation"
plot_directory = "./plots/"
os.makedirs(output_directory, exist_ok=True)

# Opening the files
losses = torch.load(model_dir+"/training_losses.pt")
loss_epochs = [losses[i]["epoch"] for i in range(len(losses))]
train_loss_per_epoch = [losses[i]["train_loss"] for i in range(len(losses))]
validation_loss_per_epoch = [losses[i]["val_loss"] for i in range(len(losses))]

predictions = torch.load(model_dir+"/test_predictions.pt")
model_probabilities = predictions["probabilities"].cpu().detach().numpy()
true_labels = predictions["labels"].cpu().detach().numpy()
reconstructed_pid = predictions["reco_pid"].cpu().detach().numpy()[:, 0]
reconstructed_momentum = predictions["reco_momentum"].cpu().detach().numpy()
RICH_PID = predictions["RICH_PID"].cpu().detach().numpy()
RICH_RQ = predictions["RICH_RQ"].cpu().detach().numpy()

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

# RICH RQ for pions
fig_pions_RQ, axs_pions_RQ = plt.subplots(nrows=5, ncols=2, figsize=(10,16))
axs_pions_RQ = axs_pions_RQ.flatten()

# For the track in each event, we have the RICH pid and the confidence in that PID
# So for every event we want to get TRUE pions, and see how often it calls it a pion or how often it calls it something else

RICH_pid_for_true_pions_per_momentum = []
RICH_RQ_for_true_pions_per_momentum = []
for i in range(len(momentum_bins)-1):
    lower_bin, upper_bin = momentum_bins[i], momentum_bins[i+1]
    momentum_mask = (reconstructed_momentum>=lower_bin) & (reconstructed_momentum < upper_bin)
    kaon_events_mask_momentum_bin = kaon_events_mask[momentum_mask]
    pion_events_mask_momentum_bin = pion_events_mask[momentum_mask]

    RICH_pid_for_true_pions = np.array(RICH_PID[momentum_mask][pion_events_mask_momentum_bin])
    RICH_RQ_for_true_pions = np.array(RICH_RQ[momentum_mask][pion_events_mask_momentum_bin])

    RICH_pid_for_true_pions_per_momentum.append(RICH_pid_for_true_pions)
    RICH_RQ_for_true_pions_per_momentum.append(RICH_RQ_for_true_pions)

    RICH_pionsRQ_for_true_pions = RICH_RQ_for_true_pions[RICH_pid_for_true_pions==211]
    RICH_kaonsRQ_for_true_pions = RICH_RQ_for_true_pions[RICH_pid_for_true_pions==321]

    total_counts = np.sum(RICH_pid_for_true_pions==211) + np.sum(RICH_pid_for_true_pions==321)
    axs_pions_RQ[i].hist(
        RICH_pionsRQ_for_true_pions,
        range=(0,1),
        bins=20,
        label="RICH PID: $pi^+$",
        alpha=0.5,
    )
    axs_pions_RQ[i].hist(
        RICH_kaonsRQ_for_true_pions,
        range=(0,1),
        bins=20,
        label="RICH PID: $K^+$",
        alpha=0.5,
    )
    axs_pions_RQ[i].legend(fontsize=12)
    axs_pions_RQ[i].set_title(f"${round(lower_bin,3)}~GeV~ \leq p < {round(upper_bin,3)}~GeV$\n Total counts={total_counts}", fontsize=12)
    axs_pions_RQ[i].set_xlabel("RQ for true $\pi$ events", fontsize=12)
    axs_pions_RQ[i].set_ylabel("Counts", fontsize=12)
plt.tight_layout()
plt.savefig(plot_directory+'RQ_pion.png')

RICH_pid_for_true_kaons_per_momentum = []
RICH_RQ_for_true_kaons_per_momentum = []
fig_kaons_RQ, axs_kaons_RQ = plt.subplots(nrows=5, ncols=2, figsize=(10,16))
axs_kaons_RQ = axs_kaons_RQ.flatten()
for i in range(len(momentum_bins)-1):
    lower_bin, upper_bin = momentum_bins[i], momentum_bins[i+1]
    momentum_mask = (reconstructed_momentum>=lower_bin) & (reconstructed_momentum < upper_bin)
    kaon_events_mask_momentum_bin = kaon_events_mask[momentum_mask]

    RICH_pid_for_true_kaons = np.array(RICH_PID[momentum_mask][kaon_events_mask_momentum_bin])
    RICH_RQ_for_true_kaons = np.array(RICH_RQ[momentum_mask][kaon_events_mask_momentum_bin])
    RICH_pid_for_true_kaons_per_momentum.append(RICH_pid_for_true_kaons)
    RICH_RQ_for_true_kaons_per_momentum.append(RICH_RQ_for_true_kaons)

    RICH_pionsRQ_for_true_kaons = RICH_RQ_for_true_kaons[RICH_pid_for_true_kaons==211]
    RICH_kaonsRQ_for_true_kaons = RICH_RQ_for_true_kaons[RICH_pid_for_true_kaons==321]

    total_counts = np.sum(RICH_pid_for_true_kaons==211) + np.sum(RICH_pid_for_true_kaons==321)

    axs_kaons_RQ[i].hist(
        RICH_pionsRQ_for_true_kaons,
        range=(0,1),
        bins=20,
        label="RICH PID: $\pi^+$",
        alpha=0.5,
    )
    axs_kaons_RQ[i].hist(
        RICH_kaonsRQ_for_true_kaons,
        range=(0,1),
        bins=20,
        label="RICH PID: $K^+$",
        alpha=0.5,
    )
    axs_kaons_RQ[i].legend(fontsize=12)
    axs_kaons_RQ[i].set_title(f"${round(lower_bin,3)}~GeV~ \leq p < {round(upper_bin,3)}~GeV$\n Total counts={total_counts}", fontsize=12)
    axs_kaons_RQ[i].set_xlabel("RQ for true $K$ events", fontsize=12)
    axs_kaons_RQ[i].set_ylabel("Counts", fontsize=12)
plt.tight_layout()
plt.savefig(plot_directory+'RQ_kaon.png')

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
    if threshold in [.5,.8,.9]:
        print(f"Average pion accuracy at threshold={threshold}:", np.mean(pion_accuracies))
        print(f"Average kaon misidentifactionat threshold={threshold}:", np.mean(kaon_misidentifications))
        # print(f"Pion accuracies at threshold={threshold}:", pion_accuracies)
        # print(f"Kaon misidentifactions at threshold={threshold}:", kaon_misidentifications)
plt.suptitle(plot_title)
plt.tight_layout()
plt.savefig(plot_directory+f'thresholds.png')

# Model efficiencies for different probability thresholds
probability_thresholds = [0.2, 0.4, 0.5, 0.7, 0.8, 0.9]
fig_thresholds_with_RQ, axs_threshold_with_RQ = plt.subplots(nrows=2, ncols=3, figsize=(16,10))
axs_threshold_with_RQ = axs_threshold_with_RQ.flatten()
for i, threshold in enumerate(probability_thresholds):
    pion_accuracies_model, pion_accuracies_RICH = [], []
    kaon_misidentifications_model, kaon_misidentifications_RICH = [], []

    for probabilities_per_momentum, pion_events_per_momentum in zip(pion_event_probabilities, pion_mask_per_momentum):
        threshold_mask = probabilities_per_momentum > threshold
        probabilities_passing_threshold = probabilities_per_momentum[threshold_mask]
        pion_events = pion_events_per_momentum[threshold_mask]

        num_correct_pion_events = len(probabilities_passing_threshold[pion_events])
        num_incorrect_kaon_events = len(probabilities_passing_threshold[~pion_events]) # num kaons that pass the pion threshold

        # probabilities_per_momentum[pion_events_per_momentum] is the number of pion events in this momentum bin
        # probabilities_per_momentum[~pion_events_per_momentum] is the number of kaon events in this momentum bin
        if len(probabilities_per_momentum[pion_events_per_momentum])>0:
            pion_accuracies_model.append(num_correct_pion_events/len(probabilities_per_momentum[pion_events_per_momentum]))
        else:
            pion_accuracies_model.append(-1)
        
        if len(probabilities_per_momentum[~pion_events_per_momentum])>0:
            kaon_misidentifications_model.append(num_incorrect_kaon_events/len(probabilities_per_momentum[~pion_events_per_momentum]))
        else:
            kaon_misidentifications_model.append(-1)
    
    for RQ_for_pions, PID_for_pions, RQ_for_kaons, PID_for_kaons in zip(RICH_RQ_for_true_pions_per_momentum, RICH_pid_for_true_pions_per_momentum, RICH_RQ_for_true_kaons_per_momentum, RICH_pid_for_true_kaons_per_momentum):
        threshold_mask_pions = RQ_for_pions > threshold
        RQ_passing_threshold_pions = RQ_for_pions[threshold_mask_pions]
        PID_passing_threshold_pions = PID_for_pions[threshold_mask_pions]

        num_correct_pion_events = np.sum(PID_passing_threshold_pions==211)
        total_pion_events = len(RQ_for_pions)
        if total_pion_events > 0:
            pion_accuracies_RICH.append(num_correct_pion_events/total_pion_events)
        else:
            pion_accuracies_RICH.append(-1)

        threshold_mask_kaons = RQ_for_kaons > threshold
        RQ_passing_threshold_kaons = RQ_for_kaons[threshold_mask_kaons]
        PID_passing_threshold_kaons = PID_for_kaons[threshold_mask_kaons]

        num_incorrect_kaon_events = np.sum(PID_passing_threshold_kaons==211)
        total_kaon_events = len(RQ_for_kaons)
        if total_kaon_events > 0:
            kaon_misidentifications_RICH.append(num_incorrect_kaon_events/total_kaon_events)
        else:
            kaon_misidentifications_RICH.append(-1)
        
        # When >threshold, how many true kaon events are called a pion
        # When >threshold, how many true pion events are called a pion

    axs_threshold_with_RQ[i].scatter(momentum_bin_centers, pion_accuracies_model, color='r', label="ML $\pi^+$")
    axs_threshold_with_RQ[i].scatter(momentum_bin_centers, kaon_misidentifications_model, color='b', label="ML $K^+$")
    axs_threshold_with_RQ[i].scatter(momentum_bin_centers, pion_accuracies_RICH, color='r',marker='s',facecolor='none', label="RICH $\pi^+$")
    axs_threshold_with_RQ[i].scatter(momentum_bin_centers, kaon_misidentifications_RICH, color='b', marker='s',facecolor='none', label="RICH $K^+$")
    axs_threshold_with_RQ[i].set_xlabel("p (GeV)", fontsize=12)
    axs_threshold_with_RQ[i].set_ylabel("Efficiency of passing $\pi^+$ cut", fontsize=12)
    axs_threshold_with_RQ[i].legend(fontsize=12)
    axs_threshold_with_RQ[i].set_title(f"Probability Threshold = {threshold}", fontsize=12)
    axs_threshold_with_RQ[i].set_ylim(-.1,1.1)
    axs_threshold_with_RQ[i].grid()
plt.suptitle(plot_title)
plt.tight_layout()
plt.savefig(plot_directory+f'thresholds_with_RQ.png')