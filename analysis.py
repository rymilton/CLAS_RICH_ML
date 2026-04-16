"""
Analysis script for RICH PID model performance.
Compares GravNet model with RICH/reconstructed PID systems.
"""

import torch
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib.colors as colors
from sklearn.metrics import roc_curve, auc
import os

hep.style.use("CMS")

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "model_dir": "/volatile/clas12/rmilton/trained_RICH_models/12_2025_dataset/RICH_models_rgaspring2018reco_crossentropy_adjustedpadding_withLRscheduler_earlystop_withlayernum_onehot_fourlayers_withdropout/",
    "plot_dir": "./plots_rgaspring2018reco_crossentropy_adjustedpadding_withLRscheduler_earlystop_withlayernum_onehot_fourlayers_withdropout_softmax/",
    "plot_title": "RICH Sector 4 positives\n Four GravNet layers + dropout",
    "momentum_bins": np.linspace(1.5, 8, 11),
    "min_multiplicity": 0,
    "probability_thresholds": [0.2, 0.4, 0.5, 0.7, 0.8, 0.9],
    "default_threshold": 0.7,
    "pion_pid": 211,
    "kaon_pid": 321,
}

os.makedirs(CONFIG["plot_dir"], exist_ok=True)


# ============================================================================
# Data Loading and Processing
# ============================================================================


def _tensor_to_numpy(tensor):
    """Convert PyTorch tensor to numpy array."""
    return tensor.cpu().detach().numpy()


def _unpack_rich_hits(hits_padded, mask_padded):
    """Unpack padded RICH hit data using masks."""
    hits = []
    for batch_hits, batch_mask in zip(hits_padded, mask_padded):
        for event_hits, event_mask in zip(batch_hits, batch_mask):
            valid = event_mask.cpu().numpy().astype(bool)
            hits.append(event_hits.cpu().numpy()[valid])
    return ak.Array(hits)


def load_predictions(model_dir):
    """Load model predictions and return processed data."""
    predictions = torch.load(os.path.join(model_dir, "test_predictions.pt"))

    # Apply Cherenkov angle cut
    cherenkov_angles = _tensor_to_numpy(predictions["RICH_cherenkov_angle"])[:, 0]
    valid_cut = cherenkov_angles > 0

    # Extract and apply cut to all quantities
    rich_pid_data = _tensor_to_numpy(predictions["RICH_PID"])[valid_cut]
    # Handle both 1D and 2D RICH_PID arrays
    if rich_pid_data.ndim > 1:
        rich_pid_data = rich_pid_data[:, 0]
    data = {
        "cherenkov_angles": cherenkov_angles[valid_cut],
        "aerogel_layers": ak.flatten(
            _tensor_to_numpy(predictions["RICH_aerogel_layer"])
        )[valid_cut],
        "model_probs": _tensor_to_numpy(predictions["probabilities"])[valid_cut],
        "true_labels": _tensor_to_numpy(predictions["labels"])[valid_cut],
        "reco_pid": _tensor_to_numpy(predictions["reco_pid"])[:, 0][valid_cut],
        "reco_momentum": _tensor_to_numpy(predictions["reco_momentum"])[valid_cut],
        "rich_pid": rich_pid_data,
        "rich_rq": _tensor_to_numpy(predictions["RICH_RQ"])[valid_cut],
        "traj_x": _tensor_to_numpy(predictions["reco_traj_x"])[valid_cut],
        "traj_y": _tensor_to_numpy(predictions["reco_traj_y"])[valid_cut],
        "traj_cx": _tensor_to_numpy(predictions["reco_traj_cx"])[valid_cut],
        "traj_cy": _tensor_to_numpy(predictions["reco_traj_cy"])[valid_cut],
        "traj_cz": _tensor_to_numpy(predictions["reco_traj_cz"])[valid_cut],
        "theta": ak.flatten(_tensor_to_numpy(predictions["reco_theta"]))[valid_cut],
        "phi": ak.flatten(_tensor_to_numpy(predictions["reco_phi"]))[valid_cut],
    }

    # Unscaling cx, cy, cz
    data["traj_cx"] = 2 * data["traj_cx"] - 1
    data["traj_cy"] = 2 * data["traj_cy"] - 1
    data["traj_cz"] = 2 * data["traj_cz"] - 1

    # Unpack padded RICH hit data
    rich_hits_x = _unpack_rich_hits(
        predictions["RICH_hits_x"], predictions["RICH_hits_mask"]
    )[valid_cut]
    rich_hits_y = _unpack_rich_hits(
        predictions["RICH_hits_y"], predictions["RICH_hits_mask"]
    )[valid_cut]
    rich_hits_time = _unpack_rich_hits(
        predictions["RICH_hits_time"], predictions["RICH_hits_mask"]
    )[valid_cut]

    data["hits_x"] = rich_hits_x
    data["hits_y"] = rich_hits_y
    data["hits_time"] = rich_hits_time
    print("done loading data")
    return data


def load_losses(model_dir):
    """Load training/validation losses."""
    losses = torch.load(os.path.join(model_dir, "training_losses.pt"))
    return {
        "epochs": [losses[i]["epoch"] for i in range(len(losses))],
        "train": [losses[i]["train_loss"] for i in range(len(losses))],
        "val": [losses[i]["val_loss"] for i in range(len(losses))],
    }


def apply_multiplicity_cut(data, min_hits=10):
    """Filter events by RICH hit multiplicity."""
    mult_mask = ak.num(data["hits_x"], axis=1) > min_hits

    for key in data:
        if isinstance(data[key], (np.ndarray, ak.Array)):
            data[key] = data[key][mult_mask]

    return data


def extract_event_masks(data):
    """Extract pion/kaon event masks from true labels."""
    pion_mask = (data["true_labels"] == [1, 0])[:, 0]
    kaon_mask = (data["true_labels"] == [0, 1])[:, 0]
    return pion_mask, kaon_mask


# ============================================================================
# Metrics Calculation
# ============================================================================


def calculate_efficiency(true_mask, pred_mask, n_total):
    """Calculate efficiency and binomial error."""
    if n_total == 0:
        return -1, -1

    n_correct = np.sum(pred_mask & true_mask)
    efficiency = n_correct / n_total
    error = np.sqrt(efficiency * (1 - efficiency) / n_total)
    return efficiency, error


def calculate_efficiencies_1d(variable, bins, true_mask, pred_mask):
    """Calculate efficiencies binned in a 1D variable.

    Args:
        variable: 1D array of values to bin
        bins: bin edges (n_bins+1 values)
        true_mask: boolean mask for true events
        pred_mask: boolean mask for predicted events

    Returns:
        dict with 'effs', 'errs', 'n_true', 'n_pred', 'bin_centers'
    """
    n_bins = len(bins) - 1
    effs = np.full(n_bins, -1.0)
    errs = np.full(n_bins, -1.0)
    n_true = np.zeros(n_bins, dtype=int)
    n_pred = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (variable >= bins[i]) & (variable < bins[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (variable >= bins[i]) & (variable <= bins[i + 1])

        n_true[i] = np.sum(true_mask & mask)
        n_correct = np.sum((true_mask & pred_mask) & mask)

        if n_true[i] > 0:
            effs[i] = n_correct / n_true[i]
            errs[i] = np.sqrt(effs[i] * (1 - effs[i]) / n_true[i])

    bin_centers = (bins[:-1] + bins[1:]) / 2

    return {
        "effs": effs,
        "errs": errs,
        "n_true": n_true,
        "bin_centers": bin_centers,
    }


def calculate_efficiencies_2d(var1, bins1, var2, bins2, true_mask, pred_mask):
    """Calculate efficiencies binned in two variables (2D).

    Args:
        var1, var2: 1D arrays of values to bin
        bins1, bins2: bin edges for each variable
        true_mask: boolean mask for true events
        pred_mask: boolean mask for predicted events

    Returns:
        dict with 'effs_2d' (n1 x n2), 'errs_2d', 'bin_centers1', 'bin_centers2', etc.
    """
    n_bins1 = len(bins1) - 1
    n_bins2 = len(bins2) - 1
    effs_2d = np.full((n_bins1, n_bins2), -1.0)
    errs_2d = np.full((n_bins1, n_bins2), -1.0)
    n_true_2d = np.zeros((n_bins1, n_bins2), dtype=int)

    for i in range(n_bins1):
        mask1 = (var1 >= bins1[i]) & (var1 < bins1[i + 1])
        if i == n_bins1 - 1:
            mask1 = (var1 >= bins1[i]) & (var1 <= bins1[i + 1])

        for j in range(n_bins2):
            mask2 = (var2 >= bins2[j]) & (var2 < bins2[j + 1])
            if j == n_bins2 - 1:
                mask2 = (var2 >= bins2[j]) & (var2 <= bins2[j + 1])

            bin_mask = mask1 & mask2
            n_true_2d[i, j] = np.sum(true_mask & bin_mask)
            n_correct_2d = np.sum((true_mask & pred_mask) & bin_mask)

            if n_true_2d[i, j] > 0:
                effs_2d[i, j] = n_correct_2d / n_true_2d[i, j]
                errs_2d[i, j] = np.sqrt(
                    effs_2d[i, j] * (1 - effs_2d[i, j]) / n_true_2d[i, j]
                )

    bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
    bin_centers2 = (bins2[:-1] + bins2[1:]) / 2

    return {
        "effs_2d": effs_2d,
        "errs_2d": errs_2d,
        "n_true_2d": n_true_2d,
        "bin_centers1": bin_centers1,
        "bin_centers2": bin_centers2,
    }


def calculate_bin_metrics(
    prob_bin,
    true_mask_bin,
    reco_pid_bin,
    rich_pid_bin,
    threshold,
    pion_pid_val,
    kaon_pid_val,
):
    """Calculate efficiency metrics for a momentum/threshold bin."""
    if len(prob_bin) == 0:
        return None

    # Model metrics
    model_pion_mask = prob_bin > threshold
    n_true_pions = np.sum(true_mask_bin)
    n_true_kaons = np.sum(~true_mask_bin)

    pion_eff, pion_err = calculate_efficiency(
        true_mask_bin, model_pion_mask, n_true_pions
    )
    kaon_misid, kaon_misid_err = calculate_efficiency(
        ~true_mask_bin, model_pion_mask, n_true_kaons
    )

    # Kaon efficiency (reverse of pion)
    kaon_eff, kaon_err = calculate_efficiency(
        ~true_mask_bin, ~model_pion_mask, n_true_kaons
    )
    pion_misid, pion_misid_err = calculate_efficiency(
        true_mask_bin, ~model_pion_mask, n_true_pions
    )

    # Reconstructed PID metrics
    reco_pion_mask = reco_pid_bin == pion_pid_val
    reco_pion_eff, reco_pion_err = calculate_efficiency(
        true_mask_bin, reco_pion_mask, n_true_pions
    )
    reco_kaon_misid, reco_kaon_misid_err = calculate_efficiency(
        ~true_mask_bin, reco_pion_mask, n_true_kaons
    )
    reco_kaon_eff, reco_kaon_err = calculate_efficiency(
        ~true_mask_bin, ~reco_pion_mask, n_true_kaons
    )
    reco_pion_misid, reco_pion_misid_err = calculate_efficiency(
        true_mask_bin, ~reco_pion_mask, n_true_pions
    )

    # RICH PID metrics
    rich_pion_mask = rich_pid_bin == pion_pid_val
    rich_pion_eff, rich_pion_err = calculate_efficiency(
        true_mask_bin, rich_pion_mask, n_true_pions
    )
    rich_kaon_misid, rich_kaon_misid_err = calculate_efficiency(
        ~true_mask_bin, rich_pion_mask, n_true_kaons
    )
    rich_kaon_eff, rich_kaon_err = calculate_efficiency(
        ~true_mask_bin, ~rich_pion_mask, n_true_kaons
    )
    rich_pion_misid, rich_pion_misid_err = calculate_efficiency(
        true_mask_bin, ~rich_pion_mask, n_true_pions
    )

    return {
        # Pion identification
        "pion_eff": pion_eff,
        "pion_err": pion_err,
        "reco_pion_eff": reco_pion_eff,
        "reco_pion_err": reco_pion_err,
        "rich_pion_eff": rich_pion_eff,
        "rich_pion_err": rich_pion_err,
        # Kaon misidentification as pion
        "kaon_misid": kaon_misid,
        "kaon_misid_err": kaon_misid_err,
        "reco_kaon_misid": reco_kaon_misid,
        "reco_kaon_misid_err": reco_kaon_misid_err,
        "rich_kaon_misid": rich_kaon_misid,
        "rich_kaon_misid_err": rich_kaon_misid_err,
        # Kaon identification
        "kaon_eff": kaon_eff,
        "kaon_err": kaon_err,
        "reco_kaon_eff": reco_kaon_eff,
        "reco_kaon_err": reco_kaon_err,
        "rich_kaon_eff": rich_kaon_eff,
        "rich_kaon_err": rich_kaon_err,
        # Pion misidentification as kaon
        "pion_misid": pion_misid,
        "pion_misid_err": pion_misid_err,
        "reco_pion_misid": reco_pion_misid,
        "reco_pion_misid_err": reco_pion_misid_err,
        "rich_pion_misid": rich_pion_misid,
        "rich_pion_misid_err": rich_pion_misid_err,
    }


# ============================================================================
# Plotting Functions
# ============================================================================


def plot_loss_curves(losses, plot_dir, title):
    """Plot training/validation loss curves."""
    fig = plt.figure(figsize=(12, 8))
    plt.plot(losses["epochs"], losses["train"], label="Training")
    plt.plot(losses["epochs"], losses["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (CrossEntropy)")
    plt.title(title)
    plt.xlim(0, 500)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "losses.png"))
    plt.close()


def plot_momentum_distributions(momentum, momentum_by_type, plot_dir, title):
    """Plot momentum distributions."""
    fig = plt.figure(figsize=(12, 8))
    plt.hist(momentum, bins=50)
    plt.xlabel("p (GeV)")
    plt.ylabel("Counts (log)")
    plt.yscale("log")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "momentum.png"))
    plt.close()

    fig = plt.figure(figsize=(12, 8))
    for label, mask in momentum_by_type.items():
        plt.hist(momentum[mask], bins=50, label=label, histtype="step")
    plt.xlabel("p (GeV)")
    plt.ylabel("Counts (log)")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "momentum_by_particle.png"))
    plt.close()


def plot_multiplicity_distribution(multiplicity, plot_dir, title):
    """Plot RICH hit multiplicity distribution."""
    fig = plt.figure(figsize=(12, 8))
    plt.hist(multiplicity, bins=np.linspace(0, 100, 101))
    plt.yscale("log")
    plt.xlabel("RICH hit multiplicity")
    plt.ylabel("Counts (log)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "multiplicity_distribution.png"))
    plt.close()


def plot_roc_curve(y_true, y_scores, plot_dir, title):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel(r"Pion misidentification ($\pi \to K$)")
    plt.ylabel(r"Kaon efficiency ($K \to K$)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    # Add threshold labels
    for i in range(0, len(thresholds), max(1, len(thresholds) // 15)):
        plt.text(
            fpr[i],
            tpr[i],
            f"{thresholds[i]:.2f}",
            fontsize=8,
            ha="right",
            va="bottom",
            rotation=45,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "ROC.png"))
    plt.close()

    return roc_auc


def plot_auc_vs_momentum(
    momentum_bins,
    pion_probs_per_bin,
    pion_masks_per_bin,
    plot_dir,
    title,
):
    """Plot AUC vs momentum"""
    from sklearn.metrics import roc_auc_score

    momentum_bin_centers = (momentum_bins[1:] + momentum_bins[:-1]) / 2
    bin_widths = (momentum_bin_centers[1] - momentum_bin_centers[0]) / 2

    aucs, auc_errs, valid_centers = [], [], []

    for i, (probs_bin, pion_mask_bin) in enumerate(
        zip(pion_probs_per_bin, pion_masks_per_bin)
    ):
        # Need both classes present
        if len(probs_bin) < 10 or len(np.unique(pion_mask_bin.astype(int))) < 2:
            continue

        labels_bin = pion_mask_bin.astype(int)
        auc_val = roc_auc_score(labels_bin, probs_bin)

        aucs.append(auc_val)
        valid_centers.append(momentum_bin_centers[i])

    valid_centers = np.array(valid_centers)
    aucs = np.array(aucs)
    auc_errs = np.array(auc_errs)

    fig = plt.figure()
    plt.errorbar(
        valid_centers,
        aucs,
        xerr=bin_widths,
        marker="o",
        color="black",
        markersize=10,
        linestyle="none",
        label="GravNet AUC",
    )
    plt.axhline(0.5, color="gray", linestyle="--", label="Random classifier")
    plt.axhline(
        1.0, color="green", linestyle="--", alpha=0.5, label="Perfect classifier"
    )
    plt.xlabel("p (GeV/c)")
    plt.ylabel("AUC")
    plt.title(f"{title}\nAUC vs Momentum")
    plt.ylim(0.4, 1.05)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "auc_vs_momentum.png"), dpi=300)
    plt.close()


def plot_auc_vs_multiplicity(
    data,
    pion_mask,
    threshold,
    plot_dir,
    title,
):
    """Plot AUC vs RICH hit multiplicity"""
    from sklearn.metrics import roc_auc_score

    mult_bins = np.linspace(0, 100, 21)  # 20 bins
    mult_bin_centers = (mult_bins[1:] + mult_bins[:-1]) / 2
    bin_widths = (mult_bin_centers[1] - mult_bin_centers[0]) / 2
    multiplicity = ak.to_numpy(ak.num(data["hits_x"], axis=1))

    aucs, auc_errs, valid_centers = [], [], []
    rng = np.random.default_rng(42)

    for i in range(len(mult_bins) - 1):
        mask = (multiplicity >= mult_bins[i]) & (multiplicity < mult_bins[i + 1])
        probs_bin = data["model_probs"][mask, 0]
        labels_bin = pion_mask[mask].astype(int)

        if len(probs_bin) < 10 or len(np.unique(labels_bin)) < 2:
            continue

        auc_val = roc_auc_score(labels_bin, probs_bin)

        aucs.append(auc_val)
        valid_centers.append(mult_bin_centers[i])

    fig = plt.figure()
    plt.errorbar(
        np.array(valid_centers),
        np.array(aucs),
        xerr=bin_widths,
        marker="o",
        color="black",
        markersize=10,
        linestyle="none",
        label="GravNet AUC",
    )
    plt.axhline(0.5, color="gray", linestyle="--", label="Random classifier")
    plt.axhline(
        1.0, color="green", linestyle="--", alpha=0.5, label="Perfect classifier"
    )
    plt.xlabel("RICH hit multiplicity")
    plt.ylabel("AUC")
    plt.title(f"{title}\nAUC vs Multiplicity")
    plt.ylim(0.4, 1.05)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "auc_vs_multiplicity.png"), dpi=300)
    plt.close()


def plot_auc_2d_momentum_multiplicity(
    data,
    pion_mask,
    momentum_bins,
    plot_dir,
    title,
):
    """Plot 2D AUC heatmap vs momentum and RICH hit multiplicity."""
    from sklearn.metrics import roc_auc_score

    mult_bins = np.linspace(0, 100, 21)
    mult_bin_centers = (mult_bins[1:] + mult_bins[:-1]) / 2
    momentum_bin_centers = (momentum_bins[1:] + momentum_bins[:-1]) / 2
    multiplicity = ak.to_numpy(ak.num(data["hits_x"], axis=1))

    n_mom = len(momentum_bins) - 1
    n_mult = len(mult_bins) - 1
    aucs_2d = np.full((n_mom, n_mult), np.nan)
    rng = np.random.default_rng(42)

    for i in range(n_mom):
        mom_mask = (data["reco_momentum"] >= momentum_bins[i]) & (
            data["reco_momentum"] < momentum_bins[i + 1]
        )
        for j in range(n_mult):
            mult_mask = (multiplicity >= mult_bins[j]) & (
                multiplicity < mult_bins[j + 1]
            )
            combined = mom_mask & mult_mask

            probs_bin = data["model_probs"][combined, 0]
            labels_bin = pion_mask[combined].astype(int)

            if len(probs_bin) < 10 or len(np.unique(labels_bin)) < 2:
                continue

            aucs_2d[i, j] = roc_auc_score(labels_bin, probs_bin)

    fig, ax = plt.subplots(figsize=(12, 9))
    masked = np.ma.masked_invalid(aucs_2d)
    im = ax.pcolormesh(
        momentum_bin_centers,
        mult_bin_centers,
        masked.T,
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        shading="auto",
    )
    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("RICH hit multiplicity")
    ax.set_title(f"{title}\nAUC vs Momentum and Multiplicity")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("AUC", fontsize=12)
    ax.grid(True, alpha=0.3, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "auc_2d_momentum_vs_multiplicity.png"), dpi=300)
    plt.close()


def plot_pion_efficiency(
    momentum_bin_centers, metrics, xerr, good_bins, plot_dir, title, threshold
):
    """Plot pion efficiency (π → π) vs momentum with model and RICH comparison."""
    fig = plt.figure()

    # Removing empty bins
    filtered = [
        (m, p, xe)
        for m, p, xe in zip(metrics, momentum_bin_centers, xerr)
        if m is not None
    ]

    metrics, momentum_bin_centers, xerr = zip(*filtered)

    metrics = list(metrics)
    momentum_bin_centers = np.array(momentum_bin_centers)
    xerr = np.array(xerr)

    # Model (GravNet) - solid red circles
    pion_effs = np.array([m["pion_eff"] for m in metrics])
    pion_errs = np.array([m["pion_err"] for m in metrics])
    plt.errorbar(
        momentum_bin_centers,
        pion_effs,
        xerr=xerr,
        yerr=pion_errs,
        label=r"GravNet $\pi^+ \to \pi^+$",
        marker="o",
        color="red",
        markersize=10,
        linestyle="none",
    )

    # RICH - open red squares
    rich_pion_effs = np.array([m["rich_pion_eff"] for m in metrics])
    rich_pion_errs = np.array([m["rich_pion_err"] for m in metrics])
    plt.errorbar(
        momentum_bin_centers,
        rich_pion_effs,
        xerr=xerr,
        yerr=rich_pion_errs,
        label=r"RICH $\pi^+ \to \pi^+$",
        marker="s",
        color="red",
        markersize=10,
        markerfacecolor="none",
        linestyle="none",
    )

    # Kaon misidentification (K → π)
    kaon_misids = np.array([m["kaon_misid"] for m in metrics])
    kaon_misid_errs = np.array([m["kaon_misid_err"] for m in metrics])
    plt.errorbar(
        momentum_bin_centers,
        kaon_misids,
        xerr=xerr,
        yerr=kaon_misid_errs,
        label=r"GravNet $K^+ \to \pi^+$",
        marker="o",
        color="blue",
        markersize=10,
        linestyle="none",
    )

    # RICH kaon misidentification
    rich_kaon_misids = np.array([m["rich_kaon_misid"] for m in metrics])
    rich_kaon_misid_errs = np.array([m["rich_kaon_misid_err"] for m in metrics])
    plt.errorbar(
        momentum_bin_centers,
        rich_kaon_misids,
        xerr=xerr,
        yerr=rich_kaon_misid_errs,
        label=r"RICH $K^+ \to \pi^+$",
        marker="s",
        color="blue",
        markersize=10,
        markerfacecolor="none",
        linestyle="none",
    )

    plt.xlabel("p (GeV/c)")
    plt.ylabel(r"Efficiency of passing $\pi^+$ cut")
    plt.title(f"{title}\nThreshold = {threshold}")
    plt.legend(fontsize=11, loc="best")
    plt.grid(True)
    plt.ylim(-0.04, 1.04)
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, f"efficiency_pion_threshold_{threshold}.png"), dpi=300
    )
    plt.close()


def plot_kaon_efficiency(
    momentum_bin_centers, metrics, xerr, good_bins, plot_dir, title
):
    """Plot kaon efficiency (K → K) vs momentum with model and RICH comparison."""
    fig = plt.figure()

    # Removing empty bins
    filtered = [
        (m, p, xe)
        for m, p, xe in zip(metrics, momentum_bin_centers, xerr)
        if m is not None
    ]

    metrics, momentum_bin_centers, xerr = zip(*filtered)

    metrics = list(metrics)
    momentum_bin_centers = np.array(momentum_bin_centers)
    xerr = np.array(xerr)

    # Model (GravNet) - solid red circles
    kaon_effs = np.array([m["kaon_eff"] for m in metrics])
    kaon_errs = np.array([m["kaon_err"] for m in metrics])
    plt.errorbar(
        momentum_bin_centers,
        kaon_effs,
        xerr=xerr,
        yerr=kaon_errs,
        label=r"GravNet $K^+ \to K^+$",
        marker="o",
        color="red",
        markersize=10,
        linestyle="none",
    )

    # RICH - open red squares
    rich_kaon_effs = np.array([m["rich_kaon_eff"] for m in metrics])
    rich_kaon_errs = np.array([m["rich_kaon_err"] for m in metrics])
    plt.errorbar(
        momentum_bin_centers,
        rich_kaon_effs,
        xerr=xerr,
        yerr=rich_kaon_errs,
        label=r"RICH $K^+ \to K^+$",
        marker="s",
        color="red",
        markersize=10,
        markerfacecolor="none",
        linestyle="none",
    )

    # Pion misidentification (π → K)
    pion_misids = np.array([m["pion_misid"] for m in metrics])
    pion_misid_errs = np.array([m["pion_misid_err"] for m in metrics])
    plt.errorbar(
        momentum_bin_centers,
        pion_misids,
        xerr=xerr,
        yerr=pion_misid_errs,
        label=r"GravNet $\pi^+ \to K^+$",
        marker="o",
        color="blue",
        markersize=10,
        linestyle="none",
    )

    # RICH pion misidentification
    rich_pion_misids = np.array([m["rich_pion_misid"] for m in metrics])
    rich_pion_misid_errs = np.array([m["rich_pion_misid_err"] for m in metrics])
    plt.errorbar(
        momentum_bin_centers,
        rich_pion_misids,
        xerr=xerr,
        yerr=rich_pion_misid_errs,
        label=r"RICH $\pi^+ \to K^+$",
        marker="s",
        color="blue",
        markersize=10,
        markerfacecolor="none",
        linestyle="none",
    )

    plt.xlabel("p (GeV/c)")
    plt.ylabel(r"Efficiency of passing $K^+$ cut")
    plt.title(f"{title}\n$K^+$ ID Efficiency")
    plt.legend(fontsize=11, loc="best")
    plt.grid(True)
    plt.ylim(-0.04, 1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "efficiency_kaon.png"), dpi=300)
    plt.close()


def plot_probability_distributions(
    momentum_bins,
    pion_probs_per_bin,
    pion_masks_per_bin,
    pion_mask,
    kaon_mask,
    plot_dir,
    title,
):
    """Plot model probability distributions binned by momentum.

    Shows 5x2 subplots for kaon and pion probabilities in each momentum bin.
    """
    momentum_bin_centers = (momentum_bins[1:] + momentum_bins[:-1]) / 2

    # Kaon probabilities (model_probs[:, 1])
    fig_kaons, axs_kaons = plt.subplots(nrows=5, ncols=2, figsize=(12, 16))
    axs_kaons = axs_kaons.flatten()

    for i, (probs_bin, pion_mask_bin) in enumerate(
        zip(pion_probs_per_bin, pion_masks_per_bin)
    ):
        if len(probs_bin) == 0:
            axs_kaons[i].text(0.5, 0.5, "No events", ha="center", va="center")
            continue

        kaon_mask_bin = ~pion_mask_bin
        kaon_probs = 1 - probs_bin  # K probability = 1 - π probability

        axs_kaons[i].hist(
            kaon_probs[kaon_mask_bin],
            range=(0, 1),
            bins=100,
            label=f"True label: $K^+$",
            alpha=0.5,
        )

        axs_kaons[i].hist(
            kaon_probs[pion_mask_bin],
            range=(0, 1),
            bins=100,
            label=f"True label: $\pi^+$",
            alpha=0.5,
        )

        lower, upper = momentum_bins[i], momentum_bins[i + 1]
        total_counts = len(probs_bin)
        axs_kaons[i].set_title(
            f"${lower:.3f} \leq p < {upper:.3f}$ GeV\nTotal = {total_counts}",
            fontsize=11,
        )
        axs_kaons[i].legend(fontsize=10)

    plt.suptitle(f"{title}\nModel Probability for $K^+$")
    plt.xlabel("Probability event contains $K^+$")
    plt.ylabel("Entries")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "kaon_probabilities.png"), dpi=300)
    plt.close()

    # Pion probabilities (model_probs[:, 0])
    fig_pions, axs_pions = plt.subplots(nrows=5, ncols=2, figsize=(12, 16))
    axs_pions = axs_pions.flatten()

    for i, (probs_bin, pion_mask_bin) in enumerate(
        zip(pion_probs_per_bin, pion_masks_per_bin)
    ):
        if len(probs_bin) == 0:
            axs_pions[i].text(0.5, 0.5, "No events", ha="center", va="center")
            continue

        kaon_mask_bin = ~pion_mask_bin

        axs_pions[i].hist(
            probs_bin[kaon_mask_bin],
            range=(0, 1),
            bins=100,
            label=f"True label: $K^+$",
            alpha=0.5,
        )

        counts, edges, _ = axs_pions[i].hist(
            probs_bin[pion_mask_bin],
            range=(0, 1),
            bins=100,
            label=f"True label: $\pi^+$",
            alpha=0.5,
        )

        lower, upper = momentum_bins[i], momentum_bins[i + 1]
        total_counts = len(probs_bin)
        pion_count = np.sum(pion_mask_bin)
        kaon_count = np.sum(kaon_mask_bin)

        axs_pions[i].set_title(
            f"${lower:.3f} \leq p < {upper:.3f}$ GeV\n"
            f"$\pi^+$ count = {pion_count}, $K^+$ count = {kaon_count}",
            fontsize=11,
        )
        axs_pions[i].legend(fontsize=10)

    plt.suptitle(f"{title}\nModel Probability for $\pi^+$")
    plt.xlabel("Probability event contains $\pi^+$")
    plt.ylabel("Entries")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "pion_probabilities.png"), dpi=300)
    plt.close()


def plot_threshold_efficiency_comparison(
    momentum_bins, pion_probs_per_bin, pion_masks_per_bin, thresholds, plot_dir, title
):
    """Plot efficiency vs momentum for different probability thresholds.

    Shows 2x3 subplots for 6 different thresholds.
    """
    momentum_bin_centers = (momentum_bins[1:] + momentum_bins[:-1]) / 2

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    axes = axes.flatten()

    for ax_idx, threshold in enumerate(thresholds):
        ax = axes[ax_idx]

        pion_effs = []
        kaon_misids = []
        valid_bins = []

        for probs_bin, pion_mask_bin in zip(pion_probs_per_bin, pion_masks_per_bin):
            if len(probs_bin) == 0:
                pion_effs.append(-1)
                kaon_misids.append(-1)
                valid_bins.append(False)
                continue

            # Apply threshold
            pion_pred_mask = probs_bin > threshold

            n_true_pions = np.sum(pion_mask_bin)
            n_true_kaons = np.sum(~pion_mask_bin)

            # Pion efficiency
            if n_true_pions > 0:
                pion_eff = np.sum(pion_pred_mask & pion_mask_bin) / n_true_pions
                pion_effs.append(pion_eff)
            else:
                pion_effs.append(-1)

            # Kaon misidentification
            if n_true_kaons > 0:
                kaon_misid = np.sum(pion_pred_mask & ~pion_mask_bin) / n_true_kaons
                kaon_misids.append(kaon_misid)
            else:
                kaon_misids.append(-1)

            valid_bins.append(True)

        # Plot valid bins only
        valid_bins = np.array(valid_bins)
        pion_effs = np.array(pion_effs)
        kaon_misids = np.array(kaon_misids)

        if np.sum(valid_bins) > 0:
            ax.scatter(
                momentum_bin_centers[valid_bins],
                pion_effs[valid_bins],
                label=r"$\pi^+$ efficiency",
                marker="o",
                color="red",
                s=100,
            )
            ax.scatter(
                momentum_bin_centers[valid_bins],
                kaon_misids[valid_bins],
                label=r"$K^+ \to \pi^+$ misid",
                marker="s",
                color="blue",
                s=100,
            )

        ax.set_xlabel("p (GeV)", fontsize=11)
        ax.set_ylabel("Efficiency", fontsize=11)
        ax.set_title(f"Threshold = {threshold}", fontsize=12)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{title}\nEfficiency vs Probability Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "threshold_efficiencies.png"), dpi=300)
    plt.close()


def plot_efficiency_1d(
    bin_centers,
    effs,
    errs,
    label_var,
    threshold,
    plot_dir,
    filename,
    title,
):
    """Plot 1D efficiency vs variable with error bars.

    Args:
        bin_centers: centers of bins
        effs: efficiency values
        errs: efficiency errors
        label_var: label for x-axis variable
        threshold: probability threshold used
        plot_dir: directory to save plot
        filename: output filename (without directory)
        title: plot title
    """
    fig = plt.figure(figsize=(12, 8))

    valid = effs > -0.5  # Filter out -1 (empty bins)
    if np.sum(valid) > 0:
        plt.errorbar(
            bin_centers[valid],
            effs[valid],
            yerr=errs[valid],
            label=r"$\pi^+$ efficiency",
            marker="o",
            color="red",
            markersize=10,
            linestyle="none",
        )

    plt.xlabel(label_var, fontsize=12)
    plt.ylabel(r"Efficiency", fontsize=12)
    plt.title(f"{title}\nThreshold = {threshold}")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.04, 1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()


def plot_efficiency_2d(
    bin_centers1,
    bin_centers2,
    effs_2d,
    label_var1,
    label_var2,
    threshold,
    plot_dir,
    filename,
    title,
):
    """Plot 2D efficiency heatmap.

    Args:
        bin_centers1, bin_centers2: bin centers for each axis
        effs_2d: 2D array of efficiencies (n1 x n2)
        label_var1, label_var2: axis labels
        threshold: probability threshold used
        plot_dir: directory to save plot
        filename: output filename
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Mask out empty bins (-1) for display
    effs_masked = np.ma.masked_where(effs_2d < 0, effs_2d)

    im = ax.pcolormesh(
        bin_centers1,
        bin_centers2,
        effs_masked.T,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        shading="auto",
    )

    ax.set_xlabel(label_var1, fontsize=12)
    ax.set_ylabel(label_var2, fontsize=12)
    ax.set_title(f"{title}\nThreshold = {threshold}")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\pi^+$ Efficiency", fontsize=12)

    # Add grid
    ax.grid(True, alpha=0.3, color="black", linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()


def plot_2d_histogram(
    var1,
    bins1,
    var2,
    bins2,
    label_var1,
    label_var2,
    plot_dir,
    filename,
    title,
):
    """Plot 2D histogram of counts.

    Args:
        var1, var2: 1D arrays of values to bin
        bins1, bins2: bin edges for each variable
        label_var1, label_var2: axis labels
        plot_dir: directory to save plot
        filename: output filename
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    h = ax.hist2d(
        np.array(var1),
        np.array(var2),
        bins=[bins1, bins2],
        cmap="viridis",
        norm=colors.LogNorm(),
    )

    ax.set_xlabel(label_var1, fontsize=12)
    ax.set_ylabel(label_var2, fontsize=12)
    ax.set_title(f"{title}\nEvent counts")

    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label("Counts (log scale)", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300)
    plt.close()


# ============================================================================
# Main Analysis
# ============================================================================


def main():
    """Run complete analysis."""
    print(f"Loading data from {CONFIG['model_dir']}")

    # Load data
    data = load_predictions(CONFIG["model_dir"])
    losses = load_losses(CONFIG["model_dir"])

    # Apply cuts
    data = apply_multiplicity_cut(data, CONFIG["min_multiplicity"])
    pion_mask, kaon_mask = extract_event_masks(data)

    print(f"Processing {len(data['reco_momentum'])} events")

    # ====== Basic Plots ======
    plot_loss_curves(losses, CONFIG["plot_dir"], CONFIG["plot_title"])

    plot_momentum_distributions(
        data["reco_momentum"],
        {
            "Pions": pion_mask,
            "Kaons": kaon_mask,
        },
        CONFIG["plot_dir"],
        CONFIG["plot_title"],
    )

    # ====== ROC Curve ======
    y_true = np.concatenate(
        [
            np.zeros(np.sum(pion_mask)),  # Pions = 0
            np.ones(np.sum(kaon_mask)),  # Kaons = 1
        ]
    )

    y_scores = np.concatenate(
        [
            data["model_probs"][pion_mask, 1],
            data["model_probs"][kaon_mask, 1],
        ]
    )
    plot_roc_curve(y_true, y_scores, CONFIG["plot_dir"], CONFIG["plot_title"])

    # ====== Multiplicity Distribution ======
    multiplicity = ak.num(data["hits_x"], axis=1)
    plot_multiplicity_distribution(
        multiplicity, CONFIG["plot_dir"], CONFIG["plot_title"]
    )

    # ====== Efficiency vs Momentum ======
    momentum_bins = CONFIG["momentum_bins"]
    momentum_bin_centers = (momentum_bins[1:] + momentum_bins[:-1]) / 2
    threshold = CONFIG["default_threshold"]

    metrics = []
    good_bins = []
    pion_probs_per_bin = []
    pion_masks_per_bin = []

    for i in range(len(momentum_bins) - 1):
        lower, upper = momentum_bins[i], momentum_bins[i + 1]
        mom_mask = (data["reco_momentum"] >= lower) & (data["reco_momentum"] < upper)

        # Store probability and mask data for probability plots
        pion_probs_per_bin.append(data["model_probs"][mom_mask, 0])
        pion_masks_per_bin.append(pion_mask[mom_mask])

        result = calculate_bin_metrics(
            data["model_probs"][mom_mask, 0],
            pion_mask[mom_mask],
            data["reco_pid"][mom_mask],
            data["rich_pid"][mom_mask],
            threshold,
            CONFIG["pion_pid"],
            CONFIG["kaon_pid"],
        )

        if result is None:
            good_bins.append(False)
        else:
            good_bins.append(True)
        metrics.append(result)

    # ====== Probability Distribution Plots ======
    plot_probability_distributions(
        momentum_bins,
        pion_probs_per_bin,
        pion_masks_per_bin,
        pion_mask,
        kaon_mask,
        CONFIG["plot_dir"],
        CONFIG["plot_title"],
    )

    # ====== Threshold Efficiency Plots ======
    plot_threshold_efficiency_comparison(
        momentum_bins,
        pion_probs_per_bin,
        pion_masks_per_bin,
        CONFIG["probability_thresholds"],
        CONFIG["plot_dir"],
        CONFIG["plot_title"],
    )

    plot_auc_vs_momentum(
        momentum_bins,
        pion_probs_per_bin,
        pion_masks_per_bin,
        CONFIG["plot_dir"],
        CONFIG["plot_title"],
    )

    plot_auc_vs_multiplicity(
        data,
        pion_mask,
        threshold,
        CONFIG["plot_dir"],
        CONFIG["plot_title"],
    )

    # ====== AUC 2D: Momentum vs Multiplicity ======
    plot_auc_2d_momentum_multiplicity(
        data,
        pion_mask,
        momentum_bins,
        CONFIG["plot_dir"],
        CONFIG["plot_title"],
    )

    # Plot efficiency curves
    good_bins = np.array(good_bins, dtype=bool)
    bin_widths = (momentum_bin_centers[1] - momentum_bin_centers[0]) / 2
    xerr = np.ones(len(momentum_bin_centers)) * bin_widths

    # Only plot for bins with valid data
    valid_counts = np.sum(good_bins)
    if valid_counts > 0:
        plot_pion_efficiency(
            momentum_bin_centers,
            metrics,
            xerr,
            good_bins,
            CONFIG["plot_dir"],
            CONFIG["plot_title"],
            threshold,
        )
        plot_kaon_efficiency(
            momentum_bin_centers,
            metrics,
            xerr,
            good_bins,
            CONFIG["plot_dir"],
            CONFIG["plot_title"],
        )

    # ====== Efficiency vs Multiplicity (1D) ======
    print("\nCalculating efficiencies vs multiplicity...")
    mult_bins = np.linspace(0, 100, 101)  # 10 bins

    pion_pred_mask = data["model_probs"][:, 0] > threshold

    eff_mult_data = calculate_efficiencies_1d(
        multiplicity, mult_bins, pion_mask, pion_pred_mask
    )

    plot_efficiency_1d(
        eff_mult_data["bin_centers"],
        eff_mult_data["effs"],
        eff_mult_data["errs"],
        "RICH hit multiplicity",
        threshold,
        CONFIG["plot_dir"],
        "efficiency_vs_multiplicity.png",
        CONFIG["plot_title"],
    )

    # ====== Efficiency vs Momentum and Multiplicity (2D) ======
    print("Calculating 2D efficiencies (momentum vs multiplicity)...")
    eff_2d_data = calculate_efficiencies_2d(
        data["reco_momentum"],
        momentum_bins,
        multiplicity,
        mult_bins,
        pion_mask,
        pion_pred_mask,
    )

    plot_efficiency_2d(
        eff_2d_data["bin_centers1"],
        eff_2d_data["bin_centers2"],
        eff_2d_data["effs_2d"],
        "Momentum (GeV/c)",
        "RICH hit multiplicity",
        threshold,
        CONFIG["plot_dir"],
        "efficiency_2d_momentum_vs_multiplicity.png",
        CONFIG["plot_title"],
    )

    # ====== 2D Histogram of Event Counts (Momentum vs Multiplicity) ======
    print("Creating 2D histogram of event counts...")
    plot_2d_histogram(
        data["reco_momentum"],
        momentum_bins,
        multiplicity,
        mult_bins,
        "Momentum (GeV/c)",
        "RICH hit multiplicity",
        CONFIG["plot_dir"],
        "histogram_2d_momentum_vs_multiplicity.png",
        CONFIG["plot_title"],
    )

    # ====== Efficiency vs Theta (1D) ======
    print("\nCalculating efficiencies vs theta...")
    theta_bins = np.linspace(0, 40, 101)  # 25 bins from 0 to 25 degrees

    eff_theta_data = calculate_efficiencies_1d(
        data["theta"], theta_bins, pion_mask, pion_pred_mask
    )

    plot_efficiency_1d(
        eff_theta_data["bin_centers"],
        eff_theta_data["effs"],
        eff_theta_data["errs"],
        r"Polar angle $\theta$ (degrees)",
        threshold,
        CONFIG["plot_dir"],
        "efficiency_vs_theta.png",
        CONFIG["plot_title"],
    )

    figure = plt.figure()
    plt.hist(data["theta"], bins=np.linspace(0, 40, 201), density=True)
    plt.xlabel("Theta")
    plt.ylabel("Normalized entries")
    plt.savefig(CONFIG["plot_dir"] + "theta.png")
    plt.close()

    # ====== Efficiency vs Momentum and Theta (2D) ======
    print("Calculating 2D efficiencies (momentum vs theta)...")
    eff_2d_theta_data = calculate_efficiencies_2d(
        data["reco_momentum"],
        momentum_bins,
        data["theta"],
        theta_bins,
        pion_mask,
        pion_pred_mask,
    )

    plot_efficiency_2d(
        eff_2d_theta_data["bin_centers1"],
        eff_2d_theta_data["bin_centers2"],
        eff_2d_theta_data["effs_2d"],
        "Momentum (GeV/c)",
        r"Polar angle $\theta$ (degrees)",
        threshold,
        CONFIG["plot_dir"],
        "efficiency_2d_momentum_vs_theta.png",
        CONFIG["plot_title"],
    )

    plot_2d_histogram(
        data["reco_momentum"],
        np.linspace(0, 12, 101),
        data["theta"],
        theta_bins,
        "Momentum (GeV/c)",
        "$\\theta$ (degrees)",
        CONFIG["plot_dir"],
        "histogram_2d_momentum_vs_theta.png",
        CONFIG["plot_title"],
    )
    theta_bins = np.linspace(0, 40, 101)  # 25 bins from 0 to 25 degrees
    plot_2d_histogram(
        data["phi"],
        np.linspace(140, 220, 201),
        data["theta"],
        theta_bins,
        "$\phi$ (degrees)",
        "$\\theta$ (degrees)",
        CONFIG["plot_dir"],
        "histogram_2d_phi_vs_theta.png",
        CONFIG["plot_title"],
    )
    print("Calculating 2D efficiencies (theta vs phi)...")

    eff_2d_theta_phi_data = calculate_efficiencies_2d(
        data["phi"],
        np.linspace(140, 220, 201),
        data["theta"],
        theta_bins,
        pion_mask,
        pion_pred_mask,
    )

    plot_efficiency_2d(
        eff_2d_theta_phi_data["bin_centers1"],
        eff_2d_theta_phi_data["bin_centers2"],
        eff_2d_theta_phi_data["effs_2d"],
        "$\Phi$ (deg)",
        r"Polar angle $\theta$ (degrees)",
        threshold,
        CONFIG["plot_dir"],
        "efficiency_2d_phi_vs_theta.png",
        CONFIG["plot_title"],
    )

    from scipy.stats import binned_statistic_2d

    print("Making cx, cy, cz plots")

    # Make a 2D histogram where each bin contains the average data["traj_cx"]. Repeat for cy and cz
    phi_bins = np.linspace(140, 220, 201)
    theta_bins = np.linspace(0, 40, 101)

    # Compute mean traj_cx in each (phi, theta) bin
    stat_cx, phi_edges, theta_edges, binnumber = binned_statistic_2d(
        np.array(data["phi"]),
        np.array(data["theta"]),
        np.array(data["traj_cx"]),
        statistic="mean",
        bins=[phi_bins, theta_bins],
    )

    fig, ax = plt.subplots(figsize=(12, 9))

    # Mask NaNs so empty bins do not show up badly
    stat_cx_masked = np.ma.masked_invalid(stat_cx)

    im = ax.pcolormesh(
        phi_edges,
        theta_edges,
        stat_cx_masked.T,
        cmap="coolwarm",
        shading="auto",
    )

    ax.set_xlabel(r"$\phi$ (degrees)", fontsize=12)
    ax.set_ylabel(r"$\theta$ (degrees)", fontsize=12)
    ax.set_title(f"{CONFIG['plot_title']}\nAverage trajectory $c_x$")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"Average cx", fontsize=12)

    ax.grid(True, alpha=0.3, color="black", linestyle="--")

    plt.tight_layout()
    plt.savefig(
        os.path.join(CONFIG["plot_dir"], "mean_traj_cx_phi_vs_theta.png"),
        dpi=300,
    )
    plt.close()

    stat_cy, phi_edges, theta_edges, binnumber = binned_statistic_2d(
        np.array(data["phi"]),
        np.array(data["theta"]),
        np.array(data["traj_cy"]),
        statistic="mean",
        bins=[phi_bins, theta_bins],
    )

    fig, ax = plt.subplots(figsize=(12, 9))

    # Mask NaNs so empty bins do not show up badly
    stat_cy_masked = np.ma.masked_invalid(stat_cy)

    im = ax.pcolormesh(
        phi_edges,
        theta_edges,
        stat_cy_masked.T,
        cmap="coolwarm",
        shading="auto",
    )

    ax.set_xlabel(r"$\phi$ (degrees)", fontsize=12)
    ax.set_ylabel(r"$\theta$ (degrees)", fontsize=12)
    ax.set_title(f"{CONFIG['plot_title']}\nAverage trajectory $c_y$")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"Average cy", fontsize=12)

    ax.grid(True, alpha=0.3, color="black", linestyle="--")

    plt.tight_layout()
    plt.savefig(
        os.path.join(CONFIG["plot_dir"], "mean_traj_cy_phi_vs_theta.png"),
        dpi=300,
    )
    plt.close()

    print("✓ Analysis complete!")
    print(f"✓ Plots saved to {CONFIG['plot_dir']}")

    stat_cz, phi_edges, theta_edges, binnumber = binned_statistic_2d(
        np.array(data["phi"]),
        np.array(data["theta"]),
        np.array(data["traj_cz"]),
        statistic="mean",
        bins=[phi_bins, theta_bins],
    )

    fig, ax = plt.subplots(figsize=(12, 9))

    # Mask NaNs so empty bins do not show up badly
    stat_cz_masked = np.ma.masked_invalid(stat_cz)

    im = ax.pcolormesh(
        phi_edges,
        theta_edges,
        stat_cz_masked.T,
        cmap="coolwarm",
        shading="auto",
    )

    ax.set_xlabel(r"$\phi$ (degrees)", fontsize=12)
    ax.set_ylabel(r"$\theta$ (degrees)", fontsize=12)
    ax.set_title(f"{CONFIG['plot_title']}\nAverage trajectory $c_z$")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"Average cz", fontsize=12)

    ax.grid(True, alpha=0.3, color="black", linestyle="--")

    plt.tight_layout()
    plt.savefig(
        os.path.join(CONFIG["plot_dir"], "mean_traj_cz_phi_vs_theta.png"),
        dpi=300,
    )
    plt.close()

    # Looking at cx, cy, cz distributions when phi > 190
    phi = np.array(data["phi"])

    masks = [
        ("phi > 0", phi > 0, "phi_gt_0"),
        ("phi > 190", phi > 190, "phi_gt_190"),
    ]

    variables = [
        ("traj_cx", r"$c_x$"),
        ("traj_cy", r"$c_y$"),
        ("traj_cz", r"$c_z$"),
    ]

    for mask_label, mask, mask_tag in masks:
        for key, label in variables:
            vals = np.array(data[key])[mask]

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.hist(vals, bins=50, histtype="step", linewidth=2)
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel("Counts", fontsize=12)
            ax.set_title(
                f"{CONFIG['plot_title']}\nTrajectory {label} for ${mask_label}$"
            )
            ax.grid(True, alpha=0.3, color="black", linestyle="--")

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    CONFIG["plot_dir"],
                    f"{key}_{mask_tag}_hist.png",
                ),
                dpi=300,
            )
            plt.close()


if __name__ == "__main__":
    main()
