import argparse
import uproot
import glob
import awkward as ak
from utils import LoadYaml
import numpy as np
import time
import h5py as h5
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config",
        default="data.yaml",
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

def open_file(file_list):
    event_dictionary = {
        "RICH_hits": ak.Array([]),
        "trajectories": ak.Array([]),
        "truth_particles": ak.Array([]),
        "reconstructed_particles": ak.Array([]),
    }
    for i, file in enumerate(file_list):
        if i%10==0:
            print(f"Opening file {i}")
        with uproot.open(f"{file}:events") as f:
            event_dictionary["RICH_hits"] = ak.concatenate((event_dictionary["RICH_hits"], f.arrays(filter_name="RICH::Hit.*")))
            event_dictionary["trajectories"] = ak.concatenate((event_dictionary["trajectories"], f.arrays(filter_name="REC::Traj.*")))
            event_dictionary["truth_particles"] = ak.concatenate((event_dictionary["truth_particles"], f.arrays(filter_name="MC::Particle.*")))
            event_dictionary["reconstructed_particles"] = ak.concatenate((event_dictionary["reconstructed_particles"], f.arrays(filter_name="REC::Particles.*")))
    return ak.Array(event_dictionary)

def select_hits(
        event_data, 
        sector,
        use_negatives = True,
        use_positives = True,
        max_num_trajectories = None
        ):

    print(f"Have {len(event_data)} events originally")
    # Removing events with no reconstructed particles
    nonempty_event_mask = ak.num(event_data["reconstructed_particles"]["REC::Particles.pid"], axis=1) > 0
    event_data = event_data[nonempty_event_mask]
    print(f"Have {len(event_data)} events after removing empty events")
    # Removing events where the first reconstructed particle isn't the trigger electron
    trigger_mask = (event_data["reconstructed_particles"]["REC::Particles.pid"][:,0]==11) & (event_data["reconstructed_particles"]["REC::Particles.status"][:,0]<0)
    event_data = event_data[trigger_mask]
    print(f"Have {len(event_data)} events after removing bad trigger events")

    # Removing invalid RICH hits and keeping desired sector hits
    hits = event_data["RICH_hits"]
    valid_hits_mask = (hits["RICH::Hit.cluster"] == 0) & (hits["RICH::Hit.xtalk"] == 0) & (hits["RICH::Hit.pmt"] > 0) & (hits["RICH::Hit.pmt"] < 392)
    hits_sector_mask = hits["RICH::Hit.sector"] == sector
    event_data["RICH_hits"] = event_data["RICH_hits"][(valid_hits_mask) & (hits_sector_mask)]
    # Removing events without a valid RICH hit
    nonempty_hits_mask = ak.num(event_data["RICH_hits"]["RICH::Hit.x"])>0
    event_data = event_data[nonempty_hits_mask]

    print(f"Have {len(event_data)} events after removing invalid RICH hits")

    # Selecting reconstructed particles that have a trajectory that hits the RICH aerogel
    RICH_detector_ID = 18
    trajectories = event_data["trajectories"]
    RICH_trajectory_mask = (trajectories["REC::Traj.detector"]==RICH_detector_ID) & (trajectories["REC::Traj.layer"]>1)
    trajectory_sector_mask = trajectories["REC::Traj.x"] > 0 if sector == 1 else trajectories["REC::Traj.x"] < 0
    event_data["trajectories"] = event_data["trajectories"][(RICH_trajectory_mask) & (trajectory_sector_mask)]

    # If the maximum number of trajectories is set, removing events with more than max_num_trajectories in RICH
    unique_pindex = [np.unique(event_pindices).to_list() for event_pindices in event_data["trajectories"]["REC::Traj.pindex"]]
    if max_num_trajectories is not None:
        num_trajectories = ak.num(unique_pindex, axis=1)
        num_trajectories_mask = (num_trajectories > 0) & (num_trajectories <= max_num_trajectories)
    else:
        num_trajectories_mask = ak.num(event_data["trajectories"]["REC::Traj.x"])>0
    event_data = event_data[num_trajectories_mask]
    # Removing events that have more than 1 trajectory for the saved particle
    invalid_multiple_trajectory_mask = ak.num(event_data["trajectories"]["REC::Traj.layer"], axis=1)==1
    event_data = event_data[invalid_multiple_trajectory_mask]
    print(f"Have {len(event_data)} events after selecting trajectories")

    # Only keeping reconstructed particles that have a trajectory that satisfied selection
    unique_pindex = [np.unique(event_pindices).to_list() for event_pindices in event_data["trajectories"]["REC::Traj.pindex"]]
    event_data["reconstructed_particles"] = event_data["reconstructed_particles"][unique_pindex]

    # Removing neutral particles from reconstructed data
    neutral_mask = (event_data["reconstructed_particles"]["REC::Particles.charge"] > 0 ) | (event_data["reconstructed_particles"]["REC::Particles.charge"] < 0)
    event_data["reconstructed_particles"] = event_data["reconstructed_particles"][neutral_mask]
    # Only keeping certain charged particles depending on user option
    if not use_negatives and not use_positives:
        raise ValueError("Must enable selection of negative or positive particles!")

    charge_mask = None
    if use_negatives:
        negative_mask = event_data["reconstructed_particles"]["REC::Particles.charge"] < 0
        charge_mask = negative_mask if charge_mask is None else (charge_mask) | (negative_mask)

    if use_positives:
        positive_mask = event_data["reconstructed_particles"]["REC::Particles.charge"] > 0
        charge_mask = positive_mask if charge_mask is None else (charge_mask) | (positive_mask)

    event_data["reconstructed_particles"] = event_data["reconstructed_particles"][charge_mask]

    # Removing invalid momentum values
    reconstructed_momentum = np.sqrt(
        event_data["reconstructed_particles"]["REC::Particles.px"]**2 +
        event_data["reconstructed_particles"]["REC::Particles.py"]**2 +
        event_data["reconstructed_particles"]["REC::Particles.pz"]**2)
    event_data["reconstructed_particles"] = ak.with_field(
        event_data["reconstructed_particles"],
        reconstructed_momentum,
        "REC::Particles.p"
    )

    momentum_mask = (reconstructed_momentum < 12) & (reconstructed_momentum > 0)
    event_data["reconstructed_particles"] = event_data["reconstructed_particles"][momentum_mask]

    # After the charge and momentum cuts, removing events without any reconstructed particles
    nonempty_recoparticles_mask = ak.num(event_data["reconstructed_particles"]["REC::Particles.pid"])>0
    event_data = event_data[nonempty_recoparticles_mask]
    print(f"Have {len(event_data)} events after charge and momentum cuts")
    return event_data
def match_to_truth(event_data, max_num_trajectories = None):

    def angular_difference(a, b):
        diff = (a - b + 180) % 360 - 180
        return diff
    def find_second_smallest(arr):
        if len(arr) < 2:
            return -1
        sorted_indices = np.argsort(arr)
        second_smallest_index = sorted_indices[1]
        return second_smallest_index
    
    rec_particles = event_data["reconstructed_particles"]
    truth_particles = event_data["truth_particles"]
    truth_particles["MC::Particle.p"] = np.sqrt(truth_particles["MC::Particle.px"]**2 + truth_particles["MC::Particle.py"]**2 + truth_particles["MC::Particle.pz"]**2)
    truth_particles = truth_particles[truth_particles["MC::Particle.p"]>0]
    truth_particles["MC::Particle.phi"] = np.arctan2(truth_particles["MC::Particle.py"], truth_particles["MC::Particle.px"])
    truth_particles["MC::Particle.phi"] = ak.where(truth_particles["MC::Particle.phi"] < 0, truth_particles["MC::Particle.phi"] + 2 * np.pi, truth_particles["MC::Particle.phi"])*180/np.pi
    rec_particles["REC::Particles.phi"] = np.arctan2(rec_particles["REC::Particles.py"], rec_particles["REC::Particles.px"])
    rec_particles["REC::Particles.phi"] = ak.where(rec_particles["REC::Particles.phi"] < 0, rec_particles["REC::Particles.phi"] + 2 * np.pi, rec_particles["REC::Particles.phi"])*180/np.pi
    truth_particles["MC::Particle.theta"] = np.arccos(truth_particles["MC::Particle.pz"]/truth_particles["MC::Particle.p"]) * 180/np.pi
    rec_particles["REC::Particles.theta"] = np.arccos(rec_particles["REC::Particles.pz"]/rec_particles["REC::Particles.p"]) * 180/np.pi
    
    delta_phi = []
    delta_theta = []
    # Cut values
    delta_theta_cut = 2
    delta_phi_cut = 4

    # For every reconstructed particle, we calculate the Δθ and Δφ w.r.t every truth particle in the event
    for event in range(len(rec_particles["REC::Particles.theta"])):
        delta_phi_event = []
        delta_theta_event = []
        for rec_phi, rec_theta in zip(rec_particles["REC::Particles.phi"][event],
                                    rec_particles["REC::Particles.theta"][event]):
            delta_phi_particle = []
            delta_theta_particle = []
            for truth_phi, truth_theta in zip(truth_particles["MC::Particle.phi"][event],
                                            truth_particles["MC::Particle.theta"][event]):
                dphi = angular_difference(rec_phi, truth_phi)
                dtheta = rec_theta - truth_theta
                delta_phi_particle.append(dphi)
                delta_theta_particle.append(dtheta)
            delta_phi_event.append(delta_phi_particle)
            delta_theta_event.append(delta_theta_particle)
        delta_phi.append(delta_phi_event)
        delta_theta.append(delta_theta_event)

    matching_index, min_delta_phi, min_delta_theta, keep_reco = [], [], [], []

    for event_i, (delta_phi_event, delta_theta_event) in enumerate(zip(delta_phi, delta_theta)):
        matching_index_event = []
        min_delta_phi_event = []
        min_delta_theta_event = []
        keep_reco_event = []

        for dphi_list, dtheta_list in zip(delta_phi_event, delta_theta_event):
            if len(dphi_list) == 0:
                print(f"Event number {event_i} had no truth particles for matching!")
                matching_index_event.append(-1)
                min_delta_phi_event.append(-1)
                min_delta_theta_event.append(-1)
                keep_reco_event.append(False)
                continue

            # Find truth particle that minimizes |Δθ| + |Δφ| (or could use another metric)
            distances = np.abs(dtheta_list) + np.abs(dphi_list)
            min_index = np.argmin(distances)
            min_dphi = dphi_list[min_index]
            min_dtheta = dtheta_list[min_index]

            # Apply the new rectangular cut
            if (abs(min_dtheta) > delta_theta_cut) or (abs(min_dphi) > delta_phi_cut):
                matching_index_event.append(-1)
                min_delta_phi_event.append(-1)
                min_delta_theta_event.append(-1)
                keep_reco_event.append(False)
                continue

            # Handle duplicates like before
            if min_index in matching_index_event:
                other_particle_index = matching_index_event.index(min_index)
                other_dphi = min_delta_phi_event[other_particle_index]
                other_dtheta = min_delta_theta_event[other_particle_index]
                other_distance = abs(other_dphi) + abs(other_dtheta)
                this_distance = abs(min_dphi) + abs(min_dtheta)

                if this_distance < other_distance:
                    # Replace with this particle
                    matching_index_event.append(min_index)
                    min_delta_phi_event.append(min_dphi)
                    min_delta_theta_event.append(min_dtheta)
                    keep_reco_event.append(True)

                    # Try second-best for the other particle
                    distances_other = np.abs(delta_theta_event[other_particle_index]) + \
                                    np.abs(delta_phi_event[other_particle_index])
                    second_smallest_index = find_second_smallest(distances_other)

                    if second_smallest_index > -1:
                        new_dphi = delta_phi_event[other_particle_index][second_smallest_index]
                        new_dtheta = delta_theta_event[other_particle_index][second_smallest_index]
                        if (abs(new_dtheta) <= delta_theta_cut) and (abs(new_dphi) <= delta_phi_cut):
                            matching_index_event[other_particle_index] = second_smallest_index
                            min_delta_phi_event[other_particle_index] = new_dphi
                            min_delta_theta_event[other_particle_index] = new_dtheta
                            keep_reco_event[other_particle_index] = True
                        else:
                            min_delta_phi_event[other_particle_index] = -1
                            min_delta_theta_event[other_particle_index] = -1
                            keep_reco_event[other_particle_index] = False
                    else:
                        min_delta_phi_event[other_particle_index] = -1
                        min_delta_theta_event[other_particle_index] = -1
                        keep_reco_event[other_particle_index] = False
                else:
                    # Current particle: try second-best
                    second_smallest_index = find_second_smallest(distances)
                    matching_index_event.append(second_smallest_index)
                    if second_smallest_index > -1:
                        new_dphi = dphi_list[second_smallest_index]
                        new_dtheta = dtheta_list[second_smallest_index]
                        if (abs(new_dtheta) <= delta_theta_cut) and (abs(new_dphi) <= delta_phi_cut):
                            min_delta_phi_event.append(new_dphi)
                            min_delta_theta_event.append(new_dtheta)
                            keep_reco_event.append(True)
                        else:
                            min_delta_phi_event.append(-1)
                            min_delta_theta_event.append(-1)
                            keep_reco_event.append(False)
                    else:
                        min_delta_phi_event.append(-1)
                        min_delta_theta_event.append(-1)
                        keep_reco_event.append(False)
            else:
                matching_index_event.append(min_index)
                min_delta_phi_event.append(min_dphi)
                min_delta_theta_event.append(min_dtheta)
                keep_reco_event.append(True)

        matching_index.append(matching_index_event)
        min_delta_phi.append(min_delta_phi_event)
        min_delta_theta.append(min_delta_theta_event)
        keep_reco.append(keep_reco_event)

    # Convert to awkward arrays as before
    matching_index = ak.Array(matching_index)
    min_delta_phi = ak.Array(min_delta_phi)
    min_delta_theta = ak.Array(min_delta_theta)
    keep_reco = ak.Array(keep_reco)

    matching_index = matching_index[keep_reco]
    min_delta_phi = min_delta_phi[keep_reco]
    min_delta_theta = min_delta_theta[keep_reco]
    event_data["reconstructed_particles"] = event_data["reconstructed_particles"][keep_reco]

    # Ordering the truth particles and removing the truth particles without a matching reco entry
    event_data["truth_particles"]=event_data["truth_particles"][matching_index]
    electron_mask = (event_data["truth_particles"]["MC::Particle.pid"] != 11) & (event_data["truth_particles"]["MC::Particle.pid"]!=-11)
    event_data["truth_particles"]=event_data["truth_particles"][electron_mask]
    event_data["reconstructed_particles"]=event_data["reconstructed_particles"][electron_mask]
    
    pion_kaon_mask = (event_data["truth_particles"]["MC::Particle.pid"] == 211) | (event_data["truth_particles"]["MC::Particle.pid"] == -211) |\
    (event_data["truth_particles"]["MC::Particle.pid"] == -321) | (event_data["truth_particles"]["MC::Particle.pid"] == 321)
    event_data["truth_particles"] =event_data["truth_particles"][pion_kaon_mask]
    event_data["reconstructed_particles"] = event_data["reconstructed_particles"][pion_kaon_mask]
    
    number_truth_particles = ak.num(event_data["truth_particles"]["MC::Particle.pid"])
    number_reconstructed_particles = ak.num(event_data["reconstructed_particles"]["REC::Particles.pid"])

    if not ak.array_equal(number_truth_particles, number_reconstructed_particles):
        raise ValueError("Some events had an unequal number of reco and truth particles!")
    nonempty_particles_mask = ak.num(event_data["truth_particles"]["MC::Particle.pid"])>0
    event_data = event_data[nonempty_particles_mask]
    if max_num_trajectories is not None:
        num_particles_mask = ak.num(event_data["reconstructed_particles"]["REC::Particles.pid"]) <= max_num_trajectories
        event_data = event_data[num_particles_mask]
    print(f"Have {len(event_data)} events after particle removal and reco/truth matching")

    # Making sure pions and kaons have the same number of events. Randomly drop events from the particle that has more
    pion_mask = (event_data["truth_particles"]["MC::Particle.pid"][:,0] == 211) | (event_data["truth_particles"]["MC::Particle.pid"][:,0] == -211)
    kaon_mask = (event_data["truth_particles"]["MC::Particle.pid"][:,0] == -321) | (event_data["truth_particles"]["MC::Particle.pid"][:,0] == 321)
    num_pions = ak.sum(pion_mask)
    num_kaons = ak.sum(kaon_mask)
    pion_kaon_balance_mask = np.ones(len(pion_mask), dtype=bool)
    if num_pions > num_kaons:
        number_difference = num_pions - num_kaons
        # Get pion indices
        pion_indices = np.where(ak.to_numpy(pion_mask))[0]
        # Randomly choose pions to drop
        drop_indices = np.random.choice(pion_indices, size=number_difference, replace=False)
        pion_kaon_balance_mask[drop_indices] = False
    elif num_kaons > num_pions:
        number_difference = num_kaons - num_pions
        # Get kaon indices
        kaon_indices = np.where(ak.to_numpy(kaon_mask))[0]
        # Randomly choose kaons to drop
        drop_indices = np.random.choice(kaon_indices, size=number_difference, replace=False)
        pion_kaon_balance_mask[drop_indices] = False
    
    event_data = event_data[pion_kaon_balance_mask]
    print(f"Have {len(event_data)} events after balancing pions and kaons")

    return event_data

# In every event, returns [0/1, 0/1] if a pion or kaon are present
def pid_to_indices(pid):
    indices = []
    for event in pid:
        type_array = np.zeros(shape=2)
        for particle in event:
            if particle==211 or particle==-211:
                type_array[0] = 1
            elif particle == 321 or particle==-321:
                type_array[1] = 1
            else:
                continue
        indices.append(type_array)
    return ak.Array(indices)
def scale_data(data, sector, data_name):

    min_max_dict = {
        "4": {
            "RICH::Hit.x": {"min": -166, "max":-37},
            "RICH::Hit.y": {"min": -81, "max":78},
            "RICH::Hit.rawtime": {"min": 125, "max":19500},
            "REC::Particles.p": {"min": 1, "max":12},
            "trajectories/RICH_aerogel_b1/REC::Traj.x": {"min": -123, "max":-35},
            "trajectories/RICH_aerogel_b1/REC::Traj.y": {"min": -45, "max":45},
            "trajectories/RICH_aerogel_b1/REC::Traj.cx": {"min": -1, "max":1},
            "trajectories/RICH_aerogel_b1/REC::Traj.cy": {"min": -1, "max":1},
            "trajectories/RICH_aerogel_b1/REC::Traj.cz": {"min": -1, "max":1},
            "trajectories/RICH_aerogel_b2/REC::Traj.x": {"min": -184, "max":-99},
            "trajectories/RICH_aerogel_b2/REC::Traj.y": {"min": -80, "max":71},
            "trajectories/RICH_aerogel_b2/REC::Traj.cx": {"min": -1, "max":1},
            "trajectories/RICH_aerogel_b2/REC::Traj.cy": {"min": -1, "max":1},
            "trajectories/RICH_aerogel_b2/REC::Traj.cz": {"min": -1, "max":1},
            "trajectories/RICH_aerogel_b3/REC::Traj.x": {"min": -244, "max":-150},
            "trajectories/RICH_aerogel_b3/REC::Traj.y": {"min": -118, "max":89},
            "trajectories/RICH_aerogel_b3/REC::Traj.cx": {"min": -1, "max":1},
            "trajectories/RICH_aerogel_b3/REC::Traj.cy": {"min": -1, "max":1},
            "trajectories/RICH_aerogel_b3/REC::Traj.cz": {"min": -1, "max":1},
        }
    }
    data_min = min_max_dict[f"{sector}"][data_name]["min"]
    data_max = min_max_dict[f"{sector}"][data_name]["max"]
    data_scaled = (data - data_min) / (data_max - data_min)
    return data_scaled
def save_data(event_data, save_dir, file_name, sector):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    output_file =  h5.File(os.path.join(save_dir, f"{file_name}.h5"), "w")

    output_RICH_hits = {}
    output_RICH_hits["RICH::Hit.x"] = scale_data(event_data["RICH_hits"]["RICH::Hit.x"], sector, "RICH::Hit.x")
    output_RICH_hits["RICH::Hit.y"] = scale_data(event_data["RICH_hits"]["RICH::Hit.y"], sector, "RICH::Hit.y")
    output_RICH_hits["RICH::Hit.rawtime"] = scale_data(event_data["RICH_hits"]["RICH::Hit.rawtime"], sector, "RICH::Hit.rawtime")
    
    float_type = h5.vlen_dtype(np.dtype('float32'))
    int_type = h5.vlen_dtype(np.dtype('int32'))
    for key, value in output_RICH_hits.items():
        dset = output_file.create_dataset(f"RICH_Hits/{key}", (len(value),), dtype=float_type)
        dset[...] = value

    output_MC_particles = {}
    output_MC_particles["MC::Particle.pid"] = pid_to_indices(event_data["truth_particles"]["MC::Particle.pid"])
    for key, value in output_MC_particles.items():
        dset = output_file.create_dataset(f"truth_particles/{key}", (len(value),), dtype=int_type)
        dset[...] = value

    output_reco_particles = {}
    output_reco_particles["REC::Particles.pid"] = event_data["reconstructed_particles"]["REC::Particles.pid"]
    output_reco_particles["REC::Particles.p"] = scale_data(event_data["reconstructed_particles"]["REC::Particles.p"], sector, "REC::Particles.p")

    for key, value in output_reco_particles.items():
        if "pid" in key:
            dset = output_file.create_dataset(f"reconstructed_particles/{key}", (len(value),), dtype=int_type)
        else:
            dset = output_file.create_dataset(f"reconstructed_particles/{key}", (len(value),), dtype=float_type)
        dset[...] = value

    output_trajectories = {}

    # Need to combine these into one trajectory
    RICH_aerogel_b1_mask = event_data["trajectories"]["REC::Traj.layer"]==2
    b1_x = scale_data(event_data["trajectories"][RICH_aerogel_b1_mask]["REC::Traj.x"], sector, "trajectories/RICH_aerogel_b1/REC::Traj.x")
    b1_y = scale_data(event_data["trajectories"][RICH_aerogel_b1_mask]["REC::Traj.y"], sector, "trajectories/RICH_aerogel_b1/REC::Traj.y")
    b1_cx = scale_data(event_data["trajectories"][RICH_aerogel_b1_mask]["REC::Traj.cx"], sector, "trajectories/RICH_aerogel_b1/REC::Traj.cx")
    b1_cy = scale_data(event_data["trajectories"][RICH_aerogel_b1_mask]["REC::Traj.cy"], sector, "trajectories/RICH_aerogel_b1/REC::Traj.cy")
    b1_cz = scale_data(event_data["trajectories"][RICH_aerogel_b1_mask]["REC::Traj.cz"], sector, "trajectories/RICH_aerogel_b1/REC::Traj.cz")

    RICH_aerogel_b2_mask = event_data["trajectories"]["REC::Traj.layer"]==3
    b2_x = scale_data(event_data["trajectories"][RICH_aerogel_b2_mask]["REC::Traj.x"], sector, "trajectories/RICH_aerogel_b2/REC::Traj.x")
    b2_y = scale_data(event_data["trajectories"][RICH_aerogel_b2_mask]["REC::Traj.y"], sector, "trajectories/RICH_aerogel_b2/REC::Traj.y")
    b2_cx = scale_data(event_data["trajectories"][RICH_aerogel_b2_mask]["REC::Traj.cx"], sector, "trajectories/RICH_aerogel_b2/REC::Traj.cx")
    b2_cy = scale_data(event_data["trajectories"][RICH_aerogel_b2_mask]["REC::Traj.cy"], sector, "trajectories/RICH_aerogel_b2/REC::Traj.cy")
    b2_cz = scale_data(event_data["trajectories"][RICH_aerogel_b2_mask]["REC::Traj.cz"], sector, "trajectories/RICH_aerogel_b2/REC::Traj.cz")

    RICH_aerogel_b3_mask = event_data["trajectories"]["REC::Traj.layer"]==4
    b3_x = scale_data(event_data["trajectories"][RICH_aerogel_b3_mask]["REC::Traj.x"], sector, "trajectories/RICH_aerogel_b3/REC::Traj.x")
    b3_y = scale_data(event_data["trajectories"][RICH_aerogel_b3_mask]["REC::Traj.y"], sector, "trajectories/RICH_aerogel_b3/REC::Traj.y")
    b3_cx = scale_data(event_data["trajectories"][RICH_aerogel_b3_mask]["REC::Traj.cx"], sector, "trajectories/RICH_aerogel_b3/REC::Traj.cx")
    b3_cy = scale_data(event_data["trajectories"][RICH_aerogel_b3_mask]["REC::Traj.cy"], sector, "trajectories/RICH_aerogel_b3/REC::Traj.cy")
    b3_cz = scale_data(event_data["trajectories"][RICH_aerogel_b3_mask]["REC::Traj.cz"], sector, "trajectories/RICH_aerogel_b3/REC::Traj.cz")

    output_trajectories["REC::Traj.x "] = np.concatenate((b1_x, b2_x, b3_x), axis=1)
    output_trajectories["REC::Traj.y "] = np.concatenate((b1_y, b2_y, b3_y), axis=1)
    output_trajectories["REC::Traj.cx"] = np.concatenate((b1_cx, b2_cx, b3_cx), axis=1)
    output_trajectories["REC::Traj.cy"] = np.concatenate((b1_cy, b2_cy, b3_cy), axis=1)
    output_trajectories["REC::Traj.cz"] = np.concatenate((b1_cz, b2_cz, b3_cz), axis=1)

    for key, value in output_trajectories.items():
        dset = output_file.create_dataset(f"trajectories/{key}", (len(value),), dtype=float_type)
        dset[...] = value

def main():
    flags = parse_arguments()
    data_parameters = LoadYaml(flags.config, flags.config_directory)
    if data_parameters["SECTOR"] != 1 and data_parameters["SECTOR"] != 4:
        raise ValueError("SECTOR must be set to either 1 or 4 in data.yaml!")
    data_files = glob.glob(data_parameters["DATA_DIRECTORY"]+"/*.root")
    data = open_file(data_files)
    data = select_hits(
        data,
        sector=data_parameters["SECTOR"],
        use_negatives = data_parameters["USE_NEGATIVE_CHARGE"],
        use_positives = data_parameters["USE_POSITIVE_CHARGE"],
        max_num_trajectories = data_parameters["MAX_NUM_TRAJECTORIES"],
        )
    data = match_to_truth(
        data,
        max_num_trajectories = data_parameters["MAX_NUM_TRAJECTORIES"],
        )
    
    if data_parameters["TRAIN_TEST_SPLIT"]:
        num_training_events = int(data_parameters["TRAINING_FRACTION"]*len(data))
        training_data = data[:num_training_events]
        test_data = data[num_training_events:]
        data_dict = {"train": training_data, "test": test_data}

        for name, split_data in data_dict.items():
            save_data(
                split_data,
                save_dir = data_parameters["SAVE_DIRECTORY"],
                file_name = data_parameters["SAVE_FILE_NAME"]+f"_{name}",
                sector=data_parameters["SECTOR"],
                )
    else:
        save_data(
            data,
            save_dir = data_parameters["SAVE_DIRECTORY"],
            file_name = data_parameters["SAVE_FILE_NAME"],
            sector=data_parameters["SECTOR"],
            )


if __name__ == "__main__":
    main()