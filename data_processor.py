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
        ):
    # Removing invalid RICH hits and keeping desired sector hits
    hits = event_data["RICH_hits"]
    valid_hits_mask = (hits["RICH::Hit.cluster"] == 0) & (hits["RICH::Hit.cluster"] == 0) & (hits["RICH::Hit.pmt"] > 0) & (hits["RICH::Hit.pmt"] < 392)
    hits_sector_mask = hits["RICH::Hit.sector"] == sector
    event_data["RICH_hits"] = event_data["RICH_hits"][(valid_hits_mask) & (hits_sector_mask)]
    nonempty_hits_mask = ak.num(event_data["RICH_hits"]["RICH::Hit.x"])>0
    event_data = event_data[nonempty_hits_mask]

    # Selecting reconstructed particles that have a trajectory that hits the RICH aerogel
    RICH_detector_ID = 18
    trajectories = event_data["trajectories"]
    RICH_trajectory_mask = (trajectories["REC::Traj.detector"]==RICH_detector_ID) & (trajectories["REC::Traj.layer"]>1)
    trajectory_sector_mask = trajectories["REC::Traj.x"] > 0 if sector == 1 else trajectories["REC::Traj.x"] < 0
    event_data["trajectories"] = event_data["trajectories"][(RICH_trajectory_mask) & (trajectory_sector_mask)]
    nonempty_trajectories_mask = ak.num(event_data["trajectories"]["REC::Traj.x"])>0
    event_data = event_data[nonempty_trajectories_mask]

    # Only keeping reconstructed particles that have a trajectory that satisfied selection
    unique_pindex = [np.unique(event_pindices).to_list() for event_pindices in event_data["trajectories"]["REC::Traj.pindex"]]
    event_data["reconstructed_particles"] = event_data["reconstructed_particles"][unique_pindex]
    # Removing neutral particles from reconstructed data
    neutral_mask = (event_data["reconstructed_particles"]["REC::Particles.charge"] > 0 ) | event_data["reconstructed_particles"]["REC::Particles.charge"] < 0
    event_data["reconstructed_particles"] = event_data["reconstructed_particles"][neutral_mask]
    if not use_negatives and not use_positives:
        raise ValueError("Must enable selection of negative or positive particles!")
    if not use_negatives:
        negatives_mask = event_data["reconstructed_particles"]["REC::Particles.charge"] > 0
        event_data["reconstructed_particles"] = event_data["reconstructed_particles"][negatives_mask]
    if not use_positives:
        positives_mask = event_data["reconstructed_particles"]["REC::Particles.charge"] < 0 
        event_data["reconstructed_particles"] = event_data["reconstructed_particles"][positives_mask]

    # Removing events that don't have any reconstructed particles. Not sure if this ever actually happens but worth checking
    nonempty_recoparticles_mask = ak.num(event_data["reconstructed_particles"]["REC::Particles.pid"])>0
    event_data = event_data[nonempty_recoparticles_mask]

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
    rec_particles["REC::Particles.p"] = np.sqrt(rec_particles["REC::Particles.px"]**2 + rec_particles["REC::Particles.py"]**2 + rec_particles["REC::Particles.pz"]**2)
    rec_particles = rec_particles[rec_particles["REC::Particles.p"]>0]
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

    event_data["truth_particles"]=event_data["truth_particles"][matching_index]
    electron_mask = (event_data["truth_particles"]["MC::Particle.pid"] != 11) & (event_data["truth_particles"]["MC::Particle.pid"]!=-11)
    event_data["truth_particles"]=event_data["truth_particles"][electron_mask]
    event_data["reconstructed_particles"]=event_data["reconstructed_particles"][electron_mask]
    
    number_truth_particles = ak.num(event_data["truth_particles"]["MC::Particle.pid"])
    number_reconstructed_particles = ak.num(event_data["reconstructed_particles"]["REC::Particles.pid"])
    if not ak.array_equal(number_truth_particles, number_reconstructed_particles):
        raise ValueError("Some events had an unequal number of reco and truth particles!")
    nonempty_particles_mask = ak.num(event_data["truth_particles"]["MC::Particle.pid"])>0
    event_data = event_data[nonempty_particles_mask]
    if max_num_trajectories is not None:
        num_particles_mask = ak.num(event_data["reconstructed_particles"]["REC::Particles.pid"]) <= max_num_trajectories
        event_data = event_data[num_particles_mask]
    print(len(event_data))
    return event_data

# In every event, returns [0/1, 0/1, 0/1] if a pion, kaon, proton are present
def pid_to_indices(pid):
    indices = []
    for event in pid:
        type_array = np.zeros(shape=3)
        for particle in event:
            if particle==211 or particle==-211:
                type_array[0] = 1
            elif particle == 321 or particle==-321:
                type_array[1] = 1
            elif particle == 2212 or particle == -2212:
                type_array[2] = 1
            else:
                continue
        indices.append(type_array)
    return ak.Array(indices)
def scale_data(data):
    data_min = ak.min(data)
    data_max = ak.max(data)
    data_scaled = (data - data_min) / (data_max - data_min)
    return data_scaled
def save_data(event_data, save_dir, file_name):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    output_file =  h5.File(os.path.join(save_dir, f"{file_name}.h5"), "w")

    output_RICH_hits = {}
    output_RICH_hits["RICH::Hit.x"] = scale_data(event_data["RICH_hits"]["RICH::Hit.x"])
    output_RICH_hits["RICH::Hit.y"] = scale_data(event_data["RICH_hits"]["RICH::Hit.y"])
    output_RICH_hits["RICH::Hit.time"] = scale_data(event_data["RICH_hits"]["RICH::Hit.time"])
    
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
    output_reco_particles["REC::Particles.px"] = scale_data(event_data["reconstructed_particles"]["REC::Particles.px"])
    output_reco_particles["REC::Particles.py"] = scale_data(event_data["reconstructed_particles"]["REC::Particles.py"])
    output_reco_particles["REC::Particles.pz"] = scale_data(event_data["reconstructed_particles"]["REC::Particles.pz"])

    for key, value in output_reco_particles.items():
        if "pid" in key:
            dset = output_file.create_dataset(f"reconstructed_particles/{key}", (len(value),), dtype=int_type)
        else:
            dset = output_file.create_dataset(f"reconstructed_particles/{key}", (len(value),), dtype=float_type)
        dset[...] = value

    output_trajectories = {}

    RICH_aerogel_b1_mask = event_data["trajectories"]["REC::Traj.layer"]==2
    output_trajectories["trajectories/RICH_aerogel_b1/REC::Traj.x"] = scale_data(event_data["trajectories"][RICH_aerogel_b1_mask]["REC::Traj.x"])
    output_trajectories["trajectories/RICH_aerogel_b1/REC::Traj.y"] = scale_data(event_data["trajectories"][RICH_aerogel_b1_mask]["REC::Traj.y"])

    RICH_aerogel_b2_mask = event_data["trajectories"]["REC::Traj.layer"]==3
    output_trajectories["trajectories/RICH_aerogel_b2/REC::Traj.x"] = scale_data(event_data["trajectories"][RICH_aerogel_b2_mask]["REC::Traj.x"])
    output_trajectories["trajectories/RICH_aerogel_b2/REC::Traj.y"] = scale_data(event_data["trajectories"][RICH_aerogel_b2_mask]["REC::Traj.y"])

    RICH_aerogel_b3_mask = event_data["trajectories"]["REC::Traj.layer"]==4
    output_trajectories["trajectories/RICH_aerogel_b3/REC::Traj.x"] = scale_data(event_data["trajectories"][RICH_aerogel_b3_mask]["REC::Traj.x"])
    output_trajectories["trajectories/RICH_aerogel_b3/REC::Traj.y"] = scale_data(event_data["trajectories"][RICH_aerogel_b3_mask]["REC::Traj.y"])

    for key, value in output_trajectories.items():
        dset = output_file.create_dataset(key, (len(value),), dtype=float_type)
        dset[...] = value

def main():
    flags = parse_arguments()
    data_parameters = LoadYaml(flags.config, flags.config_directory)
    if data_parameters["SECTOR"] != 1 and data_parameters["SECTOR"] != 4:
        raise ValueError("SECTOR must be set to either 1 or 4 in data.yaml!")
    test_file = glob.glob(data_parameters["DATA_DIRECTORY"]+"/*.root")
    print(len(test_file))
    test_data = open_file(test_file)
    test_data = select_hits(
        test_data,
        sector=data_parameters["SECTOR"],
        use_negatives = data_parameters["USE_NEGATIVE_CHARGE"],
        use_positives = data_parameters["USE_POSITIVE_CHARGE"]
        )
    test_data = match_to_truth(
        test_data,
        max_num_trajectories = data_parameters["MAX_NUM_TRAJECTORIES"],
        )
    save_data(
        test_data,
        save_dir = data_parameters["SAVE_DIRECTORY"],
        file_name = data_parameters["SAVE_FILE_NAME"]
        )


if __name__ == "__main__":
    main()