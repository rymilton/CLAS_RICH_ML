import argparse
import uproot
import glob
import awkward as ak
from utils import LoadYaml
import numpy as np
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
    start_time = time.time()
    for i, file in enumerate(file_list):
        if i%10==0:
            print(f"Up until file {i} took: ",time.time() - start_time)
            start_time = time.time()
        with uproot.open(f"{file}:events") as f:
            event_dictionary["RICH_hits"] = ak.concatenate((event_dictionary["RICH_hits"], f.arrays(filter_name="RICH::Hit.*")))
            event_dictionary["trajectories"] = ak.concatenate((event_dictionary["trajectories"], f.arrays(filter_name="REC::Traj.*")))
            event_dictionary["truth_particles"] = ak.concatenate((event_dictionary["truth_particles"], f.arrays(filter_name="MC::Particle.*")))
            event_dictionary["reconstructed_particles"] = ak.concatenate((event_dictionary["reconstructed_particles"], f.arrays(filter_name="REC::Particles.*")))
    return ak.Array(event_dictionary)

def select_hits(
        event_data, 
        sector,
        max_num_trajectories = None,
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

    # Selecting reconstructed particles that have a trajectory that hits the RICH
    RICH_detector_ID = 18
    trajectories = event_data["trajectories"]
    RICH_trajectory_mask = trajectories["REC::Traj.detector"]==RICH_detector_ID
    trajectory_sector_mask = trajectories["REC::Traj.x"] > 0 if sector == 1 else trajectories["REC::Traj.x"] < 0
    event_data["trajectories"] = event_data["trajectories"][(RICH_trajectory_mask) & (trajectory_sector_mask)]
    num_trajectories = ak.num(event_data["trajectories"]["REC::Traj.x"])
    num_trajectories_mask = num_trajectories>0
    if max_num_trajectories is not None:
        num_trajectories_mask = (num_trajectories_mask) & (num_trajectories <= max_num_trajectories)
    event_data = event_data[num_trajectories_mask]

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
def match_to_truth(event_data):

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
    
    delta_R = []
    delta_R_cut_value = 10
    # For every reconstructed particle, we calculate the Delta R value with respect to every truth particle in the event
    for event in range(len(rec_particles["REC::Particles.theta"])):
        delta_R_event = []
        for rec_phi, rec_theta in zip(rec_particles["REC::Particles.phi"][event], rec_particles["REC::Particles.theta"][event]):
            delta_R_particle = []
            for truth_phi, truth_theta in zip(truth_particles["MC::Particle.phi"][event], truth_particles["MC::Particle.theta"][event]):
                delta_R_particle.append(np.sqrt( angular_difference(rec_phi, truth_phi )**2 + (rec_theta- truth_theta)**2))
            delta_R_event.append(delta_R_particle)
        delta_R.append(delta_R_event)
    
    matching_index, min_delta_R, keep_reco = [], [], []

    for event_i, delta_R_event in enumerate(delta_R):
        matching_index_event = []
        min_delta_R_event = []
        keep_reco_event = []

        # Looping over all the delta R for each reco particle in the event
        for particle_delta_R in delta_R_event:
            if len(particle_delta_R) == 0:
                print(f"Event number {event_i} had no truth particles for DeltaR calculation!")
                matching_index_event.append(-1)
                min_delta_R_event.append(-1)
                keep_reco_event.append(False)
                continue
            # Finding the minimum delta R. i.e. the corresponding truth particle
            min_index = np.argmin(particle_delta_R)
            min_delta_R_particle = particle_delta_R[min_index]
            # If the minimum delta R value is greater than the cutoff, don't consider this particle
            if min_delta_R_particle > delta_R_cut_value:
                matching_index_event.append(-1)
                min_delta_R_event.append(-1)
                keep_reco_event.append(False)
                continue
            # Handling the case where two reco particles correspond to the same truth particle
            # In this case, keep the particle with the smaller Delta R value.
            # Give the other particle the next smallest Delta R matching. If there isn't one, append -1
            if min_index in matching_index_event:
                other_particle_index = matching_index_event.index(min_index) # Getting index of other reco particle in the event
                other_particle_deltaR = min_delta_R_event[other_particle_index]
                if min_delta_R_particle < other_particle_deltaR:
                    # Add this particle
                    matching_index_event.append(min_index)
                    min_delta_R_event.append(min_delta_R_particle)
                    keep_reco_event.append(True)
                    # Get second lowest delta R for other particle
                    all_other_delta_R = delta_R_event[other_particle_index]
                    second_smallest_index = find_second_smallest(all_other_delta_R)
                    # Getting the second smallest Delta R for this particle.
                    # If there is no second smallest, it'll have index -1 and we'll ignore this particle
                    # If there is a second smallest, check if the Delta R value satisifies the Delta R cut
                    matching_index_event[other_particle_index] = second_smallest_index
                    if second_smallest_index > -1:
                        if all_other_delta_R[second_smallest_index] < delta_R_cut_value:
                            min_delta_R_event[other_particle_index] = all_other_delta_R[second_smallest_index]
                            keep_reco_event[other_particle_index] = True
                        else:
                            min_delta_R_event[other_particle_index] = -1
                            keep_reco_event[other_particle_index] = False
                    else:
                            min_delta_R_event[other_particle_index] = -1
                            keep_reco_event[other_particle_index] = False
                else:
                    # For the current particle, get the second smallest Delta R if it exists.
                    # Make sure it satisfies the Delta R cut
                    second_smallest_index = find_second_smallest(particle_delta_R)
                    matching_index_event.append(second_smallest_index)
                    if second_smallest_index > -1:
                        if particle_delta_R[second_smallest_index] < delta_R_cut_value:
                            min_delta_R_event.append(particle_delta_R[second_smallest_index])
                            keep_reco_event.append(True)
                        else:
                            min_delta_R_event.append(-1)
                            keep_reco_event.append(False)
                    else:
                        min_delta_R_event.append(-1)
                        keep_reco_event.append(False)
            else:
                matching_index_event.append(min_index)
                min_delta_R_event.append(min_delta_R_particle)
                keep_reco_event.append(True)
        matching_index.append(matching_index_event)
        min_delta_R.append(min_delta_R_event)
        keep_reco.append(keep_reco_event)

    matching_index = ak.Array(matching_index)
    min_delta_R = ak.Array(min_delta_R)
    keep_reco = ak.Array(keep_reco)
    matching_index = matching_index[keep_reco]
    min_delta_R = min_delta_R[keep_reco]
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
    print(len(event_data))
    return event_data
def main():
    flags = parse_arguments()
    data_parameters = LoadYaml(flags.config, flags.config_directory)
    if data_parameters["SECTOR"] != 1 and data_parameters["SECTOR"] != 4:
        raise ValueError("SECTOR must be set to either 1 or 4 in data.yaml!")
    test_file = glob.glob(data_parameters["DATA_DIRECTORY"]+"/*.root")
    test_data = open_file(test_file)
    test_data = select_hits(
        test_data,
        sector=data_parameters["SECTOR"],
        max_num_trajectories = data_parameters["MAX_NUM_TRAJECTORIES"],
        use_negatives = data_parameters["USE_NEGATIVE_CHARGE"],
        use_positives = data_parameters["USE_POSITIVE_CHARGE"]
        )
    test_data = match_to_truth(test_data)


if __name__ == "__main__":
    main()