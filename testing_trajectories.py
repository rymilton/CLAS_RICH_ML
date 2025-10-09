import uproot as ur
import awkward as ak
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        default="/volatile/clas12/rmilton/RICH_data/RECO_DIS_default_skimp2p5_allOpticsNominal_1599.hipo.root",
        help="ROOT file to load in",
        type=str,
    )
    
    parser.add_argument(
        "--pindex",
        default=0,
        help="pindex to look at",
        type=int,
    )

    flags = parser.parse_args()

    return flags

def main(flags):

    with ur.open(f"{flags.file}:events") as f:
        rec_traj = f.arrays(filter_name="REC::Traj*")
        rec_particles = f.arrays(filter_name="REC::Particles*")

    pindex = flags.pindex
    # Selecting trajectories with desired pindex that appear in RICH aerogel
    b1_b2_b3_mask = (rec_traj["REC::Traj.detector"]==18) & (rec_traj["REC::Traj.pindex"]==pindex) & ((rec_traj["REC::Traj.layer"]==2) | (rec_traj["REC::Traj.layer"]==3) | (rec_traj["REC::Traj.layer"]==4))
    masked_layers = rec_traj["REC::Traj.layer"][b1_b2_b3_mask]
    masked_pindices = rec_traj["REC::Traj.pindex"][b1_b2_b3_mask]

    # Selecting events with multiple entries for the trajectory with desired pindex
    # If there are multiple entries, that means the trajectory appeared in multiple aerogel layers
    multiple_intersections_mask = ak.num(masked_pindices)>1
    masked_pindices = masked_pindices[multiple_intersections_mask]
    
    print(f"Traj. with pindex {pindex} is in layers:{masked_layers[multiple_intersections_mask]}")
    print(f"Traj. with pindex {pindex} is reconstructed with PIDs:{rec_particles['REC::Particles.pid'][multiple_intersections_mask][masked_pindices]}")

if __name__ == "__main__":
    flags = parse_arguments()
    main(flags)