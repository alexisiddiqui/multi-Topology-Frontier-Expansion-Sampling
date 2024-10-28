import os
import pymol
from pymol import cmd
def load_concatenate_dcds(pdb_file, dcd_directory):
    # Load the initial structure
    cmd.load(pdb_file)
    
    # Get the object name (assumes it's the same as the PDB filename without extension)
    obj_name = os.path.splitext(os.path.basename(pdb_file))[0]
    
    # Get all DCD files in the directory
    dcd_files = sorted([f for f in os.listdir(dcd_directory) if f.endswith('.dcd')])
    
    # Load each DCD file
    total_frames = 0
    for dcd_file in dcd_files:
        dcd_path = os.path.join(dcd_directory, dcd_file)
        
        # Load the trajectory
        cmd.load_traj(dcd_path, obj_name, state=0)
        
        # Update the total frame count
        new_total_frames = cmd.count_states(obj_name)
        frames_added = new_total_frames - total_frames
        total_frames = new_total_frames
        
        print(f"Loaded {dcd_file}. Added {frames_added} frames. Total frames: {total_frames}")
    
    print(f"All trajectories loaded. Total frames: {total_frames}")

# Example usage
pdb_file = "/home/alexi/Documents/Frontier-Expansion-Sampling/bpti.pdb"
dcd_directory = "/home/alexi/Documents/Frontier-Expansion-Sampling/10"

# Ensure PyMOL is in command mode
cmd.set("movie_panel", 0)

# Run the function
load_concatenate_dcds(pdb_file, dcd_directory)

# Set up the movie panel after loading all trajectories
cmd.set("movie_panel", 1)
cmd.mset(f"1 -{total_frames}")
