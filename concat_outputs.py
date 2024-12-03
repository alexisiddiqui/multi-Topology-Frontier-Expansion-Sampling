#!/usr/bin/env python3

import os
import json
import mdtraj as md
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import warnings
from tqdm import tqdm
import sys

def get_reference_topology(topology_file: str) -> Tuple[md.Topology, np.ndarray]:
    """
    Create a reference topology from a topology file.
    Returns the stripped topology and atom indices.
    """
    traj = md.load_prmtop(topology_file)
    # Select non-hydrogen protein atoms
    protein_heavy = traj.select("protein and not element H")
    # Create new topology with just the selected atoms
    stripped_top = traj.subset(protein_heavy)
    return stripped_top, protein_heavy

def process_trajectory(args: Tuple) -> Tuple[md.Trajectory, Dict]:
    """
    Process a single trajectory file using reference atom indices.
    """
    dcd_path, top_path, ref_topology, indices, cycle, seed, topology_name = args
    
    try:
        # Load trajectory with selected atoms
        traj = md.load(dcd_path, top=top_path, atom_indices=indices)
        # Set the reference topology
        traj.topology = ref_topology
        
        traj_info = {
            "topology": topology_name,
            "cycle": cycle,
            "seed": seed,
            "n_frames": traj.n_frames,
            "path": dcd_path
        }
        
        print(f"Processed trajectory: {topology_name} Cycle {cycle}, Seed {seed}")
        return traj, traj_info
    except Exception as e:
        print(f"Error processing trajectory {dcd_path}: {str(e)}")
        return None, None

def concatenate_protein_trajectories(base_path: str, protein: str, max_workers: int = 4) -> None:
    """
    Concatenate trajectories for a single protein using a reference topology.
    """
    protein_path = os.path.join(base_path, protein)
    output_dir = os.path.join(base_path, f"{protein}_combined")
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort directories for consistent reference topology
    items = sorted([item for item in os.listdir(protein_path) 
                   if item.startswith(f"{protein}_10_c")])
    
    # Get reference topology
    ref_topology = None
    ref_indices = None
    topology_dirs = []
    
    for item in items:
        topology_path = os.path.join(protein_path, item)
        if os.path.isdir(topology_path):
            prmtop_file = os.path.join(topology_path, f"{item}.prmtop")
            if os.path.exists(prmtop_file):
                try:
                    if ref_topology is None:
                        ref_topology, ref_indices = get_reference_topology(prmtop_file)
                        print(f"Using {item} as reference topology")
                        topology_dirs.append((item, topology_path, prmtop_file))
                    else:
                        # Verify topology matches reference
                        test_top, test_indices = get_reference_topology(prmtop_file)
                        if len(test_indices) == len(ref_indices):
                            topology_dirs.append((item, topology_path, prmtop_file))
                        else:
                            print(f"Warning: Topology {item} has different number of atoms than reference")
                except Exception as e:
                    print(f"Error processing topology {item}: {str(e)}")
                    continue
    
    if ref_topology is None:
        print(f"No valid reference topology found for {protein}")
        return
    
    # Collect all trajectories
    all_trajs = []
    for topology_name, topology_path, prmtop_file in topology_dirs:
        # Process each cycle directory
        cycle_dirs = sorted([d for d in os.listdir(topology_path) if d.isdigit()])
        for cycle_dir in cycle_dirs:
            cycle = int(cycle_dir)
            cycle_path = os.path.join(topology_path, cycle_dir)
            
            # Add each trajectory file
            dcd_files = sorted([f for f in os.listdir(cycle_path) if f.endswith('.dcd')])
            for file in dcd_files:
                seed = int(file.split('.')[0])
                dcd_path = os.path.join(cycle_path, file)
                all_trajs.append((
                    dcd_path, prmtop_file, ref_topology,
                    ref_indices, cycle, seed, topology_name
                ))
    
    if not all_trajs:
        print(f"No trajectories found for {protein}")
        return
    
    print(f"\nProcessing {len(all_trajs)} trajectories for {protein}...")
    
    # Process trajectories in parallel
    processed_trajs = []
    traj_info = []
    current_frame = 0
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_trajectory, all_trajs),
            total=len(all_trajs),
            desc="Processing trajectories"
        ))
    
    # Collect valid results
    for traj, info in results:
        if traj is not None:
            processed_trajs.append(traj)
            info["start_frame"] = current_frame
            info["end_frame"] = current_frame + info["n_frames"]
            current_frame = info["end_frame"]
            traj_info.append(info)
    
    if not processed_trajs:
        print(f"No valid processed trajectories for {protein}")
        return
    
    # Concatenate trajectories
    print(f"\nConcatenating {len(processed_trajs)} trajectories...")
    final_traj = md.join(processed_trajs)
    
    # Save concatenated trajectory as XTC
    output_traj_path = os.path.join(output_dir, f"{protein}_combined.xtc")
    final_traj.save_xtc(output_traj_path)
    print(f"Saved concatenated trajectory to {output_traj_path}")
    
    # Save reference topology as PDB
    output_top_path = os.path.join(output_dir, f"{protein}_combined.pdb")
    

    ref_traj = md.Trajectory(final_traj.xyz[0], ref_topology)
    ref_traj.save(output_top_path)



    print(f"Saved reference topology to {output_top_path}")
    
    # Save trajectory info
    info_dict = {
        "protein": protein,
        "total_frames": final_traj.n_frames,
        "n_atoms": final_traj.n_atoms,
        "reference_topology": topology_dirs[0][0],
        "frames": traj_info
    }
    
    info_path = os.path.join(output_dir, f"{protein}_trajectory_info.json")
    with open(info_path, 'w') as f:
        json.dump(info_dict, f, indent=2)
    print(f"Saved trajectory info to {info_path}")

def main():
    """Main function to process all proteins."""
    base_path = os.path.abspath("RW_11_FES_output")
    print(f"Processing trajectories in: {base_path}")
    protein_names = ["BPTI", "BRD4", "HOIP", "LXR", "MBP"][2:]
    protein_names = ["LXR", "MBP"]

    if len(sys.argv) > 1:
        proteins = [sys.argv[1]]
    else:
        proteins = [d for d in os.listdir(base_path) 
                   if os.path.isdir(os.path.join(base_path, d)) 
                   and d in protein_names]
    print(f"Found proteins: {proteins}")
    
    for protein in proteins:
        print(f"\nProcessing {protein}...")
        try:
            concatenate_protein_trajectories(base_path, protein)
        except Exception as e:
            print(f"Error processing {protein}: {str(e)}")
            continue

if __name__ == "__main__":
    main()