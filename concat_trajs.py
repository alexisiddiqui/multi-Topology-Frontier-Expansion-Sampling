import os
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter
import shutil
from tqdm import tqdm
import multiprocessing
import sys  # Added import for sys

def create_output_dirs(base_output_dir, protein_name):
    os.makedirs(base_output_dir, exist_ok=True)
    protein_dir = os.path.join(base_output_dir, protein_name)
    intermediate_dir = os.path.join(protein_dir, 'intermediate')
    final_dir = os.path.join(protein_dir, 'final')
    
    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    return intermediate_dir, final_dir

def process_topology(topology_dir, output_dir, protein_name):
    print(f"Starting processing for topology directory: {topology_dir}")
    prmtop_files = [f for f in os.listdir(topology_dir) if f.endswith('.prmtop')]
    if not prmtop_files:
        print(f"No topology file found in {topology_dir}")
        return None
    topology_file = os.path.join(topology_dir, prmtop_files[0])
    
    topology_name = os.path.basename(topology_dir)
    topology_output_dir = os.path.join(output_dir, topology_name)
    os.makedirs(topology_output_dir, exist_ok=True)
    
    trajectory_files = []
    for root, _, files in os.walk(topology_dir):
        for file in files:
            if file.endswith('.dcd'):
                trajectory_files.append(os.path.join(root, file))
    
    if not trajectory_files:
        print(f"No trajectory files found in {topology_dir}")
        return None
        
    u = mda.Universe(topology_file, trajectory_files[0])
    
    selection = ("protein and (not resname SOL WAT TIP3 HOH) and "
                "(not (name H[BGJDEFZ]* HE* HH* HD* HZ*)) and "
                "(backbone or name H HA or not name H*)")
    
    protein = u.select_atoms(selection)
    
    stripped_top_file = os.path.join(topology_output_dir, 
                                    f"{protein_name}_{topology_name}_stripped.pdb")
    protein.write(stripped_top_file)
    print(f"Saved stripped topology to {stripped_top_file}")
    
    stripped_traj_files = []
    for traj_file in trajectory_files:
        u_traj = mda.Universe(topology_file, traj_file)
        protein_traj = u_traj.select_atoms(selection)
        
        out_traj_name = f"{protein_name}_{os.path.basename(traj_file)}_stripped.xtc"
        out_traj_file = os.path.join(topology_output_dir, out_traj_name)
        
        with XTCWriter(out_traj_file, n_atoms=protein_traj.n_atoms) as w:
            for ts in u_traj.trajectory:
                w.write(protein_traj)
        stripped_traj_files.append(out_traj_file)
        print(f"Processed {traj_file}")
    
    combined_traj_name = f"{protein_name}_{topology_name}_combined_stripped.xtc"
    combined_traj_file = os.path.join(topology_output_dir, combined_traj_name)
    
    u_combined = mda.Universe(stripped_top_file, stripped_traj_files)
    with XTCWriter(combined_traj_file, n_atoms=u_combined.atoms.n_atoms) as w:
        for ts in u_combined.trajectory:
            w.write(u_combined.atoms)
    
    print(f"Processed {topology_dir}, combined trajectory saved to {combined_traj_file}")
    return combined_traj_file, stripped_top_file

def process_topology_wrapper(args):
    return process_topology(*args)

def main():
    print("Initializing processing of topology directories...")
    base_dir = '/home/lina4225/_data/multi-Topology-Frontier-Expansion-Sampling/RW_11_FES_output_bad'
    
    # Modify protein_name to use command-line argument if provided
    if len(sys.argv) > 1:
        protein_name = sys.argv[1]
    else:
        protein_name = 'BPTI'
    
    protein_dir = os.path.join(base_dir, protein_name)
    
    output_base_dir = os.path.join(base_dir, 'trajectory_outputs')
    intermediate_dir, final_dir = create_output_dirs(output_base_dir, protein_name)
    
    topology_dirs = [os.path.join(protein_dir, d) for d in os.listdir(protein_dir)
                    if os.path.isdir(os.path.join(protein_dir, d)) and d.startswith(f'{protein_name}_10_c')]
    
    combined_traj_files = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    print(f"Created multiprocessing pool with {multiprocessing.cpu_count()} processes.")
    
    results = []
    for result in tqdm(pool.imap(process_topology_wrapper, 
                                [(td, intermediate_dir, protein_name) for td in topology_dirs]), 
                      total=len(topology_dirs), 
                      desc='Processing topologies'):
        if result:
            traj_file, stripped_top_file = result
            combined_traj_files.append(traj_file)
            print(f"Added {traj_file} to combined trajectories.")
    
    pool.close()
    pool.join()
    
    if combined_traj_files and stripped_top_file:
        print("Combining all trajectories into final output...")
        final_traj_name = f'{protein_name}_overall_combined_stripped.xtc'
        final_top_name = f'{protein_name}_overall_combined_stripped.pdb'
        
        final_traj_file = os.path.join(final_dir, final_traj_name)
        final_top_file = os.path.join(final_dir, final_top_name)
        
        shutil.copy2(stripped_top_file, final_top_file)
        
        u_overall = mda.Universe(final_top_file, combined_traj_files)
        with XTCWriter(final_traj_file, n_atoms=u_overall.atoms.n_atoms) as w:
            for ts in u_overall.trajectory:
                w.write(u_overall.atoms)
        
        print(f"All trajectories combined into {final_traj_file}")
        print(f"Final topology saved to {final_top_file}")
    else:
        print("No trajectory files processed.")
    
    print("Processing complete.")

if __name__ == "__main__":
    main()