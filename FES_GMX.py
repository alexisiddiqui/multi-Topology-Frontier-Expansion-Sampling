#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Juanrong Zhang
Modified for verbosity, timing, type hints, and comments
"""

import matplotlib.pyplot as plt
import numpy as np
import mdtraj as md
import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit
import os
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull
import parmed as pmd
import time
from typing import List, Tuple

print("Imported all necessary libraries.")

# Set the name of system
system_name: str = "bpti"
print(f"System name set to: {system_name}")

# Set the number of cycles
num_cycle: int = 30
print(f"Number of cycles set to: {num_cycle}")

# The list that saves the number of seeds in each cycle
lis: List[int] = []
print("Initialized empty list to store number of seeds per cycle.")

def convert_gromacs_to_amber(top_file: str, gro_file: str) -> Tuple[str, str]:
    """
    Convert Gromacs topology and coordinate files to Amber format.
    
    Args:
    top_file (str): Path to the Gromacs topology file
    gro_file (str): Path to the Gromacs coordinate file
    
    Returns:
    Tuple[str, str]: Paths to the created Amber prmtop and inpcrd files
    """
    print("Converting Gromacs files to Amber format...")
    start_time = time.time()
    pmd.gromacs.GROMACS_TOPDIR = os.environ.get('GMXLIB', os.getcwd())
    gromacs = pmd.load_file(top_file, xyz=gro_file)
    top_path = os.path.splitext(top_file)[0]
    # Save as Amber format
    amber_prmtop = f"{top_path}.prmtop"
    amber_inpcrd = f"{top_path}.inpcrd"
    try:
        gromacs.save(amber_prmtop, format='amber')
        gromacs.save(amber_inpcrd, format='rst7')
    # save pdb
        gromacs.save(f"{system_name}.pdb", format='pdb')
    except:
        print("using existing topology")

    end_time = time.time()
    print(f"Conversion complete. Created {amber_prmtop} and {amber_inpcrd}")
    print(f"Time taken for conversion: {end_time - start_time:.2f} seconds")
    return amber_prmtop, amber_inpcrd

def build_system(prmtop: omma.AmberPrmtopFile) -> omma.Simulation:
    """
    Build the OpenMM simulation system.
    
    Args:
    prmtop (omma.AmberPrmtopFile): Amber topology file
    
    Returns:
    omma.Simulation: OpenMM simulation object
    """
    print("Building system...")
    start_time = time.time()
    system = prmtop.createSystem(nonbondedMethod=omma.PME, 
                                 nonbondedCutoff=1*unit.nanometer, 
                                 constraints=omma.HBonds)
    print("System created with PME, 1nm cutoff, and HBonds constraints.")
    
    integrator = omm.LangevinIntegrator(300*unit.kelvin, 1/unit.picoseconds, 
                                        0.002*unit.picoseconds)
    print("Langevin integrator created: 310K, 1/ps collision frequency, 2fs timestep.")
    
    platform = omm.Platform.getPlatformByName('CUDA')
    print("CUDA platform selected for GPU acceleration.")
    
    simulation = omma.Simulation(prmtop.topology, system, integrator, platform)
    print("Simulation object created.")
    
    end_time = time.time()
    print(f"Time taken to build system: {end_time - start_time:.2f} seconds")
    return simulation

def run_short_eq(simulation: omma.Simulation, inpcrd: omma.AmberInpcrdFile, steps: int) -> None:
    """
    Run a short equilibration simulation.
    
    Args:
    simulation (omma.Simulation): OpenMM simulation object
    inpcrd (omma.AmberInpcrdFile): Amber coordinate file
    steps (int): Number of simulation steps
    """
    print(f"Running short equilibration for {steps} steps...")
    start_time = time.time()
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    print("Initial positions and box vectors set.")

    print("Minimizing energy...")
    simulation.minimizeEnergy()
    print("Energy minimization complete.")

    simulation.reporters.append(omma.DCDReporter('eq.dcd', 5000))
    simulation.reporters.append(omma.StateDataReporter('eq_log.txt', 5000, 
                                                       step=True, 
                                                       potentialEnergy=True, 
                                                       temperature=True))
    print("DCD and state data reporters added.")

    print(f"Starting {steps} step equilibration...")
    simulation.step(steps)
    print("Equilibration complete.")
    end_time = time.time()
    print(f"Time taken for equilibration: {end_time - start_time:.2f} seconds")

def run_seed_simulation(simulation: omma.Simulation, inpcrd: omma.AmberInpcrdFile, steps: int, seed_index: int, coming_cycle: int) -> None:
    """
    Run a seed simulation.
    
    Args:
    simulation (omma.Simulation): OpenMM simulation object
    inpcrd (omma.AmberInpcrdFile): Amber coordinate file
    steps (int): Number of simulation steps
    seed_index (int): Index of the current seed
    coming_cycle (int): Current cycle number
    """
    print(f"Running seed simulation {seed_index} for cycle {coming_cycle}...")
    start_time = time.time()
    simulation.context.setPositions(inpcrd.positions)
    simulation.context.setVelocitiesToTemperature(300)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    print("Positions, velocities, and box vectors set.")

    dcd_file = f'./{coming_cycle}/{seed_index}.dcd'
    log_file = f'./{coming_cycle}/{seed_index}.txt'
    simulation.reporters.append(omma.DCDReporter(dcd_file, 5000))
    simulation.reporters.append(omma.StateDataReporter(log_file, 5000, 
                                                       step=True, 
                                                       potentialEnergy=True, 
                                                       temperature=True))
    print(f"Reporters added: {dcd_file} and {log_file}")

    print(f"Starting {steps} step seed simulation...")
    simulation.step(steps)
    print("Seed simulation complete.")
    end_time = time.time()
    print(f"Time taken for seed simulation: {end_time - start_time:.2f} seconds")


def get_proj_and_rst(coming_cycle: int, lis: List[int]) -> np.ndarray:
    """
    Process data for the current cycle, perform PCA on pairwise coordinates, and identify vertices.
    
    Args:
    coming_cycle (int): Current cycle number
    lis (List[int]): List of number of seeds in previous cycles
    
    Returns:
    np.ndarray: Array of vertex indices
    """
    print(f"Processing data for cycle {coming_cycle}...")
    start_time = time.time()
    os.system(f'mkdir {coming_cycle}')
    print(f"Created directory: {coming_cycle}")
    
    print("Loading ensemble and performing PCA...")
    ensemble = md.load_pdb(f'{system_name}.pdb')
    topology = ensemble.topology
    ref = md.load_pdb(f'{system_name}.pdb', atom_indices=topology.select('name CA'))
    
    if coming_cycle == 1:
        dcd_bb = md.load('./eq.dcd', top=f'{system_name}.pdb',
                         atom_indices=topology.select('name CA'))
        dcd = md.load('./eq.dcd', top=f'{system_name}.pdb')
    else:
        filenames = ['./eq.dcd'] + [f'./{index_cycle+1}/{index_seed+1}.dcd'  
                     for index_cycle in range(len(lis))
                     for index_seed in range(lis[index_cycle])]
        dcd_bb = md.load(filenames, top=f'{system_name}.pdb', atom_indices=topology.select('name CA'))
        dcd = md.load(filenames, top=f'{system_name}.pdb')
    
    print("Superposing structures...")
    dcd_bb = dcd_bb.superpose(reference=ref)
    
    coord = np.array([frame.xyz for frame in dcd_bb])
    
    # Calculate pairwise distances
    n_atoms = coord.shape[1]
    n_frames = coord.shape[0]
    pairwise_distances = np.zeros((n_frames, n_atoms * (n_atoms - 1) // 2))
    
    for i in range(n_frames):
        pairwise_distances[i] = md.geometry.compute_distances(dcd_bb[i], np.array([(j, k) for j in range(n_atoms) for k in range(j+1, n_atoms)]))
    
    print("Performing PCA on pairwise distances...")
    pca = PCA(n_components=3)
    proj = pca.fit_transform(pairwise_distances)
    print("PCA variance ratios:", pca.explained_variance_ratio_)
    
    np.savetxt(f'./{coming_cycle}/proj.txt', proj, fmt="%.2f")
    print(f"Projections saved to ./{coming_cycle}/proj.txt")
    
    print("Performing Gaussian Mixture Model analysis...")
    probability_cutoff = 0.1
    BIC_cutoff = 0.3    

    n_components = np.arange(1, 21)         
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(proj) for n in n_components]
    bic = [m.bic(proj) for m in models]
    print("BIC values:", bic)
    
    slope = (bic - min(bic)) / (max(bic) - min(bic)) < BIC_cutoff
    model_index = np.where(slope == True)[0][0]
    components = model_index + 1
    print(f"Selected number of components: {components}")

    gmm2 = models[model_index]
    prob = gmm2.fit(proj).predict_proba(proj).round(3)
    
    print("Identifying hull vertices...")
    index = []
    hull_index = []
    index_not_hull = []
    for i in range(components):
        index.append(np.argwhere((prob[:, i] > probability_cutoff) == True)[:, 0])
        hull = ConvexHull(proj[index[i]])
        hull_index_Xmoon = index[i][hull.vertices]
        hull_index.append(hull_index_Xmoon)
        index_not_hull.append(set(index[i]).difference(set(hull_index[i])))
    
    vertix_index = []
    for i in range(components):
        hull = ConvexHull(proj[index[i]])
        hull_index_res = index[i][hull.vertices]
        for j in hull_index_res:
            for k in range(components):
                mark = True
                if i == k:
                    continue
                else:
                    if j in index_not_hull[k]:
                        mark = False
                        break
            if mark:
                vertix_index.append(j)
    
    vertix_index = np.unique(vertix_index)
    np.savetxt(f'./{coming_cycle}/vertix_index.txt', vertix_index, fmt='%d')
    print(f"Vertex indices saved to ./{coming_cycle}/vertix_index.txt")
    
    dcd_save = dcd[vertix_index]
    dcd_save.save_amberrst7(f'./{coming_cycle}/rst')
    print(f"Restart files saved to ./{coming_cycle}/rst")
    
    # Plot PCA results
    plt.figure(figsize=(10, 8))
    plt.scatter(proj[:, 0], proj[:, 1], c='blue', alpha=0.5)
    plt.scatter(proj[vertix_index, 0], proj[vertix_index, 1], c='red', s=100)
    plt.title(f'PCA projection - Cycle {coming_cycle}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(f'./{coming_cycle}/pca_plot.png')
    plt.close()
    print(f"PCA plot saved to ./{coming_cycle}/pca_plot.png")
    

# Main execution
print("Starting main simulation process...")
overall_start_time = time.time()

# Set paths for input files
top_path: str = "/home/alexi/Documents/Frontier-Expansion-Sampling/BPTI_60_1_af_sample_127_10000_protonated_max_plddt_1050/BPTI_60_1_af_sample_127_10000_protonated_max_plddt_1050.top"
gro_path: str = "/home/alexi/Documents/Frontier-Expansion-Sampling/BPTI_60_1_af_sample_127_10000_protonated_max_plddt_1050/BPTI_60_1_af_sample_127_10000_protonated_max_plddt_1050_solv_ions.gro"
os.environ["GMXLIB"] = os.getcwd()

# Convert Gromacs files to Amber format
amber_prmtop, initial_inpcrd = convert_gromacs_to_amber(top_path, gro_path)

top_path = os.path.splitext(top_path)[0]
# Read prmtop and inpcrd file of system
prmtop = omma.AmberPrmtopFile(f'{top_path}.prmtop')
initial_inpcrd = omma.AmberInpcrdFile(f'{top_path}.inpcrd')
print("Loaded prmtop and inpcrd files.")

# Build the initial system
simulation = build_system(prmtop)
print("Initial system built.")

# Run 10 ns equilibration simulation
print("Running 10 ns equilibration simulation...")
run_short_eq(simulation, initial_inpcrd, 5000000)
print("Equilibration complete.")

# Main simulation loop
for i in range(num_cycle):
    cycle_start_time = time.time()
    coming_cycle = i + 1
    print(f"\nStarting cycle {coming_cycle}...")
    
    # Process data and get new vertex indices
    vertix_index = get_proj_and_rst(coming_cycle, lis)
    lis.append(len(vertix_index))
    print(f"Number of seeds for cycle {coming_cycle}: {len(vertix_index)}")
    
    # Run seed simulations
    for j in range(len(vertix_index)):
        print(f"Running seed simulation {j+1} of {len(vertix_index)}...")
        simulation_seed = build_system(prmtop)
        seed_inpcrd = omma.AmberInpcrdFile(f'./{coming_cycle}/rst.' +
                                           ("{0:0"+str(len(str(len(vertix_index))))+"d}").format(j+1))
        run_seed_simulation(simulation_seed, seed_inpcrd, 50000, j+1, coming_cycle)
    
    cycle_end_time = time.time()
    print(f"Time taken for cycle {coming_cycle}: {cycle_end_time - cycle_start_time:.2f} seconds")

overall_end_time = time.time()
print("All cycles completed. Simulation finished.")
print(f"Total time taken: {overall_end_time - overall_start_time:.2f} seconds")