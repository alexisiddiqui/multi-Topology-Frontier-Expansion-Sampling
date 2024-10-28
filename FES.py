#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Juanrong Zhang
Modified for verbosity
"""

import numpy as np
import mdtraj as md
import simtk.openmm.app as omma
import simtk.openmm as omm
import simtk.unit as unit
import os
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull

print("Imported all necessary libraries.")

# Set the name of system
system_name = "open_amber"
print(f"System name set to: {system_name}")

# Set the number of cycles
num_cycle = 30
print(f"Number of cycles set to: {num_cycle}")

# The list that save the number of seeds in each cycle
lis = []
print("Initialized empty list to store number of seeds per cycle.")

def build_system(prmtop):
    print("Building system...")
    system = prmtop.createSystem(nonbondedMethod = omma.PME, 
                                 nonbondedCutoff = 1*unit.nanometer, 
                                 constraints = omma.HBonds)
    print("System created with PME, 1nm cutoff, and HBonds constraints.")
    
    integrator = omm.LangevinIntegrator(310*unit.kelvin, 1/unit.picoseconds, 
                                        0.002*unit.picoseconds)
    print("Langevin integrator created: 310K, 1/ps collision frequency, 2fs timestep.")
    
    platform = omm.Platform.getPlatformByName('CUDA')
    print("CUDA platform selected for GPU acceleration.")
    
    simulation = omma.Simulation(prmtop.topology, system, integrator, platform)
    print("Simulation object created.")
    
    return simulation

def run_short_eq(simulation, inpcrd, steps):
    print(f"Running short equilibration for {steps} steps...")
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

def run_seed_simulation(simulation, inpcrd, steps, seed_index, coming_cycle):
    print(f"Running seed simulation {seed_index} for cycle {coming_cycle}...")
    simulation.context.setPositions(inpcrd.positions)
    simulation.context.setVelocitiesToTemperature(310)
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

def get_proj_and_rst(coming_cycle, lis):
    print(f"Processing data for cycle {coming_cycle}...")
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
    res = coord.reshape(coord.shape[0], -1)
    
    print("Performing PCA...")
    mean = np.mean(res, axis=0)
    res_new = res - mean
    pca = PCA(n_components=3)
    pca.fit(res_new)
    print("PCA variance ratios:", pca.explained_variance_ratio_)
    
    proj = np.dot(res_new, pca.components_.T)
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
    
    return vertix_index

print("Starting main simulation process...")

# Read prmtop and inpcrd file of system
prmtop = omma.AmberPrmtopFile(f'{system_name}.prmtop')
initial_inpcrd = omma.AmberInpcrdFile(f'{system_name}.inpcrd')
print("Loaded prmtop and inpcrd files.")

simulation = build_system(prmtop)
print("Initial system built.")

# Run 10 ns eq simulation
print("Running 10 ns equilibration simulation...")
run_short_eq(simulation, initial_inpcrd, 5000000)
print("Equilibration complete.")

for i in range(num_cycle):
    coming_cycle = i + 1
    print(f"\nStarting cycle {coming_cycle}...")
    
    vertix_index = get_proj_and_rst(coming_cycle, lis)
    lis.append(len(vertix_index))
    print(f"Number of seeds for cycle {coming_cycle}: {len(vertix_index)}")
    
    for j in range(len(vertix_index)):
        print(f"Running seed simulation {j+1} of {len(vertix_index)}...")
        simulation_seed = build_system(prmtop)
        seed_inpcrd = omma.AmberInpcrdFile(f'./{coming_cycle}/rst.' +
                                           ("{0:0"+str(len(str(len(vertix_index))))+"d}").format(j+1))
        run_seed_simulation(simulation_seed, seed_inpcrd, 50000, j+1, coming_cycle)

print("All cycles completed. Simulation finished.")