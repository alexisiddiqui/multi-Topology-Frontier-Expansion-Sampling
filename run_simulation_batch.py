#!/usr/bin/env python3
import json
import sys
import os
import time
from typing import List, Optional
import numpy as np
from dataclasses import dataclass
import mdtraj as md
import simtk.openmm as omm
import simtk.openmm.app as omma
import simtk.unit as unit
import multiprocessing
from contextlib import contextmanager

@dataclass
class SimulationBatch:
    batch_id: int
    system_name: str
    cycle: int
    prmtop_file: str
    num_seeds: int
    steps_per_seed: int
    output_dir: str

def initialize_cuda():
    """Initialize CUDA platform with MPS support"""
    try:
        platform = omm.Platform.getPlatformByName('CUDA')
        properties = {}
        return platform, properties
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
        return None, None

def build_system_with_platform(prmtop: omma.AmberPrmtopFile, 
                             platform: omm.Platform,
                             properties: dict) -> omma.Simulation:
    """Build OpenMM simulation system"""
    system = prmtop.createSystem(
        nonbondedMethod=omma.PME,
        nonbondedCutoff=0.8*unit.nanometer,
        constraints=omma.HBonds
    )
    
    barostat = omm.MonteCarloBarostat(
        1.0*unit.bar,
        300*unit.kelvin,
        25
    )
    system.addForce(barostat)
    
    integrator = omm.LangevinMiddleIntegrator(
        300*unit.kelvin,
        1.0/unit.picosecond,
        0.002*unit.picoseconds
    )
    
    simulation = omma.Simulation(
        prmtop.topology,
        system,
        integrator,
        platform,
        properties
    )
    
    return simulation

def run_single_seed(args):
    """Run a single seed simulation"""
    seed_idx, batch = args
    
    try:
        # Initialize CUDA
        platform, properties = initialize_cuda()
        if platform is None:
            raise RuntimeError("Failed to initialize CUDA platform")
            
        # Load topology
        prmtop = omma.AmberPrmtopFile(batch.prmtop_file)
        simulation = build_system_with_platform(prmtop, platform, properties)
        
        # Load restart file
        rst_path = os.path.join(batch.output_dir, 
                               batch.system_name,
                               str(batch.cycle),
                               f"rst.{seed_idx}")
        inpcrd = omma.AmberInpcrdFile(rst_path)
        
        # Setup simulation
        simulation.context.setPositions(inpcrd.positions)
        simulation.context.setVelocitiesToTemperature(300)
        if inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
            
        # Create output directory
        output_dir = os.path.join(batch.output_dir,
                                 batch.system_name,
                                 str(batch.cycle))
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup reporters
        dcd_file = os.path.join(output_dir, f"{seed_idx}.dcd")
        log_file = os.path.join(output_dir, f"{seed_idx}.txt")
        
        simulation.reporters.append(omma.DCDReporter(dcd_file, 5000))
        simulation.reporters.append(omma.StateDataReporter(
            log_file, 5000,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True
        ))
        
        # Run initial steps at shorter timestep
        simulation.integrator.setStepSize(0.001*unit.picoseconds)
        simulation.step(2000)
        simulation.integrator.setStepSize(0.002*unit.picoseconds)
        
        # Run production
        simulation.step(batch.steps_per_seed)
        
        # Cleanup
        del simulation.context
        del simulation
        
        return True
        
    except Exception as e:
        print(f"Error in seed {seed_idx}: {str(e)}")
        return False

def run_batch(batch: SimulationBatch):
    """Run a batch of seed simulations using MPS"""
    print(f"Starting batch {batch.batch_id} with {batch.num_seeds} seeds")
    
    # Create pool with 4 processes for MPS
    with multiprocessing.Pool(processes=4) as pool:
        seed_range = range(1, batch.num_seeds + 1)
        args = [(seed_idx, batch) for seed_idx in seed_range]
        results = pool.map(run_single_seed, args)
    
    # Check results
    successful = sum(1 for r in results if r)
    print(f"Completed {successful}/{batch.num_seeds} seeds successfully")
    return successful == batch.num_seeds

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_simulation_batch.py <config_file>")
        sys.exit(1)
        
    # Load batch configuration
    with open(sys.argv[1], 'r') as f:
        config = json.load(f)
    batch = SimulationBatch(**config)
    
    # Run the batch
    success = run_batch(batch)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()