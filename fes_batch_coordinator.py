#!/usr/bin/env python3
import os
import numpy as np
import subprocess
from pathlib import Path
import json
import time
from typing import List, Dict
from dataclasses import dataclass, asdict
import math

@dataclass
class SimulationBatch:
    batch_id: int
    system_name: str
    cycle: int
    prmtop_file: str
    num_seeds: int
    steps_per_seed: int
    output_dir: str

def create_simulation_batches(systems: List[SystemConfig], 
                            system_vertices: Dict[str, np.ndarray],
                            cycle: int,
                            seeds_per_batch: int = 50,
                            output_dir: str = None) -> List[SimulationBatch]:
    """
    Create batches of simulation tasks, with each batch handling 50 seeds
    """
    batches = []
    batch_id = 0
    
    for system in systems:
        if system.system_name not in system_vertices:
            continue
            
        vertices = system_vertices[system.system_name]
        if len(vertices) == 0:
            continue
            
        # Calculate how many batches we need for this system
        num_vertices = len(vertices)
        num_batches = math.ceil(num_vertices / seeds_per_batch)
        
        for i in range(num_batches):
            batch = SimulationBatch(
                batch_id=batch_id,
                system_name=system.system_name,
                cycle=cycle,
                prmtop_file=system.amber_prmtop,
                num_seeds=min(seeds_per_batch, num_vertices - i * seeds_per_batch),
                steps_per_seed=50000,
                output_dir=output_dir
            )
            batches.append(batch)
            batch_id += 1
            
    return batches

def write_batch_config(batch: SimulationBatch, output_dir: str) -> str:
    """Write batch configuration to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f'batch_{batch.batch_id}_config.json')
    
    with open(config_path, 'w') as f:
        json.dump(asdict(batch), f, indent=2)
    
    return config_path

def submit_batch_job(config_path: str, log_dir: str):
    """Submit a SLURM job for a batch of simulations"""
    job_script = f"""#!/bin/bash
#SBATCH --job-name=FES_batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
#SBATCH --partition=medium
#SBATCH --gres=gpu:v100:1
#SBATCH --clusters=htc
#SBATCH --output={log_dir}/slurm_%j.out
#SBATCH --error={log_dir}/slurm_%j.err
#SBATCH --constraint="[scratch:weka]"

module load CUDA/12.0.0
module load Anaconda3/2022.10
source activate FES

# Set environment variables
export OMP_NUM_THREADS=4
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# Start CUDA MPS
nvidia-cuda-mps-control -d

# Set working directory to SCRATCH
cd $SCRATCH

# Copy necessary files
echo "Setting up working directory..."
WORK_DIR="${{SCRATCH}}/FES_batch_${{SLURM_JOB_ID}}"
mkdir -p ${{WORK_DIR}}
cd ${{WORK_DIR}}

# Run the batch simulation script
python {os.path.abspath('run_simulation_batch.py')} {config_path}

# Cleanup MPS
echo quit | nvidia-cuda-mps-control

# Cleanup working directory
cd $SCRATCH
rm -rf ${{WORK_DIR}}
"""
    
    job_script_path = config_path.replace('.json', '.sh')
    with open(job_script_path, 'w') as f:
        f.write(job_script)
    
    subprocess.run(['sbatch', job_script_path])
    print(f"Submitted batch job: {job_script_path}")

def main():
    """Main execution function"""
    # Your existing PCA and analysis code here
    coords, system_indices = combine_trajectories(systems, cycle, source_dir)
    pca, proj = perform_shared_pca(coords)
    system_vertices = analyze_projections(proj, system_indices, cycle)
    
    # Create output directories
    log_dir = os.path.join(source_dir, 'logs')
    config_dir = os.path.join(source_dir, 'batch_configs')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    
    # Create and submit batch jobs
    batches = create_simulation_batches(
        systems=systems,
        system_vertices=system_vertices,
        cycle=cycle,
        output_dir=source_dir
    )
    
    print(f"Created {len(batches)} simulation batches")
    for batch in batches:
        config_path = write_batch_config(batch, config_dir)
        submit_batch_job(config_path, log_dir)
        time.sleep(2)  # Avoid overwhelming scheduler
        
    print("All batch jobs submitted successfully")

if __name__ == "__main__":
    main()