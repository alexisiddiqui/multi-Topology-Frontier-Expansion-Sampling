#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified version of the script to perform PCA on combined trajectories from all systems
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
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import glob
from dataclasses import dataclass, field
import scipy



@dataclass
class SystemConfig:
    system_name: str
    top_file: str
    gro_file: str
    amber_prmtop: Optional[str] = None
    amber_inpcrd: Optional[str] = None
    xtc_file: Optional[str] = None  # Added field for XTC file
    num_cycle: int = 30
    lis: List[int] = field(default_factory=list)
    pdb_path: Optional[str] = None
    ca_indices: Optional[np.ndarray] = None

class SharedPCASpace:
    def __init__(self):
        self.pca: Optional[PCA] = None
        self.combined_coords: Optional[np.ndarray] = None
        self.system_indices: Dict[str, np.ndarray] = {}
        self.reference_structure = None
        self.ca_indices = None

    def initialize_reference(self, reference_pdb: str) -> None:
        """
        Initialize the reference structure for alignment.
        
        Args:
        reference_pdb (str): Path to reference PDB file
        """
        self.reference_structure = md.load_pdb(reference_pdb)
        self.ca_indices = self.reference_structure.topology.select('name CA')

def find_input_systems(base_dir: str) -> List[SystemConfig]:
    """
    Find all topology directories within the experiment directory.
    Now also checks for XTC files.
    Returns a list of SystemConfig objects.
    """
    systems = []
    experiment_name = os.path.basename(base_dir)
    topology_dirs = glob.glob(os.path.join(base_dir, f"{experiment_name}_*_c*"))
    topology_dirs.sort()
    
    for topology_dir in topology_dirs:
        topology_name = os.path.basename(topology_dir)
        
        # Find the required files
        top_files = glob.glob(os.path.join(topology_dir, "*.top"))
        gro_files = glob.glob(os.path.join(topology_dir, "*_solv_ions.gro"))
        xtc_files = glob.glob(os.path.join(topology_dir, "*_concatenated.xtc"))
        
        if top_files and gro_files:
            system = SystemConfig(
                system_name=topology_name,
                top_file=top_files[0],
                gro_file=gro_files[0],
                xtc_file=xtc_files[0] if xtc_files else None
            )
            systems.append(system)
    
    print(f"Found {len(systems)} topology systems in {base_dir}:")
    for system in systems:
        print(f"  - {system.system_name}")
        print(f"    Top: {system.top_file}")
        print(f"    Gro: {system.gro_file}")
        print(f"    Xtc: {system.xtc_file if system.xtc_file else 'Not found - will run equilibration'}")
    
    return systems

def convert_xtc_to_dcd(system: SystemConfig, output_dir: str) -> None:
    """
    Convert XTC trajectory to DCD format using MDTraj.
    
    Args:
    system (SystemConfig): System configuration object
    output_dir (str): Directory to store output files
    """
    if not system.xtc_file:
        return
    
    print(f"Converting XTC to DCD for {system.system_name}...")
    start_time = time.time()
    
    try:
        # Create equilibration directory
        eq_dir = os.path.join(output_dir, system.system_name, 'equilibration')
        os.makedirs(eq_dir, exist_ok=True)
        
        # Load XTC with GRO topology
        traj = md.load(system.xtc_file, top=system.gro_file)
        
        # Save as DCD
        dcd_path = os.path.join(eq_dir, 'eq.dcd')
        traj.save_dcd(dcd_path)
        
        print(f"Successfully converted XTC to DCD: {dcd_path}")
        print(f"Trajectory contains {traj.n_frames} frames")
        
    except Exception as e:
        print(f"Error converting XTC to DCD: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        raise
    
    end_time = time.time()
    print(f"Conversion complete. Time taken: {end_time - start_time:.2f} seconds")

def setup_output_directories(base_output_dir: str, systems: List[SystemConfig]) -> None:
    """
    Create output directory structure for all systems.
    
    Args:
    base_output_dir (str): Base directory for output
    systems (List[SystemConfig]): List of systems to process
    """
    for system in systems:
        # Create system directory and equilibration subdirectory
        system_dir = os.path.join(base_output_dir, system.system_name)
        eq_dir = os.path.join(system_dir, 'equilibration')
        os.makedirs(eq_dir, exist_ok=True)
        print(f"Created directories for {system.system_name}")

def convert_gromacs_to_amber(system: SystemConfig, output_dir: str) -> None:
    """
    Convert Gromacs topology and coordinate files to Amber format using ParmEd.
    
    Args:
    system (SystemConfig): System configuration object
    output_dir (str): Directory to store output files
    """
    print(f"Converting Gromacs files to Amber format for {system.system_name}...")
    start_time = time.time()
    
    try:
        # Set GROMACS topology directory
        pmd.gromacs.GROMACS_TOPDIR = os.environ.get('GMXLIB', os.getcwd())
        
        # Load the Gromacs files
        print(f"Loading Gromacs topology: {system.top_file}")
        print(f"Loading coordinates: {system.gro_file}")
        gromacs = pmd.load_file(system.top_file, xyz=system.gro_file)
        
        # Create system directory
        system_dir = os.path.join(output_dir, system.system_name)
        os.makedirs(system_dir, exist_ok=True)
        
        # Define output files with absolute paths
        amber_prmtop = os.path.join(system_dir, f"{system.system_name}.prmtop")
        amber_rst7 = os.path.join(system_dir, f"{system.system_name}.rst7")  # Standard Amber restart
        pdb_file = os.path.join(system_dir, f"{system.system_name}.pdb")
        
        # Save files using ParmEd's built-in writers
        print("Saving Amber topology file...")
        gromacs.save(amber_prmtop, format='amber')
        
        print("Saving Amber restart file...")
        gromacs.save(amber_rst7, format='rst7')
        
        print("Saving PDB file...")
        gromacs.save(pdb_file, format='pdb')
        
        # Update system config with absolute paths
        system.amber_prmtop = os.path.abspath(amber_prmtop)
        system.amber_inpcrd = os.path.abspath(amber_rst7)
        system.pdb_path = os.path.abspath(pdb_file)
        
        print(f"Files saved successfully:")
        print(f"  Topology: {system.amber_prmtop}")
        print(f"  Restart:  {system.amber_inpcrd}")
        print(f"  PDB:      {system.pdb_path}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        print(f"Detailed error: {type(e).__name__}: {str(e)}")
        print("Using existing topology if available")
        
        if hasattr(e, '__traceback__'):
            import traceback
            print("Full traceback:")
            traceback.print_tb(e.__traceback__)

    end_time = time.time()
    print(f"Conversion complete. Time taken: {end_time - start_time:.2f} seconds")


def process_system_trajectories(system: SystemConfig, base_output_dir: str) -> None:
    """
    Process system trajectories - either convert XTC or run equilibration.
    
    Args:
    system (SystemConfig): System configuration object
    base_output_dir (str): Base output directory
    """
    print(f"\nProcessing trajectories for system: {system.system_name}")
    
    if system.xtc_file and os.path.exists(system.xtc_file):
        # Convert XTC to DCD
        convert_xtc_to_dcd(system, base_output_dir)
    else:
        # Run equilibration as before
        print("No XTC file found, running equilibration simulation...")
        prmtop = omma.AmberPrmtopFile(system.amber_prmtop)
        initial_inpcrd = omma.AmberInpcrdFile(system.amber_inpcrd)
        simulation = build_system(prmtop)
        run_short_eq(simulation, initial_inpcrd, 5000, system)



def setup_system_directory(system: SystemConfig) -> None:
    """
    Create necessary directories for a system.
    
    Args:
    system (SystemConfig): System configuration object
    """
    os.makedirs(system.system_name, exist_ok=True)
    os.chdir(system.system_name)

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
    print("Langevin integrator created: 300K, 1/ps collision frequency, 2fs timestep.")
    
    platform = omm.Platform.getPlatformByName('CUDA')
    print("CUDA platform selected for GPU acceleration.")
    
    simulation = omma.Simulation(prmtop.topology, system, integrator, platform)
    print("Simulation object created.")
    
    end_time = time.time()
    print(f"Time taken to build system: {end_time - start_time:.2f} seconds")
    return simulation

def run_seed_simulation(simulation: omma.Simulation, 
                       inpcrd: omma.AmberInpcrdFile, 
                       steps: int, 
                       seed_index: int, 
                       cycle: int,
                       system_name: str,
                       output_dir: str) -> None:
    """
    Run a seed simulation with correct path handling.
    
    Args:
    simulation (omma.Simulation): OpenMM simulation object
    inpcrd (omma.AmberInpcrdFile): Amber coordinate file
    steps (int): Number of simulation steps
    seed_index (int): Index of the current seed
    cycle (int): Current cycle number
    system_name (str): Name of the system being simulated
    output_dir (str): Base output directory
    """
    print(f"Running seed simulation {seed_index} for cycle {cycle}...")
    start_time = time.time()
    
    try:
        simulation.context.setPositions(inpcrd.positions)
    except Exception as e:
        print(f"Error setting positions: {e}")
        print(f"Inpcrd file positions shape: {inpcrd.positions.shape}")
        raise
        
    simulation.context.setVelocitiesToTemperature(300)
    
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    print("Positions, velocities, and box vectors set.")

    # Create cycle directory in system-specific location
    cycle_dir = os.path.abspath(os.path.join(output_dir, system_name, str(cycle)))
    os.makedirs(cycle_dir, exist_ok=True)

    dcd_file = os.path.join(cycle_dir, f"{seed_index}.dcd")
    log_file = os.path.join(cycle_dir, f"{seed_index}.txt")
    
    simulation.reporters.append(omma.DCDReporter(dcd_file, 5000))
    simulation.reporters.append(omma.StateDataReporter(log_file, 5000, 
                                                     step=True, 
                                                     potentialEnergy=True, 
                                                     temperature=True,
                                                     speed=True))
    print(f"Reporters added: {dcd_file} and {log_file}")

    print(f"Starting {steps} step seed simulation...")
    simulation.step(steps)
    print("Seed simulation complete.")
    end_time = time.time()
    print(f"Time taken for seed simulation: {end_time - start_time:.2f} seconds")

def run_short_eq(simulation: omma.Simulation, inpcrd: omma.AmberInpcrdFile, steps: int, system: SystemConfig) -> None:
    """
    Run a short equilibration simulation and save in system-specific directory.
    
    Args:
    simulation (omma.Simulation): OpenMM simulation object
    inpcrd (omma.AmberInpcrdFile): Amber coordinate file
    steps (int): Number of simulation steps
    system (SystemConfig): System configuration object
    """
    print(f"Running short equilibration for {system.system_name} for {steps} steps...")
    start_time = time.time()
    
    # Create equilibration directory in FES_output directory
    eq_dir = os.path.join('BPTI_FES_output', system.system_name, 'equilibration')
    os.makedirs(eq_dir, exist_ok=True)
    
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    print("Initial positions and box vectors set.")

    print("Minimizing energy...")
    simulation.minimizeEnergy()
    print("Energy minimization complete.")
    print("Starting 5000 step equilibration...")
    print("Equilibration step 1 - not saving")
    simulation.step(steps//2)

    eq_dcd = os.path.join(eq_dir, 'eq.dcd')
    eq_log = os.path.join(eq_dir, 'eq_log.txt')
    
    simulation.reporters.append(omma.DCDReporter(eq_dcd, 100))
    simulation.reporters.append(omma.StateDataReporter(eq_log, 100, 
                                                     step=True, 
                                                     potentialEnergy=True, 
                                                     temperature=True,
                                                     speed=True))
    print("DCD and state data reporters added.")

    print(f"Starting {steps} step pt II equilibration...")
    simulation.step(steps//2)
    print("Equilibration complete.")
    end_time = time.time()
    print(f"Time taken for equilibration: {end_time - start_time:.2f} seconds")
    
    
def combine_trajectories(systems: List[SystemConfig], cycle: int, source_dir: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Combine trajectories from all systems for the current cycle with improved path handling.
    
    Args:
    systems (List[SystemConfig]): List of all systems
    cycle (int): Current cycle number
    source_dir (str): Base output directory for all trajectory files
    
    Returns:
    Tuple[np.ndarray, Dict[str, np.ndarray]]: Combined coordinates and indices mapping
    """
    print("Combining trajectories from all systems...")
    all_coords = []
    system_indices = {}
    start_idx = 0
    
    for system in systems:
        print(f"\nProcessing trajectories for {system.system_name}")
        
        # Ensure we have absolute paths
        if not os.path.isabs(system.pdb_path):
            system.pdb_path = os.path.abspath(system.pdb_path)
        if not os.path.exists(system.pdb_path):
            raise OSError(f"PDB file not found: {system.pdb_path}")
            
        print(f"Using PDB file: {system.pdb_path}")
        print(f"CA indices: {system.ca_indices}")
        
        # Create list of trajectory files using absolute paths
        dcd_files = []
        
        # Always add equilibration trajectory from output directory
        eq_dcd = os.path.abspath(os.path.join(source_dir, system.system_name, 'equilibration', 'eq.dcd'))
        if os.path.exists(eq_dcd):
            dcd_files.append(eq_dcd)
            print(f"Found equilibration trajectory: {eq_dcd}")
        else:
            print(f"Warning: Equilibration DCD not found: {eq_dcd}")
            continue
        
        # For cycles beyond first, add previous cycle trajectories from output directory
        if cycle > 1:
            for prev_cycle in range(1, cycle):
                if prev_cycle <= len(system.lis) and system.lis[prev_cycle-1] > 0:
                    print(f"Adding {system.lis[prev_cycle-1]} trajectories from cycle {prev_cycle}")
                    for seed in range(1, system.lis[prev_cycle-1] + 1):
                        cycle_dcd = os.path.abspath(os.path.join(
                            source_dir,
                            system.system_name,
                            str(prev_cycle),
                            f"{seed}.dcd"
                        ))
                        if os.path.exists(cycle_dcd):
                            dcd_files.append(cycle_dcd)
                            print(f"Found trajectory: {cycle_dcd}")
                        else:
                            print(f"Warning: DCD file not found: {cycle_dcd}")
        
        if not dcd_files:
            print(f"No valid trajectory files found for {system.system_name}")
            continue
            
        print(f"Loading {len(dcd_files)} trajectories...")
        try:
            # Load trajectories sequentially to save memory
            traj = md.load(dcd_files[0], top=system.pdb_path, atom_indices=system.ca_indices)
            print(f"Loaded first trajectory with shape: {traj.xyz.shape}")
            
            for dcd in dcd_files[1:]:
                try:
                    additional_traj = md.load(dcd, top=system.pdb_path, atom_indices=system.ca_indices)
                    traj = traj.join(additional_traj)
                    print(f"Successfully joined trajectory: {dcd}")
                except Exception as e:
                    print(f"Error loading trajectory {dcd}: {str(e)}")
                    continue
                    
            print(f"Loaded combined trajectory with shape: {traj.xyz.shape}")
            
            # Superpose to reference
            print("Superposing trajectory...")
            traj.superpose(reference=traj[0])
            
            # Store coordinates and index mapping
            coords = traj.xyz
            print(f"Adding coordinates with shape: {coords.shape}")
            all_coords.append(coords)
            end_idx = start_idx + coords.shape[0]
            system_indices[system.system_name] = np.arange(start_idx, end_idx)
            start_idx = end_idx
            
        except Exception as e:
            print(f"Error processing trajectories for {system.system_name}: {str(e)}")
            continue
    
    if not all_coords:
        raise RuntimeError("No valid coordinates found for any system")
        
    combined_coords = np.concatenate(all_coords, axis=0)
    print(f"Final combined coordinates shape: {combined_coords.shape}")
    return combined_coords, system_indices




def perform_shared_pca(coords: np.ndarray) -> Tuple[PCA, np.ndarray]:
    """
    Perform PCA on combined coordinates.
    
    Args:
    coords (np.ndarray): Combined coordinate array
    
    Returns:
    Tuple[PCA, np.ndarray]: PCA object and projections
    """
    print("Performing PCA on combined coordinates...")
    
    # Calculate pairwise distances
    n_atoms = coords.shape[1]
    n_frames = coords.shape[0]
    
    # Create atom pairs array properly shaped for mdtraj
    pairs = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pairs.append([i, j])
    atom_pairs = np.array(pairs)
    
    print(f"Computing distances for {len(pairs)} atom pairs over {n_frames} frames...")
    pairwise_distances = np.zeros((n_frames, len(pairs)))
    
    for i in range(n_frames):
        traj_frame = md.Trajectory(xyz=coords[i].reshape(1, n_atoms, 3), 
                                 topology=None)
        pairwise_distances[i] = md.geometry.compute_distances(traj_frame, atom_pairs)[0]
    
    # Perform PCA
    print("Fitting PCA...")
    pca = PCA(n_components=3)
    proj = pca.fit_transform(pairwise_distances)
    print("PCA variance ratios:", pca.explained_variance_ratio_)
    
    return pca, proj
def analyze_projections(proj: np.ndarray, system_indices: Dict[str, np.ndarray], cycle: int) -> Dict[str, np.ndarray]:
    """
    Analyze PCA projections and identify vertices across all systems using a global convex hull.
    Added robust handling of nearly coplanar points.
    """
    print("Analyzing projections across all systems...")
    
    # GMM parameters
    probability_cutoff = 0.1
    BIC_cutoff = 0.3
    
    # Fit GMM to all data
    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(proj) 
              for n in n_components]
    bic = [m.bic(proj) for m in models]
    
    slope = (bic - min(bic)) / (max(bic) - min(bic)) < BIC_cutoff
    model_index = np.where(slope == True)[0][0]
    components = model_index + 1
    print(f"Selected {components} components for GMM")
    
    gmm = models[model_index]
    prob = gmm.fit(proj).predict_proba(proj).round(3)
    
    # Find vertices globally for each component
    all_vertex_indices = set()  # Use set to avoid duplicates
    
    def compute_hull_with_fallback(points: np.ndarray) -> np.ndarray:
        """Try different ConvexHull options if the initial attempt fails."""
        try:
            # First try with default options
            hull = ConvexHull(points)
            return hull.vertices
        except scipy.spatial.QhullError:
            try:
                # Try with QJ option (jitter input points)
                hull = ConvexHull(points, qhull_options='QJ')
                return hull.vertices
            except scipy.spatial.QhullError:
                try:
                    # Try with scaled points
                    hull = ConvexHull(points, qhull_options='QbB')
                    return hull.vertices
                except scipy.spatial.QhullError:
                    # If all else fails, use QJ and QbB together
                    try:
                        hull = ConvexHull(points, qhull_options='QJ QbB')
                        return hull.vertices
                    except scipy.spatial.QhullError as e:
                        print(f"Failed to compute convex hull even with all options: {str(e)}")
                        # Fall back to selecting points with maximum/minimum coordinates
                        extrema = []
                        for dim in range(points.shape[1]):
                            extrema.extend([np.argmin(points[:, dim]), np.argmax(points[:, dim])])
                        return np.unique(extrema)

    for i in range(components):
        # Get points belonging to this component across all systems
        component_indices = np.argwhere((prob[:, i] > probability_cutoff) == True)[:, 0]
        if len(component_indices) > 0:
            print(f"\nProcessing component {i} with {len(component_indices)} points")
            
            # Get the actual points for this component
            component_points = proj[component_indices]
            
            # Check if points are too close together
            point_spread = np.ptp(component_points, axis=0)
            if np.any(point_spread < 1e-10):
                print(f"Warning: Component {i} has very small spread in some dimensions: {point_spread}")
                # Add points with extreme values for each dimension
                for dim in range(component_points.shape[1]):
                    idx_min = component_indices[np.argmin(component_points[:, dim])]
                    idx_max = component_indices[np.argmax(component_points[:, dim])]
                    all_vertex_indices.update([idx_min, idx_max])
            else:
                # Compute convex hull with fallback options
                try:
                    hull_vertices = compute_hull_with_fallback(component_points)
                    all_vertex_indices.update(component_indices[hull_vertices])
                except Exception as e:
                    print(f"Error computing hull for component {i}: {str(e)}")
                    # Fallback: add points with extreme values
                    for dim in range(component_points.shape[1]):
                        idx_min = component_indices[np.argmin(component_points[:, dim])]
                        idx_max = component_indices[np.argmax(component_points[:, dim])]
                        all_vertex_indices.update([idx_min, idx_max])
    
    print(f"Found {len(all_vertex_indices)} total vertices across all systems")
    
    # Convert to sorted numpy array for consistency
    all_vertex_indices = np.array(sorted(list(all_vertex_indices)))
    
    # Distribute vertices to their respective systems
    system_vertices = {}
    for system_name, indices in system_indices.items():
        # Find vertices that belong to this system
        system_vertex_indices = all_vertex_indices[np.isin(all_vertex_indices, indices)]
        system_vertices[system_name] = system_vertex_indices
        print(f"System {system_name}: {len(system_vertex_indices)} vertices")
    
    # Additional diagnostics
    print("\nVertex distribution by system:")
    for system_name, vertices in system_vertices.items():
        if len(vertices) > 0:
            print(f"{system_name}:")
            print(f"  Number of vertices: {len(vertices)}")
            print(f"  Global indices: {vertices}")
            
            # Calculate which GMM components these vertices belong to
            vertex_components = gmm.predict(proj[vertices])
            component_counts = np.bincount(vertex_components)
            print("  Component distribution:")
            for comp, count in enumerate(component_counts):
                if count > 0:
                    print(f"    Component {comp}: {count} vertices")
    
    return system_vertices

def visualize_pca_results(proj: np.ndarray, 
                         system_indices: Dict[str, np.ndarray], 
                         system_vertices: Dict[str, np.ndarray],
                         cycle: int,
                         output_dir: str) -> None:
    """
    Create visualization of PCA results showing projections and selected vertices,
    focusing on PC1 vs PC2.
    
    Args:
    proj (np.ndarray): PCA projections
    system_indices (Dict[str, np.ndarray]): Mapping of system names to their frame indices
    system_vertices (Dict[str, np.ndarray]): Mapping of system names to their vertex indices
    cycle (int): Current cycle number
    output_dir (str): Directory to save plots
    """
    # Create figure for PC1 vs PC2
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Set background color and grid
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Color palette for different systems
    colors = plt.cm.tab10(np.linspace(0, 1, len(system_indices)))
    
    # Plot each system's data
    for (system_name, indices), color in zip(system_indices.items(), colors):
        # Plot all points for this system
        ax.scatter(proj[indices, 0], 
                  proj[indices, 1],
                  color=color,
                  alpha=0.3,
                  label=system_name,
                  s=20)
        
        # Highlight vertices if any exist
        if system_name in system_vertices:
            vertices = system_vertices[system_name]
            if len(vertices) > 0:
                ax.scatter(proj[vertices, 0],
                         proj[vertices, 1],
                         color=color,
                         s=100,
                         marker='*',
                         edgecolor='black',
                         linewidth=1,
                         label=f'{system_name} vertices')
    
    # Customize appearance
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'PCA Projections - Cycle {cycle}', fontsize=14, pad=20)
    
    # Add legend with transparent background
    legend = ax.legend(bbox_to_anchor=(1.05, 1), 
                      loc='upper left', 
                      borderaxespad=0.,
                      framealpha=0.8)
    legend.get_frame().set_facecolor('white')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'pca_cycle_{cycle}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA visualization to: {plot_path}")

def analyze_vertex_distribution(system_vertices: Dict[str, np.ndarray],
                              proj: np.ndarray,
                              cycle: int,
                              output_dir: str) -> None:
    """
    Create analysis plots for vertex distribution in PCA space.
    
    Args:
    system_vertices (Dict[str, np.ndarray]): Mapping of system names to their vertex indices
    proj (np.ndarray): PCA projections
    cycle (int): Current cycle number
    output_dir (str): Directory to save plots
    """
    # Calculate vertex statistics for each system
    systems = []
    vertices_count = []
    
    for system_name, vertices in system_vertices.items():
        if len(vertices) > 0:
            systems.append(system_name)
            vertices_count.append(len(vertices))
    
    if not systems:  # If no vertices found
        print("No vertices found for analysis")
        return
        
    # Create bar plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Set background color
    ax.set_facecolor('white')
    
    # Create bars with custom style
    bars = ax.bar(systems, vertices_count, 
                  color='lightblue',
                  edgecolor='navy',
                  alpha=0.7)
    
    # Customize appearance
    ax.set_xlabel('Systems', fontsize=12)
    ax.set_ylabel('Number of vertices', fontsize=12)
    ax.set_title(f'Number of Vertices per System - Cycle {cycle}', 
                 fontsize=14, pad=20)
    
    # Rotate x-labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'vertex_count_cycle_{cycle}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved vertex count analysis to: {plot_path}")
    
@dataclass
class TrajectoryInfo:
    """Track frame information for each system's trajectories"""
    system_name: str
    frame_counts: Dict[str, int] = field(default_factory=dict)  # maps trajectory path to frame count
    cumulative_offset: int = 0  # offset in combined trajectory
    total_frames: int = 0  # total frames for this system

def track_trajectory_frames(systems: List[SystemConfig], cycle: int, source_dir: str) -> Dict[str, TrajectoryInfo]:
    """
    Build a mapping of frame counts and offsets for all trajectories.
    
    Args:
        systems: List of system configurations
        cycle: Current cycle number
        source_dir: Base output directory
    
    Returns:
        Dict mapping system names to their trajectory information
    """
    trajectory_info = {}
    cumulative_offset = 0
    
    for system in systems:
        info = TrajectoryInfo(system_name=system.system_name)
        
        # Load and track frames for each trajectory
        dcd_files = []
        
        # Add equilibration trajectory
        eq_dcd = os.path.abspath(os.path.join(source_dir, system.system_name, 'equilibration', 'eq.dcd'))
        if os.path.exists(eq_dcd):
            dcd_files.append(('eq', eq_dcd))
            
        # Add trajectories from previous cycles
        if cycle > 1:
            for prev_cycle in range(1, cycle):
                if prev_cycle-1 >= len(system.lis):
                    continue
                n_seeds = system.lis[prev_cycle-1]
                for seed in range(1, n_seeds + 1):
                    dcd_path = os.path.abspath(os.path.join(
                        source_dir, 
                        system.system_name,
                        str(prev_cycle),
                        f"{seed}.dcd"
                    ))
                    if os.path.exists(dcd_path):
                        dcd_files.append((f"c{prev_cycle}s{seed}", dcd_path))
        
        # Load each trajectory and count frames
        total_frames = 0
        for traj_id, dcd_path in dcd_files:
            try:
                traj = md.load(dcd_path, top=system.pdb_path)
                n_frames = traj.n_frames
                info.frame_counts[dcd_path] = n_frames
                total_frames += n_frames
                print(f"{system.system_name} - {traj_id}: {n_frames} frames from {dcd_path}")
            except Exception as e:
                print(f"Error loading {dcd_path}: {str(e)}")
                continue
        
        info.total_frames = total_frames
        info.cumulative_offset = cumulative_offset
        cumulative_offset += total_frames
        
        trajectory_info[system.system_name] = info
        print(f"\n{system.system_name} summary:")
        print(f"Total frames: {total_frames}")
        print(f"Cumulative offset: {cumulative_offset}")
        print("Individual trajectories:")
        for path, frames in info.frame_counts.items():
            print(f"  {os.path.basename(path)}: {frames} frames")
        print()
        
    return trajectory_info

def save_system_vertices(system: SystemConfig, vertices: np.ndarray, cycle: int, 
                        all_systems: List[SystemConfig], output_dir: str,
                        traj_info: Dict[str, TrajectoryInfo]) -> None:
    """
    Save vertex information for a system using robust frame tracking.
    """
    cycle_dir = os.path.abspath(os.path.join(output_dir, system.system_name, str(cycle)))
    os.makedirs(cycle_dir, exist_ok=True)
    
    # Save original vertex indices for reference
    vertex_file = os.path.join(cycle_dir, 'vertex_index.txt')
    np.savetxt(vertex_file, vertices, fmt='%d')
    
    print(f"\nProcessing vertices for {system.system_name}")
    print(f"Global vertex indices: {vertices}")
    
    # Get system's trajectory information
    system_info = traj_info[system.system_name]
    system_offset = system_info.cumulative_offset
    
    # Convert global vertices to local indices
    local_vertices = vertices[vertices >= system_offset]
    local_vertices = local_vertices[local_vertices < (system_offset + system_info.total_frames)]
    local_vertices = local_vertices - system_offset
    
    print(f"System frame information:")
    print(f"Total frames: {system_info.total_frames}")
    print(f"System offset in combined trajectory: {system_offset}")
    print(f"Local vertex indices after adjustment: {local_vertices}")
    
    if len(local_vertices) == 0:
        print(f"Warning: No valid vertices found for system {system.system_name} in cycle {cycle}")
        return

    try:
        # Load trajectories
        dcd_files = []
        frame_map = {}  # Maps global frame index to (trajectory, local_frame) pairs
        current_offset = 0
        
        # Build frame mapping
        for dcd_path, n_frames in system_info.frame_counts.items():
            dcd_files.append(dcd_path)
            for i in range(n_frames):
                frame_map[current_offset + i] = (dcd_path, i)
            current_offset += n_frames
        
        print("\nFrame mapping for vertices:")
        for i, global_idx in enumerate(local_vertices):
            traj_path, local_idx = frame_map[global_idx]
            print(f"Vertex {i+1}: Global frame {global_idx} -> {os.path.basename(traj_path)} frame {local_idx}")
        
        # Load and process trajectories
        print("\nLoading trajectories...")
        traj = md.load(dcd_files[0], top=system.pdb_path)
        for dcd in dcd_files[1:]:
            traj = traj.join(md.load(dcd, top=system.pdb_path))
        
        print(f"Loaded combined trajectory with {traj.n_frames} frames")
        
        # Verify frame count matches our tracking
        if traj.n_frames != system_info.total_frames:
            print(f"Warning: Loaded frames ({traj.n_frames}) doesn't match tracked frames ({system_info.total_frames})")
        
        # Save restart files
        for i, vertex_idx in enumerate(local_vertices, 1):
            if vertex_idx >= traj.n_frames:
                print(f"Warning: Vertex index {vertex_idx} out of bounds for trajectory with {traj.n_frames} frames")
                continue
                
            vertex_traj = traj[vertex_idx:vertex_idx+1]
            rst_path = os.path.join(cycle_dir, f"rst.{i}")
            print(f"Saving restart file {i} from frame {vertex_idx} to: {rst_path}")
            vertex_traj.save_amberrst7(rst_path)
            
    except Exception as e:
        print(f"Error processing trajectory for {system.system_name}: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        raise


def process_cycles(systems: List[SystemConfig], num_cycles: int, source_dir: str) -> None:
    """Process all cycles with robust frame tracking."""
    vis_dir = os.path.join(source_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    for cycle in range(1, num_cycles + 1):
        print(f"\n{'='*80}")
        print(f"Starting cycle {cycle}")
        print(f"{'='*80}\n")
        
        # Initialize simulations
        simulations = {}
        for system in systems:
            try:
                if not os.path.exists(system.amber_prmtop):
                    print(f"Error: Topology file not found for {system.system_name}: {system.amber_prmtop}")
                    continue
                prmtop = omma.AmberPrmtopFile(system.amber_prmtop)
                simulation = build_system(prmtop)
                simulations[system.system_name] = simulation
                print(f"Initialized simulation for {system.system_name}")
            except Exception as e:
                print(f"Error initializing simulation for {system.system_name}: {str(e)}")
                continue
        
        try:
            # Track all trajectory frames
            print("\nTracking trajectory frames...")
            traj_info = track_trajectory_frames(systems, cycle, source_dir)
            
            # Combine trajectories and perform PCA
            coords, system_indices = combine_trajectories(systems, cycle, source_dir)
            print(f"Combined coordinates shape: {coords.shape}")
            
            # Verify combined coordinates match our frame tracking
            total_tracked_frames = sum(info.total_frames for info in traj_info.values())
            if coords.shape[0] != total_tracked_frames:
                print(f"Warning: Combined coordinates frames ({coords.shape[0]}) "
                      f"doesn't match tracked frames ({total_tracked_frames})")
            
            pca, proj = perform_shared_pca(coords)
            system_vertices = analyze_projections(proj, system_indices, cycle)
            
            # Create visualizations
            cycle_vis_dir = os.path.join(vis_dir, f'cycle_{cycle}')
            os.makedirs(cycle_vis_dir, exist_ok=True)
            visualize_pca_results(proj, system_indices, system_vertices, cycle, cycle_vis_dir)
            analyze_vertex_distribution(system_vertices, proj, cycle, cycle_vis_dir)
            
            # Process each system
            for system in systems:
                print(f"\n{'*'*40}")
                print(f"Processing {system.system_name} for cycle {cycle}")
                print(f"{'*'*40}")
                
                if system.system_name not in system_vertices:
                    print(f"No vertices found for {system.system_name}")
                    system.lis.append(0)
                    continue
                
                vertices = system_vertices[system.system_name]
                if len(vertices) == 0:
                    print(f"No vertices found for {system.system_name}")
                    system.lis.append(0)
                    continue
                
                # Save vertices using tracked frame information
                save_system_vertices(system, vertices, cycle, systems, source_dir, traj_info)
                system.lis.append(len(vertices))
                
                if system.system_name not in simulations:
                    print(f"Error: No simulation object for {system.system_name}")
                    continue
                
                # Run simulations for vertices
                simulation = simulations[system.system_name]
                for j, vertex_idx in enumerate(vertices, 1):
                    rst_file = os.path.join(source_dir, system.system_name, str(cycle), f"rst.{j}")
                    if not os.path.exists(rst_file):
                        print(f"Warning: Restart file not found: {rst_file}")
                        continue
                    
                    try:
                        print(f"Running simulation for {system.system_name} vertex {j}")
                        inpcrd = omma.AmberInpcrdFile(rst_file)
                        run_seed_simulation(
                            simulation=simulation,
                            inpcrd=inpcrd,
                            steps=5000,
                            seed_index=j,
                            cycle=cycle,
                            system_name=system.system_name,
                            output_dir=source_dir
                        )
                    except Exception as e:
                        print(f"Error running simulation for vertex {j}: {str(e)}")
                        continue
                
        except Exception as e:
            print(f"Error processing cycle {cycle}: {str(e)}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
            continue
            
        print(f"\nCycle {cycle} completed")
        
        # Clean up simulations
        for simulation in simulations.values():
            del simulation.context
        simulations.clear()

def main():
    """
    Updated main execution function with XTC handling.
    """
    print("Starting multi-topology shared-PCA analysis...")
    overall_start_time = time.time()
    
    # Get absolute paths for directories
    current_dir = os.getcwd()
    base_input_dir = os.path.join(current_dir, "RW_10/BPTI")
    base_output_dir = os.path.join(current_dir, "BPTI_FES_output")
    
    print(f"Input directory: {base_input_dir}")
    print(f"Output directory: {base_output_dir}")
    
    # Remove previous output if it exists
    if os.path.exists(base_output_dir):
        print(f"Removing previous output directory: {base_output_dir}")
        os.system(f"rm -rf {base_output_dir}")
    
    # Create output directory
    os.makedirs(base_output_dir)
    print(f"Created output directory: {base_output_dir}")
    
    # Set up environment
    os.environ["GMXLIB"] = os.getcwd()
    
    # Find all topology systems
    systems = find_input_systems(base_input_dir)
    if not systems:
        print("No valid topology systems found!")
        return
    
    # Create output directories
    setup_output_directories(base_output_dir, systems)
    
    # Process each system's files
    for system in systems:
        print(f"\nProcessing system: {system.system_name}")
        system_dir = os.path.join(base_output_dir, system.system_name)
        
        # Convert files to Amber format
        convert_gromacs_to_amber(system, base_output_dir)
        
        # Initialize CA indices for this topology
        if not os.path.exists(system.pdb_path):
            raise OSError(f"PDB file not found: {system.pdb_path}")
            
        topology = md.load(system.pdb_path).topology
        system.ca_indices = topology.select('name CA')
        
        # Process trajectories - either convert XTC or run equilibration
        process_system_trajectories(system, base_output_dir)
    
    # Process all cycles using shared PCA space
    os.chdir(base_output_dir)
    process_cycles(systems, num_cycles=3, source_dir=base_output_dir)
    
    overall_end_time = time.time()
    print("\nAll systems processed successfully.")
    print(f"Total time taken: {overall_end_time - overall_start_time:.2f} seconds")

    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()