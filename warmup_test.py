import simtk.openmm as omm
import simtk.openmm.app as omma
import simtk.unit as unit
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os
import parmed as pmd
from typing import List, Dict, Tuple

class WarmupTest:
    def __init__(self, base_dir: str = "RW_10/BPTI", output_dir: str = "warmup_test_results", n_replicates: int = 5):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.timesteps = [0.5, 1.0, 1.5, 2.0]  # in fs
        self.simulation_time = 5  # ps
        self.n_replicates = n_replicates
        os.makedirs(output_dir, exist_ok=True)
        # plt.style.use('seaborn')

    def create_simulation(self, prmtop_path: str, timestep: float) -> omma.Simulation:
        """Create a new simulation instance"""
        prmtop = omma.AmberPrmtopFile(prmtop_path)
        system = prmtop.createSystem(
            nonbondedMethod=omma.PME,
            nonbondedCutoff=1*unit.nanometer,
            constraints=omma.HBonds
        )
        
        integrator = omm.LangevinIntegrator(
            300*unit.kelvin,
            1.0/unit.picoseconds,
            timestep*unit.femtoseconds
        )
        
        platform = omm.Platform.getPlatformByName('CUDA')
        properties = {'Precision': 'mixed'}
        return omma.Simulation(prmtop.topology, system, integrator, platform, properties)

    def setup_system(self, topology_dir: str, timestep: float) -> Tuple[str, List[np.ndarray], List[np.ndarray]]:
        """Setup system and return prmtop path and coordinates"""
        top_file = next(f for f in os.listdir(topology_dir) if f.endswith('.top'))
        gro_file = next(f for f in os.listdir(topology_dir) if f.endswith('_solv_ions.gro'))
        xtc_file = next(f for f in os.listdir(topology_dir) if f.endswith('.xtc'))
        
        pmd.gromacs.GROMACS_TOPDIR = os.environ.get('GMXLIB', os.getcwd())

        traj = md.load(os.path.join(topology_dir, xtc_file),
                      top=os.path.join(topology_dir, gro_file))
        
        positions = [frame.xyz[0] for frame in traj[-self.n_replicates:]]
        box_vectors = [frame.unitcell_vectors[0] for frame in traj[-self.n_replicates:]]

        print(f"Converting {os.path.basename(topology_dir)} to Amber format...")
        gromacs = pmd.load_file(os.path.join(topology_dir, top_file),
                               xyz=os.path.join(topology_dir, gro_file))
        
        temp_dir = os.path.join(self.output_dir, "temp")
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        prmtop_path = os.path.join(temp_dir, "system.prmtop")
        gromacs.save(prmtop_path, format='amber')
        
        return prmtop_path, positions, box_vectors

    def run_warmup_test(self, prmtop_path: str, position: np.ndarray, 
                       box_vector: np.ndarray, timestep: float, system_name: str,
                       replicate: int) -> Dict:
        """Run a single warm-up test"""
        print(f"\nTesting {timestep} fs timestep for {system_name} (replicate {replicate + 1})")
        
        # Create new simulation instance for each replicate
        simulation = self.create_simulation(prmtop_path, timestep)
        n_steps = int(self.simulation_time * 1000 / timestep)  # convert ps to fs
        
        # Initialize system
        simulation.context.setPositions(position)
        simulation.context.setPeriodicBoxVectors(*box_vector)
        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
        
        # Collect data
        report_interval = max(1, n_steps // 100)  # Get ~100 data points
        energies = []
        temperatures = []
        steps = []
        
        class DataReporter(omma.StateDataReporter):
            def __init__(self, data_lists, reportInterval):
                super().__init__(None, reportInterval, step=True, 
                               potentialEnergy=True, temperature=True)
                self.data_lists = data_lists
                
            def report(self, simulation, state):
                potential_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                temperature = state.getKineticEnergy() / (0.5 * unit.MOLAR_GAS_CONSTANT_R * 
                             simulation.system.getNumParticles() * 3)
                temperature = temperature.value_in_unit(unit.kelvin)
                
                self.data_lists['energies'].append(potential_energy)
                self.data_lists['temperatures'].append(temperature)
                self.data_lists['steps'].append(simulation.currentStep)
        
        data = {'energies': energies, 'temperatures': temperatures, 'steps': steps}
        reporter = DataReporter(data, report_interval)
        simulation.reporters.append(reporter)
        
        try:
            simulation.step(2000)

            # set to 2fs timestep for the rest of the simulation
            simulation.integrator.setStepSize(2*unit.femtoseconds)
            simulation.step(10000)


            return data
        except Exception as e:
            print(f"Simulation failed at timestep {timestep} fs (replicate {replicate + 1}): {str(e)}")
            return None

    def plot_results(self, all_results: Dict[float, List[Dict]], system_name: str):
        """Create plots comparing different timesteps with means and standard deviations"""
        plt.rcParams.update({'font.size': 10,
                           'axes.labelsize': 12,
                           'axes.titlesize': 12,
                           'xtick.labelsize': 10,
                           'ytick.labelsize': 10,
                           'legend.fontsize': 10})

        # First set of plots (trajectories)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle(f'Simulation Results for {system_name}', y=0.95, fontsize=14)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.timesteps)))
        
        # Plot trajectories with means and std devs
        for idx, (timestep, replicates) in enumerate(all_results.items()):
            color = colors[idx]
            valid_replicates = [r for r in replicates if r is not None]
            
            if not valid_replicates:
                continue
                
            n_points = min(len(r['steps']) for r in valid_replicates)
            common_x = np.linspace(0, 1, n_points)
            
            interp_energies = []
            interp_temps = []
            
            for i, replicate in enumerate(valid_replicates):
                steps = np.array(replicate['steps'])
                x = steps / steps[-1]
                
                interp_energy = np.interp(common_x, x, replicate['energies'])
                interp_temp = np.interp(common_x, x, replicate['temperatures'])
                
                interp_energies.append(interp_energy)
                interp_temps.append(interp_temp)
                
                ax1.plot(x, replicate['energies'], color=color, alpha=0.15, 
                        linewidth=1, label=f'{timestep} fs (Rep {i+1})' if idx == 0 and i == 0 else None)
                ax2.plot(x, replicate['temperatures'], color=color, alpha=0.15, 
                        linewidth=1, label=f'{timestep} fs (Rep {i+1})' if idx == 0 and i == 0 else None)
            
            energy_mean = np.mean(interp_energies, axis=0)
            energy_std = np.std(interp_energies, axis=0)
            temp_mean = np.mean(interp_temps, axis=0)
            temp_std = np.std(interp_temps, axis=0)
            
            ax1.plot(common_x, energy_mean, color=color, linewidth=2, 
                    label=f'{timestep} fs (mean)')
            ax2.plot(common_x, temp_mean, color=color, linewidth=2, 
                    label=f'{timestep} fs (mean)')
            
            ax1.fill_between(common_x, energy_mean - energy_std, 
                           energy_mean + energy_std, color=color, alpha=0.3)
            ax2.fill_between(common_x, temp_mean - temp_std, 
                           temp_mean + temp_std, color=color, alpha=0.3)
        
        ax1.set_xlabel('Fraction of Simulation Complete')
        ax1.set_ylabel('Potential Energy (kJ/mol)')
        ax1.set_title('Energy Evolution', pad=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax2.set_xlabel('Fraction of Simulation Complete')
        ax2.set_ylabel('Temperature (K)')
        ax2.set_title('Temperature Evolution', pad=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylim(220, 240)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
        plt.savefig(os.path.join(self.output_dir, f'{system_name}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Create enhanced statistical analysis plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        fig.suptitle(f'Stability Analysis for {system_name}', y=0.95, fontsize=14)
        
        timesteps = []
        energy_stats = []  # Will contain (mean, std, individual_values)
        temp_stats = []    # Will contain (mean, std, individual_values)
        
        for timestep, replicates in all_results.items():
            valid_replicates = [r for r in replicates if r is not None]
            if valid_replicates:
                timesteps.append(timestep)
                
                # Calculate standard deviation for each replicate
                e_stds = [np.std(r['energies'][-10:]) for r in valid_replicates]
                t_stds = [np.std(r['temperatures'][-10:]) for r in valid_replicates]
                
                # Store statistics
                energy_stats.append((np.mean(e_stds), np.std(e_stds), e_stds))
                temp_stats.append((np.mean(t_stds), np.std(t_stds), t_stds))
        
        timesteps = np.array(timesteps)
        
        # Plot Energy Statistics
        for i, (mean, std, values) in enumerate(energy_stats):
            # Plot individual points
            ax1.scatter([timesteps[i]] * len(values), values, 
                       color='blue', alpha=0.3, s=50)
        
        # Plot means and error bars
        means = [stat[0] for stat in energy_stats]
        stds = [stat[1] for stat in energy_stats]
        ax1.errorbar(timesteps, means, yerr=stds, color='blue', 
                    linewidth=2, marker='o', markersize=8, 
                    capsize=5, capthick=2, label='Mean ± Std Dev')
        
        # Plot Temperature Statistics
        for i, (mean, std, values) in enumerate(temp_stats):
            # Plot individual points
            ax2.scatter([timesteps[i]] * len(values), values, 
                       color='red', alpha=0.3, s=50)
        
        # Plot means and error bars
        means = [stat[0] for stat in temp_stats]
        stds = [stat[1] for stat in temp_stats]
        ax2.errorbar(timesteps, means, yerr=stds, color='red', 
                    linewidth=2, marker='o', markersize=8, 
                    capsize=5, capthick=2, label='Mean ± Std Dev')
        
        # Format plots
        ax1.set_xlabel('Timestep (fs)')
        ax1.set_ylabel('Energy Std Dev (kJ/mol)')
        ax1.set_title('Energy Fluctuations', pad=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        ax2.set_xlabel('Timestep (fs)')
        ax2.set_ylabel('Temperature Std Dev (K)')
        ax2.set_title('Temperature Fluctuations', pad=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, f'{system_name}_final_fluctuations.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


    def run_all_tests(self):
        """Run tests for each topology"""
        topology_dir = os.path.join(self.base_dir, "BPTI_10_c0")
        system_name = os.path.basename(topology_dir)
        
        print(f"\nTesting system: {system_name}")
        
        results = {}
        # First setup the system and get coordinates
        prmtop_path, positions, box_vectors = self.setup_system(topology_dir, self.timesteps[0])
        
        for timestep in self.timesteps:
            replicate_results = []
            for i in range(self.n_replicates):
                result = self.run_warmup_test(prmtop_path, positions[i], box_vectors[i], 
                                            timestep, system_name, i)
                replicate_results.append(result)
            
            results[timestep] = replicate_results
            
        self.plot_results(results, system_name)
        
        with open(os.path.join(self.output_dir, f'{system_name}_summary.txt'), 'w') as f:
            f.write("Timestep (fs)\tTotal Steps\tSuccessful Replicates\tEnergy StdDev\tTemp StdDev\n")
            for timestep, replicates in results.items():
                valid_replicates = [r for r in replicates if r is not None]
                if valid_replicates:
                    n_steps = valid_replicates[0]['steps'][-1]
                    e_stds = [np.std(r['energies']) for r in valid_replicates]
                    t_stds = [np.std(r['temperatures']) for r in valid_replicates]
                    avg_e_std = np.mean(e_stds)
                    avg_t_std = np.mean(t_stds)
                    f.write(f"{timestep}\t{n_steps}\t{len(valid_replicates)}/{self.n_replicates}\t"
                           f"{avg_e_std:.2f}\t{avg_t_std:.2f}\n")
                else:
                    f.write(f"{timestep}\tN/A\tFAILED\tN/A\tN/A\n")

if __name__ == "__main__":
    tester = WarmupTest()
    tester.run_all_tests()