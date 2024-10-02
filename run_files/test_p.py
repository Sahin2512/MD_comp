import os
from openmm import *
from openmm.app import *
from openmm.unit import *
import psutil
import time

# Define directories for input and output files
input_dir = 'input_files'
output_dir = 'output_files'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

psf = CharmmPsfFile('1eru.psf')
crd = CharmmCrdFile('1eru.crd')
pdb = PDBFile('1eru.pdb')
print("This are the pdbreader files")
forceField = ('/charmm.xml/charmm36.xml')
param_list=['toppar_water_ions.str', 'toppar_ions_won.str', 'toppar_dum_noble_gases.str', 'toppar_all36_synthetic_polymer_patch.str', 'toppar_all36_synthetic_polymer.str', 'toppar_all36_prot_retinol.str', 'toppar_all36_prot_na_combined.str', 'toppar_all36_prot_modify_res.str', 'toppar_all36_prot_model.str', 'toppar_all36_prot_heme.str', 'toppar_all36_prot_fluoro_alkanes.str', 'toppar_all36_prot_c36m_d_aminoacids.str', 'toppar_all36_prot_arg0.str', 'toppar_all36_polymer_solvent.str', 'toppar_all36_na_rna_modified.str', 'toppar_all36_nano_lig_patch.str', 'toppar_all36_nano_lig.str', 'toppar_all36_na_nad_ppi.str', 'toppar_all36_moreions.str', 'toppar_all36_lipid_yeast.str', 'toppar_all36_lipid_tag.str', 'toppar_all36_lipid_sphingo.str', 'toppar_all36_lipid_prot.str', 'toppar_all36_lipid_oxidized.str', 'toppar_all36_lipid_mycobacterial.str', 'toppar_all36_lipid_model.str', 'toppar_all36_lipid_miscellaneous.str', 'toppar_all36_lipid_lps.str', 'toppar_all36_lipid_lnp.str', 'toppar_all36_lipid_inositol.str', 'toppar_all36_lipid_hmmm.str', 'toppar_all36_lipid_ether.str', 'toppar_all36_lipid_detergent.str', 'toppar_all36_lipid_dag.str', 'toppar_all36_lipid_cholesterol.str', 'toppar_all36_lipid_cardiolipin.str', 'toppar_all36_lipid_bacterial.str', 'toppar_all36_lipid_archaeal.str', 'toppar_all36_label_spin.str', 'toppar_all36_label_fluorophore.str', 'toppar_all36_carb_imlab.str', 'toppar_all36_carb_glycopeptide.str', 'toppar_all36_carb_glycolipid.str', 'top_interface.rtf', 'top_all36_prot.rtf', 'top_all36_na.rtf', 'top_all36_lipid.rtf', 'top_all36_cgenff.rtf', 'top_all36_carb.rtf', 'par_interface.prm', 'par_all36_na.prm', 'par_all36m_prot.prm', 'par_all36_lipid.prm', 'par_all36_cgenff.prm', 'par_all36_carb.prm', 'cam.str']
param_files= ["params/"+ filename for filename in param_list]
params = CharmmParameterSet(*param_files)

# System configuration
nonbondedMethod = PME
nonbondedCutoff = 1.0 * nanometers
ewaldErrorTolerance = 0.0005
constraints = HBonds
rigidWater = True
constraintTolerance = 0.000001
hydrogenMass = 1.5 * amu

is_periodic = psf.box_vectors is not None

# Periodic box vectors
if not is_periodic:
    sizebox = 6.3
    psf.setBox(sizebox * nanometer, sizebox * nanometer, sizebox * nanometer)

# Integrators
dt = 0.004 * picoseconds
temperature = 300 * kelvin
friction = 1.0 / picosecond
pressure = 1.0 * atmospheres
barostatInterval = 25

# Simulation options
steps = 500000
equilibrationSteps = 500000

# Prepare the simulation
def print_memory_usage():
    process = psutil.Process(os.getpid())
    print('Memory usage:', process.memory_info().rss / (1024 * 1024), 'MB')  # in MB

start1 = time.time()
print('Building system...')
topology = psf.topology
positions = pdb.positions
system = psf.createSystem(params, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
                          constraints=constraints, rigidWater=rigidWater, ewaldErrorTolerance=ewaldErrorTolerance, hydrogenMass=hydrogenMass)
system.addForce(MonteCarloBarostat(pressure, temperature, barostatInterval))
integrator = LangevinMiddleIntegrator(temperature, friction, dt)
integrator.setConstraintTolerance(constraintTolerance)
simulation = Simulation(topology, system, integrator)
simulation.context.setPositions(positions)

# Minimize and equilibrate
print_memory_usage()
print('Initial potential energy:')
print('Performing energy minimization...')
simulation.minimizeEnergy()
print('Equilibrating...')
simulation.context.setVelocitiesToTemperature(temperature)
simulation.step(equilibrationSteps)
print("Equilibration complete")

# Save the state after equilibration
equilibrated_state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)
with open(os.path.join(output_dir, 'equilibrated_state.xml'), 'w') as f:
    f.write(XmlSerializer.serialize(equilibrated_state))

# Prepare for the main simulation run
dcdReporter = DCDReporter(os.path.join(output_dir, 'test.dcd'), 10000)
dataReporter = StateDataReporter(os.path.join(output_dir, 'test.txt'), 10000, totalSteps=steps,
                                 step=True, speed=True, progress=True, potentialEnergy=True, temperature=True, separator='\t')
checkpointReporter = CheckpointReporter(os.path.join(output_dir, 'checkpoint.chk'), 10000)

# Running the Main Simulation
print('Running main simulation...')
simulation.reporters.append(dcdReporter)
simulation.reporters.append(dataReporter)
simulation.reporters.append(checkpointReporter)

# Load the equilibrated state
with open(os.path.join(output_dir, 'equilibrated_state.xml'), 'r') as f:
    state = XmlSerializer.deserialize(f.read())
simulation.context.setState(state)

simulation.step(steps)
print('Simulation complete.')
