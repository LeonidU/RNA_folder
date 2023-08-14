import openmm
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

def openmm_augmentation(input_filename, output_filename, steps=10000, temperature=300):

    pdb = PDBFile(input_filename)

    forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, model='tip3p', padding=1*nanometer)
    modeller.addHydrogens(forcefield)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, nonbondedCutoff=1*nanometer, constraints=HBonds)
    integrator = LangevinIntegrator(temperature*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    simulation.reporters.append(PDBReporter(output_filename, steps))
#    simulation.reporters.append(StateDataReporter(stdout, 200, step=True, potentialEnergy=True, temperature=True)) #Here is strings for logging and reporting. Uncomment this line to log temperature and energy every 1000 steps 
    simulation.step(steps)

openmm_augmentation("R2-006.pdb", "R2-006_augmented.pdb")