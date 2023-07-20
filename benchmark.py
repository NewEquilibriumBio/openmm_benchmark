"""
Run benchmarks on given platform.
"""

import time

import tap
from openmm import LangevinMiddleIntegrator, OpenMMException, Platform
from openmm.app import PME, ForceField, Modeller, PDBFile, Simulation
from openmm.unit import femtosecond, kelvin, kilojoules, mole, nanometer, picosecond
from openmmplumed import PlumedForce


class ArgumentParser(tap.Tap):
    """
    Run benchmarks on given platform.
    """

    platform: str  # Select to run benchmark on the cuda or hip platform


def get_model(topology: str, water_model: str) -> Modeller:
    """
    Load the starting structure and convert waters if needed.

    :param topology: The input structure
    :param water_model: the water model to use
    """
    protein = PDBFile(f"files/{topology}")
    modeller = Modeller(protein.topology, protein.positions)
    modeller.convertWater(water_model)
    return modeller


def run_simulation(  # pylint: disable=too-many-locals
    topology: str,
    water_model: str,
    precision: str,
    platform_name: str,
    plumed: bool = False,
) -> float:
    """
    Run a simulation with the given settings and return the estimated ns/day.

    :param topology: The input structure
    :param water_model: the water model to use
    :param precision: the precision to use
    :param platform_name: the platform to use
    :param plumed: whether or not to use openmmplumed
    """
    forcefield = ForceField("amber99sbildn.xml", f"{water_model}.xml")
    model = get_model(topology, water_model)
    integrator = LangevinMiddleIntegrator(300.0 * kelvin, 1.0 / picosecond, 1.0 * femtosecond)

    platform = Platform.getPlatformByName(platform_name.upper())
    platform_properties = {"Precision": precision}

    system = forcefield.createSystem(
        topology=model.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
    )

    if plumed is True:
        with open("files/plumed.inp", "r", encoding="utf-8") as f:
            script = f.read()
            script = script.replace("STRUCTURE=", f"STRUCTURE=files/{topology}")
        system.addForce(PlumedForce(script))

    simulation = Simulation(
        model.topology, system, integrator, platform=platform, platformProperties=platform_properties
    )

    simulation.context.setPositions(model.positions)
    simulation.minimizeEnergy(tolerance=20 * kilojoules / mole)

    # Warmup
    simulation.step(1000)

    n_steps = 100000
    start_time = time.time()
    simulation.step(n_steps)
    end_time = time.time()
    ns_per_day = (n_steps / 1e6) / ((end_time - start_time) / (60 * 60 * 24))
    return ns_per_day


def main(platform: str) -> None:
    """
    Run the benchmarks.

    :param platform: the platform to run on
    """
    with open(f"results_{platform}.csv", "w", encoding="utf-8") as f:
        f.write("topology,water model,precision,ns/day\n")
        for topology in ("small.pdb", "medium.pdb", "large.pdb"):
            for water_model in ("tip3p", "tip4pew"):
                for precision in ("single", "mixed", "double"):
                    ns_per_day = run_simulation(topology, water_model, precision, platform)
                    f.write(f"{topology},{water_model},{precision},{ns_per_day:.2f}\n")

        try:
            run_simulation("small.pdb", "tip3p", "mixed", platform, plumed=True)
            f.write("OpenMM-PLUMED: Success")
        except OpenMMException:
            f.write("OpenMM-PLUMED: Failed")


if __name__ == "__main__":
    parser = ArgumentParser()
    arguments = parser.parse_args()
    main(arguments.platform)
