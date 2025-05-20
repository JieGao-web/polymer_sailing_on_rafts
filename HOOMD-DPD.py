######################################################################################
# DPD simulations of single polymer chain diffusion on a two-component lipid bilayer #
# Written by Jie Gao & Jinglei Hu (hujinglei@nju.edu.cn)                             #
# Initial configuration file in gsd format and trajectory in dcd format              #
######################################################################################
#!/usr/bin/python
import os, re, random, datetime
import glob
import hoomd
import numpy as np
from hoomd import md

# ============ USER PARAMETERS ==================================

RESTART = False                       # If True, you are continuing from a previous simulation
TEMP = 1.0                            # DPD Default Temperature
DELTA_T = 0.01                        # temperature 1.01, with 1.0 % deviation from the specified value of TEMP = 1.0
N_TIMESTEPS = 10000000                # Number of integration steps
FREQ_TRAJ = 10000                     # Frequency of saving trajectory (period)
FREQ_SORT = 200                       # Sorting frequency
PULL_FX = 0.5                         # Pulling force along x for the polymer
RAND_SEED = random.randint(0, 65535)  # Random seed (replace with your value)
TRAJ_NAME = "traj.dcd"                # DCD trajectory filename
CFG_PREFIX = "cpt"
CFG_PREFIX_without_W = "solute"
RUN_LOG = "run.log"
LOG_GSD = "log_thermo.gsd"

## Initialize ##
### Fetch initial configuration gsd file ###
gsd_list = []
for file in os.listdir(os.getcwd()):
    if file.startswith(CFG_PREFIX) and file.endswith(".gsd"):
        gsd_list.append(file)
gsd_list.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)])
INIT_GSD = str(gsd_list[-1])
gsd_list = []


# ============ INITIALIZE HOOMD and LOAD SNAPSHOT =========================
gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu, seed=RAND_SEED)
if RESTART:
    sim.create_state_from_gsd(filename=INIT_GSD)
else:
    sim.timestep = 0
    sim.create_state_from_gsd(filename=INIT_GSD)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)


# ============ BUILD FORCES & BOND POTENTIALS =============================
dpd = md.pair.DPD(nlist=md.nlist.Cell(buffer=0.05,exclusions=(), default_r_cut=1.0), kT=TEMP, default_r_cut=1.0)  # neighbor list

dpd.params[('H', 'H')] = dict(A=25.0, gamma=4.5)
dpd.params[('H', 'T')] = dict(A=200.0, gamma=4.5)
dpd.params[('H', 'C')] = dict(A=29.0, gamma=4.5)
dpd.params[('H', 'O')] = dict(A=200.0, gamma=4.5)
dpd.params[('H', 'W')] = dict(A=25.0, gamma=4.5)
dpd.params[('H', 'P')] = dict(A=10.0, gamma=4.5)
dpd.params[('T', 'T')] = dict(A=25.0, gamma=4.5)
dpd.params[('T', 'C')] = dict(A=200.0, gamma=4.5)
dpd.params[('T', 'O')] = dict(A=29.0, gamma=4.5)
dpd.params[('T', 'W')] = dict(A=200.0, gamma=4.5)
dpd.params[('T', 'P')] = dict(A=200.0, gamma=4.5)
dpd.params[('C', 'C')] = dict(A=25.0, gamma=4.5)
dpd.params[('C', 'O')] = dict(A=200.0, gamma=4.5)
dpd.params[('C', 'W')] = dict(A=25.0, gamma=4.5)
dpd.params[('C', 'P')] = dict(A=5.0, gamma=4.5)
dpd.params[('O', 'O')] = dict(A=25.0, gamma=4.5)
dpd.params[('O', 'W')] = dict(A=200.0, gamma=4.5)
dpd.params[('O', 'P')] = dict(A=200.0, gamma=4.5)
dpd.params[('W', 'W')] = dict(A=25.0, gamma=4.5)
dpd.params[('W', 'P')] = dict(A=22.5, gamma=4.5)
dpd.params[('P', 'P')] = dict(A=25.0, gamma=4.5)



# 2) Bonded forces: FENE + Harmonic
# FENE:
fene = md.bond.FENEWCA()
fene.params['lipid'] = dict(k=0.0, r0=1.5, epsilon=0.0, sigma=1.0, delta=0.0)
fene.params['polymer'] = dict(k=30.0, r0=1.5, epsilon=0.0, sigma=1.0, delta=0.0)

# Harmonic:
harm = md.bond.Harmonic()
harm.params["lipid"] = dict(k=100.0, r0=0.45)
harm.params["polymer"] = dict(k=0.0, r0=0.45)

# ============ BUILD EXTERNAL FORCES (CONSTANT) ==========================
group_P = hoomd.filter.Type(["P"])
num_P = len(group_P(sim.state))

snapshot = sim.state.get_snapshot()
fx = PULL_FX / num_P
force_vector = (fx, 0.0, 0.0)
const_force_polymer = md.force.Constant(filter=group_P)
const_force_polymer.constant_force["P"] = (fx, 0, 0)


N = snapshot.particles.N
group_W_tags = list(range(N - num_P, N))
group_W = hoomd.filter.Tags(group_W_tags)
const_force_W = md.force.Constant(filter=group_W)
const_force_W.constant_force["W"] = (-fx, 0, 0)

# ============ CHOOSING AN INTEGRATOR FOR DPD ============================

nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
integrator = hoomd.md.Integrator(dt=DELTA_T, methods=[nve], forces=[dpd, fene, harm, const_force_polymer, const_force_W])
sim.operations.integrator = integrator

group_all = hoomd.filter.All()
thermo = md.compute.ThermodynamicQuantities(filter=group_all)
sim.operations.computes.append(thermo)

## Memory sorting ##
sorter = hoomd.tune.ParticleSorter(FREQ_SORT)
sim.operations.tuners.append(sorter)

## zero center-of-mass linear momentum during relaxation run, but not during production run ##
if not RESTART:
    zero_momentum = md.update.ZeroMomentum(N_TIMESTEPS // 100)
    sim.operations.updaters.append(zero_momentum)


# ============ LOGGING AND TRAJECTORY OUTPUT ============================
# 1) DCD or GSD trajectory
traj_format = TRAJ_NAME[-3:]
if  traj_format == "dcd":
    dcd_writer = hoomd.write.DCD(
        filename=TRAJ_NAME,
        trigger=hoomd.trigger.Periodic(FREQ_TRAJ),
        filter=hoomd.filter.Type(["H", "T", "C", "O", "P"]),
        unwrap_full=True,
    )
    sim.operations.writers.append(dcd_writer)
elif  traj_format == "gsd":
    gsd_writer = hoomd.write.GSD(
        filename=TRAJ_NAME,
        trigger=hoomd.trigger.Periodic(FREQ_TRAJ),
        filter=hoomd.filter.Type(["H", "T", "C", "O", "P"]),
        mode="ab",
    )
    gsd_writer.dynamic = ["property","particles/velocity"]
    sim.operations.writers.append(gsd_writer)


# 2) A standard logger for textual or tabular output
thermo_logger = hoomd.logging.Logger()
thermo_logger.add(thermo)
gsd_writer = hoomd.write.GSD(
    filename=LOG_GSD, trigger=hoomd.trigger.Periodic(FREQ_TRAJ), mode="ab", filter=hoomd.filter.Null(), logger=thermo_logger
)
sim.operations.writers.append(gsd_writer)


class Status:
    def __init__(self, sim):
        self.sim = sim

    @property
    def seconds_remaining(self):
        try:
            return (self.sim.final_timestep - self.sim.timestep) / self.sim.tps
        except ZeroDivisionError:
            return 0

    @property
    def etr(self):
        return str(datetime.timedelta(seconds=self.seconds_remaining)).split(".")[0]


table_logger = hoomd.logging.Logger(categories=["scalar", "string"])
table_logger.add(sim, quantities=["timestep", "tps"])
table_logger[("Status", "etr")] = (Status(sim), "etr", "string")
table_logger.add(thermo, ["kinetic_temperature", "pressure", "kinetic_energy", "potential_energy", "pressure_tensor"])
table_logger["total_energy"] = (lambda: thermo.kinetic_energy + thermo.potential_energy, "scalar")
table_writer = hoomd.write.Table(
    output=open(RUN_LOG, mode="x", newline="\n"),
    trigger=hoomd.trigger.Periodic(FREQ_TRAJ),
    logger=table_logger,
    max_header_len=1,
)
sim.operations.writers.append(table_writer)


# ============ RUN SIMULATION ============================
sim.run(N_TIMESTEPS)

# ============ FINAL PRINTOUTS ===========================
files = glob.glob("cpt.*.gsd")
pattern = re.compile(r"cpt\.(\d+)\.gsd")
max_number = 0
for file in files:
    match = pattern.match(file)
    if match:
        # 将字符串数字转换为整数
        current_number = int(match.group(1))
        if current_number > max_number:
            max_number = current_number


CPT_FILE = CFG_PREFIX + "." + str(sim.timestep).zfill(10) + ".gsd"
hoomd.write.GSD.write(state=sim.state, mode="wb", filename=CPT_FILE)
CPT_FILE_without_W = CFG_PREFIX_without_W + "." + str(sim.timestep).zfill(10) + ".gsd"
hoomd.write.GSD.write(state=sim.state,filter=hoomd.filter.Type(["H", "T", "C", "O", "P"]), mode="wb", filename=CPT_FILE_without_W)
