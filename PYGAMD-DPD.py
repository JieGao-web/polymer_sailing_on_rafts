######################################################################################
# DPD simulations of single polymer chain diffusion on a two-component lipid bilayer #
# Written by Jie Gao & Jinglei Hu (hujinglei@nju.edu.cn)                             #
# Initial configuration file in xml format and trajectory in dcd format              #
# NOTE: 1. DO NOT change N_TIMESTEPS & FREQ_TRAJ for restart simulations!!!          #
#       2. Last frame of a dcd file overlaps with first frame of consecutive dcd!!!  #
######################################################################################
#!/usr/bin/python
import os, re, random
from poetry import cu_gala as gala
from poetry import _options

## >> Parameters & Constants << ##
RESTART     = False                          # restart simulation
TEMP        = 1.0                            # DPD default temperature = 1.0
DELTA_T     = 0.04                           # temperature 0.99, with 1.0 % deviation from the specified value of TEMP = 1.0
N_TIMESTEPS = 10000000                       # number of integration steps
FREQ_TRAJ   = 10000                          # frequency of saving trajectory, i.e., sampling
FREQ_SORT   = 500                            # frequency of memory sorting
PULL_FX     = 0.5                            # pulling force on polymer chain along x-direction, the opposite force is distributed over lipid beads. No net force on the system!
DUMP_FILE   = 'dump.tsv'                     # dump file incl. timestep, momentum, pressure tensor, temperature, totoal potential, pressure
TRAJ_NAME   = 'traj'                         # name of trajectory
XML_NAME    = 'cpt'                          # name of checkpoint xml file (naming style: cpt.0000000000.xml)
RAND_SEED   = random.randint(1000, 99999999) # seed of pseudo random number generator

## Initialize ##
### Fetch initial configuration xml file ###
xml_list= []
for file in os.listdir(os.getcwd()):
    if file.startswith(XML_NAME) and file.endswith('.xml'):
        xml_list.append(file)
xml_list.sort(key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
INIT_XML = str(xml_list[-1])
xml_list = []

### Reset time_step="0" in cpt.0000000000.xml ###
create_time = os.path.getctime(INIT_XML)
if INIT_XML == XML_NAME + '.0000000000.xml':
    with open(INIT_XML,"r") as file:
        original_contents = file.read()
    new_contents = re.sub(r'time_step="[^\"]*"', 'time_step="0"', original_contents)
    with open(INIT_XML, 'w') as file:
        file.write(new_contents)
modification_time = os.path.getmtime(INIT_XML)
os.utime(INIT_XML, (modification_time, create_time))

### Read xml ###
build_method = gala.XMLReader(INIT_XML)
perform_config = gala.PerformConfig(_options.gpu)
all_info = gala.AllInfo(build_method, perform_config)
app = gala.Application(all_info, DELTA_T)

## >> Force field << ##
### DPD pairwise forces ###
### Bead type: LA-Head: H, LA-Tail: T, LB-Head: C, LB-Tail: O, Water: W, Polymer: P ###
DPD_ParamTabs = [
    { 'pair': {'atom1': 'H', 'atom2': 'H'}, 'aij':  25.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'H', 'atom2': 'T'}, 'aij': 200.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'H', 'atom2': 'C'}, 'aij':  29.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'H', 'atom2': 'O'}, 'aij': 200.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'H', 'atom2': 'W'}, 'aij':  25.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'H', 'atom2': 'P'}, 'aij':  10.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'T', 'atom2': 'T'}, 'aij':  25.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'T', 'atom2': 'C'}, 'aij': 200.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'T', 'atom2': 'O'}, 'aij':  29.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'T', 'atom2': 'W'}, 'aij': 200.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'T', 'atom2': 'P'}, 'aij': 200.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'C', 'atom2': 'C'}, 'aij':  25.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'C', 'atom2': 'O'}, 'aij': 200.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'C', 'atom2': 'W'}, 'aij':  25.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'C', 'atom2': 'P'}, 'aij':  5.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'O', 'atom2': 'O'}, 'aij':  25.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'O', 'atom2': 'W'}, 'aij': 200.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'O', 'atom2': 'P'}, 'aij': 200.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'W', 'atom2': 'W'}, 'aij':  25.0, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'W', 'atom2': 'P'}, 'aij':  22.5, 'sij': 3.0 }, \
    { 'pair': {'atom1': 'P', 'atom2': 'P'}, 'aij':  25.0, 'sij': 3.0 }]

### Bonded force ###
BF_Tabs = [ {'type':'H-T', 'FENE':{'K':  0.0, 'rm': 1.5}, 'HARM':{'K': 100.0, 'r0': 0.45 }}, \
            {'type':'T-T', 'FENE':{'K':  0.0, 'rm': 1.5}, 'HARM':{'K': 100.0, 'r0': 0.45 }}, \
            {'type':'C-O', 'FENE':{'K':  0.0, 'rm': 1.5}, 'HARM':{'K': 100.0, 'r0': 0.45 }}, \
            {'type':'O-O', 'FENE':{'K':  0.0, 'rm': 1.5}, 'HARM':{'K': 100.0, 'r0': 0.45 }}, \
            {'type':'P-P', 'FENE':{'K': 30.0, 'rm': 1.5}, 'HARM':{'K':   0.0, 'r0': 0.45 }} ]

NL_RCUT = 1.0
NL_RBUF = 0.05
NL = gala.NeighborList(all_info, NL_RCUT, NL_RBUF)
dpd = gala.DPDForce(all_info, NL, NL_RCUT, RAND_SEED)
for n in range(0, len(DPD_ParamTabs)):
  table = DPD_ParamTabs[n]
  dpd.setParams(table['pair']['atom1'], table['pair']['atom2'], table['aij'], table['sij'])
app.add(dpd)

## Bonded forces ##
### FENE ###
BFF = gala.BondForceFENE(all_info)
for n in range(0, len(BF_Tabs)):
  table = BF_Tabs[n]
  BFF.setParams(table['type'], table['FENE']['K'], table['FENE']['rm'])
app.add(BFF)

### Harmonic ###
BFH = gala.BondForceHarmonic(all_info)
for n in range(0, len(BF_Tabs)):
  table = BF_Tabs[n]
  BFH.setParams(table['type'], table['HARM']['K'], table['HARM']['r0'])
app.add(BFH)

## External forces ##
### Polymer beads ###
group_P = gala.ParticleSet(all_info, 'P')
fx_Pbeads = gala.VariantConst(PULL_FX / group_P.getNumMembers()) # external force on each monomer
EF_P = gala.ExternalForce(all_info, group_P)
EF_P.setForce(fx_Pbeads, 'X')
app.add(EF_P)

group_all = gala.ParticleSet(all_info, 'all') # a collection of particles

### Water beads ### !NOTE: the opposite force should be distributed over the same amount of other beads in order to avoid the accumulation of numerical errors!
IDX_MAX = group_all.getNumMembers() - 1
group_W = gala.ParticleSet(all_info, IDX_MAX-group_P.getNumMembers()+1, IDX_MAX)
fx_Wbeads = gala.VariantConst(-PULL_FX / group_W.getNumMembers()) # external force on each lipid bead
EF_W = gala.ExternalForce(all_info, group_W)
EF_W.setForce(fx_Wbeads, 'X')
app.add(EF_W)

## Integrate Newton's equation of motion ##
comp_info = gala.ComputeInfo(all_info, group_all)  # calculating system informations, such as temperature, pressure, and momentum
Gwvv = gala.DPDGWVV(all_info, group_all) # integration method with GWVV algorithm
app.add(Gwvv)

## Memory sorting ##
sort_method = gala.Sort(all_info)
sort_method.setPeriod(FREQ_SORT)
app.add(sort_method)

## zero center-of-mass linear momentum during relaxation run, but not during production run ##
if RESTART != True:
  ZM = gala.ZeroMomentum(all_info)
  ZM.setPeriod(N_TIMESTEPS // 100)
  app.add(ZM)

## Dump momentum, pressure tensor, temperature, energy, etc. ##
DInfo = gala.DumpInfo(all_info, comp_info, DUMP_FILE)
DInfo.setPeriod(FREQ_TRAJ)
DInfo.dumpPressTensor()
app.add(DInfo)

## Write trajectory of membrane and polymer without water ##
group_NoWater = gala.ParticleSet(all_info, ['H', 'T', 'C', 'O', 'P'])
dcd_dump = gala.DCDDump(all_info, group_NoWater, TRAJ_NAME, True)
dcd_dump.setPeriod(FREQ_TRAJ)
dcd_dump.unpbc(True) # output continous coordinates without applying periodic boundary condition
app.add(dcd_dump)

xml_NoWater = gala.XMLDump(all_info, group_NoWater, 'solute')
xml_NoWater.setPrecision(7)
xml_NoWater.setPeriod(N_TIMESTEPS)
xml_NoWater.setOutput(['position', 'image', 'velocity', 'mass', 'bond', 'angle', 'force'])
app.add(xml_NoWater)

## save checkpoint for restart ##
xml_dump = gala.XMLDump(all_info, XML_NAME)
xml_dump.setPrecision(7)
xml_dump.setPeriod(N_TIMESTEPS)
xml_dump.setOutput(['position', 'image', 'velocity', 'mass', 'bond', 'angle', 'force'])
app.add(xml_dump)

## Run ##
app.run(N_TIMESTEPS)

## Write other info. ##
NL.printStats() # output neighbor list info.
print('INFO : --- Other parameters:')
print('INFO : Restart from previous simulation:', RESTART)
print('INFO : Temperature:', TEMP)
print('INFO : Integration stepsize:', DELTA_T)
print('INFO : Time steps:', N_TIMESTEPS)
print('INFO : Random seed:', RAND_SEED)
