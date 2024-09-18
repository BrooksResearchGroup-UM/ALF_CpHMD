# In[1]: Importing modules
import os
import sys
import subprocess
import alf.GetLambda
import numpy as np
import pandas as pd
import shlex
import re
import itertools
import nglview as nv
import ipywidgets
import random
import argparse
import shutil
import time
import alf
import pycharmm
import pycharmm.read as read
import pycharmm.lingo as lingo
import pycharmm.generate as gen
import pycharmm.settings as settings
import pycharmm.write as write
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.select as select
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.param as param
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_fix as cons_fix
import pycharmm.shake as shake
import pycharmm.scalar as scalar
import pycharmm.charmm_file as charmm_file

############################################
# Set up global parameters
input_folder = 'his'
toppar = 'toppar'
topology_files = [
    'top_all36_prot.rtf',
    'par_all36m_prot.prm',
    'toppar_water_ions.str',
    'top_all36_cgenff.rtf',
    'par_all36_cgenff.prm',
    'my_files/titratable_residues.str'
]

# topology_files = [
#     'top_all22_prot.rtf',
#     'par_all22_prot.prm',
#     'toppar_water_ions.str',
#     'my_files/titratable_residues_c22.str'
# ]

# non-bonded conditions
nb_fswitch = False
nb_pme = True
cutnb = 14
cutim = cutnb
ctofnb = 12
ctonnb = 10

# dynamics conditions and paramaters
cpt_on = True # run with CPT for NPT?
temperature = 298.15
pH = None
hmr = False
cent_ncres = False
hydrogen = False

# ALF run parameters
start = 1
end = 20
phase = 1
phase_runs = 5


# Set up non-bonded parameters
nb_param = {'elec': True, 'atom': True,
            'cdie': True,'eps': 1,
            'cutnb': cutnb,'cutim': cutim,
            'ctofnb': ctofnb,'ctonnb': ctonnb,
            'inbfrq': -1,'imgfrq': -1
            }
if nb_pme:
    nb_param.update({'switch': True, 'vfswitch': True,
                     'ewald': True, 'pmewald': True,
                    'kappa': 0.320, 'order': 6
                    }) 
elif nb_fswitch:
    nb_param.update({'fswitch': True, 'vfswitch': True,
                     'ewald': False, 'pmewald': False
                    })
else:
    print('No non-bonded conditions set')


def check_charmm():    
    global toppar
    # Checking CHARMM executable
    path_charmm_match = re.search(r"'(.*?)'", 
                                  str(pycharmm.lib.__dict__.get('charmm')))
    if path_charmm_match:
        path_charmm = path_charmm_match.group(1)
        print('CHARMM executable is:', path_charmm)
    else:
        print('CHARMM executable path not found.')
        sys.exit()

    # Check that topology_folder is defined and exists
    if 'toppar' not in globals():
        print('Topology folder not defined properly. Using default topology folder.')
        #Find in 1-4 levels of parent folder of CHARMM executable
        max_search = 4
        for _ in range(max_search):
            charmm_folder = os.path.dirname(path_charmm)
            toppar = os.path.join(charmm_folder, 'toppar')
            if os.path.exists(toppar):
                print('Topology folder is:', toppar)
                break
        else:
            print('Topology folder not found. Please check the path.')
            sys.exit()
    return None

def read_topology_files(verbose=True):
    if not verbose:
        lingo.charmm_script('prnlev -1')
    
    prm_files = [file for file in topology_files if file.endswith('.prm')]
    rtf_files = [file for file in topology_files if file.endswith('.rtf')]
    str_files = [file for file in topology_files if file.endswith('.str')]
    lingo.charmm_script('bomblevel -2')
    settings.set_warn_level(-1)
    if rtf_files:
        read.rtf(os.path.join(toppar, rtf_files[0]))
        for file in rtf_files[1:]:
            read.rtf(os.path.join(toppar, file), append=True)
            
    if prm_files:
        read.prm(os.path.join(toppar, prm_files[0]), flex=True)
        for file in prm_files[1:]:
            read.prm(os.path.join(toppar, file), flex=True, append=True)
            
    for file in str_files:
        lingo.charmm_script(f'stream {os.path.join(toppar, file)}')
    
    settings.set_warn_level(5)
    lingo.charmm_script('bomblevel 0')
    if not verbose:
        lingo.charmm_script('prnlev 5')
    lingo.charmm_script('IOFOrmat EXTEnded')
    return None

def parser():
    global input_folder, temperature, pH, hmr, cent_ncres, hydrogen, start, end, phase
    p = argparse.ArgumentParser(description='Run ALF simulations')
    p.add_argument('-i', '--input',
                        dest='input_folder',
                        default=input_folder,
                        help='Input folder')
    p.add_argument('-t', '--temperature',
                        dest='temperature',
                        type=float,
                        default=temperature,
                        help='Temperature (in K)')
    p.add_argument('-pH', '--pH',
                        dest='pH',
                        type=float,
                        default=pH,
                        help='pH')
    p.add_argument('-hmr', '--hmr',
                        dest='hmr',
                        type=bool,
                        default=hmr,
                        help='Hydrogen mass repartitioning (True or False)',
                        action=argparse.BooleanOptionalAction)
    p.add_argument('-c', '--cent_ncres',
                        dest='cent_ncres',
                        type=int,
                        default=cent_ncres,
                        help='reCENTering command allows to recenter the system at the geometric center of the first NCRES residues in the psf file')
    p.add_argument('-s', '--start',
                        dest='start',
                        type=int,
                        default=start,
                        help='Start run')
    p.add_argument('-e', '--end',
                        dest='end',
                        type=int,
                        default=end,
                        help='End run')
    p.add_argument('-p', '--phase',
                        dest='phase',
                        type=int,
                        default=phase,
                        help='Phase (1, 2 or 3)')
    p.add_argument('-H', '--hydrogen',
                        dest='hydrogen',
                        type=bool,
                        default=hydrogen,
                        help='Restrain hydrogen atoms in BLOC (default: False)')

    args, unknown = p.parse_known_args()
    # Reassign arguments
    input_folder = args.input_folder
    temperature = args.temperature
    pH = args.pH
    hmr = args.hmr
    cent_ncres = args.cent_ncres
    hydrogen = args.hydrogen
    start = args.start
    end = args.end
    phase = args.phase
        
    required_files = ['system.crd', 'system.psf', 'patches.dat', 'box.dat', 'fft.dat']

    # check if all required files are present and not empty
    for file in required_files:
        if not os.path.isfile(os.path.join(input_folder, 'prep', file)):
            raise FileNotFoundError(f'File {file} not found in {input_folder}')
        if os.path.getsize(os.path.join(input_folder, 'prep', file)) == 0:
            raise FileNotFoundError(f'File {file} is empty')
    return None

# name is folder name of input_folder

# In[2]: arguments

# Get arguments as flags from command line

parser()

name = input_folder.split('/')[-1]

sys.stdout = open(f'{input_folder}/python_log.out', 'w')


# In[3]: CHARMM topology and parameter files
check_charmm()
read_topology_files()


# In[4]: Simulation class
class ALF_simulation:
    def __init__(self, **kwargs):
        # global temperature, pH, phase, start, end
        self.pH = None
        self.temperature = pH
        self.phase = phase
        self.start = start
        self.end = end
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.fix_bias = False
        

        
        self.box_size = [0, 0, 0]
        self.type = None
        self.angles = [90, 90, 90]
        
        # CHARMM Units
        self.log_unit = 30
        self.dcd_unit = 40
        self.lmd_unit = 50
        self.rpr_unit = 60
        self.rst_unit = 70
        self.restart_run = None
        self.phase_runs = phase_runs

        # CHARMM Output
        self.clog = None

        self.homedir = os.getcwd()

        # Initialize ALF
        self.alf_initilize()

    def run(self):

        # Find last run
        self.start = self.find_last_run()
        
        # Prevent re-running the same simulation
        if os.path.isfile(f'{input_folder}/variables{self.end}.inp'):
            return print('Simulation already executed up to run {}'.format(self.end))
        # Run simulation
        
        for i in range(self.start, self.end+1):
            start_time = time.time()
            if i < 20:
                self.phase_runs = 1
            else:
                self.phase_runs = phase_runs
            for j in range(self.phase_runs):
                print('Run', i, 'Phase', j+1)
                if self.phase_runs > 1:
                    letter = '_'+self.alphabet[j]
                else:
                    letter = ''
                self.run_mkdir(i)
                self.redirect_output(i, letter)
                self.crystal(i, letter)
                print('Creating block file... for run', i)
                self.block(i, letter)
                self.scat()
                self.minimization(i)
                self.dynamics(i, letter)
                self.return_output()
            self.alf_analysis(i)
            end_time = time.time()
            print(f'Run {i} completed in {round(end_time-start_time, 1)} seconds')
                
    

    def redirect_output(self, run, letter=''):
        self.clog = charmm_file.CharmmFile(file_name=f'{input_folder}/run{run}/log{letter}.out', file_unit=self.log_unit , read_only=False, formatted=True)
        pycharmm.charmm_script(f'OUTUnit {str(self.log_unit)}')
    
    def return_output(self):
        if self.clog:
            pycharmm.charmm_script('OUTUnit 6')
            self.clog.close()
        

    def find_last_run(self):
        for i in range(self.end+1,self.start-1, -1):
            if os.path.isfile(f'{input_folder}/variables{i}.inp'):
                if i == 1:
                    print('No previous runs found, starting from run 1')
                    return 1
                else:
                    print(f'Found parameters from run {i-1}, starting from run {i}')
                    return i
        else:
            raise FileNotFoundError(f'File {input_folder}/variables1.inp not found')

    def crystal(self, run, letter=''):
        global nb_param
        settings.set_verbosity(5)
        if run == self.start and (letter == '' or letter == '_a'):
            try:
                psf.delete_atoms(pycharmm.SelectAtoms.all_atoms())
                pycharmm.charmm_script('IMAGe FIXEd sele ALL end')
                pycharmm.charmm_script('CRYSTAL FREE')
            except:
                pass
            if hmr:
                read.psf_card(f'{input_folder}/prep/system_hmr.psf')
            else:
                read.psf_card(f'{input_folder}/prep/system.psf')
            self.read_selections(f'{input_folder}/prep/patches.dat')
            self.alf_initilize()

        if run > 5:
            self.restart_run = random.randint(run-5, run-1)
        else: 
            self.restart_run = 1
        
        # check that the previous run folder exists
        file = open(f'{input_folder}/prep/box.dat').readlines()
        self.type = file[0].strip()
        self.box_size = list(map(float, file[1].strip().split()))
        self.angles = list(map(float, file[2].strip().split()))
        if hmr:
            read.coor_card(f'{input_folder}/prep/system_hmr.crd')
        else:
            read.coor_card(f'{input_folder}/prep/system.crd')
        if self.restart_run != 1:
            if os.path.isfile(f'{input_folder}/run{self.restart_run}/prod.crd{letter}'):
                read.coor_card(f'{input_folder}/run{self.restart_run}/prod.crd{letter}')
            elif os.path.isfile(f'{input_folder}/run{self.restart_run}/prod.crd'):
                read.coor_card(f'{input_folder}/run{self.restart_run}/prod.crd')
            # else:
            #     read.coor_card(f'{input_folder}/run{self.restart_run}/res/{name}_flat.crd')
            if os.path.isfile(f'{input_folder}/run{self.restart_run}/box.dat{letter}'):
                file = open(f'{input_folder}/run{self.restart_run}/box.dat{letter}').readlines()
            else:
                file = open(f'{input_folder}/run{self.restart_run}/box.dat').readlines()
            self.box_size = list(map(float, file[1].strip().split()))
            
        
        # print(f'Crystal type: {self.type}, Box size: {self.box_size}, Angles: {self.angles}')
        # if self.type.upper().startswith('CUBI'):
        #     crystal.define_cubic(self.box_size[0])
        # elif self.type.upper().startswith('OCTA'):
        #     crystal.define_octa(self.box_size[0])
        # elif self.type.upper().startswith('RHOM'):
        #     crystal.define_rhdo(self.box_size[0])
        if run != self.start:
            pycharmm.charmm_script('CRYSTAL FREE')
        pycharmm.charmm_script(f'CRYSTAL DEFINE {self.type} {" ".join(map(str, self.box_size))} {" ".join(map(str, self.angles))}')
            
        crystal.build(nb_param['cutim'])
        # pycharmm.charmm_script(script=f'CRYSTAL BUILD NOPEr 0 CUTOff {nb_param['cutim']}')
        if not cent_ncres:
            pycharmm.charmm_script('IMAGE BYRESid SELE        segid SOLV .or. segid IONS  END')
            pycharmm.charmm_script('IMAGE BYSEGid SELE .not. (segid SOLV .or. segid IONS) END')
        
        fft = open(f'{input_folder}/prep/fft.dat').read().strip().split()
        nb_param.update({'fftx': fft[0], 'ffty': fft[1], 'fftz': fft[2]})
        pycharmm.NonBondedScript(**nb_param).run()

    def minimization(self, run):
        
        if int(run) < 6:
            if os.path.isfile(f'{input_folder}/prep/system_min.crd') == True:
                print('Minimization already done, skipping.')
                read.coor_card(f'{input_folder}/prep/system_min.crd')
            else:
                minimize.run_abnr(nstep = 1000, tolenr = 1e-3, tolgrd=1e-3)
                write.coor_card(f'{input_folder}/prep/system_min.crd')     
                energy.show()
                #pycharmm.charmm_script('energy domdec gpu only dlb off ndir 1 1 1 ')
                pycharmm.charmm_script('energy blade')

    def dyn_init(self):
        pycharmm.charmm_script('faster on')
        pycharmm.charmm_script('blade on')
        # pycharmm.charmm_script('shake fast bonh param')
        shake.on(fast=True, bonh=True, param=True, tol=1e-7)
        # n = psf.get_natom()
        # scalar.set_fbetas([1.0] * n)
        gscale = 0.1
        dyn.set_fbetas(np.full(psf.get_natom(), gscale, dtype=float))

    def run_mkdir(self, run):
        os.makedirs(f'{input_folder}/run{run}', exist_ok=True)
        os.makedirs(f'{input_folder}/run{run}/dcd', exist_ok=True)
        os.makedirs(f'{input_folder}/run{run}/res', exist_ok=True)
        # os.mkdir(f'{input_folder}/run{run}')
        # os.mkdir(f'{input_folder}/run{run}/dcd')
        # os.mkdir(f'{input_folder}/run{run}/res')


    def read_selections(self, selections_file):
        print(self.patch_info)
        for index, row in self.patch_info.iterrows():
            name = row['SELECT']
            segid = row['SEGID']
            resid = row['RESID']
            resname = row['PATCH']
            atoms = '-\n(type ' + " -\n .or. type ".join(row["ATOMS"].split(' ')) + ')'
            pycharmm.charmm_script(f'DEFine {name} SELEction SEGID {segid} .AND. RESId {resid} .AND. RESName {resname} .AND. {atoms} END')


    def alf_initilize(self):

         # Read patches.info
        self.patch_info = pd.read_csv(os.path.join(input_folder, 'prep', 'patches.dat'), sep=',')
        self.patch_info[['site', 'sub']] = self.patch_info['SELECT'].str.extract(r's(\d+)s(\d+)')

        required_files = ['name', 'nsubs', 'nblocks', 'nreps', 'ncentral', 'engine']
        # if one of the required files is missing in f'{input_folder}/prep', initialize alf_info
        if not all([file in required_files for file in os.listdir(os.path.join(input_folder, 'prep'))]):


            self.alf_info = {}
            self.alf_info['name'] = input_folder.split('/')[-1]
            self.alf_info['nsubs'] = np.array([],dtype=int)
            self.alf_info['nblocks'] = 0
            self.alf_info['nreps'] = 1
            self.alf_info['ncentral'] = 0
            self.alf_info['nnodes'] = 1
            self.alf_info['temp'] = 298.15
            self.alf_info['engine'] = 'charmm'
            for site in self.patch_info['site'].unique():
                self.alf_info['nblocks'] += len(self.patch_info[self.patch_info['site'] == site]['sub'].unique()) 
                # how many subsites in this block
                self.alf_info['nsubs'] = np.append(self.alf_info['nsubs'], len(self.patch_info[self.patch_info['site'] == site]['sub'].unique()))
                
            for key in self.alf_info.keys():

                f = open(f'{input_folder}/prep/{key}', 'w') 
                if key == 'nsubs':
                    f.write(' '.join(map(str, self.alf_info[key])))
                else:
                    f.write(str(self.alf_info[key]))
                f.close()
            # copy a folder 'G_imp' from alf module to current directory
            try:
                alf_location = os.path.dirname(alf.__file__)
                shutil.copytree(f'{alf_location}/G_imp', f'{input_folder}/G_imp')
            except: 
                pass 
            home_dir = os.getcwd()
            try:
                os.chdir(f'{input_folder}')
                alf.InitVars(self.alf_info)
                alf.SetVars(self.alf_info,1)
            finally:
                os.chdir(home_dir)

    def manual_bias_adjustment(self, lam: list):
        # change the fixed bias in variables1.inp
        with open(f'{input_folder}/variables1.inp', 'r') as f:
            lines = f.readlines() 
        np.savetxt(f'{input_folder}/analysis0/b.dat', np.reshape(np.array(lam),(1,-1)))
        # np.savetxt(f'{pdb_name}/analysis0/b_prev.dat', np.reshape(np.array(lam),(1,-1)))
        # np.savetxt(f'{pdb_name}/nbshift/b_shift.dat', np.array(lam) ,fmt="%.18e", newline=" ")
        np.savetxt(f'{input_folder}/analysis0/b_sum.dat',np.reshape(np.array(lam),(1,-1)),fmt=' %7.2f')
        

        updated_lines = []    
        for line in lines:
            if line.startswith('set lams'):
                line_parts = line.split('=')
                line_prefix = line_parts[0].strip()
                current_value = float(line_parts[1].strip())
                new_value = '{:.2f}'.format(lam.pop(0))
                line = f"{line_prefix}= {new_value}\n"
            updated_lines.append(line)
            
        #Write the updated content back to variables1.inp
        with open(f'{input_folder}/variables1.inp', 'w') as file:
            file.writelines(updated_lines)

    def block(self, i, letter=''):
        run = i
        def read_variable_file(i):
            """
            This function reads variables{i}.inp file and returns a dictionary of variable names and values.
            i is the variable in the filename.
            """
            filename = f'{input_folder}/variables{i}.inp'
            variables = {}
            with open(filename, "r") as f:
                for line in f:
                    if line.startswith("set"):
                        # Remove the "set" keyword and any whitespace characters
                        line = line.replace("set", "").strip()
                        # Split the line into variable name and value
                        var_name, var_value = line.split("=")
                        # Remove any whitespace characters from variable name and value
                        var_name = var_name.strip()
                        var_value = var_value.strip()
                        # Convert variable value to float if possible
                        try:
                            var_value = float(var_value)
                        except ValueError:
                            pass
                        # Add variable to the dictionary
                        variables[var_name] = var_value
            return variables
        

        if i != self.start:
            block_command = 'BLOCK\nCLEAR\nEND\n'
        else:
            block_command = ''

        variables = read_variable_file(i)
        
        
        block_command += f'BLOCK {len(self.patch_info)+1} '
        if variables['nreps'] > 1:
            block_command += f'NREP {variables["nreps"]}\n\n'
        else:
            block_command += '\n'
        
        # Define blocks
        block_command += '!----------------------------------------\n'
        block_command += '! Set up l-dynamics by setting BLOCK parameters\n'
        block_command += '!----------------------------------------\n\n'
        
        # for index, row in self.patch_info.iterrows():
        #     block_command += f'CALL {index+2} SELEct {row["SELECT"]} END\n'
        block_command += ''.join(
            f'CALL {index+2} SELECT {row["SELECT"]} END\n'
            for index, row in self.patch_info.iterrows()
        )
        
        
        # Exclude blocks from each other
        block_command += '!----------------------------------------\n'
        block_command += '! Exclude blocks from each other\n'
        block_command += '!----------------------------------------\n\n'
        
        # exclude blocks with same RESID from each other
        # for index, row in self.patch_info.iterrows():
        #     for index2, row2 in self.patch_info.iterrows():
        #         if index2 > index and row['RESID'] == row2['RESID']:
        #             block_command += 'adexcl {:<3} {:<3}\n'.format(index+2, index2+2)
        
        block_command += ''.join(
            'adexcl {:<3} {:<3}\n'.format(index+2, index2+2)
            for index, row in self.patch_info.iterrows()
            for index2, row2 in self.patch_info.iterrows()
            if index2 > index and row['site'] == row2['site']
        )

        
        block_command += '\n'            
        block_command += '!----------------------------------------\n'
        block_command += '!QLDM turns on lambda-dynamics option\n'
        block_command += '!----------------------------------------\n\n'
        block_command += 'QLDM THETa\n\n'
        
        block_command += '!----------------------------------------\n'
        block_command += '!LANGEVIN turns on the langevin heatbath\n'
        block_command += '!----------------------------------------\n\n'
        block_command += 'LANG TEMP 298.15\n\n'
        
        if self.pH != None:
            block_command += '!----------------------------------------\n'
            block_command += '!Setup initial pH from PHMD\n'
            block_command += '!----------------------------------------\n\n'
            block_command += f'PHMD pH {self.pH}\n'
            
        block_command += '!----------------------------------------\n'
        block_command += '!Soft-core potentials\n'
        block_command += '!----------------------------------------\n\n'
        block_command += 'SOFT ON\n'
        
        block_command += '!----------------------------------------\n'
        block_command += '!lambda-dynamics energy constrains (from ALF) AKA fixed bias\n'
        block_command += '!----------------------------------------\n\n'
        
        
        if self.pH != None:
            block_command += 'LDIN {:<4} {:<4} {:<4} {:<4} {:<2} {:<4} {:<5}\n'.format(1, 1, 0.0, 12.0, 0.0, 5.0, 'NONE')
            
            for index, row in self.patch_info.iterrows():
                l0 = round(1/len(self.patch_info.loc[self.patch_info['site'] == row['site']]),2)
                block_command += 'LDIN {:<4} {:<4} {:<4} {:<4} {:<2} {:<4} {:<5}\n'.format(index+2, l0, 0.0, 12.0, variables['lam'+str(row['SELECT'])], 5.0, row['TAG'])                                                                    
                
        else:
            block_command += 'LDIN {:<4} {:<4} {:<4} {:<4} {:<2} {:<4}\n'.format(1, 1, 0.0, 12.0, 0.0, 5.0)
            for index, row in self.patch_info.iterrows():
                l0 = round(1/len(self.patch_info.loc[self.patch_info['site'] == row['site']]),2)
                block_command += 'LDIN {:<4} {:<4} {:<4} {:<4} {:<2} {:<4}\n'.format(index+2, l0, 0.0, 12.0, variables['lam'+str(row['SELECT'])], 5.0)
            
        block_command += '!------------------------------------------\n'
        block_command += '! All bond/angle/dihe terms treated at full str (no scaling),\n'
        block_command += '! prevent unphysical results'
        block_command += '!------------------------------------------\n\n'
        block_command += 'rmla bond theta impr\n\n'
        
        
        block_command += '!------------------------------------------\n'
        block_command += '! Selects MSLD, the numbers assign each block to the specified site on the core\n'
        block_command += '!------------------------------------------\n\n'
        block_command += 'MSLD 0 - \n'
        
        index = 1
        for select in self.patch_info['SELECT']:
            block_command += f'{select.split("s")[1]} '
            if select.split("s")[1] != self.patch_info['SELECT'].iloc[-1].split("s")[1]:
                block_command += ' -\n'
        block_command += '-\nfnex 5.5 \n\n'
            
        block_command += '!------------------------------------------\n'
        block_command += '! Constructs the interaction matrix and assigns lambda & theta values for each block\n'
        block_command += '!------------------------------------------\n\n'
        block_command += 'MSMA\n\n'
        
        block_command += '!------------------------------------------\n'
        block_command += '! PME for electrostatics\n'
        block_command += '!------------------------------------------\n\n'
        block_command += 'PMEL EX\n\n'
        
        block_command += '!------------------------------------------\n'
        block_command += '! Variable lambda potentials\n'
        block_command += '!------------------------------------------\n\n'
        
        # iterate over f with same resid
        ldbi = '!------------------------------------------\n'
        ldbi += '! Quadratic Barriers\n'
        ldbi += '!------------------------------------------\n\n'
        i = 0
        # for resid in self.patch_info['RESID'].unique():
        #     iter = self.patch_info.loc[self.patch_info['RESID'] == resid]

        for site in self.patch_info['site'].unique():
            iter = self.patch_info.loc[self.patch_info['site'] == site]
            for index, row in iter.iterrows():
                for index2, row2 in iter.iterrows():
                    if index < index2:
                        index_1 = index + 2
                        index_2 = index2 + 2
                        ldbv = str(row['SELECT'])+str(row2['SELECT'])
                        if index2 > index:
                            try:
                                i += 1
                                ldbi += 'ldbv {:<3} {:<2} {:<2} {:<4} {:<8} {:<6} {:<1}\n'.format(i, index_1, index_2, 6, 0.0, variables["c"+ldbv], 0)
                            except: pass
        
        ldbi += '!------------------------------------------\n'
        ldbi += '!End point Potentials\n'
        ldbi += '!------------------------------------------\n\n'
        for site in self.patch_info['site'].unique():
            iter = self.patch_info.loc[self.patch_info['site'] == site]
            for index, row in iter.iterrows():
                for index2, row2 in iter.iterrows():
                    index_1 = index + 2
                    index_2 = index2 + 2
                    ldbv = str(row['SELECT'])+str(row2['SELECT'])
                    
                    if index2 != index:
                        try:
                            i += 1
                            ldbi += 'ldbv {:<3} {:<2} {:<2} {:<4} {:<8} {:<6} {:<1}\n'.format(i, index_1, index_2, 8, 0.017, variables["s"+ldbv], 0)
                        except: pass
        ldbi += '!------------------------------------------\n'
        ldbi += '! Skew Potentials\n'
        ldbi += '!------------------------------------------\n\n'                  
        for site in self.patch_info['site'].unique():
            iter = self.patch_info.loc[self.patch_info['site'] == site]
            for index, row in iter.iterrows():
                for index2, row2 in iter.iterrows():
                    index_1 = index + 2
                    index_2 = index2 + 2
                    ldbv = str(row['SELECT'])+str(row2['SELECT'])
                    if index2 != index:
                        try:
                            i += 1
                            ldbi += 'ldbv {:<3} {:<2} {:<2} {:<4} {:<8} {:<6} {:<1}\n'.format(i, index_1, index_2, 10, -5.56, variables["x"+ldbv], 0)
                        except: pass
        block_command += f'LDBI {i}\n' +ldbi + '\n'
        block_command += 'END'
        # print(f'{input_folder}/run{run}/block{letter}.str')
        with open(f'{input_folder}/run{run}/block{letter}.str', 'w') as f:
            f.write(block_command)
        # print(block_command)
        # settings.set_bomb_level(-5)
        pycharmm.charmm_script(block_command)
        # settings.set_bomb_level(0)
        
    def scat(self, hydrogen = False):
        
        scat_command = 'BLOCK\n scat on\nscat k 300\n'        
        for site in self.patch_info['site'].unique():
            atoms =  self.patch_info.loc[self.patch_info['site'] == site]['ATOMS']
            # get all unique atoms in the site
            atoms = set([atom for atom in atoms.str.split().sum()])
            if hydrogen == False:
                atoms = [atom for atom in atoms if atom.startswith('H') == False]
            for atom in atoms:
                scat_command += f'cats SELE type {atom} .and. ({" .or. ".join(map(str, self.patch_info.loc[self.patch_info["site"] == site]["SELECT"]))}) END\n'

        scat_command += 'END'
        if os.path.isfile(f'{input_folder}/prep/restrains.str') == False:
            with open(f'{input_folder}/prep/restrains.str', 'w') as f:
                f.write(scat_command)
        pycharmm.charmm_script(scat_command)
    
        
        
    
    def dynamics(self, run, letter=''):
        global cpt_on, temperature
        if run == self.start:
            settings.set_verbosity(5)
            self.dyn_init()
        else: settings.set_verbosity(1)
        
        if self.phase == 1:
            nsteps_eq = 10000 # 20 ps
            nsteps_prod = 40000 # 80 ps
            nsavc = 1000 # 2 ps
            nsavl = 1 # 2 fs

        elif self.phase == 2:
            nsteps_eq = 100000 # 200 ps
            nsteps_prod = 400000 # 800 ps
            nsavc = 10000 # 2 ps
            nsavl = 1 # 2 fs

        elif self.phase == 3:
            nsteps_eq = 0 # 0 ps
            nsteps_prod = 5000000 # 1000 ps
            nsavc = 10000 # 2 ps
            nsavl = 10 # 2 fs
        
        else:
            raise ValueError('Phase must be 1, 2 or 3')
        
        # Set up dynamics parameters
        # dyn_param = {'start': True, 
        #              'restart': False,
        #              'blade': True,
        #              'cpt': cpt_on, 
        #              'leap': True,
        #             'langevin': True, 
        #             'timestep': 0.002,
        #             'firstt': temperature, 
        #             'finalt': temperature,
        #             'tstruc': temperature, 
        #             'tbath': temperature,
        #             'iasors': 1, 
        #             'iasvel': 1, 
        #             'iscvel': 1, 
        #             'iscale': 0,
        #             'ihtfrq': 0, 
        #             'ieqfrq': 0, 
        #             'ichecw': 0,
        #             'imgfrq': -1, 
        #             'ilbfrq': 0,
        #             'echeck': -1, 
        #             'iunldm': self.lmd_unit,
        #             'iunwri': self.rst_unit,
        #             'iuncrd': self.dcd_unit
        #             }
        
        dyn_param = {
            'start': True, 
            'restart': False,
            'blade': True,
            'prmc': True,
            'iprs': 100,
            'prdv': 100,
            'cpt': cpt_on, 
            'timestep': 0.002,
            'firstt': temperature, 
            'finalt': temperature,
            'tstruc': temperature, 
            'tbath': temperature,
            'ichecw': 0, #do not scale velocities to final temp (i.e. equilibrate)
            'ihtfrq': 0, #frequency of heating
            'ieqfrq': 0, #frequence of scaling/assigning velocities
            'iasors': 1, #assign velocities during heating (0 will be scale velocities)
            'iasvel': 1, #using gaussian distribution to assign velocities
            'iscvel': 0,
            'inbfrq': 0, # BLaDE does it's own neighbor searching
            'ilbfrq': 0,
            'imgfrq': 0, # BLADE does it's own neighbor searching
            'ntrfrq': 0,
            'echeck': -1, # energy tolerance before crash
            'iunldm': self.lmd_unit,
            'iunwri': self.rst_unit,
            'iuncrd': self.dcd_unit
                    }
        # 'leap': True, # some code doesn't have this flag
        # 'langevin': True, # some code doesn't have this flag
        # 'iscale': 0, # some code doesn't have this flag
        
        if cpt_on:
            dyn_param.update({
                'pconstant': True, 'pmass': psf.get_natom() * 0.12,
                'pref': 1.0, 'pgamma': 20.0,
                'hoover': True, 'reft': temperature, 'tmass': 1000
            })
        

        # add item to dyn_param
        if cent_ncres:
            molecule = (pycharmm.SelectAtoms().by_seg_id('SOLV') | 
                        pycharmm.SelectAtoms().by_seg_id('IONS'))
            molecule = molecule.__invert__()
            n_molecule = len(list(set(molecule.get_res_ids())))
            molecule.unstore()
            dyn_param.update({'cent ncres': n_molecule})
        
        
        if hmr:
            nsteps_eq = nsteps_eq // 2
            nsteps_prod = nsteps_prod // 2
            nsavc = nsavc // 2
            nsavl = nsavl // 2
            nsavl = max(nsavl, 1)
            dyn_param.update({'timestep': 0.004})
        
        if self.phase == 3:
            dyn_param.update({
                'start': False,
                'restart': True,
                'iunrea': self.rpr_unit})
            
        dyn_param.update({'nsavc': nsavc, 'nsavl': nsavl,
                         'nprint': nsavc,'iprfrq': nsavc,
                         'isvfrq': nsavc
                         })
        
        # Equalibration Run
        if nsteps_eq > 0:
            dcd_fn = f'{input_folder}/run{run}/dcd/{name}_eq.dcd{letter}'
            rst_fn = f'{input_folder}/run{run}/res/{name}_eq.rst{letter}'
            lmd_fn = f'{input_folder}/run{run}/res/{name}_eq.lmd{letter}'

            dcd = pycharmm.CharmmFile(file_name = dcd_fn, file_unit = self.dcd_unit, read_only = False, formatted = False)
            rst = pycharmm.CharmmFile(file_name = rst_fn, file_unit = self.rst_unit, read_only = False, formatted = 'formatted')
            # lingo.charmm_script(f'OPEN WRITE UNIT {self.rst_unit} CARD NAME {rst_fn}')
            lmd = pycharmm.CharmmFile(file_name = lmd_fn, file_unit = self.lmd_unit, read_only = False, formatted = False)

            dyn_param.update({'nstep': nsteps_eq})
            pycharmm.DynamicsScript(**dyn_param).run()
            dcd.close()
            rst.close()
            lmd.close()
            if not cent_ncres:
                dcd = pycharmm.CharmmFile(f'{input_folder}/run{run}/dcd/{name}_eq.dcd{letter}', 
                                          file_unit=self.dcd_unit,read_only=True, formatted=False)
                dcd_new = pycharmm.CharmmFile(f'{input_folder}/run{run}/dcd/{name}_eq_1.dcd{letter}',
                                              self.dcd_unit+5, False, False)
                lingo.charmm_script(f'merge first {self.dcd_unit} nunit 1 output {self.dcd_unit+5} sele all end recenter -\n sele .not. (segid SOLV .or. segid IONS) .and. .not. hydrogen end')
                os.remove(f'{input_folder}/run{run}/dcd/{name}_eq.dcd{letter}')
                os.rename(f'{input_folder}/run{run}/dcd/{name}_eq_1.dcd{letter}', f'{input_folder}/run{run}/dcd/{name}_eq.dcd{letter}')
                dcd.close()
                dcd_new.close()
        # lingo.charmm_script('energy blade')
        
        # Production Run
        if nsteps_prod > 0:
            if self.phase ==1 or self.phase == 2:
                sim_type = 'flat'
            elif self.phase == 3:
                sim_type = 'prod'
            dcd_fn = f'{input_folder}/run{run}/dcd/{name}_{sim_type}.dcd{letter}'
            lmd_fn = f'{input_folder}/run{run}/res/{name}_{sim_type}.lmd{letter}'
            rst_fn = f'{input_folder}/run{run}/res/{name}_{sim_type}.rst{letter}'
            dyn_param.update({'start': False, 'restart': True, 'iunrea': self.rpr_unit})
            
            if sim_type == 'flat' and os.path.isfile(f'{input_folder}/run{run}/res/{name}_eq.rst{letter}'):
                rpr_fn = f'{input_folder}/run{run}/res/{name}_eq.rst{letter}'
            elif os.path.isfile(f'{input_folder}/run{self.restart_run}/res/{name}_prod.rst{letter}'):
                rpr_fn = f'{input_folder}/run{self.restart_run}/res/{name}_prod.rst{letter}'
            elif os.path.isfile(f'{input_folder}/run{self.restart_run}/res/{name}_flat.rst{letter}'):
                rpr_fn = f'{input_folder}/run{self.restart_run}/res/{name}_flat.rst{letter}'
            elif os.path.isfile(f'{input_folder}/run{self.restart_run}/res/{name}_prod.rst'):
                rpr_fn = f'{input_folder}/run{self.restart_run}/res/{name}_prod.rst'
            elif os.path.isfile(f'{input_folder}/run{self.restart_run}/res/{name}_flat.rst'):
                rpr_fn = f'{input_folder}/run{self.restart_run}/res/{name}_flat.rst'
            else:
                print(f'No restart file found for {name}')
                dyn_param.update({'start': True, 'restart': False})
                dyn_param.popitem('iunrea')
            rpr = pycharmm.CharmmFile(file_name = rpr_fn, file_unit = self.rpr_unit, read_only = True, formatted = 'formatted')
            dcd = pycharmm.CharmmFile(file_name = dcd_fn, file_unit = self.dcd_unit, read_only = False, formatted = False)
            rst = pycharmm.CharmmFile(file_name = rst_fn, file_unit = self.rst_unit, read_only = False, formatted = 'formatted')
            lmd = pycharmm.CharmmFile(file_name = lmd_fn, file_unit = self.lmd_unit, read_only = False, formatted = False)

            dyn_param.update({'nstep': nsteps_prod})

            pycharmm.DynamicsScript(**dyn_param).run()
            dcd.close()
            lmd.close()
            rst.close()
            rpr.close()
            if not cent_ncres:
                dcd = pycharmm.CharmmFile(f'{input_folder}/run{run}/dcd/{name}_{sim_type}.dcd{letter}', 
                                          file_unit=self.dcd_unit,read_only=True, formatted=False)
                dcd_new = pycharmm.CharmmFile(f'{input_folder}/run{run}/dcd/{name}_{sim_type}_1.dcd{letter}',
                                              self.dcd_unit+5, False, False)
                lingo.charmm_script(f'merge first {self.dcd_unit} nunit 1 output {self.dcd_unit+5} sele all end recenter -\n sele .not. (segid SOLV .or. segid IONS) .and. .not. hydrogen end')
                os.remove(f'{input_folder}/run{run}/dcd/{name}_{sim_type}.dcd{letter}')
                os.rename(f'{input_folder}/run{run}/dcd/{name}_{sim_type}_1.dcd{letter}', f'{input_folder}/run{run}/dcd/{name}_{sim_type}.dcd{letter}')
                dcd.close()
                dcd_new.close()
            pycharmm.charmm_script('! Simulation Complete')
            settings.set_verbosity(5)
            lmd = pycharmm.CharmmFile(file_name=lmd_fn, file_unit = self.lmd_unit, read_only = True, formatted = False)
            # pycharmm.charmm_script(f'traj lamb print ctlo 0.95 cthi 0.99 first {self.lmd_unit} nunit 1')
            lmd.close()
        #BOLD RED checkmark: calculating the box size
        self.box_size[0] = pycharmm.get_energy_value('XTLA')
        self.box_size[1] = pycharmm.get_energy_value('XTLB')
        self.box_size[2] = pycharmm.get_energy_value('XTLC')
        self.box_size = [max(self.box_size), max(self.box_size), max(self.box_size)]
        write.coor_card(f'{input_folder}/run{run}/prod.crd{letter}')
        f = open(f'{input_folder}/run{run}/box.dat{letter}', 'w')
        f.write(f'{self.type}\n')
        f.write(f'{self.box_size[0]} {self.box_size[1]} {self.box_size[2]}\n')
        f.write(f'{self.angles[0]} {self.angles[1]} {self.angles[2]}')
        f.close()

    def alf_analysis(self,run):
        try:
            home_path = os.getcwd()
            os.chdir(f'{input_folder}')
            print(os.getcwd())
            im5 = max(run-5,1)
            N = run - im5 + 1
            print(f'Analysis{run} started')
            if not os.path.exists(f'analysis{run}'):
                os.mkdir(f'analysis{run}')
            shutil.copy(f'analysis{run-1}/b_sum.dat',f'analysis{run}/b_prev.dat')
            shutil.copy(f'analysis{run-1}/c_sum.dat',f'analysis{run}/c_prev.dat')
            shutil.copy(f'analysis{run-1}/x_sum.dat',f'analysis{run}/x_prev.dat')
            shutil.copy(f'analysis{run-1}/s_sum.dat',f'analysis{run}/s_prev.dat')

            # Weighted addition of the bias
            # for site in self.patch_info['site'].unique():p
            #     tags = self.patch_info.loc[self.patch_info['site'] == site]['TAG']
                # tags can be 'NONE', 'UPOS int' or 'UNEG int'
                # if it it is UPOS, it's positive value of pKa, if it is UNEG, it's negative value of pKa
                # if it is NONE, it's reference state
                

            if not os.path.exists(f'analysis{run}/G_imp'):
                G_imp_dir = os.path.join(os.getcwd(),'G_imp')
                target_path = os.path.join(os.getcwd(),f'analysis{run}/G_imp')
                try:
                    os.symlink(G_imp_dir,target_path)
                except FileExistsError:
                    pass

            os.chdir(path=f'analysis{run}')
            self.alf_info['nreps'] = self.phase_runs
            os.makedirs('data', exist_ok=True)
            for j in range(0,self.phase_runs):
                if self.phase_runs == 1:
                    letter = ''
                else:
                    letter = '_' + self.alphabet[j]
                if self.phase in [1,2]:
                    fnmsin = [f'../run{run}/res/{name}_flat.lmd{letter}']
                else:
                    fnmsin = [f'../run{run}/res/{name}_prod.lmd{letter}']
                fnmout = f'data/Lambda.0.{j}.dat'
                alf.GetLambda.GetLambda(self.alf_info, fnmout, fnmsin)
            # check how many previous folders have same structure
            alf.GetEnergy(self.alf_info, im5, run)

            # load all Lambda values
            lambda_data = None
            l_files = os.listdir('data')
            l_files = [file for file in l_files if file.startswith('Lambda')]
            for file in l_files:
                l = np.loadtxt(f'data/{file}')
                if lambda_data is None:
                    lambda_data = l
                else:
                    lambda_data = np.concatenate((lambda_data, l), axis=0)
            col_count = np.sum(lambda_data > 0.99, axis=0) / lambda_data.shape[0]
            # if np.all(np.abs(col_count - col_count[0]) < 0.03) and self.phase != 1:
            #     self.fix_bias = True
            #     print('All columns are converged. Fixing G bias.')
            if np.all(np.abs(np.subtract.outer(col_count, col_count)) < 0.1) and np.all(col_count > 0.03) and self.phase == 1:
                self.phase = 2
                print(f'Run {run}: Moving to phase 2.')
            # if np.all(np.abs(np.subtract.outer(col_count, col_count)) < 0.03) and np.all(col_count > 0.03) and self.phase == 2:
                # self.phase = 3
                # print(f'Run {run}: Moving to phase 3.')

            # fpout = open('output.dat','w')
            # fperr = open('error.dat','w')
            # check how many variables alf.RunWham takes, it should be 3 or 4
            N_sims = len(os.listdir('Energy'))
            # if alf.RunWham.__code__.co_argcount == 3 and self.fix_bias == False:
            #     alf.RunWham(nf=N*self.alf_info['nreps'], 0, 0)
            # elif alf.RunWham.__code__.co_argcount == 3 and self.fix_bias == True:
            #     alf.RunWham(N*self.alf_info['nreps'], 1, 0)
            # if alf.RunWham.__code__.co_argcount == 4 and self.fix_bias == False:
            #     alf.RunWham(N_sims, self.alf_info['temp'], 0, 0)
            # elif alf.RunWham.__code__.co_argcount == 4 and self.fix_bias == True:
            #     alf.RunWham(N*self.alf_info['nreps'], self.alf_info['temp'], 1, 0)
            # else:
                # raise ValueError('RunWham function takes wrong number of arguments')
            fpout = open('output.dat','w')
            fperr = open('error.dat','w')
            cmd = f"python -c 'import alf; alf.RunWham({N*self.alf_info['nreps']},{self.alf_info['temp']}, 0, 0)'"
            cmd = f"python -c 'import alf; alf.RunWham({N_sims},{self.alf_info['temp']}, 0, 0)'"
            subprocess.call(cmd, shell=True, stdout=fpout, stderr=fperr)
            fpout.close()
            fperr.close()
            if self.phase == 1 and run < 10:
                alf.GetFreeEnergy5(alf_info=self.alf_info, ms=0,msprof=0, cutb=3.5)
            elif self.phase == 1 and run >= 10 and run < 50:
                alf.GetFreeEnergy5(self.alf_info, ms=0,msprof=0, cutb=2.5)
            elif self.phase == 1:
                alf.GetFreeEnergy5(self.alf_info, ms=0,msprof=0, cutb=1)
            if self.phase == 2:
                alf.GetFreeEnergy5(self.alf_info, ms=0,msprof=0, cutb=0.05)
            if self.phase == 3:
                alf.GetFreeEnergy5(self.alf_info, ms=0,msprof=0, cutb=0.02)
            alf.SetVars(self.alf_info, run+1)
            try:
                shutil.rmtree('Energy')
                shutil.rmtree('Lambda')
            except FileNotFoundError:
                pass  # Ignore if the directory doesn't exist

        except:
            print(f'Analysis{run} failed')
            raise Exception(f'Analysis{run} failed')
        finally:
            os.chdir(home_path)
            sys.stdout.flush()
            # if run > 5:
            #     shutil.rmtree(f'{input_folder}/analysis{run-6}')
        

            

# md1 = ALF_simulation()
# md1.run()

ALF_simulation().run()
# %%

# %%
sys.stdout.close()