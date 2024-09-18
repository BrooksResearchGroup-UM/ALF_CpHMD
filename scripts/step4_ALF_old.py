# In[1]: Importing modules
import os
import sys
import subprocess
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
import pycharmm.nbonds as nbonds
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

# Checking CHARMM executable
path_charmm = re.search(r"'(.*?)'", str(pycharmm.lib.__dict__['charmm']))
if path_charmm is not None:
    print('CHARMM executable is: ', path_charmm.group(1))
else:
    print('CHARMM executable path not found.')

#
#
# In[2]: arguments
args = {
    'input_folder': 'preparation/his'
}


# Get arguments as flags from command line
parser = argparse.ArgumentParser(description='ALF for single AA')
parser.add_argument('-i', '--input', dest='input_folder', default=args['input_folder'], help='Input folder')
# skip unknown arhuments
args = parser.parse_known_args()[0]

# input folder
input_folder = args.input_folder
print('Input folder: ', input_folder)
name = input_folder.split('/')[-1]

required_files = ['system.crd', 'system.psf', 'patches.dat', 'box.dat', 'fft.dat']

# check if all required files are present and not empty
for file in required_files:
    if not os.path.isfile(os.path.join(input_folder, 'prep', file)):
        raise FileNotFoundError(f'File {file} not found in {input_folder}')
    if os.path.getsize(os.path.join(input_folder, 'prep', file)) == 0:
        raise FileNotFoundError(f'File {file} is empty')
    


# In[3]: CHARMM topology and parameter files
if __name__ == '__main__':
    # Set up the CHARMM environment and read in the parameters
    lingo.charmm_script('prnlev 1')
    read.rtf('toppar/top_all36_prot.rtf')
    read.prm('toppar/par_all36m_prot.prm', flex=True)
    lingo.charmm_script('stream toppar/toppar_water_ions.str')
    lingo.charmm_script('stream toppar_my/protpatch_protein_segments.str')
    lingo.charmm_script('prnlev 5')
    lingo.charmm_script('IOFOrmat EXTEnded')

# In[4]: Simulation class
class ALF_simulation:
    def __init__(self, temperature = 298.15, pH = None, phase = 1, start = 1, end = 50):
        self.pH = pH
        self.temperature = temperature
        self.phase = phase
        self.start = start
        self.end = end
        self.box_size = [0, 0, 0]
        self.type = None
        self.angles = [90, 90, 90]

        # CHARMM Units
        self.log_unit = 30
        self.dcd_unit = 40
        self.lmd_unit = 50
        self.rpr_unit = 60
        self.rst_unit = 70

        # CHARMM Output
        self.clog = None

        self.homedir = os.getcwd()


        # Read patches dictionary
        self.patches = {}
        self.atom_groups = {}
        self.patch_dict()

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
            print(f'Running simulation {i}')
            start_time = time.time()
            self.run_mkdir(i)
            # self.redirect_output(i)
            self.crystal(i)
            self.block(i)
            self.scat()
            self.minimization(i)
            self.dynamics(i)
            self.alf_analysis(i)
            # self.return_output()
            end_time = time.time()
            print(f'Run {i} completed in {round(end_time-start_time, 1)} seconds')
    


    def redirect_output(self, run):
        self.clog = charmm_file.CharmmFile(file_name=f'{input_folder}/run{run}/log.out', file_unit=self.log_unit , read_only=False, formatted=True)
        pycharmm.charmm_script(f'OUTUnit {str(self.log_unit)}')
    
    def return_output(self):
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

    def crystal(self, run):
        settings.set_verbosity(5)

        if run == self.start:
            try:
                psf.delete_atoms(pycharmm.SelectAtoms.all_atoms())
                pycharmm.charmm_script('CRYSTAL FREE')
            except:
                pass
            read.psf_card(f'{input_folder}/prep/system.psf')
            self.read_selections(f'{input_folder}/prep/patches.dat')
            self.alf_initilize()

        if run > 5:
            restart_run = random.randint(run-5, run-1)
        else: 
            restart_run = 1

        file = open(f'{input_folder}/prep/box.dat').readlines()
        self.type = file[0].strip()
        self.box_size = list(map(float, file[1].strip().split()))
        self.angles = list(map(float, file[2].strip().split()))
        if restart_run == 1:
            read.coor_card(f'{input_folder}/prep/system.crd')
        else:
            pycharmm.charmm_script('CRYSTAL FREE')
            read.coor_card(f'{input_folder}/run{restart_run}/prod.crd')
            file = open(f'{input_folder}/run{restart_run}/box.dat').readlines()
            self.box_size = list(map(float, file[0].strip().split()))
            
        
        
        # crystal.define_cubic(box_size)
        # crystal.build(12)
        
        pycharmm.charmm_script(f'CRYSTAL DEFINE {self.type} {" ".join(map(str, self.box_size))} {" ".join(map(str, self.angles))}')
        pycharmm.charmm_script(f'CRYSTAL BUILD NOPEr 0')

        pycharmm.charmm_script('IMAGE BYRESid SELE        segid SOLV .or. segid IONS  END')
        pycharmm.charmm_script('IMAGE BYSEGid SELE .not. (segid SOLV .or. segid IONS) END')

        fft = open(f'{input_folder}/prep/fft.dat').read().strip().split()

        nbonds = {'elec': True,
                'atom': True,
                'cdie': True,
                'eps': 1,
                'vdw': True,
                'vatom': True,
                'vfswitch': True,
                'cutnb': 15,
                'cutim': 15,
                'ctofnb': 12,
                'ctonnb': 10,
                'ewald': True,
                'pmewald': True,
                'kappa': 0.320,
                'order': 6,
                'fftx': fft[0],
                'ffty': fft[1],
                'fftz': fft[2]
                }
        pycharmm.NonBondedScript(**nbonds).run()

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
        pycharmm.charmm_script('shake fast bonh param')
        n = psf.get_natom()
        scalar.set_fbetas([1.0] * n)

    def run_mkdir(self, run):
        try:
            os.mkdir(f'{input_folder}/run{run}')
            os.mkdir(f'{input_folder}/run{run}/dcd')
            os.mkdir(f'{input_folder}/run{run}/res')
        except:
            pass

    def read_selections(self, selections_file):
        with open(selections_file) as f:
            lines = f.readlines()
        for line in lines[1:]:
            segid, resid, resname, name = line.split(',')[0:4]
            pycharmm.charmm_script(f'DEFine {name} SELEction SEGID {segid} .AND. RESId {resid} .AND. RESName {resname} END')

    def patch_dict(self):
            
            #patches 
            with open('toppar_my/protpatch_protein_segments.str') as f:
                lines = f.readlines()

            for i in range(len(lines)):
                # search for resname in ( ), e.g. (GLU)
                if lines[i].startswith("!") and (lines[i].endswith("PATCHES\n") or lines[i].endswith("patches\n")):
                    
                    # find matching resname 
                    resname = re.search(r'\((\w+)\)',lines[i])
                    if resname is not None:
                        resname = resname.group(1).upper()
                    else:
                        continue
                    
                    # add resname to dictionary if not present
                    if resname not in self.patches:
                        self.patches[resname] = []
                    if resname not in self.atom_groups:
                        self.atom_groups[resname] = []
                    
                    # find patch names for each resname
                    pka = {}
                    for j in range(i+1, len(lines)):
                        if lines[j].startswith("!") and (lines[j].endswith("PATCHES\n") or lines[j].endswith("patches\n")):
                                    break
                        if lines[j].startswith("pres") or lines[j].startswith("PRES"):
                            patch_name = lines[j].split()[1].upper()
                                    
                            if patch_name not in self.patches[resname]:
                                self.patches[resname].append(patch_name)
                            if patch_name not in self.atom_groups:
                                self.atom_groups[patch_name] = []

                            # find atom names for each patch
                            for k in range(j+1, len(lines)):
                                if lines[k].startswith("pres") or lines[k].startswith("PRES"):
                                    break
                                if lines[k].lower().__contains__("pka"):
                                    pka[patch_name] = lines[k].split('=')[1].strip()
                                if lines[k].startswith("atom") or lines[k].startswith("ATOM"):
                                    atom = lines[k].split()[1].upper()
                                    self.atom_groups[patch_name].append(atom)
                                    if atom not in self.atom_groups[resname]:
                                        self.atom_groups[resname].append(atom)
                            
            # Now we have dictionary with resnames as keys
            # and patch names as values and dictionary with patch names as keys and atom names as values
            # The values for resnames are the same as for patch names, so we need to check it with original topology file


            # find lines starting with 'RESI {amino_acid}', look for 'BOND', 
            # and found ones which are have connections to atoms and are hydrogens in patches
            # BOND CB CA  CG CB  CD CG  OE2 CD  -> each two atoms are connected


            with open('toppar/top_all36_prot.rtf') as f:
                lines = f.readlines()

            for i in range(len(lines)):
                if lines[i].startswith(f"RESI") or lines[i].startswith(f"resi"):
                    resname = lines[i].split()[1].upper()
                    if resname not in self.patches.keys():
                        continue
                    atoms = []
                    # print(f'Checking {resname}')
                    for j in range(i+1, len(lines)):
                        # break if next resname is found
                        if lines[j].startswith("RESI") or lines[j].startswith("PRES"):
                            break
                        # find atoms in resname
                        if lines[j].startswith("ATOM"):
                            atoms.append(lines[j].split()[1].upper())
                        # find bonds in resname
                        if lines[j].startswith("BOND"):
                            bonds = lines[j].split()[1:]
                            for k in range(0, len(bonds), 2):
                                if bonds[k] in self.atom_groups[resname] and bonds[k].startswith('H') == False:
                                    if bonds[k+1] not in self.atom_groups[resname] and bonds[k+1].startswith('H'):
                                        # print(f'added {bonds[k+1]} to {resname}:{patches[resname]}')
                                        self.atom_groups[resname].append(bonds[k+1])
                                if bonds[k+1] in self.atom_groups[resname] and bonds[k+1].startswith('H') == False:
                                    if bonds[k] not in self.atom_groups[resname] and bonds[k].startswith('H'):
                                        # print(f'added {bonds[k]} to {resname}:{patches[resname]}')
                                        self.atom_groups[resname].append(bonds[k])
                    
                    
                    # remove atoms which are not in the residue
                    self.atom_groups[resname] = [atom for atom in self.atom_groups[resname] if atom in atoms]

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

    def block(self, i):
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
        
        for index, row in self.patch_info.iterrows():
            block_command += f'CALL {index+2} SELEct {row["SELECT"]} END\n'
        
        
        # Exclude blocks from each other
        block_command += '!----------------------------------------\n'
        block_command += '! Exclude blocks from each other\n'
        block_command += '!----------------------------------------\n\n'
        
        # exclude blocks with same RESID from each other
        for index, row in self.patch_info.iterrows():
            for index2, row2 in self.patch_info.iterrows():
                if index2 > index and row['RESID'] == row2['RESID']:
                    block_command += 'adexcl {:<3} {:<3}\n'.format(index+2, index2+2)
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
                l0 = round(1/len(self.patch_info.loc[self.patch_info['RESID'] == row['RESID']]),2)
                block_command += 'LDIN {:<4} {:<4} {:<4} {:<4} {:<2} {:<4} {:<5}\n'.format(index+2, l0, 0.0, 12.0, variables['lam'+str(row['SELECT'])], 5.0, row['TAG'])                                                                    
                
        else:
            block_command += 'LDIN {:<4} {:<4} {:<4} {:<4} {:<2} {:<4}\n'.format(1, 1, 0.0, 12.0, 0.0, 5.0)
            for index, row in self.patch_info.iterrows():
                l0 = round(1/len(self.patch_info.loc[self.patch_info['RESID'] == row['RESID']]),2)
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
        for resid in self.patch_info['RESID'].unique():
            iter = self.patch_info.loc[self.patch_info['RESID'] == resid]
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
        for resid in self.patch_info['RESID'].unique():
            iter = self.patch_info.loc[self.patch_info['RESID'] == resid]
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
        for resid in self.patch_info['RESID'].unique():
            iter = self.patch_info.loc[self.patch_info['RESID'] == resid]
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
        # print(block_command)
        # settings.set_bomb_level(-5)
        pycharmm.charmm_script(block_command)
        # settings.set_bomb_level(0)
        
    def scat(self, hydrogen = False):
        
        scat_command = 'BLOCK\n scat on\nscat k 300\n'
        for resid in self.patch_info['RESID'].unique():
            atoms = []
            resname = self.patch_info.loc[self.patch_info['RESID'] == resid]['PATCH'].iloc[0][:-1]
            if resname == 'ASH':
                resname = 'ASP'
            elif resname == 'GLH':
                resname = 'GLU'
            for patch in [resname]+self.patches[resname]:
                atoms += self.atom_groups[patch]
            atoms = set(atoms)
            for atom in atoms:
                if atom.startswith('H') == False and hydrogen == False:
                    scat_command += f'cats SELE type {atom} .and. resid {resid} .and. segid PROA END\n'
        
        scat_command += 'END'
        pycharmm.charmm_script(scat_command)
        
    def dynamics(self, run):
        if run == self.start:
            settings.set_verbosity(5)
            self.dyn_init()
        else: settings.set_verbosity(1)
        
        if self.phase == 1:
            nsteps_eq = 10000 # 20 ps
            nsteps_prod = 40000 # 80 ps
            nsavc = 1000 # 2 ps
            nsavl = 10 # 2 fs

        elif self.phase == 2:
            nsteps_eq = 100000 # 200 ps
            nsteps_prod = 400000 # 800 ps
            nsavc = 10000 # 2 ps
            nsavl = 10 # 2 fs

        elif self.phase == 3:
            nsteps_eq = 0 # 0 ps
            nsteps_prod = 5000000 # 1000 ps
            nsavc = 10000 # 2 ps
            nsavl = 10 # 2 fs
        
        else:
            raise ValueError('Phase must be 1, 2 or 3')
        

        dyn_dict = {
                'leap': True,
                'verlet': False,
                'cpt': False,
                'new': False,
                'langevin': False,
                'omm': False,
                'blade': True,
                'start': True,
                'timestep': 0.002,
                'nstep': 0,
                'nsavc': 0,
                'nsavv': 0,
                'nsavl':  0,    # frequency for saving lambda values in lamda-dynamics
                'nprint': 0,  # Frequency to write to output
                'iprfrq': 0,  # Frequency to calculate averages
                'isvfrq': 0,  # Frequency to save restart file
                'ntrfrq': 0,
                'inbfrq': -1,
                'ihbfrq': 0,
                'ilbfrq': 0,
                'imgfrq': -1,
                'iunrea': -1,
                'iunwri': self.rst_unit,
                'iuncrd': self.dcd_unit,
                'iunldm': self.lmd_unit,
                'firstt': self.temperature,
                'finalt': self.temperature,
                'tstruct': self.temperature,
                'tbath': self.temperature,
                'prmc': True,
                'iprs': 10,  # Monte carlo pressure frequency (50 is default)
                'pref': 1,   # reference pressure (atmospheres)
                'prdv': 100,  # RMS proposed volume change (A cubed)
                'iasors': 1,  # assign velocities
                # method for assignment of velocities during heating & equil when IASORS is nonzero.
                'iasvel': 1,
                            # This option also controls the initial assignment of velocities
                'iscale': 0,  # not scale velocities on a restart
                'scale': 1,  # scaling factor for velocity scaling
                'ichecw': 0,  # not check temperature
                'echeck': -1  # not check energy
            }
        
        if self.phase == 3:
            dyn_dict.update({
                'start': False,
                'restart': True,
                'iunrea': self.rpr_unit})
            
        dyn_dict.update({'nsavc': nsavc, 
                         'nsavl': nsavl, 
                         'nprint': nsavc,
                         'iprfrq': nsavc,
                         'isvfrq': nsavc
                         })
        
        # Equalibration Run
        if nsteps_eq > 0:
            dcd_fn = f'{input_folder}/run{run}/dcd/{name}_eq.dcd'
            rst_fn = f'{input_folder}/run{run}/res/{name}_eq.rst'
            lmd_fn = f'{input_folder}/run{run}/res/{name}_eq.lmd'

            dcd = pycharmm.CharmmFile(file_name = dcd_fn, file_unit = self.dcd_unit, read_only = False, formatted = False)
            rst = pycharmm.CharmmFile(file_name = rst_fn, file_unit = self.rst_unit, read_only = False, formatted = 'formatted')
            lmd = pycharmm.CharmmFile(file_name = lmd_fn, file_unit = self.lmd_unit, read_only = False, formatted = False)

            dyn_dict.update({'nstep': nsteps_eq})
            pycharmm.DynamicsScript(**dyn_dict).run()

            dcd.close()
            rst.close()
            lmd.close()

        # Production Run
        if nsteps_prod > 0:
            if self.phase ==1 or self.phase == 2:
                dcd_fn = f'{input_folder}/run{run}/dcd/{name}_flat.dcd'
                rst_fn = f'{input_folder}/run{run}/res/{name}_flat.rst'
                lmd_fn = f'{input_folder}/run{run}/res/{name}_flat.lmd'
            elif self.phase == 3:
                dcd_fn = f'{input_folder}/run{run}/dcd/{name}_prod.dcd'
                rst_fn = f'{input_folder}/run{run}/res/{name}_prod.rst'
                lmd_fn = f'{input_folder}/run{run}/res/{name}_prod.lmd'
                # Need to work on this
                if run > 1:
                    dyn_dict.update({'start': False, 'restart': True, 'iunrea': self.rpr_unit})
                    rpr_fn = f'{input_folder}/run{run}/res/{name}_prod.rpr'
                    rpr = pycharmm.CharmmFile(file_name = rpr_fn, file_unit = self.rpr_unit, read_only = False, formatted = False)

            dcd = pycharmm.CharmmFile(file_name = dcd_fn, file_unit = self.dcd_unit, read_only = False, formatted = False)
            rst = pycharmm.CharmmFile(file_name = rst_fn, file_unit = self.rst_unit, read_only = False, formatted = 'formatted')
            lmd = pycharmm.CharmmFile(file_name = lmd_fn, file_unit = self.lmd_unit, read_only = False, formatted = False)

            dyn_dict.update({'nstep': nsteps_prod})

            pycharmm.DynamicsScript(**dyn_dict).run()

            dcd.close()
            lmd.close()
            rst.close()
            pycharmm.charmm_script('! Simulation Complete')
            settings.set_verbosity(5)
            lmd = pycharmm.CharmmFile(file_name=lmd_fn, file_unit = self.lmd_unit, read_only = True, formatted = False)
            pycharmm.charmm_script(f'traj lamb print ctlo 0.95 cthi 0.99 first {self.lmd_unit} nunit 1')
            lmd.close()
         
        self.box_size[0] = pycharmm.get_energy_value('XTLA')
        self.box_size[1] = pycharmm.get_energy_value('XTLB')
        self.box_size[2] = pycharmm.get_energy_value('XTLC')
        # setup a maximum box size from all values 
        self.box_size = [max(self.box_size)]*3
        write.coor_card(f'{input_folder}/run{run}/prod.crd')
        f = open(f'{input_folder}/run{run}/box.dat', 'w')
        f.write(f'{self.box_size[0]} {self.box_size[1]} {self.box_size[2]}')
        f.close()

    def alf_analysis(self,run):
        try:
            os.chdir(f'{input_folder}')
            im5 = max(run-5,1)
            N = run - im5 + 1
            print(f'Analysis{run} started')
            if not os.path.exists(f'analysis{run}'):
                os.mkdir(f'analysis{run}')
            shutil.copy(f'analysis{run-1}/b_sum.dat',f'analysis{run}/b_prev.dat')
            shutil.copy(f'analysis{run-1}/c_sum.dat',f'analysis{run}/c_prev.dat')
            shutil.copy(f'analysis{run-1}/x_sum.dat',f'analysis{run}/x_prev.dat')
            shutil.copy(f'analysis{run-1}/s_sum.dat',f'analysis{run}/s_prev.dat')

            if not os.path.exists(f'analysis{run}/G_imp'):
                G_imp_dir = os.path.join(os.getcwd(),'G_imp')
                target_path = os.path.join(os.getcwd(),f'analysis{run}/G_imp')
                try:
                    os.symlink(G_imp_dir,target_path)
                except FileExistsError:
                    pass

            os.chdir(f'analysis{run}')
            alf.GetLambdas(self.alf_info, run)
            alf.GetEnergy(self.alf_info, im5, run)

            fpout = open('output.dat','w')
            fperr = open('error.dat','w')
            cmd = f"python -c 'import alf; alf.RunWham({N*self.alf_info['nreps']},{self.alf_info['temp']}, 0, 0)'"
            subprocess.call(cmd, shell=True, stdout=fpout, stderr=fperr)
            alf.GetFreeEnergy5(self.alf_info, 0,0)
            alf.SetVars(self.alf_info, run+1)
        except:
            print(f'Analysis{run} failed')
            raise Exception(f'Analysis{run} failed')
        finally:
            os.chdir('../..')
            # if run > 5:
            #     shutil.rmtree(f'{input_folder}/analysis{run-5}')
        

            

md1 = ALF_simulation(phase=1, start=1, end=100)
md1.run()
# %%
