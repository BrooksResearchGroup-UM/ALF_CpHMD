"""Dynamics execution for ALF simulations.

Handles CHARMM output redirection, crystal/legacy setup, BLOCK commands,
restraints, minimization, and production dynamics via BLADE GPU.
"""

import logging
import re

import numpy as np


class DynamicsRunner:
    """Executes molecular dynamics using pyCHARMM with BLADE GPU acceleration.

    Receives ``config`` and ``state`` by reference — writes directly to
    ``state.structure_loaded``, ``state.restart_run``, ``state.box_size``, etc.

    Args:
        config: ALFConfig instance
        state: SimulationState instance (mutated in-place)
    """

    LOG_UNIT = 99  # File unit for CHARMM log redirection
    ALPHABET = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, config, state):
        self.config = config
        self.state = state
        self._log_file = None

    # ------------------------------------------------------------------
    # Output redirection
    # ------------------------------------------------------------------

    def redirect_output(self, run_idx: int, k: int, replica_idx: int):
        """Redirect CHARMM output to a separate log file for this replica/repeat."""
        import pycharmm
        import pycharmm.lingo as lingo

        run_dir = self.config.input_folder / f"run{run_idx}"
        log_path = run_dir / f"log.{k}.{replica_idx}.out"

        self._log_file = pycharmm.CharmmFile(
            file_name=str(log_path),
            file_unit=self.LOG_UNIT,
            read_only=False,
            formatted=True,
        )
        lingo.charmm_script(f"OUTUnit {self.LOG_UNIT}")

    def return_output(self):
        """Return CHARMM output to standard output (unit 6)."""
        import pycharmm.lingo as lingo

        if self._log_file is not None:
            lingo.charmm_script("OUTUnit 6")
            self._log_file.close()
            self._log_file = None

    # ------------------------------------------------------------------
    # Crystal / default setup
    # ------------------------------------------------------------------

    def setup_crystal(self, run_idx: int, letter: str, k: int, replica_idx: int):
        """Setup crystal/periodic boundary conditions."""
        import random

        import pycharmm.read as read
        import pycharmm.settings as settings

        from .charmm_utils import (
            BoxParameters,
            FFTParameters,
            NonBondedConfig,
            clear_block,
            clear_crystal,
            define_selections,
            setup_crystal,
            setup_nonbonded,
        )

        prep_dir = self.config.input_folder / "prep"
        is_first_run = not self.state.structure_loaded and k == 0

        if k > 0:
            return

        if is_first_run:
            settings.set_bomb_level(-1)
            if self.config.hmr:
                psf_file = prep_dir / "system_hmr.psf"
                if not psf_file.exists():
                    psf_file = prep_dir / "system.psf"
                    read.psf_card(str(psf_file))
                    import pycharmm.psf as pycharmm_psf
                    pycharmm_psf.hmr(newpsf=str(prep_dir / "system_hmr.psf"))
                else:
                    read.psf_card(str(psf_file))
            else:
                psf_file = prep_dir / "system.psf"
                read.psf_card(str(psf_file))
            settings.set_bomb_level(0)

            if self.state.patch_info is not None:
                define_selections(self.state.patch_info)

            self.state.structure_loaded = True
        else:
            clear_block()
            if self.config.restrains == "NOE":
                from .charmm_utils import clear_noe
                clear_noe()
            clear_crystal()

        if run_idx > 5:
            self.state.restart_run = random.randint(run_idx - 5, run_idx - 1)
        else:
            self.state.restart_run = 1

        box_params = BoxParameters.from_file(prep_dir / "box.dat")
        self.state.crystal_type = box_params.crystal_type
        self.state.box_size = box_params.dimensions
        self.state.box_angles = box_params.angles

        if (prep_dir / "system_min.crd").exists():
            crd_file = prep_dir / "system_min.crd"
        elif self.config.hmr:
            crd_file = prep_dir / "system_hmr.crd"
        else:
            crd_file = prep_dir / "system.crd"

        read.coor_card(str(crd_file))

        if self.state.restart_run != 1:
            restart_candidates = [
                f"run{self.state.restart_run}/prod.{k}.{replica_idx}.crd",
                f"run{self.state.restart_run}/prod.crd{letter}",
                f"run{self.state.restart_run}/prod.crd",
            ]
            for crd_name in restart_candidates:
                crd_path = self.config.input_folder / crd_name
                if crd_path.exists():
                    read.coor_card(str(crd_path))
                    break

        fft_params = FFTParameters.from_file(prep_dir / "fft.dat")

        nb_config = NonBondedConfig(
            cutnb=self.config.cutnb,
            cutim=self.config.cutnb,
            ctofnb=self.config.ctofnb,
            ctonnb=self.config.ctonnb,
            elec_type=self.config.elec_type,
            vdw_type=self.config.vdw_type,
            fftx=fft_params.fftx,
            ffty=fft_params.ffty,
            fftz=fft_params.fftz,
        )

        setup_crystal(box_params, nb_config, use_image_centering=not self.config.cent_ncres)
        setup_nonbonded(nb_config)

    # ------------------------------------------------------------------
    # Legacy setup
    # ------------------------------------------------------------------

    def setup_legacy(self, run_idx: int, letter: str, k: int, replica_idx: int):
        """Set up CHARMM session by streaming a legacy setup script."""
        import random

        import pycharmm.lingo as lingo
        import pycharmm.settings as settings

        from .charmm_utils import (
            NonBondedConfig,
            clear_block,
            clear_crystal,
            setup_nonbonded,
        )

        prep_dir = self.config.input_folder / "prep"
        is_first_run = not self.state.structure_loaded and k == 0

        if k > 0:
            return

        if not is_first_run:
            clear_block()
            if self.config.restrains == "NOE":
                from .charmm_utils import clear_noe
                clear_noe()
            clear_crystal()
            lingo.charmm_script("delete atom sele all end")

        if run_idx > 5:
            self.state.restart_run = random.randint(run_idx - 5, run_idx - 1)
        else:
            self.state.restart_run = 1

        var_file = self.config.input_folder / f"variables{run_idx}.inp"
        if not var_file.exists():
            raise FileNotFoundError(f"Variables file not found: {var_file}")
        lingo.charmm_script(f"stream {var_file}")

        setup_script = prep_dir / self.config.legacy_setup_script
        script_text = setup_script.read_text()

        box = self.state.alf_info["box"]
        script_text = re.sub(
            r"(?i)set\s+box\s*=\s*\S+",
            f"set box = {box}",
            script_text,
        )
        script_text = re.sub(
            r"(?i)set\s+builddir\s*=\s*\S+",
            f"set builddir = {prep_dir}",
            script_text,
        )

        processed_script = self.config.input_folder / f".setup_{run_idx}.inp"
        processed_script.write_text(script_text)

        settings.set_bomb_level(-2)
        lingo.charmm_script(f"stream {processed_script}")
        settings.set_bomb_level(0)

        processed_script.unlink(missing_ok=True)

        self.state.structure_loaded = True

        import pycharmm.read as pycharmm_read

        min_crd = prep_dir / "system_min.crd"
        if min_crd.exists():
            pycharmm_read.coor_card(str(min_crd))

        if self.state.restart_run and self.state.restart_run != 1:
            restart_candidates = [
                f"run{self.state.restart_run}/prod.{k}.{replica_idx}.crd",
                f"run{self.state.restart_run}/prod.crd{letter}",
                f"run{self.state.restart_run}/prod.crd",
            ]
            for crd_name in restart_candidates:
                crd_path = self.config.input_folder / crd_name
                if crd_path.exists():
                    pycharmm_read.coor_card(str(crd_path))
                    break

        nb_config = NonBondedConfig(
            cutnb=self.config.cutnb,
            cutim=self.config.cutnb,
            ctofnb=self.config.ctofnb,
            ctonnb=self.config.ctonnb,
            elec_type=self.config.elec_type,
            vdw_type=self.config.vdw_type,
        )
        setup_nonbonded(nb_config)

        self.apply_legacy_restraints()

    def apply_legacy_restraints(self):
        """Stream user-provided restraints for legacy mode."""
        import pycharmm.lingo as lingo

        prep_dir = self.config.input_folder / "prep"
        restraint_file = prep_dir / "restrains.str"

        if restraint_file.exists():
            lingo.charmm_script(f"stream {restraint_file}")
            return

        if self.config.restrains:
            logging.getLogger(__name__).warning(
                "Restraints requested (restrains=%r) but prep/restrains.str not found. "
                "Legacy (msld-py-prep) systems require a hand-written restraints file. "
                "See the example in examples/88_marcella/prep/restrains.str.",
                self.config.restrains,
            )

    # ------------------------------------------------------------------
    # BLOCK commands
    # ------------------------------------------------------------------

    def build_block_commands(self, run_idx: int, letter: str, k: int, replica_idx: int):
        """Build and execute BLOCK/MSLD commands for lambda dynamics."""
        from .block_builder import BlockConfig, build_block_command, read_variable_file
        from .charmm_utils import execute_block_command
        from .cphmd_params import (
            compute_all_site_parameters,
            compute_bias_shifts,
            get_delta_pKa_for_phase,
            write_bias_files,
        )

        if k > 0:
            return

        if self.state.patch_info is None:
            raise ValueError("patch_info not loaded")

        var_file = self.config.input_folder / f"variables{run_idx}.inp"
        variables = read_variable_file(var_file)

        effective_pH = self.config.pH
        delta_pKa = get_delta_pKa_for_phase(self.state.phase)

        if self.config.pH is not None:
            cphmd_params = compute_all_site_parameters(
                self.state.patch_info,
                self.config.temperature,
                self.config.pH,
            )
            effective_pH = cphmd_params.effective_pH

            if self.config.no_pka_bias:
                nblocks = self.state.alf_info["nblocks"]
                b_shift = np.zeros(nblocks)
                b_fix_shift = np.zeros(nblocks)
            else:
                b_shift, b_fix_shift = compute_bias_shifts(
                    cphmd_params,
                    self.state.patch_info,
                    delta_pKa,
                    replica_idx,
                )
            write_bias_files(self.config.input_folder, b_shift, b_fix_shift)

        block_config = BlockConfig(
            temperature=self.config.temperature,
            pH=self.config.pH,
            effective_pH=effective_pH,
            delta_pKa=delta_pKa,
            use_cphmd=(self.config.pH is not None and delta_pKa != 0 and not self.config.no_pka_bias),
            initial_lambdas=self.state.forced_initial_lambdas,
            lambda_mass=self.config.lambda_mass,
            lambda_fbeta=self.config.lambda_fbeta,
        )

        block_cmd = build_block_command(
            self.state.patch_info, variables, block_config,
            fnex=self.config.fnex,
            chi_offset=self.config.chi_offset,
            omega_decay=self.config.omega_decay,
        )

        block_file = self.config.input_folder / f"run{run_idx}" / f"block.{k}.{replica_idx}.str"
        block_file.write_text(block_cmd)

        execute_block_command(block_cmd)

    # ------------------------------------------------------------------
    # Restraints
    # ------------------------------------------------------------------

    def apply_restraints(self, run_idx: int, k: int = 0):
        """Apply SCAT or NOE restraints to titratable atoms."""
        import pycharmm.lingo as lingo

        from .restraints import generate_noe_restraints, generate_scat_restraints

        if k > 0:
            return

        if self.state.patch_info is None:
            raise ValueError("patch_info not loaded")

        include_hydrogen = self.config.restrain_hydrogens

        if self.config.restrains == "NOE":
            restraint_cmd = generate_noe_restraints(
                self.state.patch_info, include_hydrogen
            )
        else:
            restraint_cmd = generate_scat_restraints(
                self.state.patch_info, include_hydrogen
            )

        restraint_file = self.config.input_folder / "prep" / "restrains.str"
        restraint_file.write_text(restraint_cmd)

        lingo.charmm_script(restraint_cmd)

    # ------------------------------------------------------------------
    # Minimization
    # ------------------------------------------------------------------

    def run_minimization(self, run_idx: int, replica_idx: int):
        """Run energy minimization before dynamics."""
        import pycharmm.energy as energy
        import pycharmm.minimize as minimize
        import pycharmm.shake as shake
        import pycharmm.write as write

        min_crd = self.config.input_folder / "prep" / "system_min.crd"
        if min_crd.exists():
            return

        print(f"Running minimization for run {run_idx}...")

        shake.on(fast=True, bonh=True, param=True, tol=1e-7)
        minimize.run_sd(nstep=250, nprint=50, step=0.005, tolenr=1e-3, tolgrd=1e-3)
        write.coor_card(str(min_crd))
        print(f"Minimized coordinates saved to {min_crd}")

        energy.show()

    # ------------------------------------------------------------------
    # Production dynamics
    # ------------------------------------------------------------------

    def run_dynamics(self, run_idx: int, letter: str, k: int, replica_idx: int):
        """Execute molecular dynamics with BLADE GPU acceleration."""
        import pycharmm
        import pycharmm.lingo as lingo
        import pycharmm.psf as psf

        dcd_unit = 51
        rst_unit = 52
        lmd_unit = 53
        rpr_unit = 54

        if self.state.phase == 1:
            nsteps_eq = 0
            nsteps_prod = 50000
            nsavc = 0
            nsavl = 1
        elif self.state.phase == 2:
            nsteps_eq = 10000
            nsteps_prod = 500000
            nsavc = 0
            nsavl = 1
        else:
            nsteps_eq = 0
            nsteps_prod = 1000000
            nsavc = 0
            nsavl = 1

        timestep = 0.002
        if self.config.hmr:
            nsteps_eq //= 2
            nsteps_prod //= 2
            nsavc //= 2
            nsavl = max(nsavl, 1)
            timestep = 0.004

        if self.config.prep_format == "legacy":
            nsavl = 10

        import pycharmm.dynamics as dyn
        import pycharmm.shake as shake

        shake.on(fast=True, bonh=True, param=True, tol=1e-7)
        lingo.charmm_script("faster on")

        gpuid = replica_idx % 8
        lingo.charmm_script(f"blade on gpuid {gpuid}")

        dyn.set_fbetas(np.full(psf.get_natom(), self.config.gscale, dtype=float))

        lingo.charmm_script("energy blade")

        dyn_param = {
            "start": True,
            "restart": False,
            "blade": True,
            "prmc": True,
            "iprs": 100,
            "prdv": 100,
            "cpt": True,
            "timestep": timestep,
            "firstt": self.config.temperature,
            "finalt": self.config.temperature,
            "tstruc": self.config.temperature,
            "tbath": self.config.temperature,
            "ichecw": 0,
            "ihtfrq": 0,
            "ieqfrq": 0,
            "iasors": 1,
            "iasvel": 1,
            "iscvel": 0,
            "inbfrq": -1,
            "ilbfrq": 0,
            "imgfrq": -1,
            "ntrfrq": 0,
            "echeck": -1,
            "iunldm": lmd_unit,
            "iunwri": rst_unit,
            "iuncrd": dcd_unit if nsavc > 0 else -1,
            "nsavc": nsavc,
            "nsavl": nsavl,
            "nprint": 10000,
            "iprfrq": 10000,
            "isvfrq": 0,
        }

        dyn_param.update({
            "pconstant": True,
            "pmass": psf.get_natom() * 0.12,
            "pref": 1.0,
            "pgamma": 20.0,
            "hoover": True,
            "reft": self.config.temperature,
            "tmass": 1000,
        })

        name = self.config.input_folder.name
        run_dir = self.config.input_folder / f"run{run_idx}"
        res_dir = run_dir / "res"
        dcd_dir = run_dir / "dcd"
        if nsavc > 0:
            dcd_dir.mkdir(exist_ok=True)

        # === Equilibration Run ===
        if nsteps_eq > 0:
            rst_fn = str(res_dir / f"{name}_eq.{k}.{replica_idx}.rst")
            lmd_fn = str(res_dir / f"{name}_eq.{k}.{replica_idx}.lmd")

            dcd = None
            if nsavc > 0:
                dcd_fn = str(dcd_dir / f"{name}_eq.{k}.{replica_idx}.dcd")
                dcd = pycharmm.CharmmFile(file_name=dcd_fn, file_unit=dcd_unit,
                                          read_only=False, formatted=False)
            rst = pycharmm.CharmmFile(file_name=rst_fn, file_unit=rst_unit,
                                      read_only=False, formatted=True)
            lmd = pycharmm.CharmmFile(file_name=lmd_fn, file_unit=lmd_unit,
                                      read_only=False, formatted=False)

            dyn_param.update({"nstep": nsteps_eq, "isvfrq": nsteps_eq})
            pycharmm.DynamicsScript(**dyn_param).run()

            if dcd is not None:
                dcd.close()
            rst.close()
            lmd.close()

        lingo.charmm_script("energy blade")

        # === Production Run ===
        if nsteps_prod > 0:
            sim_type = "flat" if self.state.phase in (1, 2) else "prod"

            rst_fn = str(res_dir / f"{name}_{sim_type}.{k}.{replica_idx}.rst")
            lmd_fn = str(res_dir / f"{name}_{sim_type}.{k}.{replica_idx}.lmd")

            dyn_param.update({"start": False, "restart": True, "iunrea": rpr_unit})

            rpr_fn = None
            if sim_type == "flat":
                candidates = [
                    res_dir / f"{name}_eq.{k}.{replica_idx}.rst",
                    res_dir / f"{name}_eq.rst",
                ]
            else:
                restart_run = run_idx - 1
                candidates = [
                    self.config.input_folder / f"run{restart_run}" / "res" / f"{name}_prod.{k}.{replica_idx}.rst",
                    self.config.input_folder / f"run{restart_run}" / "res" / f"{name}_flat.{k}.{replica_idx}.rst",
                ]

            for candidate in candidates:
                if candidate.exists():
                    rpr_fn = str(candidate)
                    break

            if rpr_fn is None and sim_type == "flat":
                dyn_param.update({"start": True, "restart": False})
                dyn_param.pop("iunrea", None)
                rpr = None
            elif rpr_fn is None:
                raise RuntimeError(f"No restart file found for production run {run_idx}")
            else:
                rpr = pycharmm.CharmmFile(file_name=rpr_fn, file_unit=rpr_unit,
                                          read_only=True, formatted=True)

            dcd = None
            if nsavc > 0:
                dcd_fn = str(dcd_dir / f"{name}_{sim_type}.{k}.{replica_idx}.dcd")
                dcd = pycharmm.CharmmFile(file_name=dcd_fn, file_unit=dcd_unit,
                                          read_only=False, formatted=False)
            rst = pycharmm.CharmmFile(file_name=rst_fn, file_unit=rst_unit,
                                      read_only=False, formatted=True)
            lmd = pycharmm.CharmmFile(file_name=lmd_fn, file_unit=lmd_unit,
                                      read_only=False, formatted=False)

            dyn_param.update({"nstep": nsteps_prod, "isvfrq": nsteps_prod})
            pycharmm.DynamicsScript(**dyn_param).run()

            if dcd is not None:
                dcd.close()
            rst.close()
            lmd.close()
            if rpr is not None:
                rpr.close()

        lingo.charmm_script("blade off")
