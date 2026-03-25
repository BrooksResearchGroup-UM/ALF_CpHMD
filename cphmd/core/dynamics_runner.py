"""Dynamics execution for ALF simulations.

Handles CHARMM output redirection, crystal/legacy setup, BLOCK commands,
restraints, minimization, and production dynamics via BLADE GPU.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np

from cphmd.utils.charmm_path import qpath


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
            file_name=qpath(log_path),
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

    def setup_crystal(
        self, run_idx: int, letter: str, k: int, replica_idx: int,
        force: bool = False,
    ):
        """Setup crystal/periodic boundary conditions.

        Args:
            force: If True, run setup even for k > 0 (used in exchange mode
                so that k=1 gets a fresh crystal/coordinate state).
        """
        import random

        import pycharmm.lingo as lingo
        import pycharmm.read as read
        import pycharmm.settings as settings

        from .charmm_utils import (
            BoxParameters,
            clear_block,
            clear_crystal,
            define_selections,
        )

        prep_dir = self.config.input_folder / "prep"
        is_first_run = not self.state.structure_loaded and k == 0

        if k > 0 and not force:
            return

        if is_first_run:
            settings.set_bomb_level(-1)
            # Disable autogeneration before PSF read — prevents CHARMM from
            # regenerating angles/dihedrals that were intentionally deleted
            # during patching (e.g., cross-sub terms removed by dele connectivity).
            lingo.charmm_script("AUTO OFF")
            prnlvl = settings.set_verbosity(1) if not self.config.debug else None
            if self.config.hmr:
                psf_file = prep_dir / "system_hmr.psf"
                if not psf_file.exists():
                    psf_file = prep_dir / "system.psf"
                    read.psf_card(qpath(psf_file))
                    import pycharmm.psf as pycharmm_psf
                    pycharmm_psf.hmr(newpsf=str(prep_dir / "system_hmr.psf"))
                else:
                    read.psf_card(qpath(psf_file))
            else:
                psf_file = prep_dir / "system.psf"
                read.psf_card(qpath(psf_file))
            if prnlvl is not None:
                settings.set_verbosity(prnlvl)
            settings.set_bomb_level(0)

            if (self.state.patch_info is not None
                    and self.config.elec_type == "pmeex"):
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
            self.state.restart_run = 0

        box_params = BoxParameters.from_file(prep_dir / "box.dat")
        self.state.crystal_type = box_params.crystal_type
        self.state.box_size = box_params.dimensions
        self.state.box_angles = box_params.angles

        if (prep_dir / "system_min.crd").exists():
            crd_file = prep_dir / "system_min.crd"
        else:
            crd_file = prep_dir / "system.crd"

        # Lower bomblevel for CRD read — converted systems may have
        # resname mismatches that trigger sequence warnings (coordinates
        # are assigned sequentially regardless)
        settings.set_bomb_level(-1)
        read.coor_card(qpath(crd_file))
        settings.set_bomb_level(0)

        if self.state.restart_run > 0:
            restart_candidates = [
                f"run{self.state.restart_run}/prod.{k}.{replica_idx}.crd",
                f"run{self.state.restart_run}/prod.crd{letter}",
                f"run{self.state.restart_run}/prod.crd",
            ]
            for crd_name in restart_candidates:
                crd_path = self.config.input_folder / crd_name
                if crd_path.exists():
                    read.coor_card(qpath(crd_path))
                    break

    def setup_crystal_nonbonded(self, run_idx: int, letter: str, k: int,
                                replica_idx: int, force: bool = False):
        """Set up crystal symmetry and nonbonded parameters.

        Called after BLOCK/MSLD setup so CHARMM knows the block count
        and allocates sufficient per-atom bond arrays in MAKINB.
        """
        from .charmm_utils import (
            BoxParameters,
            FFTParameters,
            NonBondedConfig,
            setup_crystal,
            setup_nonbonded,
        )

        if k > 0 and not force:
            return

        prep_dir = self.config.input_folder / "prep"
        box_params = BoxParameters.from_file(prep_dir / "box.dat")
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
            self.state.restart_run = 0

        var_file = self.config.input_folder / f"variables{run_idx}.inp"
        if not var_file.exists():
            raise FileNotFoundError(f"Variables file not found: {var_file}")
        lingo.charmm_script(f"stream {qpath(var_file)}")

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
        lingo.charmm_script(f"stream {qpath(processed_script)}")
        settings.set_bomb_level(0)

        processed_script.unlink(missing_ok=True)

        self.state.structure_loaded = True

        import pycharmm.read as pycharmm_read

        min_crd = prep_dir / "system_min.crd"
        if min_crd.exists():
            pycharmm_read.coor_card(qpath(min_crd))

        if self.state.restart_run and self.state.restart_run > 0:
            restart_candidates = [
                f"run{self.state.restart_run}/prod.{k}.{replica_idx}.crd",
                f"run{self.state.restart_run}/prod.crd{letter}",
                f"run{self.state.restart_run}/prod.crd",
            ]
            for crd_name in restart_candidates:
                crd_path = self.config.input_folder / crd_name
                if crd_path.exists():
                    pycharmm_read.coor_card(qpath(crd_path))
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
            lingo.charmm_script(f"stream {qpath(restraint_file)}")
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

    def build_block_commands(
        self, run_idx: int, letter: str, k: int, replica_idx: int,
        force: bool = False,
    ):
        """Build and execute BLOCK/MSLD commands for lambda dynamics.

        Args:
            force: If True, run setup even for k > 0 (used in exchange mode
                so that k=1 gets a fresh BLOCK/PHMD state).
        """
        from .block_builder import BlockConfig, build_block_command, read_variable_file
        from .charmm_utils import execute_block_command
        from .cphmd_params import (
            adjust_tags_for_effective_ph,
            compute_all_site_parameters,
            compute_per_unit_shift,
            get_delta_pKa_for_phase,
            write_bias_files,
        )
        from .cphmd_params import (
            replica_pH as compute_replica_pH,
        )

        if k > 0 and not force:
            return

        if self.state.patch_info is None:
            raise ValueError("patch_info not loaded")

        var_file = self.config.input_folder / f"variables{run_idx}.inp"
        variables = read_variable_file(var_file)

        rep_pH = None
        delta_pKa = get_delta_pKa_for_phase(self.state.phase)
        # Start with original patch_info; adjusted below if CpHMD is active
        block_patch_info = self.state.patch_info

        if self.config.pH:
            cphmd_params = compute_all_site_parameters(
                self.state.patch_info,
                self.config.temperature,
            )
            ncentral = self.state.alf_info["ncentral"]

            # Each replica runs at a different pH, fanning out from effective_pH
            rep_pH = compute_replica_pH(
                cphmd_params.effective_pH, delta_pKa, replica_idx, ncentral,
            )

            if self.config.no_pka_bias:
                nblocks = self.state.alf_info["nblocks"]
                b_shift = np.zeros(nblocks).tolist()
                b_fix_shift = np.zeros(nblocks).tolist()
            else:
                b_shift, b_fix_shift = compute_per_unit_shift(
                    cphmd_params,
                    self.state.patch_info,
                    delta_pKa,
                )
            # Only rank 0 writes shift files (all ranks compute same per-unit values)
            if replica_idx == 0:
                write_bias_files(self.config.input_folder, b_shift, b_fix_shift)

            # Adjust TAG pKa values relative to effective_pH (multi-site correction).
            # For single-site systems this is a no-op (effective_pH == site pH₀).
            block_patch_info = adjust_tags_for_effective_ph(
                self.state.patch_info, cphmd_params,
            )

        block_config = BlockConfig(
            temperature=self.config.temperature,
            pH=self.config.pH,
            effective_pH=rep_pH,
            delta_pKa=delta_pKa,
            use_cphmd=(self.config.pH and delta_pKa != 0 and not self.config.no_pka_bias),
            initial_lambdas=self.state.forced_initial_lambdas,
            lambda_mass=self.config.lambda_mass,
            lambda_fbeta=self.config.lambda_fbeta,
        )

        block_cmd = build_block_command(
            block_patch_info, variables, block_config,
            fnex=self.config.fnex,
            chi_offset=self.config.chi_offset,
            omega_decay=self.config.omega_decay,
            chi_offset_t=self.config.chi_offset_t,
            chi_offset_u=self.config.chi_offset_u,
            no_t_bias=self.config.no_t_bias,
            no_u_bias=self.config.no_u_bias,
            electrostatics=self.config.elec_type,
        )

        block_file = self.config.input_folder / f"run{run_idx}" / f"block.{k}.{replica_idx}.str"
        block_file.parent.mkdir(parents=True, exist_ok=True)
        block_file.write_text(block_cmd)

        execute_block_command(block_cmd)

    # ------------------------------------------------------------------
    # Restraints
    # ------------------------------------------------------------------

    def apply_restraints(self, run_idx: int, k: int = 0, force: bool = False):
        """Apply SCAT or NOE restraints to titratable atoms."""
        import pycharmm.lingo as lingo

        from .restraints import generate_noe_restraints, generate_scat_restraints

        if k > 0 and not force:
            return

        if self.config.restrains == "none":
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
                self.state.patch_info, include_hydrogen,
                force_constant=self.config.scat_force_constant,
            )

        restraint_file = self.config.input_folder / "prep" / "restrains.str"
        restraint_file.write_text(restraint_cmd)

        lingo.charmm_script(restraint_cmd)

    # ------------------------------------------------------------------
    # Minimization
    # ------------------------------------------------------------------

    def run_minimization(self, run_idx: int, replica_idx: int):
        """Run short CPU minimization before BLaDE (bonded terms only).

        In MPI mode, only rank 0 minimizes; others wait at the barrier.
        """
        import pycharmm.minimize as minimize
        import pycharmm.shake as shake

        min_crd = self.config.input_folder / "prep" / "system_min.crd"
        if min_crd.exists():
            return

        if self.state.rank == 0:
            print(f"Running CPU pre-minimization (100 steps) for run {run_idx}...")
            sys.stdout.flush()

            shake.on(fast=True, bonh=True, param=True, tol=1e-7)
            minimize.run_sd(nstep=100, nprint=50, step=0.005, tolenr=1e-3, tolgrd=1e-3)

    def run_blade_minimization(self, run_idx: int, replica_idx: int):
        """Run BLaDE GPU minimization (includes EXTERN VDW/ELEC).

        Rank 0 minimizes for 2000 steps and saves system_min.crd.
        All other ranks (and subsequent runs) read the saved coordinates.
        In MPI mode, a barrier ensures the file is written before others read it.
        """
        import pycharmm.energy as energy
        import pycharmm.minimize as minimize
        import pycharmm.read as read
        import pycharmm.write as write

        from .charmm_utils import qpath

        min_crd = self.config.input_folder / "prep" / "system_min.crd"
        if min_crd.exists():
            print(f"Reading minimized coordinates from {min_crd}")
            read.coor_card(qpath(min_crd))
            sys.stdout.flush()
            return

        if self.state.rank == 0:
            print(f"Running BLaDE minimization (2000 steps) for run {run_idx}...")
            print("Energy BEFORE BLaDE minimization:")
            energy.show()
            sys.stdout.flush()

            minimize.run_sd(nstep=2000, nprint=200, step=0.005, tolenr=1e-3, tolgrd=1e-3)
            write.coor_card(str(min_crd))
            print(f"Minimized coordinates saved to {min_crd}")

            print("Energy AFTER BLaDE minimization:")
            energy.show()
            sys.stdout.flush()

        # MPI barrier: wait for rank 0 to write before others read
        if self.state.size > 1:
            from mpi4py import MPI
            MPI.COMM_WORLD.Barrier()

            if self.state.rank != 0:
                print(f"Reading minimized coordinates from {min_crd}")
                read.coor_card(qpath(min_crd))
                sys.stdout.flush()

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
            nsteps_eq = 0
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

        lingo.charmm_script(f"blade on gpuid {self.state.gpuid}")

        dyn.set_fbetas(np.full(psf.get_natom(), self.config.gscale, dtype=float))

        lingo.charmm_script("energy blade")

        # BLaDE minimization: resolves EXTERN VDW clashes that CPU minimizer cannot see
        if run_idx <= 5:
            self.run_blade_minimization(run_idx, replica_idx)

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

        run_dir = self.config.input_folder / f"run{run_idx}"
        res_dir = run_dir / "res"
        dcd_dir = run_dir / "dcd"
        if nsavc > 0:
            dcd_dir.mkdir(exist_ok=True)

        # === Equilibration Run ===
        if nsteps_eq > 0:
            rst_fn = str(res_dir / f"eq.{k}.{replica_idx}.rst")
            lmd_fn = str(res_dir / f"eq.{k}.{replica_idx}.lmd")

            dcd = None
            if nsavc > 0:
                dcd_fn = str(dcd_dir / f"eq.{k}.{replica_idx}.dcd")
                dcd = pycharmm.CharmmFile(file_name=qpath(dcd_fn), file_unit=dcd_unit,
                                          read_only=False, formatted=False)
            rst = pycharmm.CharmmFile(file_name=qpath(rst_fn), file_unit=rst_unit,
                                      read_only=False, formatted=True)
            lmd = pycharmm.CharmmFile(file_name=qpath(lmd_fn), file_unit=lmd_unit,
                                      read_only=False, formatted=False)

            # For k>0, restart from k=0's production restart to get
            # CPT-evolved crystal params.  blade off after k=0 may not
            # transfer box dimensions back to CPU, so starting fresh
            # would use stale crystal → image clashes.  iasvel=1
            # (already set) randomizes velocities for independence.
            rpr = None
            if k > 0:
                sim_type_k0 = "flat" if self.state.phase in (1, 2) else "prod"
                k0_rst = res_dir / f"{sim_type_k0}.0.{replica_idx}.rst"
                if k0_rst.exists():
                    dyn_param.update({"start": False, "restart": True,
                                      "iunrea": rpr_unit})
                    rpr = pycharmm.CharmmFile(
                        file_name=qpath(k0_rst), file_unit=rpr_unit,
                        read_only=True, formatted=True,
                    )

            dyn_param.update({"nstep": nsteps_eq, "isvfrq": nsteps_eq})
            pycharmm.DynamicsScript(**dyn_param).run()

            if dcd is not None:
                dcd.close()
            rst.close()
            lmd.close()
            if rpr is not None:
                rpr.close()

        lingo.charmm_script("energy blade")

        # === Production Run ===
        if nsteps_prod > 0:
            sim_type = "flat" if self.state.phase in (1, 2) else "prod"

            rst_fn = str(res_dir / f"{sim_type}.{k}.{replica_idx}.rst")
            lmd_fn = str(res_dir / f"{sim_type}.{k}.{replica_idx}.lmd")

            dyn_param.update({"start": False, "restart": True, "iunrea": rpr_unit})

            rpr_fn = None
            if sim_type == "flat":
                candidates = [
                    res_dir / f"eq.{k}.{replica_idx}.rst",
                    res_dir / "eq.rst",
                ]
            else:
                candidates = []
                # For k>0, prefer current run's k=0 production restart
                # (has CPT-evolved crystal params from this run's k=0).
                if k > 0:
                    candidates.append(
                        res_dir / f"prod.0.{replica_idx}.rst"
                    )
                restart_run = run_idx - 1
                candidates.extend([
                    self.config.input_folder / f"run{restart_run}" / "res" / f"prod.{k}.{replica_idx}.rst",
                    self.config.input_folder / f"run{restart_run}" / "res" / f"flat.{k}.{replica_idx}.rst",
                ])

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
                rpr = pycharmm.CharmmFile(file_name=qpath(rpr_fn), file_unit=rpr_unit,
                                          read_only=True, formatted=True)

            dcd = None
            if nsavc > 0:
                dcd_fn = str(dcd_dir / f"{sim_type}.{k}.{replica_idx}.dcd")
                dcd = pycharmm.CharmmFile(file_name=qpath(dcd_fn), file_unit=dcd_unit,
                                          read_only=False, formatted=False)
            rst = pycharmm.CharmmFile(file_name=qpath(rst_fn), file_unit=rst_unit,
                                      read_only=False, formatted=True)
            lmd = pycharmm.CharmmFile(file_name=qpath(lmd_fn), file_unit=lmd_unit,
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

    # ------------------------------------------------------------------
    # Segmented production dynamics (for replica exchange)
    # ------------------------------------------------------------------

    def get_production_steps(self) -> tuple[int, int, int, float]:
        """Return (nsteps_eq, nsteps_prod, nsavl, timestep) for the current phase.

        Applies HMR and legacy adjustments. Shared between run_dynamics
        and run_dynamics_segment so step counts are consistent.
        """
        if self.state.phase == 1:
            nsteps_eq = 0
            nsteps_prod = 50000
            nsavl = 1
        elif self.state.phase == 2:
            nsteps_eq = 0
            nsteps_prod = 500000
            nsavl = 1
        else:
            nsteps_eq = 0
            nsteps_prod = 1000000
            nsavl = 1

        timestep = 0.002
        if self.config.hmr:
            nsteps_eq //= 2
            nsteps_prod //= 2
            nsavl = max(nsavl, 1)
            timestep = 0.004

        if self.config.prep_format == "legacy":
            nsavl = 10

        return nsteps_eq, nsteps_prod, nsavl, timestep

    def run_equilibration(
        self, run_idx: int, k: int, replica_idx: int,
        restart_from: Path | None = None,
    ) -> Path | None:
        """Run equilibration dynamics (Phase 2 only), returning the restart path.

        This is extracted from the eq section of ``run_dynamics()`` so that
        the exchange path can run eq separately before segmented production.

        Args:
            restart_from: Optional restart file to read instead of starting
                fresh. Used for k>0 in exchange mode to restore diverse lambda
                states after k=0's exchange homogenizes all replicas.

        Returns:
            Path to the eq restart file, or None if nsteps_eq == 0.
        """
        import pycharmm
        import pycharmm.dynamics as dyn
        import pycharmm.lingo as lingo
        import pycharmm.psf as psf
        import pycharmm.shake as shake

        nsteps_eq, _, nsavl, timestep = self.get_production_steps()
        if nsteps_eq == 0:
            return None

        rst_unit = 52
        lmd_unit = 53
        rpr_unit = 54

        run_dir = self.config.input_folder / f"run{run_idx}"
        res_dir = run_dir / "res"

        shake.on(fast=True, bonh=True, param=True, tol=1e-7)
        lingo.charmm_script("faster on")

        lingo.charmm_script(f"blade on gpuid {self.state.gpuid}")

        dyn.set_fbetas(np.full(psf.get_natom(), self.config.gscale, dtype=float))
        lingo.charmm_script("energy blade")

        # BLaDE minimization: resolves EXTERN VDW clashes that CPU minimizer cannot see
        if run_idx <= 5:
            self.run_blade_minimization(run_idx, replica_idx)

        # Use restart if provided, otherwise start fresh
        use_restart = restart_from is not None and Path(restart_from).exists()

        dyn_param = {
            "start": not use_restart,
            "restart": use_restart,
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
            "iuncrd": -1,
            "nsavc": 0,
            "nsavl": nsavl,
            "nprint": 10000,
            "iprfrq": 10000,
            "nstep": nsteps_eq,
            "isvfrq": nsteps_eq,
            "pconstant": True,
            "pmass": psf.get_natom() * 0.12,
            "pref": 1.0,
            "pgamma": 20.0,
            "hoover": True,
            "reft": self.config.temperature,
            "tmass": 1000,
        }

        if use_restart:
            dyn_param["iunrea"] = rpr_unit

        rst_fn = res_dir / f"eq.{k}.{replica_idx}.rst"
        lmd_fn = res_dir / f"eq.{k}.{replica_idx}.lmd"

        rpr = None
        if use_restart:
            rpr = pycharmm.CharmmFile(
                file_name=qpath(restart_from), file_unit=rpr_unit,
                read_only=True, formatted=True,
            )

        rst = pycharmm.CharmmFile(
            file_name=qpath(rst_fn), file_unit=rst_unit,
            read_only=False, formatted=True,
        )
        lmd = pycharmm.CharmmFile(
            file_name=qpath(lmd_fn), file_unit=lmd_unit,
            read_only=False, formatted=False,
        )

        pycharmm.DynamicsScript(**dyn_param).run()

        rst.close()
        lmd.close()
        if rpr is not None:
            rpr.close()

        lingo.charmm_script("energy blade")

        return rst_fn

    def run_dynamics_segment(
        self,
        run_idx: int,
        letter: str,
        k: int,
        replica_idx: int,
        segment_idx: int,
        nsteps: int,
        restart_from: Path | None = None,
        is_first_segment: bool = False,
        blade_ready: bool = False,
    ) -> tuple[Path, Path]:
        """Run a single dynamics segment for replica exchange.

        Each segment produces a restart file and an LMD file.
        Subsequent segments (or swapped partners) chain via restart files.

        Args:
            run_idx: ALF run index.
            letter: Replica letter suffix.
            k: Repeat index.
            replica_idx: Replica index.
            segment_idx: Segment index within this run.
            nsteps: Number of MD steps for this segment.
            restart_from: Path to restart file to read (partner's or
                          previous segment's). None for first segment.
            is_first_segment: Whether this is the very first segment
                              in the run (may need ``start=True``).
            blade_ready: If True, BLADE/shake are already configured
                         (e.g. by a prior equilibration call).

        Returns:
            Tuple of (rst_path, lmd_path) for the completed segment.
        """
        import pycharmm
        import pycharmm.lingo as lingo
        import pycharmm.psf as psf

        rst_unit = 52
        lmd_unit = 53
        rpr_unit = 54

        _, _, nsavl, timestep = self.get_production_steps()

        run_dir = self.config.input_folder / f"run{run_idx}"
        res_dir = run_dir / "res"

        sim_type = "flat" if self.state.phase in (1, 2) else "prod"
        seg_tag = f"seg{segment_idx:04d}"

        rst_fn = res_dir / f"{sim_type}.{k}.{replica_idx}.{seg_tag}.rst"
        lmd_fn = res_dir / f"{sim_type}.{k}.{replica_idx}.{seg_tag}.lmd"

        # Build base dynamics params (same as run_dynamics production section)
        import pycharmm.dynamics as dyn
        import pycharmm.shake as shake

        if is_first_segment and segment_idx == 0 and not blade_ready:
            shake.on(fast=True, bonh=True, param=True, tol=1e-7)
            lingo.charmm_script("faster on")

            lingo.charmm_script(f"blade on gpuid {self.state.gpuid}")

            dyn.set_fbetas(np.full(psf.get_natom(), self.config.gscale, dtype=float))
            lingo.charmm_script("energy blade")

            # BLaDE minimization: resolves EXTERN VDW clashes
            if run_idx <= 5:
                self.run_blade_minimization(run_idx, replica_idx)

        dyn_param = {
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
            "iuncrd": -1,
            "nsavc": 0,
            "nsavl": nsavl,
            "nprint": min(nsteps, 10000),
            "iprfrq": min(nsteps, 10000),
            "nstep": nsteps,
            "isvfrq": nsteps,  # write restart at end of segment
            "pconstant": True,
            "pmass": psf.get_natom() * 0.12,
            "pref": 1.0,
            "pgamma": 20.0,
            "hoover": True,
            "reft": self.config.temperature,
            "tmass": 1000,
        }

        # Determine start vs restart
        rpr = None
        if restart_from is not None and Path(restart_from).exists():
            dyn_param["start"] = False
            dyn_param["restart"] = True
            dyn_param["iunrea"] = rpr_unit
            rpr = pycharmm.CharmmFile(
                file_name=qpath(restart_from),
                file_unit=rpr_unit,
                read_only=True,
                formatted=True,
            )
        elif is_first_segment:
            # First segment of run: look for eq restart or previous run restart
            rpr_fn = self._find_segment_restart(run_idx, k, replica_idx, res_dir)
            if rpr_fn is not None:
                dyn_param["start"] = False
                dyn_param["restart"] = True
                dyn_param["iunrea"] = rpr_unit
                rpr = pycharmm.CharmmFile(
                    file_name=qpath(rpr_fn),
                    file_unit=rpr_unit,
                    read_only=True,
                    formatted=True,
                )
            else:
                dyn_param["start"] = True
                dyn_param["restart"] = False
        else:
            # Should not happen — subsequent segments always have restart_from
            raise RuntimeError(
                f"No restart file for segment {segment_idx} "
                f"(run {run_idx}, replica {replica_idx})"
            )

        rst = pycharmm.CharmmFile(
            file_name=qpath(rst_fn), file_unit=rst_unit,
            read_only=False, formatted=True,
        )
        lmd = pycharmm.CharmmFile(
            file_name=qpath(lmd_fn), file_unit=lmd_unit,
            read_only=False, formatted=False,
        )

        pycharmm.DynamicsScript(**dyn_param).run()

        rst.close()
        lmd.close()
        if rpr is not None:
            rpr.close()

        return (rst_fn, lmd_fn)

    def _find_segment_restart(
        self,
        run_idx: int,
        k: int,
        replica_idx: int,
        res_dir: "Path",
    ) -> "str | None":
        """Find a restart file for the first segment of a run.

        Searches for eq restart (flat runs) or previous run's final restart.
        """
        sim_type = "flat" if self.state.phase in (1, 2) else "prod"

        if sim_type == "flat":
            candidates = [
                res_dir / f"eq.{k}.{replica_idx}.rst",
                res_dir / "eq.rst",
            ]
        else:
            restart_run = run_idx - 1
            prev_res = self.config.input_folder / f"run{restart_run}" / "res"
            candidates = [
                prev_res / f"prod.{k}.{replica_idx}.rst",
                prev_res / f"flat.{k}.{replica_idx}.rst",
            ]

        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def finish_blade(self):
        """Turn off BLADE GPU acceleration (call after all segments done)."""
        import pycharmm.lingo as lingo

        lingo.charmm_script("blade off")
