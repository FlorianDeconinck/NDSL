import gc
import numbers
import os
from pathlib import Path
from pprint import pformat
from typing import Any

from dace import SDFG, DeviceType, dtypes, nodes
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.dtypes import DeviceType as DaceDeviceType
from dace.dtypes import ScheduleType
from dace.dtypes import StorageType as DaceStorageType
from dace.sdfg.analysis.schedule_tree import passes
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.utils import fuse_states, inline_sdfgs
from dace.transformation.auto.auto_optimize import make_transients_persistent
from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap
from dace.transformation.helpers import get_parent_map

import ndsl.dsl.dace.builder.stree.optimizations as ndsl_passes
from ndsl import OptimizationConfig, ndsl_log
from ndsl.dsl.dace.builder.cache import BuildInfo
from ndsl.dsl.dace.dace_config import DaceConfig
from ndsl.dsl.dace.hardware_config import get_gpu_hardware_defaults
from ndsl.dsl.dace.utils import (
    DaCeProgress,
    memory_static_analysis,
    report_memory_static_analysis,
)


def _to_gpu(sdfg: SDFG) -> None:
    """Flag memory in SDFG to GPU.
    Force deactivate OpenMP sections for sanity."""

    # Gather all maps
    allmaps = [
        (me, state)
        for me, state in sdfg.all_nodes_recursive()
        if isinstance(me, nodes.MapEntry)
    ]
    topmaps = [
        (me, state) for me, state in allmaps if get_parent_map(state, me) is None
    ]

    # Set storage of arrays to GPU, scalarizable arrays will be set on registers
    for _sd, _aname, arr in sdfg.arrays_recursive():
        if arr.shape == (1,):
            arr.storage = dtypes.StorageType.Register
        else:
            arr.storage = dtypes.StorageType.GPU_Global

    # All maps will be schedule on GPU
    for mapentry, _state in topmaps:
        mapentry.schedule = dtypes.ScheduleType.GPU_Device

    # Deactivate OpenMP sections
    for sd in sdfg.all_sdfgs_recursive():
        sd.openmp_sections = False


def _simplify(
    sdfg: SDFG,
    *,
    validate: bool = False,
    validate_all: bool = False,
    verbose: bool = False,
) -> dict | None:
    return sdfg.simplify(
        validate=validate,
        validate_all=validate_all,
        verbose=verbose,
        # We disable ScalarToSymbolPromotion because it might push symbols onto edges
        # that DaCe itself can't parse anymore later, e.g. casts,  inlined function
        # calls or (complicated) field accesses.
        # We disable LiftTrivialIf because it takes long on bigger graphs and we estimate
        # the potential speed gains to be minimal anyway.
        skip={"ScalarToSymbolPromotion", "LiftTrivialIf"},
    )


def _tree_as_sdfg(stree: tn.ScheduleTreeRoot) -> SDFG:
    """
    Convert the given ScheduleTree to SDFG.

    This function wraps `stree.as_sdfg()` with a configuration that is suitable for
    NDSL, e.g. skipping certain passes of `sdfg.simplify()`.
    """
    # We disable ScalarToSymbolPromotion because it might push symbols onto edges
    # that DaCe itself can't parse anymore later, e.g. casts,  inlined function
    # calls or (complicated) field accesses.
    # We disable ControlFlowRaising because tree -> sdfg outputs control flow graphs.
    # We disable LiftTrivialIf because it takes long on bigger graphs and we estimate
    # the potential speed gains to be minimal anyway.
    return stree.as_sdfg(
        validate=False,
        simplify=False,  # D_SW failed validation on merging
        skip={"ScalarToSymbolPromotion", "ControlFlowRaising", "LiftTrivialIf"},
    )


def _optimize_for_serial_cpu(
    config: DaceConfig,
    sdfg: SDFG,
    device_type: DaceDeviceType,
    validate: bool = True,
) -> SDFG:
    """
    Applies a series of passes aimed at leveraging (AVX) auto-vectorization on modern hardware and
    compiler.

    The passes will clean up dead code, remove/hoist branching, favor loop making over loop/guard combo,
    look to merge loops to increase transient locality, refine transients and generally aims at presenting
    a better vectorizable code. Target compilers were GCC (mildly successful) and LLVM (successfull)

    :param sdfg: The frozen SDFG to optimize (left unchanged).
    :param verbose: Print how many changes each pass made.
    :return: The optimized SDFG.
    """

    # Dev NOTE: we work with a DaCe branch not merged with `main` which causes issue for their CI.
    #           We will check that the transform of this branch `function-regions` actually exists
    #           and bail if they don't. PredicateToIntegerArray is representative of the entire
    #           workflow.
    try:
        from dace.transformation.passes import PredicateToIntegerArray
    except ImportError:
        ndsl_log.warning(
            "Schedule tree canonicalization passes are not available, "
            "serial CPU optimization skipped."
        )
        return sdfg

    with DaCeProgress(config.get_orchestrate(), "Go to schedule tree"):
        stree = sdfg.as_schedule_tree()

    with DaCeProgress(
        config.get_orchestrate(),
        "Prepare tree for serial CPU optimziation",
    ):
        inner_schedule = (
            ScheduleType.GPU_Device
            if device_type is DeviceType.GPU
            else ScheduleType.Default
        )
        pipeline = [
            (
                "expand_maps",
                lambda t: ndsl_passes.expand_maps(t, inner_schedule=inner_schedule),
            ),
            ("toloops", passes.convert_map_to_loop),
        ]
        for name, apply in pipeline:
            r = apply(stree)
            ndsl_log.debug(f"{name}: {r}")

    with DaCeProgress(
        config.get_orchestrate(),
        "Apply canonicalization and massage loop for vectorization",
    ):
        pipeline = [
            # Temporaries: fuse vertical loops that share them, then shrink them (before splitting separates iterations)
            (
                "fusion",
                lambda t: passes.fuse_loops_for_reuse(
                    t, cache_bytes=256 * 1024, min_trip_count=2
                ),
            ),
            ("refine", passes.refine_loop_local_transients),
            # Guard cleanup
            ("pair", passes.pair_complementary_guards),
            ("forward", passes.forward_substitute_conditions),
            ("deadassign", passes.remove_dead_assignments),
            ("fold", passes.fold_guards),
            ("unswitch", passes.unswitch_invariant_guards),
            ("split", passes.split_iteration_spaces),
            ("selects", passes.convert_diamonds_to_selects),
            ("deadstores", passes.remove_dead_stores),
            ("merge", passes.merge_contiguous_loops),
            ("reroll", passes.reroll_statements),
            ("fuserolled", passes.fuse_rolled_loops),
            ("hoist", passes.hoist_select_arms),
            # Rarely true guards (before hoist_condition_reads, which adds unguarded statements before conditions)
            ("sink", passes.sink_into_guards),
            (
                "move",
                passes.move_statements_to_readers,
            ),  # Past nests that keep statements from their readers" guards
            ("sink2", passes.sink_into_guards),
            ("coarsen", passes.coarsen_guards),
            # Skip resets of containers nothing wrote since they were last reset (before reuse, which would share them)
            ("resets", passes.skip_redundant_resets),
            # Conditions read transient arrays through memlets, so that the memory passes see those reads
            ("hoistcond", passes.hoist_condition_reads),
            # Memory: share slots between temporaries, then put small ones on the stack
            ("reuse", passes.reuse_transients),
            (
                "stack",
                lambda t: passes.move_small_transients_to_stack(
                    t,
                    max_array_bytes=4096,
                    max_total_bytes=128 * 1024,
                    zero_read_before_written=True,
                ),
            ),
        ]
        for name, apply in pipeline:
            r = apply(stree)
            ndsl_log.debug(f"{name}: {r}")

    with DaCeProgress(config.get_orchestrate(), "Go back to SDFG"):
        result = stree.as_sdfg(validate=False, simplify=False)
        PredicateToIntegerArray().apply_pass(
            result, {}
        )  # Boolean mask arrays become int32 (vectorizes better)

    if validate:
        with DaCeProgress(config.get_orchestrate(), "Validate"):
            result.validate()
    return result


def _optimize_for_gpu(
    dace_config: DaceConfig,
    sdfg: SDFG,
    device_type: DaceDeviceType,
    validate: bool = True,
) -> SDFG:
    """Default optimization pipeline for GPU:
    - Maximize the 3D kernel coverage
    - Add thread/blocks configuration by reading the hardware
    - Move persistent memory into a memory pool
    """

    with DaCeProgress(dace_config.get_orchestrate(), "Go to schedule tree"):
        stree = sdfg.as_schedule_tree()

    with DaCeProgress(
        dace_config.get_orchestrate(),
        "Maximize kernel coverage",
    ):
        inner_schedule = (
            ScheduleType.GPU_Device
            if device_type is DeviceType.GPU
            else ScheduleType.Default
        )
        pipeline = [
            (
                "expand_maps",
                lambda t: ndsl_passes.expand_maps(t, inner_schedule=inner_schedule),
            ),
            # To avoid global sync in kernel, we always kernalize on GPU
            (
                "kernalize",
                lambda t: ndsl_passes.kernalize(t, backend=dace_config.get_backend()),
            ),
            ("collaps_maps", ndsl_passes.collapse_maps),
        ]
        for name, apply in pipeline:
            r = apply(stree)
            ndsl_log.debug(f"{name}: {r}")

    with DaCeProgress(dace_config.get_orchestrate(), "Go back to SDFG"):
        result_sdfg = stree.as_sdfg(validate=False, simplify=False)

    with DaCeProgress(
        dace_config.get_orchestrate(), "Make residency of GPU memory correct & leaner"
    ):
        # Make the transients array persistents
        _to_gpu(result_sdfg)
        make_transients_persistent(sdfg=result_sdfg, device=device_type)

    # TODO: this is the correct pattern but somehow it fails on HPC because of the parsing
    #       of CopyCorners - we rely on the default...
    # Set block size on GPU maps and collect callback tasklets to exclude next
    # gpu_defaults = get_gpu_hardware_defaults()
    # exclude_tasklets_list = []
    # for me, _state in result_sdfg.all_nodes_recursive():
    #     if (
    #         isinstance(me, nodes.MapEntry)
    #         and me.map.schedule == ScheduleType.GPU_Device
    #     ) and me.map.gpu_block_size is None:
    #         me.map.gpu_block_size = gpu_defaults.block_size

    #     if isinstance(me, nodes.Tasklet) and "callback_" in me.label:
    #         exclude_tasklets_list.append(me.label)

    # result_sdfg.apply_transformations_repeated(
    #     AddThreadBlockMap, print_report=False, validate=False
    # )

    # Pool memory using the cuda-given pool
    with DaCeProgress(
        dace_config.get_orchestrate(), "Turn Persistents into pooled Scope"
    ):
        memory_pooled = 0.0
        for _sd, _aname, arr in result_sdfg.arrays_recursive():
            # Change Persistent memory (sub-SDFG) into Scope and flag it.
            if arr.lifetime == dtypes.AllocationLifetime.Persistent:
                arr.pool = True
                memory_pooled += arr.total_size * arr.dtype.bytes
                arr.lifetime = dtypes.AllocationLifetime.Scope
        memory_pooled = float(memory_pooled) / (1024 * 1024)
        ndsl_log.debug(f"Pooled {memory_pooled:.2f} mb")

    if validate:
        with DaCeProgress(dace_config.get_orchestrate(), "Validate"):
            result_sdfg.validate()

    return result_sdfg


def _prepare_sdfg(config: DaceConfig, sdfg: SDFG, device_type: DaceDeviceType):
    """Mandatory clean up and specialization of the SDFG to get it ready for further optimization:
    - Specialize remaining symbols
    - [GPU Only] Swap schedule to GPU_Device on parallel maps
    - Default memory residency to optimize space for runtime without checking usage
    - NDSL's restricted simplify

    SDFG is modified in-place.
    """

    # Fully specialize all known symbols and then propagate these changes in the simplify
    # pass that follows. This is not only a smart idea in general, but also simplifies (haha)
    # the schedule tree (optimization) roundtrip.
    with DaCeProgress(config.get_orchestrate(), "fully specialize symbols"):
        for my_sdfg in sdfg.all_sdfgs_recursive():
            if my_sdfg.parent_nsdfg_node is not None:
                repl_dict: dict[str, str] = {}
                for sym, val in my_sdfg.parent_nsdfg_node.symbol_mapping.items():
                    if isinstance(val, numbers.Number):
                        repl_dict[sym] = str(val)
                my_sdfg.replace_dict(repl_dict)

        if config.verbose_orchestration:
            ndsl_log.debug("Saving parsed_sdfg.sdfgz")
            sdfg.save(
                os.path.abspath(f"{sdfg.build_folder}/parsed_sdfg.sdfgz"),
                compress=True,
            )

    if config.is_gpu_backend():
        with DaCeProgress(
            config.get_orchestrate(), "configure maps to run on GPU"
        ):
            for this_sdfg in sdfg.all_sdfgs_recursive():
                for state in this_sdfg.states():
                    for node in state.nodes():
                        if (
                            isinstance(node, nodes.EntryNode)
                            and node.schedule != ScheduleType.Sequential
                        ):
                            node.schedule = ScheduleType.GPU_Device

    with DaCeProgress(
        config.get_orchestrate(), "make residency of memory leaner"
    ):
        # - Scalar are registers
        # - Transient are persistent (cache miss but not runtime reallocation)
        for _sd, _aname, arr in sdfg.arrays_recursive():
            if arr.shape == (1,):
                arr.storage = DaceStorageType.Register
        make_transients_persistent(sdfg=sdfg, device=device_type)

    with DaCeProgress(config.get_orchestrate(), "simplify"):
        _simplify(sdfg)


def _compile(dace_config: DaceConfig, sdfg: SDFG) -> CompiledSDFG:
    """Compile the SDFG into a CompiledSDFG. First clean up the states and nested sdfgs, report statistics
    on the SDFG and memory.
    """

    # Clean up SDFG
    with DaCeProgress(dace_config.get_orchestrate(), "Inline SDFGs & Fuse states"):
        ctr_sdfg = inline_sdfgs(sdfg)
        ctr_state = fuse_states(sdfg)
        ndsl_log.debug(f"Inline SDFGs | Fuse states counts: {ctr_sdfg} | {ctr_state}")

    # Compile
    with DaCeProgress(dace_config.get_orchestrate(), "Codegen & compile"):
        ndsl_log.debug(
            "Stats on SDFG before compilation:\n"
            f"  Arrays: {len([name for _, name, _ in sdfg.arrays_recursive(include_nested_data=True)])}\n"
            f"  SDFGs : {len([sdfg for sdfg in sdfg.all_sdfgs_recursive()])}\n"
            f"  States: {len([state for state in sdfg.states()])}\n"
            f"  CFGs  : {len([cfg for cfg in sdfg.all_control_flow_regions(recursive=True)])}\n"
        )
        compiled_sdfg = sdfg.compile()

    # Printing analysis of the compiled SDFG
    with DaCeProgress(
        dace_config.get_orchestrate(), "Build finished. Running memory static analysis"
    ):
        report = report_memory_static_analysis(
            sdfg, memory_static_analysis(sdfg), False
        )
        ndsl_log.info(report)

    # Store build info in the common cache directory
    BuildInfo.save(
        sdfg,
        dace_config.layout,
        dace_config.tile_resolution,
        report,
        dace_config.get_backend(),
    )

    return compiled_sdfg


def optimize_full_program_sdfg(
    parsed_sdfg: SDFG,
    config: DaceConfig,
    optimization_config: OptimizationConfig | None,
    args: Any,
    kwargs: Any,
) -> CompiledSDFG:
    """Optimize and compile the SDFG (creating the .dacecache and with the source and dynamic library)
    from a parsed SDFG (e.g. python code + gt4py stencils).
    """
    device_type = DaceDeviceType.GPU if config.is_gpu_backend() else DaceDeviceType.CPU

    # Enforce cache directory made so all downstream caching file
    # won't hit an non existing directory
    Path(parsed_sdfg.build_folder).mkdir(parents=True, exist_ok=True)

    # Resolve optimization configuration
    if optimization_config is None:
        ndsl_log.debug(f"Using default optimization config for {parsed_sdfg.label}.")
        optimization_config = OptimizationConfig.get_default()
    optimization_config.concretize(config.get_backend())
    ndsl_log.debug(f"Compiling config:\n{pformat(optimization_config, indent=2)}")

    # Clean up and prepare SDFG for optimization
    with DaCeProgress(config.get_orchestrate(), "Prepare SDFG"):
        _prepare_sdfg(config, parsed_sdfg, device_type)

    # Core optimization pipelines
    if optimization_config.serial_cpu_vectorization:
        with DaCeProgress(config.get_orchestrate(), "Fixed pipeline for serial CPU"):
            parsed_sdfg = _optimize_for_serial_cpu(
                config,
                parsed_sdfg,
                device_type,
                validate=False,
            )
    elif config.is_gpu_backend():
        with DaCeProgress(config.get_orchestrate(), "Fixed pipeline for GPU"):
            parsed_sdfg = _optimize_for_gpu(
                config,
                parsed_sdfg,
                device_type,
                validate=False,
            )

    # Clean up and compile the SDFG (also additional feedback to the user)
    with DaCeProgress(config.get_orchestrate(), "Compile"):
        compiled_sdfg = _compile(config, parsed_sdfg)

    # The full process of orchestration and compile is very memory hungry, since we are about to
    # begin runtime execution in earnest, let's give python a beat to clean up
    gc.collect(2)

    return compiled_sdfg
