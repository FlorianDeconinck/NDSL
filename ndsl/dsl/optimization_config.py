import enum
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from ndsl import Backend


class OptimizationHint(enum.Enum):
    """Hint for the configuration system that will drive the OptimizationOption.AUTO value"""

    AUTO = enum.auto()
    "Automatic strategy resolved by querying software and hardware capabilities"
    SERIAL = enum.auto()
    "Suitable for many-cores (serial) CPU strategy."
    PARALLEL = enum.auto()
    "Suitable for GPU or many-thread (parallelization) CPU strategy."

    def __repr__(self) -> str:
        return self.name


class OptimizationOption(enum.Enum):
    """Options for configuration element. AUTO will rely on the best guess default"""

    AUTO = enum.auto()
    "Best guess relying on the OptimizationHint"
    APPLY = enum.auto()
    "Pass will always be applied"
    DO_NOT_APPLY = enum.auto()
    "Pass will never be applied"

    def __repr__(self) -> str:
        return self.name


@dataclass
class OptimizationConfig:
    """Configuration for running the full-program optimization"""

    @dataclass
    class Tree:
        """Optimization using the Schedule Tree IR"""

        @dataclass
        class Merger:
            enabled: bool = True
            """Enable cartesian axis merging."""

            overcompute_vertical: bool = True
            """
            When merging allow vertical maps (K) of different sizes to merge by inserting an `if` guard.
            """

            overcompute_horizontal: bool = False
            """
            When merging allow horizontal maps (I and J) of different sizes to
            merge by inserting an `if` guard.
            """

            align_lhs_on_center_vertical: bool = False
            """
            Attempt on aligning the LHS for K axis on center so we can further merging.
            EXPERIMENTAL, therefore defaulting to False.
            """

            align_lhs_on_center_horizontal: bool = False
            """
            Attempt on aligning the LHS for I and J axis on center so we can further merging.
            EXPERIMENTAL, therefore defaulting to False.
            """

            order: str = "default"
            """
            Allows to manually override the merging order (e.g. `KJI` will merge `K`, then `J`, then `I`).
            The default follows loop order of the backend given to `CartesianMerge`.
            """

        enabled: bool = os.getenv("NDSL_STREE_OPT", "False").lower() == "true"
        """Enable Schedule Tree transformations."""

        # TODO: Is it safe? Deactivate by default for now
        inline_K_loops_size_one: bool = False
        """"Remove serial for loops of size one in the K-axis."""

        kernelize: OptimizationOption = OptimizationOption.AUTO
        """Enable maximizing 3-axis kernelization by duplicating maps."""

        merger: Merger = field(default_factory=Merger)
        """Configuration object for cartesian axis merging."""

        refine_transients: bool = True
        """Reduce dimensionality of transient arrays based on their usage."""

    @dataclass
    class GPU:
        """Optimization dedicated for GPU"""

        common_gpu_xforms: bool = False
        """DaCe common xforms bundled in `apply_gpu_transformations`"""

    stree: Tree = field(default_factory=Tree)
    """Schedule Tree optimization options"""

    array_access_via_cursor_arithmetic: bool = False
    """
    Transform array access from the basic strided access to a arithmetic on pointer (cursor) strategy
    EXPERIMENTAL, therefore defaulting to False.
    """

    gpu: GPU = field(default_factory=GPU)
    """GPU-only optimization options"""

    hint: OptimizationHint = OptimizationHint.AUTO
    """Hint for all optimizations passes"""

    def is_concretized(self) -> bool:
        """Return whether no nested configuration contains an automatic hint."""

        def contains_auto(value: Any) -> bool:
            if is_dataclass(value):
                return any(
                    contains_auto(getattr(value, config_field.name))
                    for config_field in fields(value)
                )
            return value is OptimizationHint.AUTO

        return not contains_auto(self)

    def concretize(self, backend: Backend) -> None:
        """Concretize the remaining AUTO argument based on various metric."""

        if self.hint == OptimizationHint.AUTO:
            if backend.is_gpu_backend():
                self.hint = OptimizationHint.PARALLEL
            else:
                # Check for OMP_NUM_THREAD to gather if we can multithread or if
                # we should follow a
                omp_num_thread = os.getenv("OMP_NUM_THREAD")
                if omp_num_thread is None or int(omp_num_thread) <= 1:
                    self.hint = OptimizationHint.SERIAL
                else:
                    self.hint = OptimizationHint.PARALLEL

        if self.stree.kernelize == OptimizationOption.AUTO:
            self.stree.kernelize = (
                OptimizationOption.APPLY
                if self.hint == OptimizationHint.PARALLEL
                else OptimizationOption.DO_NOT_APPLY
            )
