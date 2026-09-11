import copy
import enum
import itertools

from dace.data import Array, ArrayView
from dace.dtypes import AllocationLifetime
from dace.memlet import Memlet
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.analysis.schedule_tree.treenodes import ViewNode
from dace.sdfg.graph import MultiConnectorEdge

from ndsl import Backend, ndsl_log
from ndsl.dsl.dace.builder.stree.common import (
    AxisIterator,
    is_axis_map,
    replace_variable_name,
)


def _make_rank_list(values_to_rank: list[int]) -> list[int]:
    """Gives back the rank of the values of list.

    E.g. [10, 0, 20, 40] into [1, 0, 2, 3]
    """
    rank = {value: index for index, value in enumerate(sorted(values_to_rank))}
    return [rank[value] for value in values_to_rank]


class MemletDirection(enum.Enum):
    INPUT = enum.auto()
    OUTPUT = enum.auto()


class HoistPointerToMap(tn.ScheduleNodeVisitor):
    """Attempt to enforce a left hand side write on center, per axis, by moving the bounds
    of the map and local offsets on array access.

    This pass is written defensively and will only apply if all the offset are the same on all
    inputs and outputs of the tasklet under a cartesian map."""

    def __init__(self, backend: Backend, axis: AxisIterator) -> None:
        super().__init__()
        self._axis = axis
        self._aligned_maps = 0
        self._backend = backend
        self._hoisted_pointer = 0

    def __str__(self) -> str:
        return f"HoistPointerToMap{self._axis.as_str().lower()}"

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:

        for child in node.children:
            self.visit(child, memlet_replace={})

        ndsl_log.debug(f"🚀 Hoisted {self._hoisted_pointer} pointers")

    def visit_MapScope(
        self, node: tn.MapScope, memlet_replace: dict[Memlet, Memlet]
    ) -> None:
        if is_axis_map(node, self._axis):
            local_memlet_replace: dict[Memlet, Memlet] = {}
            for memlet in node.input_memlets():
                self._hoist_pointer(
                    node, memlet, MemletDirection.INPUT, local_memlet_replace
                )
            for memlet in node.output_memlets():
                self._hoist_pointer(
                    node, memlet, MemletDirection.OUTPUT, local_memlet_replace
                )

            memlet_replace = local_memlet_replace

        for child in node.children:
            self.visit(child, memlet_replace=memlet_replace)

    def visit_TaskletNode(
        self, node: tn.TaskletNode, memlet_replace: dict[Memlet, Memlet]
    ) -> None:
        for old_memlet, new_memlet in memlet_replace.items():
            # breakpoint()
            for tasklet_name, tasklet_memlet in node.in_memlets.items():
                if tasklet_memlet.data != old_memlet.data:
                    continue
                node.in_memlets[tasklet_name] = new_memlet

            for tasklet_name, tasklet_memlet in node.out_memlets.items():
                if tasklet_memlet.data != old_memlet.data:
                    continue
                node.out_memlets[tasklet_name] = new_memlet

    def visit_IfScope(
        self, node: tn.IfScope, memlet_replace: dict[Memlet, Memlet]
    ) -> None:
        for memlet in itertools.chain(node.input_memlets(), node.output_memlets()):
            name = memlet.data
            if name not in memlet_replace:
                continue

            # Update the conditional code (and memlet ?)
            replace_variable_name(node.condition, name, memlet_replace[memlet].data)

        for child in node.children:
            self.visit(child, memlet_replace=memlet_replace)

    def _hoist_pointer(
        self,
        node: tn.MapScope,
        memlet: Memlet,
        direction: MemletDirection,
        local_memlet_replace: dict[Memlet, Memlet],
    ) -> None:
        array_name = memlet.data
        this_data = node.get_root().containers[array_name]
        if not isinstance(this_data, (Array, ArrayView)):
            return

        # Skip non 3D because it's difficult to now the cartesian-ness just with
        # the data shape, strides or else
        if len(this_data.shape) > 3:
            ndsl_log.debug(f"Data dimensions aren't supported: {array_name}, skipping.")
            return

        # Escape when the cartesian axis is not covered in shape
        # ⚠️ ⚠️ This is buggy because we cannot really differentiate cartesian dimensions
        # and data dimensions since the information doesn't carry through ⚠️ ⚠️
        if self._axis.as_cartesian_index() > len(this_data.shape) - 1:
            return

        # Make an array view by pop'ing the axis from the shape
        # (and recomputing the strides and size)
        # It will be name OldName_Xv with X the axis
        viewed_data = copy.copy(this_data)
        viewed_data.lifetime = AllocationLifetime.Scope
        shape = list(viewed_data.shape)
        shape.pop(self._axis.as_cartesian_index())
        strides = list(viewed_data.strides)
        strides.pop(self._axis.as_cartesian_index())
        viewed_data.set_shape(tuple(shape))
        viewed_data.set_strides_from_layout(*_make_rank_list(strides))
        array_view = ArrayView.view(viewed_data)
        array_view_name = f"{memlet.data}{self._axis.as_str().upper()[1:]}v"
        # array_view.set_shape(new_shape=(array_view.shape[1],), strides=(1,))

        # Record the ArrayView and insert the view node
        view_node = ViewNode(
            target=array_view_name,
            source=memlet.data,
            memlet=Memlet(expr=f"{array_name}[{self._axis.as_str()}]"),
            src_desc=this_data,
            view_desc=array_view,
        )
        # build Memlet (by hand, based on direction)
        if direction == MemletDirection.INPUT:
            view_node.memlet._edge = MultiConnectorEdge(
                src=node,
                src_conn=f"OUT_{array_name}",  # TODO: only applies of isinstance(src, MapEntry) - otherwise None
                dst=view_node,
                dst_conn="views",
                data=view_node.memlet,
                key=0,  # not sure - dummy value
            )
        else:
            view_node.memlet._edge = MultiConnectorEdge(
                src=view_node,
                src_conn="views",
                dst=node,
                dst_conn=f"IN_{array_name}",  # TODO: only applies if isinstance(dst, MapExit) - otherwise None
                data=view_node.memlet,
                key=0,  # not sure - dummy value
            )
        node.get_root().containers[array_view_name] = array_view
        node.children.insert(0, view_node)

        # We will need to replace the memlet downstream with a new memlet where
        # the subset has have the axis removed
        view_subset = memlet.subset.string_list()
        view_subset.pop(self._axis.as_cartesian_index())
        # new_subset = memlet.subset.string_list().pop(1)
        local_memlet_replace[memlet] = Memlet(
            expr=f"{array_view_name}[{','.join(view_subset)}]"
        )

        # Record for feedback
        self._hoisted_pointer += 1
