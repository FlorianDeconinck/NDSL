# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Expansion of multidimensional maps in schedule trees."""

from dace import dtypes, subsets
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def expand_maps(
    stree: tn.ScheduleTreeNode,
    expansion_limit: int | None = None,
    inner_schedule: dtypes.ScheduleType | None = None,
) -> int:
    """Expand multidimensional maps into nested maps, outermost dimension first.

    The first map dimension retains the original schedule; subsequent dimensions use ``inner_schedule`` if given, or
    the original schedule otherwise. If ``expansion_limit`` is set, that many dimensions are expanded, leaving any
    remaining dimensions in an innermost multidimensional map. Non-positive limits expand all dimensions.

    Unlike SDFG MapExpansion, this pass only restructures MapScopes: map-boundary memlets are represented implicitly
    in the schedule tree and are regenerated during lowering.

    :param stree: The schedule tree or subtree to modify in place.
    :param expansion_limit: Maximum number of dimensions to expand, or ``None`` for all dimensions.
    :param inner_schedule: Schedule for maps created inside the outermost map.
    :return: The number of maps expanded.
    :raises TypeError: If ``expansion_limit`` is not an integer or ``None``.
    """
    expanded = 0

    def visit(scope: tn.ScheduleTreeScope) -> None:
        nonlocal expanded
        for child in list(scope.children):
            if isinstance(child, tn.ScheduleTreeScope):
                visit(child)
            if not isinstance(child, tn.MapScope):
                continue
            current_map = child.node.map
            dimensions = current_map.get_param_num()
            if dimensions <= 1:
                continue
            if expansion_limit is not None and not isinstance(expansion_limit, int):
                raise TypeError(
                    f"Does not know how to handle type {type(expansion_limit).__name__}"
                )

            full_expand = (
                expansion_limit is None
                or expansion_limit <= 0
                or dimensions - expansion_limit <= 1
            )
            count = dimensions if full_expand else expansion_limit
            schedule = inner_schedule or current_map.schedule
            maps = []
            for dim in range(count):
                param = current_map.params[dim]
                map_schedule = current_map.schedule if dim == 0 else schedule
                maps.append(
                    nodes.Map(
                        current_map.label + "_" + str(param),
                        [param],
                        subsets.Range([current_map.range.ranges[dim]]),
                        schedule=map_schedule,
                    )
                )
                maps[-1].range.tile_sizes[0] = current_map.range.tile_sizes[dim]
            if not full_expand:
                maps.append(
                    nodes.Map(
                        current_map.label,
                        current_map.params[count:],
                        current_map.range.ranges[count:],
                        schedule=schedule,
                    )
                )
                maps[-1].range.tile_sizes = current_map.range.tile_sizes[count:]

            child.node.map = maps[0]
            body: list[tn.ScheduleTreeNode] = list(child.children)
            for map_node in reversed(maps[1:]):
                body = [
                    tn.MapScope(
                        node=nodes.MapEntry(map_node), children=body, state=child.state
                    )
                ]
            child.children = []
            child.add_children(body)
            expanded += 1

        # Recursion can replace children, but not their order; reparenting keeps the tree invariant explicit.
        children = list(scope.children)
        scope.children = []
        scope.add_children(children)

    if not isinstance(stree, tn.ScheduleTreeScope):
        return 0
    visit(stree)
    return expanded
