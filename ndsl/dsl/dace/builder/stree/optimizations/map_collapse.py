# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Collapse directly nested maps in schedule trees."""

import copy

from dace import symbolic
from dace.sdfg.analysis.schedule_tree import treenodes as tn


def collapse_maps(stree: tn.ScheduleTreeNode) -> int:
    """Collapse directly nested maps into one multidimensional map.

    Two maps can be collapsed when the inner map is the outer map's only child, its range is independent of the outer
    map parameters, and both maps have the same schedule. These conditions preserve the iteration domain and avoid
    needing to rewrite intervening statements or dynamic-range inputs.

    :param stree: The schedule tree or subtree to modify in place.
    :return: The number of map pairs collapsed.
    """
    collapsed = 0

    def visit(scope: tn.ScheduleTreeScope) -> None:
        nonlocal collapsed
        for child in list(scope.children):
            if isinstance(child, tn.ScheduleTreeScope):
                visit(child)
            if not isinstance(child, tn.MapScope) or len(child.children) != 1:
                continue
            inner = child.children[0]
            if not isinstance(inner, tn.MapScope):
                continue

            outer_map, inner_map = child.node.map, inner.node.map
            if outer_map.schedule != inner_map.schedule:
                continue
            outer_params = set(outer_map.params)
            inner_dependencies = {
                str(symbol)
                for dim in inner_map.range
                for symbol in symbolic.symlist(dim)
            }
            if outer_params & inner_dependencies:
                continue

            merged_map = copy.deepcopy(outer_map)
            merged_map.params = outer_map.params + inner_map.params
            merged_map.range = type(outer_map.range)(
                outer_map.range.ranges + inner_map.range.ranges
            )
            merged_map.range.tile_sizes = (
                outer_map.range.tile_sizes + inner_map.range.tile_sizes
            )
            child.node.map = merged_map

            body = list(inner.children)
            child.children = []
            child.add_children(body)
            inner.children = []
            inner.parent = None
            collapsed += 1

        children = list(scope.children)
        scope.children = []
        scope.add_children(children)

    if isinstance(stree, tn.ScheduleTreeScope):
        visit(stree)
    return collapsed
