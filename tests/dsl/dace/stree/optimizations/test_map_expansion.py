from dace import dtypes, nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl.dsl.dace.builder.stree.optimizations.map_expansion import expand_maps


def _map_scope(params: list[str]) -> tn.MapScope:
    ranges = [(0, 8, 1) for _ in params]
    map_node = nodes.Map("compute", params, ranges, schedule=dtypes.ScheduleType.Sequential)
    map_node.range.tile_sizes = [2 for _ in params]
    return tn.MapScope(node=nodes.MapEntry(map_node), children=[tn.StateBoundaryNode()])


def test_expand_maps_all_dimensions() -> None:
    stree = tn.ScheduleTreeRoot(name="tester", children=[_map_scope(["i", "j", "k"])])

    assert expand_maps(stree) == 1

    outer = stree.children[0]
    assert isinstance(outer, tn.MapScope)
    assert outer.node.map.params == ["i"]
    assert outer.node.map.schedule == dtypes.ScheduleType.Sequential
    assert outer.node.map.range.tile_sizes == [2]

    middle = outer.children[0]
    assert isinstance(middle, tn.MapScope)
    assert middle.node.map.params == ["j"]
    assert middle.node.map.schedule == dtypes.ScheduleType.Sequential
    assert middle.node.map.range.tile_sizes == [2]

    inner = middle.children[0]
    assert isinstance(inner, tn.MapScope)
    assert inner.node.map.params == ["k"]
    assert inner.node.map.range.tile_sizes == [2]
    assert isinstance(inner.children[0], tn.StateBoundaryNode)


def test_expand_maps_with_limit_leaves_inner_map() -> None:
    stree = tn.ScheduleTreeRoot(name="tester", children=[_map_scope(["i", "j", "k"])])

    assert expand_maps(stree, expansion_limit=1, inner_schedule=dtypes.ScheduleType.CPU_Multicore) == 1

    outer = stree.children[0]
    assert isinstance(outer, tn.MapScope)
    assert outer.node.map.params == ["i"]
    assert outer.node.map.schedule == dtypes.ScheduleType.Sequential

    inner = outer.children[0]
    assert isinstance(inner, tn.MapScope)
    assert inner.node.map.params == ["j", "k"]
    assert inner.node.map.schedule == dtypes.ScheduleType.CPU_Multicore
    assert inner.node.map.range.tile_sizes == [2, 2]
    assert isinstance(inner.children[0], tn.StateBoundaryNode)


def test_expand_maps_skips_single_dimension() -> None:
    map_scope = _map_scope(["i"])
    stree = tn.ScheduleTreeRoot(name="tester", children=[map_scope])

    assert expand_maps(stree) == 0
    assert stree.children[0] is map_scope
    assert map_scope.node.map.params == ["i"]