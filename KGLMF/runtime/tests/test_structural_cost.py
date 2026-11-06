"""Active-edge costs: eligibility, topology and propagation depth invariants."""
import pytest
import torch

from ..memory import LatentMemory, MemoryGraph, STRUCTURAL_COST_VERSION
from ..aligned_memory import AlignedMemory
from ..online import OnlineEngine


@pytest.fixture(params=[LatentMemory, AlignedMemory])
def memory(request):
    torch.manual_seed(42)
    return request.param(8, 12)


def graph_with_edges():
    graph = MemoryGraph(8)
    for i in range(4):
        graph.add(torch.randn(16, 8), [f's{i}'], 'math')
    # Three real edges plus an isolated node; not a complete graph.
    graph.edges = {('n0', 'n1'): .7, ('n1', 'n2'): .6, ('n0', 'n2'): .5}
    return graph


def test_edges_count_once_not_per_round_and_cost_reacts_to_density(memory):
    graph = graph_with_edges()
    query = torch.randn(8)
    costs = []
    for hops in (1, 4, 8):
        _, meta = memory(query, graph, budget=16, hops=hops)
        assert meta['active_nodes'] == 4
        assert meta['active_edges'] == 3
        assert meta['graph_depth'] == hops
        assert meta['cost_version'] == STRUCTURAL_COST_VERSION
        costs.append(meta['structural_cost'])
    assert costs == [23., 23., 23.]
    graph.edges[('n2', 'n3')] = .2
    _, meta = memory(query, graph, budget=16, hops=8)
    assert meta['structural_cost'] == 24.
    assert meta['active_edges'] == 4


def test_filtered_subgraph_and_self_transitions(memory):
    graph = graph_with_edges()
    route = memory.route(torch.randn(8), graph, excluded_nodes=['n1'], hops=8)
    assert set(route['ids']) == {'n0', 'n2', 'n3'}
    assert route['active_edges'] == 1
    graph.edges = {}
    route = memory.route(torch.randn(8), graph, hops=8)
    assert route['active_edges'] == 0
    assert torch.allclose(route['scores'], route['initial'], atol=1e-6)
    assert torch.isfinite(route['scores']).all()


@pytest.mark.parametrize('topology,hops,expected', [('flat', 8, 0), ('graph', 0, 0), ('shuffled', 8, 3)])
def test_topology_and_zero_rounds(memory, topology, hops, expected):
    route = memory.route(torch.randn(8), graph_with_edges(), topology=topology, hops=hops)
    assert route['active_edges'] == expected


def test_empty_retrieval_and_domain_filter(memory):
    graph = graph_with_edges()
    prefix, meta = memory(torch.randn(8), graph, domains=['unseen'])
    assert prefix is None and meta['active_edges'] == 0
    assert meta['active_nodes'] == 0
    assert meta['cost_version'] == STRUCTURAL_COST_VERSION


def test_fixed_edge_cost_is_accounting_not_a_parameter_gradient(memory):
    cost = memory.structural_cost(16, active_nodes=4, active_edges=10)
    assert cost.item() == 30. and not cost.requires_grad
    loss = memory.pool.square().sum() + .001*cost
    loss.backward()
    assert torch.allclose(memory.pool.grad, 2*memory.pool.detach())
    assert memory.fixed_cost_weights.grad is None
    assert torch.equal(memory.cost_coefficients(), torch.ones(2))


@pytest.mark.parametrize('version', [None, 'graph-depth-v1', 'active-edges-v1'])
def test_historical_cost_checkpoints_rejected_before_state_mutation(version):
    # Version rejection must happen before accessing or replacing engine state.
    with pytest.raises(ValueError, match='Structural cost version changed'):
        OnlineEngine.load_state(None, {'structural_cost_version': version})
