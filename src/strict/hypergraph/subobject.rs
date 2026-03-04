use crate::array::{Array, ArrayKind, NaturalArray};
use crate::finite_function::FiniteFunction;

use super::Hypergraph;
use num_traits::{One, Zero};

// A monic subobject of a host hypergraph, represented by masks on wires and edges.
// i.e. morphism: H -> host where H is the graph obtained by G removing nodes/edges in the mask.
//
// Note: to avoid dangling hyper graphs that are not valid, we keep nodes of kept hyperedges
// we could fail the construction if there are dangling hyperedges instead of restore their nodes
// but the rewrite wouldn't be complete
//
// This avoids materializing the subgraph morphism data (new node/edge arrays + maps) up front:
// we keep just the host reference and two boolean masks and compute images/injections on demand.
// Performance notes:
// - masks are compact and cheap to copy compared to rebuilding incidence arrays
// - dangling avoidance can be done by scanning incidence once (preserve incident nodes)
// - the concrete subgraph hypergraph H is only built when explicitly requested
pub(crate) struct SubgraphMorphism<'a, K: ArrayKind, O, A> {
    host: &'a Hypergraph<K, O, A>,
    remove_node_mask: K::Type<bool>,
    remove_edge_mask: K::Type<bool>,
}

impl<'a, K: ArrayKind, O, A> SubgraphMorphism<'a, K, O, A>
where
    K::Type<bool>: Array<K, bool>,
{
    pub(crate) fn from_masks(
        host: &'a Hypergraph<K, O, A>,
        mut remove_node_mask: K::Type<bool>,
        remove_edge_mask: K::Type<bool>,
    ) -> Self
    where
        K::Type<K::I>: NaturalArray<K>,
        K::Type<O>: Array<K, O>,
        K::Type<A>: Array<K, A>,
    {
        // Ensure the resulting subgraph is non-dangling by keeping any node
        // incident to a remaining (non-removed) hyperedge.
        keep_incident_nodes(host, &remove_edge_mask, &mut remove_node_mask);

        Self {
            host,
            remove_node_mask,
            remove_edge_mask,
        }
    }

    pub(crate) fn as_hypergraph_with_injections(
        &self,
    ) -> Option<(Hypergraph<K, O, A>, FiniteFunction<K>, FiniteFunction<K>)>
    where
        K::Type<K::I>: NaturalArray<K>,
        K::Type<O>: Array<K, O>,
        K::Type<A>: Array<K, A>,
        K::Type<bool>: Array<K, bool>,
        for<'b> K::Slice<'b, K::I>: From<&'b [K::I]>,
    {
        // Since this subgraph is monic and non-dangling (by construction),
        // its image is a valid hypergraph.
        // Complexity: O(|W| + |X| + |incidence|) to build injections and remap incidence.
        let host = self.host;

        // Compute kept node/edge indices.
        let mut kept_nodes = Vec::<K::I>::new();
        let mut i = K::I::zero();
        while i < host.w.len() {
            if !self.remove_node_mask.get(i.clone()) {
                kept_nodes.push(i.clone());
            }
            i = i + K::I::one();
        }

        let mut kept_edges = Vec::<K::I>::new();
        let mut e = K::I::zero();
        while e < host.x.len() {
            if !self.remove_edge_mask.get(e.clone()) {
                kept_edges.push(e.clone());
            }
            e = e + K::I::one();
        }

        // Injections from subgraph into original.
        let kept_w_inj = FiniteFunction::new(index_from_vec::<K>(&kept_nodes), host.w.len())?;
        let kept_x_inj = FiniteFunction::new(index_from_vec::<K>(&kept_edges), host.x.len())?;

        // Build labels for the subgraph image.
        let new_w = (&kept_w_inj >> &host.w)?;
        let new_x = (&kept_x_inj >> &host.x)?;

        if kept_nodes.is_empty() {
            // No nodes remain. For a non-dangling subgraph this can only yield an empty hypergraph.
            if !kept_edges.is_empty() {
                return None;
            }
            return Some((Hypergraph::empty(), kept_w_inj, kept_x_inj));
        }

        // Total inverse with explicit fill outside image(kept_w_inj).
        // The fill value is never observed here because filtered incidence values
        // lie in image(kept_w_inj) by construction.
        let kept_w_inv = kept_w_inj.inverse_with_fill(K::I::zero())?;

        // Rebuild incidence by reindexing directly along the kept-edge injection,
        // then factor values through the node injection kept_w_inj : kept_nodes -> host_nodes.
        let new_s = host.s.map_indexes(&kept_x_inj)?.map_values(&kept_w_inv)?;
        let new_t = host.t.map_indexes(&kept_x_inj)?.map_values(&kept_w_inv)?;

        let remainder = Hypergraph {
            s: new_s,
            t: new_t,
            w: new_w,
            x: new_x,
        };

        Some((remainder, kept_w_inj, kept_x_inj))
    }
}

// Keep any node incident to a remaining (non-removed) hyperedge.
fn keep_incident_nodes<K: ArrayKind, O, A>(
    host: &Hypergraph<K, O, A>,
    remove_edge_mask: &K::Type<bool>,
    remove_node_mask: &mut K::Type<bool>,
) where
    K::Type<K::I>: NaturalArray<K>,
    K::Type<bool>: Array<K, bool>,
    K::Type<O>: Array<K, O>,
    K::Type<A>: Array<K, A>,
{
    let s_ptr = host.s.sources.table.cumulative_sum();
    let t_ptr = host.t.sources.table.cumulative_sum();
    let mut edge = K::I::zero();
    while edge < host.x.len() {
        if !remove_edge_mask.get(edge.clone()) {
            let s_start = s_ptr.get(edge.clone());
            let s_end = s_ptr.get(edge.clone() + K::I::one());
            let mut k = s_start.clone();
            while k < s_end {
                let v = host.s.values.table.get(k.clone());
                let value = K::Type::<bool>::fill(false, K::I::one());
                remove_node_mask.set_range(v.clone()..v.clone() + K::I::one(), &value);
                k = k + K::I::one();
            }

            let t_start = t_ptr.get(edge.clone());
            let t_end = t_ptr.get(edge.clone() + K::I::one());
            let mut k = t_start.clone();
            while k < t_end {
                let v = host.t.values.table.get(k.clone());
                let value = K::Type::<bool>::fill(false, K::I::one());
                remove_node_mask.set_range(v.clone()..v.clone() + K::I::one(), &value);
                k = k + K::I::one();
            }
        }
        edge = edge + K::I::one();
    }
}

fn index_from_vec<K: ArrayKind>(v: &Vec<K::I>) -> K::Index
where
    K::Index: Array<K, K::I>,
    for<'a> K::Slice<'a, K::I>: From<&'a [K::I]>,
{
    let slice: K::Slice<'_, K::I> = v.as_slice().into();
    K::Index::from_slice(slice)
}
