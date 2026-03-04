use crate::array::*;
use crate::category::Arrow;
use crate::finite_function::FiniteFunction;
use crate::strict::hypergraph::arrow::is_convex_subgraph_morphism;
use crate::strict::hypergraph::subobject::SubgraphMorphism;
use crate::strict::hypergraph::Hypergraph;
use crate::strict::open_hypergraph::OpenHypergraph;
use num_traits::Zero;

/// A rewrite rule for strict open hypergraphs under rewriting with
/// symmetric monoidal structure (SMC).
///
pub struct SmcRewriteRule<K: ArrayKind, O, A> {
    lhs: OpenHypergraph<K, O, A>,
    rhs: OpenHypergraph<K, O, A>,
}

impl<K: ArrayKind, O, A> SmcRewriteRule<K, O, A>
where
    K::Type<K::I>: NaturalArray<K>,
    K::Type<bool>: Array<K, bool>,
    K::Type<O>: Array<K, O> + PartialEq,
    K::Type<A>: Array<K, A>,
{
    /// Create a new rewrite rule if the boundaries of `lhs` and `rhs` match.
    pub fn new(lhs: OpenHypergraph<K, O, A>, rhs: OpenHypergraph<K, O, A>) -> Option<Self> {
        if Self::validate(&lhs, &rhs) {
            Some(Self { lhs, rhs })
        } else {
            None
        }
    }

    pub fn validate(lhs: &OpenHypergraph<K, O, A>, rhs: &OpenHypergraph<K, O, A>) -> bool {
        Self::boundaries_match(lhs, rhs)
            && Self::boundary_legs_injective(lhs, rhs)
            && Self::lhs_boundary_legs_disjoint(lhs)
            && Self::sides_monogamous_acyclic(lhs, rhs)
    }

    pub fn boundaries_match(lhs: &OpenHypergraph<K, O, A>, rhs: &OpenHypergraph<K, O, A>) -> bool {
        lhs.source() == rhs.source() && lhs.target() == rhs.target()
    }

    pub fn sides_monogamous_acyclic(
        lhs: &OpenHypergraph<K, O, A>,
        rhs: &OpenHypergraph<K, O, A>,
    ) -> bool {
        lhs.is_monogamous() && lhs.is_acyclic() && rhs.is_monogamous() && rhs.is_acyclic()
    }

    fn boundary_legs_injective(
        lhs: &OpenHypergraph<K, O, A>,
        rhs: &OpenHypergraph<K, O, A>,
    ) -> bool {
        lhs.s.is_injective() && lhs.t.is_injective() && rhs.s.is_injective() && rhs.t.is_injective()
    }

    fn lhs_boundary_legs_disjoint(lhs: &OpenHypergraph<K, O, A>) -> bool {
        lhs.s.has_disjoint_image(&lhs.t)
    }
}

impl<K: ArrayKind, O, A> Clone for SmcRewriteRule<K, O, A>
where
    K::Type<O>: Clone,
    K::Type<A>: Clone,
    K::Type<K::I>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            lhs: self.lhs.clone(),
            rhs: self.rhs.clone(),
        }
    }
}

impl<K: ArrayKind, O: core::fmt::Debug, A: core::fmt::Debug> core::fmt::Debug
    for SmcRewriteRule<K, O, A>
where
    K::Index: core::fmt::Debug,
    K::Type<K::I>: core::fmt::Debug,
    K::Type<O>: core::fmt::Debug,
    K::Type<A>: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SmcRewriteRule")
            .field("lhs", &self.lhs)
            .field("rhs", &self.rhs)
            .finish()
    }
}

/// A validated rewrite match witness for SMC rewriting, represented by maps on
/// wires and operations.
pub struct SmcRewriteMatch<'a, K: ArrayKind, O, A> {
    rule: &'a SmcRewriteRule<K, O, A>,
    host: &'a OpenHypergraph<K, O, A>,
    w: FiniteFunction<K>,
    x: FiniteFunction<K>,
}

impl<'a, K: ArrayKind, O, A> SmcRewriteMatch<'a, K, O, A> {
    pub fn new(
        rule: &'a SmcRewriteRule<K, O, A>,
        host: &'a OpenHypergraph<K, O, A>,
        w: FiniteFunction<K>,
        x: FiniteFunction<K>,
    ) -> Option<Self>
    where
        K::Type<K::I>: NaturalArray<K>,
        K::Type<bool>: Array<K, bool>,
        K::Type<O>: Array<K, O> + PartialEq,
        K::Type<A>: Array<K, A> + PartialEq,
    {
        if !host.is_monogamous() || !host.is_acyclic() {
            return None;
        }
        if is_convex_subgraph_morphism(&rule.lhs.h, &host.h, &w, &x) {
            Some(Self { rule, host, w, x })
        } else {
            None
        }
    }

    pub fn w(&self) -> &FiniteFunction<K> {
        &self.w
    }

    pub fn x(&self) -> &FiniteFunction<K> {
        &self.x
    }

    pub fn rule(&self) -> &SmcRewriteRule<K, O, A> {
        self.rule
    }

    pub fn host(&self) -> &OpenHypergraph<K, O, A> {
        self.host
    }
}

impl<'a, K: ArrayKind, O, A> Clone for SmcRewriteMatch<'a, K, O, A>
where
    K::Type<K::I>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            rule: self.rule,
            host: self.host,
            w: self.w.clone(),
            x: self.x.clone(),
        }
    }
}

impl<'a, K: ArrayKind, O: core::fmt::Debug, A: core::fmt::Debug> core::fmt::Debug
    for SmcRewriteMatch<'a, K, O, A>
where
    K::Index: core::fmt::Debug,
    K::Type<K::I>: core::fmt::Debug,
    K::Type<O>: core::fmt::Debug,
    K::Type<A>: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SmcRewriteMatch")
            .field("w", &self.w)
            .field("x", &self.x)
            .finish()
    }
}

/// Apply a rewrite rule to `host` using a match `m : L -> host` where `L` is the apex of `lhs`.
///
/// Returns `None` if the rewrite is invalid.
pub fn apply_smc_rewrite<'a, K: ArrayKind, O, A>(
    m: &SmcRewriteMatch<'a, K, O, A>,
) -> Option<OpenHypergraph<K, O, A>>
where
    K::Type<K::I>: NaturalArray<K>,
    K::Type<bool>: Array<K, bool>,
    K::Type<O>: Array<K, O> + PartialEq,
    K::Type<A>: Array<K, A> + PartialEq,
    O: PartialEq,
    for<'b> K::Slice<'b, K::I>: From<&'b [K::I]>,
{
    let rule = m.rule();
    let host = m.host();
    let lhs = rule.lhs();
    let rhs = rule.rhs();

    // Compute where the LHS boundary lands in the host.
    let lhs_inputs_in_host = (&lhs.s >> m.w())?;
    let lhs_outputs_in_host = (&lhs.t >> m.w())?;

    let mut in_match_mask = K::Type::<bool>::fill(false, host.h.w.len());
    if m.w().table.len() != K::I::zero() {
        in_match_mask.scatter_assign_constant(&m.w().table, true);
    }

    // Build the remainder by removing the matched interior, but keep outer interface
    // and images of the LHS boundary ports.
    let mut remove_node_mask = in_match_mask.clone();
    clear_mask_at(&mut remove_node_mask, &host.s);
    clear_mask_at(&mut remove_node_mask, &host.t);
    clear_mask_at(&mut remove_node_mask, &lhs_inputs_in_host);
    clear_mask_at(&mut remove_node_mask, &lhs_outputs_in_host);

    let mut remove_edge_mask = K::Type::<bool>::fill(false, host.h.x.len());
    if m.x().table.len() != K::I::zero() {
        remove_edge_mask.scatter_assign_constant(&m.x().table, true);
    }

    let remainder = SubgraphMorphism::from_masks(&host.h, remove_node_mask, remove_edge_mask);
    let (remainder, kept_w_inj, _kept_x_inj) = remainder.as_hypergraph_with_injections()?;

    // Factor boundary maps through the remainder injection to build the context L⊥.
    let host_inputs = host.s.factor_through_injective(&kept_w_inj);
    let host_outputs = host.t.factor_through_injective(&kept_w_inj);
    let lhs_outputs = lhs_outputs_in_host.factor_through_injective(&kept_w_inj);
    let lhs_inputs = lhs_inputs_in_host.factor_through_injective(&kept_w_inj);

    let s_ctx = (&host_inputs + &lhs_outputs)?;
    let t_ctx = (&host_outputs + &lhs_inputs)?;
    let context = OpenHypergraph::new(s_ctx, t_ctx, remainder).ok()?;
    if !context.is_monogamous() {
        return None;
    }

    let h_in = host.s.source();
    let h_out = host.t.source();
    let l_in = lhs.s.source();
    let l_out = lhs.t.source();

    pushout_rewrite(&context, rhs, h_in, h_out, l_in, l_out)
}

impl<K: ArrayKind, O, A> SmcRewriteRule<K, O, A> {
    pub fn lhs(&self) -> &OpenHypergraph<K, O, A> {
        &self.lhs
    }

    pub fn rhs(&self) -> &OpenHypergraph<K, O, A> {
        &self.rhs
    }
}

// Clear mask entries at the image of a finite function.
fn clear_mask_at<K: ArrayKind>(mask: &mut K::Type<bool>, f: &FiniteFunction<K>)
where
    K::Type<bool>: Array<K, bool>,
{
    if f.table.len() != K::I::zero() {
        mask.scatter_assign_constant(&f.table, false);
    }
}

// Slice a finite function along a contiguous range of its source.
fn slice_map<K: ArrayKind>(
    f: &FiniteFunction<K>,
    start: K::I,
    len: K::I,
) -> Option<FiniteFunction<K>>
where
    K::Type<K::I>: NaturalArray<K>,
{
    let end = start.clone() + len.clone();
    if end > f.source() {
        return None;
    }
    let table = K::Index::from_slice(f.table.get_range(start..end));
    FiniteFunction::new(table, f.target())
}

// Build the pushout of the context and RHS along the shared boundary, then
// reconstruct the outer interface from the context's host boundary segment.
fn pushout_rewrite<K: ArrayKind, O, A>(
    context: &OpenHypergraph<K, O, A>,
    rhs: &OpenHypergraph<K, O, A>,
    host_inputs: K::I,
    host_outputs: K::I,
    lhs_inputs: K::I,
    lhs_outputs: K::I,
) -> Option<OpenHypergraph<K, O, A>>
where
    K::Type<K::I>: NaturalArray<K>,
    K::Type<O>: Array<K, O> + PartialEq,
    K::Type<A>: Array<K, A> + PartialEq,
{
    // Slice out the LHS boundary segments from the context interface:
    // - outputs of LHS live in the tail of context.s
    // - inputs of LHS live in the tail of context.t
    let c_s_out = slice_map(&context.s, host_inputs.clone(), lhs_outputs.clone())?;
    let c_t_in = slice_map(&context.t, host_outputs.clone(), lhs_inputs.clone())?;

    // Build the span into the coproduct of wires:
    // - f maps LHS boundary ports into the context part
    // - g maps LHS boundary ports into the RHS part
    let f_in = c_t_in.inject0(rhs.h.w.len());
    let f_out = c_s_out.inject0(rhs.h.w.len());
    let f = (&f_in + &f_out)?;

    let g_in = rhs.s.inject1(context.h.w.len());
    let g_out = rhs.t.inject1(context.h.w.len());
    let g = (&g_in + &g_out)?;

    // Pushout the span to glue RHS into the hole.
    let (h, left_arrow, _right_arrow) = Hypergraph::pushout_along_span(&context.h, &rhs.h, &f, &g)?;

    // The outer interface is the prefix of the context boundary.
    // Reindex that prefix through the left arrow into the pushout.
    let s_host = slice_map(&context.s, K::I::zero(), host_inputs)?;
    let t_host = slice_map(&context.t, K::I::zero(), host_outputs)?;
    let s = s_host.compose(&left_arrow.w)?;
    let t = t_host.compose(&left_arrow.w)?;

    OpenHypergraph::new(s, t, h).ok()
}
