/// Named-shape compression certified after the graph has been learned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphCompressionKind {
    Circle,
    Interval,
    FiniteSet,
    /// A contractible bounded surface (`χ = 1`, orientable, `b₁ = 0`, `b₂ = 0`):
    /// the topological type of a sampled sheet or disk. This is what a swiss roll
    /// glues to — the fold unrolls into one flat chart with no handle and no
    /// closed 2-cycle (#2280 acceptance: "swiss roll → sheet").
    Disk,
    Cylinder,
    /// The non-orientable bounded surface with one boundary loop (`χ = 0`,
    /// non-orientable, `b₁ = 1`, `b₂ = 0`): a cylinder's orientation cocycle with
    /// a single sign reversal around the loop. Recognized by the orientability
    /// certificate — the "half-twist as a discrete sign" (#2280 acceptance:
    /// "Möbius band → holonomy sign detected").
    MobiusStrip,
    Torus,
    Sphere,
    ProjectivePlane,
    KleinBottle,
    Graph,
}

/// MDL read-out for whether the learned edge set earns a standard name.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphCompressionReport {
    pub kind: GraphCompressionKind,
    pub name: &'static str,
    pub generic_edge_bits: f64,
    pub named_bits: f64,
    pub bits_saved: f64,
}

impl GraphCompressionReport {
    pub fn certified(
        kind: GraphCompressionKind,
        name: &'static str,
        generic_edge_bits: f64,
        named_bits: f64,
    ) -> Self {
        Self {
            kind,
            name,
            generic_edge_bits,
            named_bits,
            bits_saved: generic_edge_bits - named_bits,
        }
    }

    pub fn unnamed(generic_edge_bits: f64) -> Self {
        Self {
            kind: GraphCompressionKind::Graph,
            name: "structure without a standard name",
            generic_edge_bits,
            named_bits: generic_edge_bits,
            bits_saved: 0.0,
        }
    }

    pub fn earns_standard_name(&self) -> bool {
        self.kind != GraphCompressionKind::Graph && self.bits_saved > 0.0
    }
}
