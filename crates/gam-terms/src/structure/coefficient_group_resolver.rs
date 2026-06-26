//! Carrier-agnostic coefficient-group resolver.
//!
//! User-declared coefficient groups are realized into penalty components and
//! rho-prior entries in two places: standard term collections (carrier =
//! columns of the realized design matrix, [`crate::smooth`]) and custom
//! families (carrier = `(block, column)` coordinates of parameter blocks,
//! [`crate::families::custom_family`]). The carrier differs but the *policy* is
//! identical: validate labels, build the parent/child hierarchy, reject cycles,
//! require each child to be a subset of its parent, require an interior node's
//! coefficients to be exactly the union of its children, and expand interior
//! nodes into the concatenation of their recursively resolved child components.
//!
//! This module hosts that shared policy once, generic over the coordinate type
//! `C` (the carrier). Each caller resolves its own selectors into `C` sets and
//! lays out its own penalty matrices and rho coordinates from the resolved
//! components — only the carrier-specific layout stays on the caller side.

use std::collections::{BTreeMap, BTreeSet};

/// A coefficient group after its selectors have been resolved by the carrier
/// into a concrete set of carrier coordinates `C`.
///
/// `coordinates` is the full coordinate set the group selects; the resolver
/// validates hierarchy relationships against it and never reinterprets the
/// coordinates themselves.
#[derive(Debug, Clone)]
pub struct ResolvedGroup<C: Ord + Clone> {
    pub label: String,
    pub parent: Option<String>,
    pub coordinates: BTreeSet<C>,
}

/// Validated coefficient-group hierarchy over carrier coordinates `C`.
///
/// Construction enforces the full group policy (unique non-empty labels,
/// non-empty coordinate sets, acyclic parent chains terminating at known
/// groups, child ⊆ parent, interior coordinates == union of children). After
/// construction, callers walk `groups` in their original order and request the
/// concatenated penalty components per group.
pub struct ResolvedGroupHierarchy<C: Ord + Clone> {
    groups: Vec<ResolvedGroup<C>>,
    coordinates_by_label: BTreeMap<String, BTreeSet<C>>,
    children_by_parent: BTreeMap<String, Vec<String>>,
}

impl<C: Ord + Clone> std::fmt::Debug for ResolvedGroupHierarchy<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedGroupHierarchy")
            .field("group_count", &self.groups.len())
            .field(
                "labels",
                &self.coordinates_by_label.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl<C: Ord + Clone> ResolvedGroupHierarchy<C> {
    /// Validate the carrier-resolved groups and build the hierarchy.
    ///
    /// `groups` must already have had each selector resolved into `coordinates`
    /// by the carrier. The order of `groups` is preserved for [`Self::groups`].
    pub fn build(groups: Vec<ResolvedGroup<C>>) -> Result<Self, String> {
        let mut seen = BTreeSet::<String>::new();
        for group in &groups {
            if group.label.trim().is_empty() {
                return Err("coefficient group label must not be empty".to_string());
            }
            if !seen.insert(group.label.clone()) {
                return Err(format!(
                    "duplicate coefficient group label '{}'",
                    group.label
                ));
            }
            if group.coordinates.is_empty() {
                return Err(format!(
                    "coefficient group '{}' contains no coefficients",
                    group.label
                ));
            }
        }

        let coordinates_by_label: BTreeMap<String, BTreeSet<C>> = groups
            .iter()
            .map(|group| (group.label.clone(), group.coordinates.clone()))
            .collect();
        let parent_by_label: BTreeMap<String, Option<String>> = groups
            .iter()
            .map(|group| (group.label.clone(), group.parent.clone()))
            .collect();
        let mut children_by_parent = BTreeMap::<String, Vec<String>>::new();
        for group in &groups {
            if let Some(parent) = group.parent.as_ref() {
                children_by_parent
                    .entry(parent.clone())
                    .or_default()
                    .push(group.label.clone());
            }
        }

        for group in &groups {
            let mut path = BTreeSet::<String>::new();
            let mut cursor = Some(group.label.as_str());
            while let Some(label) = cursor {
                if !path.insert(label.to_string()) {
                    return Err(format!(
                        "coefficient group hierarchy contains a cycle involving '{label}'"
                    ));
                }
                cursor = parent_by_label
                    .get(label)
                    .ok_or_else(|| {
                        format!("coefficient group hierarchy references unknown group '{label}'")
                    })?
                    .as_deref();
            }
            if let Some(parent) = group.parent.as_ref() {
                let parent_set = coordinates_by_label.get(parent).ok_or_else(|| {
                    format!(
                        "coefficient group '{}' references unknown parent group '{parent}'",
                        group.label
                    )
                })?;
                let child_set = coordinates_by_label
                    .get(&group.label)
                    .expect("resolved group coordinates should exist");
                if !child_set.is_subset(parent_set) {
                    return Err(format!(
                        "coefficient group '{}' is not a subset of parent group '{parent}'",
                        group.label
                    ));
                }
            }
            if let Some(children) = children_by_parent.get(&group.label) {
                let mut child_union = BTreeSet::<C>::new();
                for child in children {
                    let child_set = coordinates_by_label
                        .get(child)
                        .expect("child group coordinates should exist after resolution");
                    child_union.extend(child_set.iter().cloned());
                }
                let parent_set = coordinates_by_label
                    .get(&group.label)
                    .expect("parent group coordinates should exist after resolution");
                if &child_union != parent_set {
                    return Err(format!(
                        "coefficient group '{}' has children but its coefficients are not exactly the union of its child groups; nested supergroups concatenate child coefficients",
                        group.label
                    ));
                }
            }
        }

        Ok(Self {
            groups,
            coordinates_by_label,
            children_by_parent,
        })
    }

    /// The resolved groups, in their original declaration order.
    pub fn groups(&self) -> &[ResolvedGroup<C>] {
        &self.groups
    }

    /// Concatenated penalty components for `label`, recursively expanded.
    ///
    /// A leaf group yields a single component (its own coordinate set). An
    /// interior node yields the concatenation of its children's components,
    /// expanding recursively when a child is itself interior. This realizes the
    /// hierarchical-Gamma identity in which an interior node's coefficient
    /// vector is the concatenation of its child vectors under one precision:
    /// overlapping children stay separate factors so their log normalizers and
    /// quadratic contributions both add — it is not a block-sum shortcut.
    pub fn concatenated_penalty_components(&self, label: &str) -> Vec<BTreeSet<C>> {
        let Some(children) = self.children_by_parent.get(label) else {
            return vec![
                self.coordinates_by_label
                    .get(label)
                    .expect("coefficient group coordinates should exist")
                    .clone(),
            ];
        };
        let mut components = Vec::new();
        for child in children {
            components.extend(self.concatenated_penalty_components(child));
        }
        components
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group<C: Ord + Clone>(
        label: &str,
        parent: Option<&str>,
        coords: impl IntoIterator<Item = C>,
    ) -> ResolvedGroup<C> {
        ResolvedGroup {
            label: label.to_string(),
            parent: parent.map(str::to_string),
            coordinates: coords.into_iter().collect(),
        }
    }

    /// Equivalent group declarations on the two real carriers — standard-term
    /// columns (`usize`) and custom-family `(block, column)` coordinates — must
    /// produce identical resolved components and hierarchy structure. We encode
    /// the same declaration on both carriers under the bijection
    /// `col <-> (col / BLOCK_WIDTH, col % BLOCK_WIDTH)` and assert the
    /// concatenated penalty components correspond column-for-column.
    #[test]
    fn carriers_produce_matching_concatenated_components() {
        const BLOCK_WIDTH: usize = 4;
        let to_pair = |c: usize| (c / BLOCK_WIDTH, c % BLOCK_WIDTH);

        // Hierarchy: `root` = union of `left` and `right`; `left` is itself an
        // interior node = union of `leaf_a` and `leaf_b`. This exercises
        // recursive child expansion and concatenation (overlap-free here, so
        // the parent is the disjoint union of its leaves).
        let column_groups = vec![
            group::<usize>("leaf_a", Some("left"), [0usize, 1]),
            group::<usize>("leaf_b", Some("left"), [2usize, 3]),
            group::<usize>("left", Some("root"), [0usize, 1, 2, 3]),
            group::<usize>("right", Some("root"), [4usize, 5]),
            group::<usize>("root", None, [0usize, 1, 2, 3, 4, 5]),
        ];
        let pair_groups: Vec<ResolvedGroup<(usize, usize)>> = column_groups
            .iter()
            .map(|g| ResolvedGroup {
                label: g.label.clone(),
                parent: g.parent.clone(),
                coordinates: g.coordinates.iter().copied().map(to_pair).collect(),
            })
            .collect();

        let column_hierarchy =
            ResolvedGroupHierarchy::build(column_groups).expect("column carrier valid");
        let pair_hierarchy =
            ResolvedGroupHierarchy::build(pair_groups).expect("pair carrier valid");

        for label in ["leaf_a", "leaf_b", "left", "right", "root"] {
            let column_components = column_hierarchy.concatenated_penalty_components(label);
            let pair_components = pair_hierarchy.concatenated_penalty_components(label);
            // Map the column components through the carrier bijection and
            // require exact equality with the pair-carrier components.
            let mapped: Vec<BTreeSet<(usize, usize)>> = column_components
                .iter()
                .map(|component| component.iter().copied().map(to_pair).collect())
                .collect();
            assert_eq!(
                mapped, pair_components,
                "carrier components diverged for group '{label}'"
            );
        }

        // `root` is interior: it must expand into the four leaf columns of
        // `left` plus the two columns of `right`, as separate components per
        // child node (not a single merged set), proving recursive expansion.
        let root_components = column_hierarchy.concatenated_penalty_components("root");
        assert_eq!(
            root_components,
            vec![
                BTreeSet::from([0usize, 1]),
                BTreeSet::from([2usize, 3]),
                BTreeSet::from([4usize, 5]),
            ],
            "interior node must concatenate recursively expanded child components"
        );
        // A leaf yields exactly one component: its own coordinate set.
        assert_eq!(
            column_hierarchy.concatenated_penalty_components("leaf_a"),
            vec![BTreeSet::from([0usize, 1])]
        );
    }

    /// The shared policy must reject the same malformed declarations on every
    /// carrier with identical messages, so standard terms and custom families
    /// cannot diverge on hierarchy rules.
    #[test]
    fn policy_violations_are_carrier_agnostic() {
        // Child not a subset of parent.
        let err = ResolvedGroupHierarchy::build(vec![
            group::<usize>("child", Some("parent"), [0usize, 9]),
            group::<usize>("parent", None, [0usize, 1]),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            "coefficient group 'child' is not a subset of parent group 'parent'"
        );

        // Interior node whose coefficients are not exactly the child union.
        let err = ResolvedGroupHierarchy::build(vec![
            group::<usize>("child", Some("parent"), [0usize]),
            group::<usize>("parent", None, [0usize, 1]),
        ])
        .unwrap_err();
        assert!(
            err.contains("not exactly the union of its child groups"),
            "unexpected message: {err}"
        );

        // Cycle in the parent chain.
        let err = ResolvedGroupHierarchy::build(vec![
            group::<usize>("a", Some("b"), [0usize]),
            group::<usize>("b", Some("a"), [0usize]),
        ])
        .unwrap_err();
        assert!(
            err.contains("contains a cycle involving"),
            "unexpected message: {err}"
        );

        // Duplicate label.
        let err = ResolvedGroupHierarchy::build(vec![
            group::<usize>("g", None, [0usize]),
            group::<usize>("g", None, [1usize]),
        ])
        .unwrap_err();
        assert_eq!(err, "duplicate coefficient group label 'g'");

        // Empty coordinate set.
        let err = ResolvedGroupHierarchy::build(vec![group::<usize>("g", None, [])]).unwrap_err();
        assert_eq!(err, "coefficient group 'g' contains no coefficients");

        // Unknown parent reference.
        let err =
            ResolvedGroupHierarchy::build(vec![group::<usize>("g", Some("missing"), [0usize])])
                .unwrap_err();
        assert_eq!(
            err,
            "coefficient group hierarchy references unknown group 'missing'"
        );

        // Identical violation, identical message on the `(block, column)`
        // carrier — confirming the policy is genuinely carrier-agnostic.
        let err = ResolvedGroupHierarchy::build(vec![
            group::<(usize, usize)>("child", Some("parent"), [(0usize, 0usize), (1, 1)]),
            group::<(usize, usize)>("parent", None, [(0usize, 0usize)]),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            "coefficient group 'child' is not a subset of parent group 'parent'"
        );
    }
}
