//! Typed Linux cgroup memory observations.
//!
//! `sysinfo::CGroupLimits` is intentionally not used here. Its aggregate loses
//! the distinction between the literal cgroup-v2 `memory.max = max` token and a
//! finite limit, derives "free" as `max - memory.current` even though
//! `memory.current` includes reclaimable file cache, and probes a fixed cgroup
//! path rather than the current process' hierarchy. Memory admission needs the
//! kernel contract before any of that information is erased.

use std::fmt;

/// The exact syntax of one cgroup-v2 `memory.max` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CgroupMemoryLimit {
    /// The literal `max` token: this level imposes no hard memory ceiling.
    Unlimited,
    /// A finite hard ceiling in bytes. Zero is valid and authoritative.
    Finite(u64),
}

/// Why a live cgroup memory hierarchy could not be observed safely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CgroupMemoryProbeFailureKind {
    MalformedMembership,
    MissingUnifiedMount,
    MalformedMountInfo,
    Io,
    InvalidLimit,
    InvalidCounter,
    MissingCounter,
    InconsistentCounters,
    UnsupportedCgroupV1,
}

impl fmt::Display for CgroupMemoryProbeFailureKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::MalformedMembership => "malformed-membership",
            Self::MissingUnifiedMount => "missing-unified-mount",
            Self::MalformedMountInfo => "malformed-mountinfo",
            Self::Io => "io",
            Self::InvalidLimit => "invalid-limit",
            Self::InvalidCounter => "invalid-counter",
            Self::MissingCounter => "missing-counter",
            Self::InconsistentCounters => "inconsistent-counters",
            Self::UnsupportedCgroupV1 => "unsupported-cgroup-v1",
        };
        formatter.write_str(name)
    }
}

/// Fail-closed evidence from an active cgroup controller probe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CgroupMemoryProbeFailure {
    kind: CgroupMemoryProbeFailureKind,
    path: Box<str>,
    detail: Box<str>,
}

impl CgroupMemoryProbeFailure {
    pub const fn kind(&self) -> CgroupMemoryProbeFailureKind {
        self.kind
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for CgroupMemoryProbeFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at {}: {}",
            self.kind, self.path, self.detail
        )
    }
}

/// Reclaim-aware headroom under the binding finite cgroup-v2 ancestor.
///
/// `working_set_bytes = memory.current - inactive_file`. Only inactive file
/// cache is credited as reclaimable; active file cache and reclaimable slab are
/// deliberately left in the working set. The governor's separate 1/4 headroom
/// remains available for reclaim latency, allocator slack, and untracked work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CgroupMemoryAvailability {
    binding_path: Box<str>,
    limit_bytes: u64,
    current_bytes: u64,
    inactive_file_bytes: u64,
    working_set_bytes: u64,
    available_bytes: u64,
    inspected_levels: usize,
}

impl CgroupMemoryAvailability {
    pub fn binding_path(&self) -> &str {
        &self.binding_path
    }

    pub const fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }

    pub const fn current_bytes(&self) -> u64 {
        self.current_bytes
    }

    pub const fn inactive_file_bytes(&self) -> u64 {
        self.inactive_file_bytes
    }

    pub const fn working_set_bytes(&self) -> u64 {
        self.working_set_bytes
    }

    pub const fn available_bytes(&self) -> u64 {
        self.available_bytes
    }

    pub const fn inspected_levels(&self) -> usize {
        self.inspected_levels
    }
}

#[cfg(test)]
mod tests_fixtures {
    use super::*;

    impl CgroupMemoryProbeFailure {
        pub(crate) fn fixture(
            kind: CgroupMemoryProbeFailureKind,
            path: impl Into<Box<str>>,
            detail: impl Into<Box<str>>,
        ) -> Self {
            Self {
                kind,
                path: path.into(),
                detail: detail.into(),
            }
        }
    }

    impl CgroupMemoryAvailability {
        pub(crate) fn fixture(
            binding_path: impl Into<Box<str>>,
            limit_bytes: u64,
            current_bytes: u64,
            inactive_file_bytes: u64,
            inspected_levels: usize,
        ) -> Self {
            let working_set_bytes = current_bytes
                .checked_sub(inactive_file_bytes)
                .expect("cgroup test fixture counters must be internally consistent");
            Self {
                binding_path: binding_path.into(),
                limit_bytes,
                current_bytes,
                inactive_file_bytes,
                working_set_bytes,
                available_bytes: limit_bytes.saturating_sub(working_set_bytes),
                inspected_levels,
            }
        }
    }
}

impl fmt::Display for CgroupMemoryAvailability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "finite cgroup-v2 ceiling at {} (limit={}, current={}, inactive_file={}, working_set={}, available={}, visible_levels={})",
            self.binding_path,
            self.limit_bytes,
            self.current_bytes,
            self.inactive_file_bytes,
            self.working_set_bytes,
            self.available_bytes,
            self.inspected_levels,
        )
    }
}

/// The process' typed cgroup memory provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CgroupMemoryObservation {
    /// No active memory controller applies on this platform/hierarchy.
    NotPresent,
    /// A cgroup-v2 hierarchy was found, but every visible hard limit was the
    /// literal `max` token (or the process is at the unconstrained root).
    V2Unbounded {
        cgroup_path: Box<str>,
        inspected_levels: usize,
    },
    /// At least one finite hard ceiling applies. This carries the ancestor with
    /// the least reclaim-aware headroom.
    V2Limited(CgroupMemoryAvailability),
    /// A memory controller appears active but its semantics could not be read
    /// exactly. Admission must fail closed rather than inherit host memory.
    ProbeFailed(CgroupMemoryProbeFailure),
}

impl fmt::Display for CgroupMemoryObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPresent => formatter.write_str("no active cgroup memory controller"),
            Self::V2Unbounded {
                cgroup_path,
                inspected_levels,
            } => write!(
                formatter,
                "unbounded cgroup-v2 hierarchy at {cgroup_path} ({inspected_levels} memory.max levels)"
            ),
            Self::V2Limited(observation) => observation.fmt(formatter),
            Self::ProbeFailed(failure) => write!(formatter, "cgroup probe failed: {failure}"),
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn detect_cgroup_memory() -> CgroupMemoryObservation {
    CgroupMemoryObservation::NotPresent
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::ffi::OsString;
    use std::fs;
    use std::io;
    use std::os::unix::ffi::OsStringExt;
    use std::path::{Component, Path, PathBuf};

    impl CgroupMemoryProbeFailure {
        fn new(
            kind: CgroupMemoryProbeFailureKind,
            path: impl Into<Box<str>>,
            detail: impl Into<Box<str>>,
        ) -> Self {
            Self {
                kind,
                path: path.into(),
                detail: detail.into(),
            }
        }
    }

    impl CgroupMemoryAvailability {
        fn from_consistent_counters(
            binding_path: impl Into<Box<str>>,
            limit_bytes: u64,
            current_bytes: u64,
            inactive_file_bytes: u64,
            inspected_levels: usize,
        ) -> Option<Self> {
            let working_set_bytes = current_bytes.checked_sub(inactive_file_bytes)?;
            Some(Self {
                binding_path: binding_path.into(),
                limit_bytes,
                current_bytes,
                inactive_file_bytes,
                working_set_bytes,
                available_bytes: limit_bytes.saturating_sub(working_set_bytes),
                inspected_levels,
            })
        }
    }

    const PROC_SELF_CGROUP: &str = "/proc/self/cgroup";
    const PROC_SELF_MOUNTINFO: &str = "/proc/self/mountinfo";

    #[derive(Debug)]
    struct UnifiedMount {
        root: PathBuf,
        mount_point: PathBuf,
    }

    pub(super) fn detect() -> CgroupMemoryObservation {
        detect_from_proc_files(
            Path::new(PROC_SELF_CGROUP),
            Path::new(PROC_SELF_MOUNTINFO),
        )
        .unwrap_or_else(CgroupMemoryObservation::ProbeFailed)
    }

    fn detect_from_proc_files(
        cgroup_file: &Path,
        mountinfo_file: &Path,
    ) -> Result<CgroupMemoryObservation, CgroupMemoryProbeFailure> {
        let membership_text = read_required(cgroup_file)?;
        let membership = match parse_unified_membership(&membership_text, cgroup_file)? {
            Some(path) => path,
            None => return Ok(CgroupMemoryObservation::NotPresent),
        };
        let mountinfo_text = read_required(mountinfo_file)?;
        let mount = select_unified_mount(&mountinfo_text, &membership, mountinfo_file)?;
        let relative = membership.strip_prefix(&mount.root).map_err(|_| {
            failure(
                CgroupMemoryProbeFailureKind::MissingUnifiedMount,
                mountinfo_file,
                format!(
                    "cgroup path {} is outside selected mount root {}",
                    membership.display(),
                    mount.root.display()
                ),
            )
        })?;
        let relative = relative
            .strip_prefix(Path::new("/"))
            .unwrap_or(relative);
        let leaf = mount.mount_point.join(relative);
        inspect_visible_hierarchy(&leaf, &mount.mount_point)
    }

    fn parse_unified_membership(
        text: &str,
        source: &Path,
    ) -> Result<Option<PathBuf>, CgroupMemoryProbeFailure> {
        let mut unified = None;
        let mut legacy_memory = false;
        for line in text.lines().filter(|line| !line.is_empty()) {
            let mut fields = line.splitn(3, ':');
            let hierarchy = fields.next();
            let controllers = fields.next();
            let path = fields.next();
            let (Some(hierarchy), Some(controllers), Some(path)) =
                (hierarchy, controllers, path)
            else {
                return Err(failure(
                    CgroupMemoryProbeFailureKind::MalformedMembership,
                    source,
                    format!("invalid /proc/self/cgroup record {line:?}"),
                ));
            };
            if controllers.split(',').any(|name| name == "memory") {
                legacy_memory = true;
            }
            if hierarchy == "0" && controllers.is_empty() {
                if unified.is_some() {
                    return Err(failure(
                        CgroupMemoryProbeFailureKind::MalformedMembership,
                        source,
                        "multiple unified cgroup-v2 memberships",
                    ));
                }
                let path = PathBuf::from(path);
                validate_absolute_cgroup_path(&path, source)?;
                unified = Some(path);
            }
        }
        if legacy_memory {
            return Err(failure(
                CgroupMemoryProbeFailureKind::UnsupportedCgroupV1,
                source,
                "an active cgroup-v1 memory controller has no cgroup-v2 limit semantics",
            ));
        }
        Ok(unified)
    }

    fn validate_absolute_cgroup_path(
        path: &Path,
        source: &Path,
    ) -> Result<(), CgroupMemoryProbeFailure> {
        if !path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(failure(
                CgroupMemoryProbeFailureKind::MalformedMembership,
                source,
                format!("invalid unified cgroup path {}", path.display()),
            ));
        }
        Ok(())
    }

    fn select_unified_mount(
        text: &str,
        membership: &Path,
        source: &Path,
    ) -> Result<UnifiedMount, CgroupMemoryProbeFailure> {
        let mut selected: Option<UnifiedMount> = None;
        for line in text.lines().filter(|line| !line.is_empty()) {
            let Some((before_separator, after_separator)) = line.split_once(" - ") else {
                return Err(failure(
                    CgroupMemoryProbeFailureKind::MalformedMountInfo,
                    source,
                    format!("mountinfo record has no separator: {line:?}"),
                ));
            };
            let after_fields = after_separator.split_whitespace().collect::<Vec<_>>();
            if after_fields.first().copied() != Some("cgroup2") {
                continue;
            }
            let before_fields = before_separator.split_whitespace().collect::<Vec<_>>();
            if before_fields.len() < 6 {
                return Err(failure(
                    CgroupMemoryProbeFailureKind::MalformedMountInfo,
                    source,
                    format!("short cgroup2 mountinfo record: {line:?}"),
                ));
            }
            let root = decode_mountinfo_path(before_fields[3], source)?;
            let mount_point = decode_mountinfo_path(before_fields[4], source)?;
            if !root.is_absolute() || !mount_point.is_absolute() {
                return Err(failure(
                    CgroupMemoryProbeFailureKind::MalformedMountInfo,
                    source,
                    format!(
                        "cgroup2 mount paths must be absolute (root={}, mount_point={})",
                        root.display(),
                        mount_point.display()
                    ),
                ));
            }
            if membership.strip_prefix(&root).is_err() {
                continue;
            }
            let candidate = UnifiedMount { root, mount_point };
            let candidate_depth = candidate.root.components().count();
            let selected_depth = selected
                .as_ref()
                .map_or(0, |mount| mount.root.components().count());
            if selected.is_none() || candidate_depth > selected_depth {
                selected = Some(candidate);
            }
        }
        selected.ok_or_else(|| {
            failure(
                CgroupMemoryProbeFailureKind::MissingUnifiedMount,
                source,
                format!(
                    "no cgroup2 mount covers process membership {}",
                    membership.display()
                ),
            )
        })
    }

    fn decode_mountinfo_path(
        raw: &str,
        source: &Path,
    ) -> Result<PathBuf, CgroupMemoryProbeFailure> {
        let bytes = raw.as_bytes();
        let mut decoded = Vec::with_capacity(bytes.len());
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] != b'\\' {
                decoded.push(bytes[index]);
                index += 1;
                continue;
            }
            let octal = bytes
                .get(index + 1..index + 4)
                .filter(|digits| digits.iter().all(|digit| (b'0'..=b'7').contains(digit)));
            let Some(octal) = octal else {
                return Err(failure(
                    CgroupMemoryProbeFailureKind::MalformedMountInfo,
                    source,
                    format!("invalid mountinfo path escape in {raw:?}"),
                ));
            };
            let value = u16::from(octal[0] - b'0') * 64
                + u16::from(octal[1] - b'0') * 8
                + u16::from(octal[2] - b'0');
            let value = u8::try_from(value).map_err(|_| {
                failure(
                    CgroupMemoryProbeFailureKind::MalformedMountInfo,
                    source,
                    format!("mountinfo path escape exceeds one byte in {raw:?}"),
                )
            })?;
            decoded.push(value);
            index += 4;
        }
        Ok(PathBuf::from(OsString::from_vec(decoded)))
    }

    fn inspect_visible_hierarchy(
        leaf: &Path,
        mount_point: &Path,
    ) -> Result<CgroupMemoryObservation, CgroupMemoryProbeFailure> {
        if !leaf.starts_with(mount_point) {
            return Err(failure(
                CgroupMemoryProbeFailureKind::MissingUnifiedMount,
                leaf,
                format!("leaf is outside mount point {}", mount_point.display()),
            ));
        }
        let leaf_metadata = fs::metadata(leaf).map_err(|error| io_failure(leaf, error))?;
        if !leaf_metadata.is_dir() {
            return Err(failure(
                CgroupMemoryProbeFailureKind::MissingUnifiedMount,
                leaf,
                "resolved process cgroup is not a directory",
            ));
        }
        let mut directory = leaf.to_path_buf();
        let mut inspected_levels = 0usize;
        let mut binding: Option<CgroupMemoryAvailability> = None;
        loop {
            let max_path = directory.join("memory.max");
            let current_path = directory.join("memory.current");
            let stat_path = directory.join("memory.stat");
            match read_optional(&max_path)? {
                Some(raw_limit) => {
                    inspected_levels = inspected_levels.saturating_add(1);
                    match parse_limit(&raw_limit, &max_path)? {
                        CgroupMemoryLimit::Unlimited => {}
                        CgroupMemoryLimit::Finite(limit_bytes) => {
                            let current_before_raw =
                                read_optional(&current_path)?.ok_or_else(|| {
                                    failure(
                                        CgroupMemoryProbeFailureKind::MissingCounter,
                                        &current_path,
                                        "finite memory.max requires memory.current",
                                    )
                                })?;
                            let current_before =
                                parse_counter(&current_before_raw, &current_path)?;
                            let stat_raw = read_optional(&stat_path)?.ok_or_else(|| {
                                failure(
                                    CgroupMemoryProbeFailureKind::MissingCounter,
                                    &stat_path,
                                    "finite memory.max requires memory.stat",
                                )
                            })?;
                            let inactive_file_bytes =
                                parse_stat_counter(&stat_raw, "inactive_file", &stat_path)?;
                            // `memory.current` and `memory.stat` are live files,
                            // not an atomic snapshot. Bracket the stat read and
                            // use the larger current value: this is conservative
                            // for admission and prevents an ordinary concurrent
                            // charge/uncharge from looking like malformed data.
                            let current_after_raw =
                                read_optional(&current_path)?.ok_or_else(|| {
                                    failure(
                                        CgroupMemoryProbeFailureKind::MissingCounter,
                                        &current_path,
                                        "memory.current disappeared during cgroup probe",
                                    )
                                })?;
                            let current_after =
                                parse_counter(&current_after_raw, &current_path)?;
                            let current_bytes = current_before.max(current_after);
                            if inactive_file_bytes > current_bytes {
                                return Err(failure(
                                    CgroupMemoryProbeFailureKind::InconsistentCounters,
                                    &stat_path,
                                    format!(
                                        "inactive_file={inactive_file_bytes} exceeds memory.current={current_bytes}"
                                    ),
                                ));
                            }
                            let candidate = CgroupMemoryAvailability::from_consistent_counters(
                                directory.display().to_string().into_boxed_str(),
                                limit_bytes,
                                current_bytes,
                                inactive_file_bytes,
                                0,
                            )
                            .ok_or_else(|| {
                                failure(
                                    CgroupMemoryProbeFailureKind::InconsistentCounters,
                                    &stat_path,
                                    "memory counters became inconsistent during construction",
                                )
                            })?;
                            if binding.as_ref().map_or(true, |current| {
                                candidate.available_bytes() < current.available_bytes()
                            }) {
                                binding = Some(candidate);
                            }
                        }
                    }
                }
                None => {
                    let has_current = read_optional(&current_path)?.is_some();
                    let has_stat = read_optional(&stat_path)?.is_some();
                    // The cgroup-v2 root is exempt from resource control and
                    // therefore has accounting files but no `memory.max`.
                    // Every non-root level with controller accounting must
                    // expose its hard-limit file; otherwise the observation is
                    // incomplete and cannot safely inherit host capacity.
                    if directory != mount_point && (has_current || has_stat) {
                        return Err(failure(
                            CgroupMemoryProbeFailureKind::MissingCounter,
                            &max_path,
                            "memory controller files are present but memory.max is missing",
                        ));
                    }
                }
            }
            if directory == mount_point {
                break;
            }
            let Some(parent) = directory.parent() else {
                return Err(failure(
                    CgroupMemoryProbeFailureKind::MissingUnifiedMount,
                    &directory,
                    "cgroup hierarchy ended before its mount point",
                ));
            };
            directory = parent.to_path_buf();
        }
        if let Some(mut binding) = binding {
            binding.inspected_levels = inspected_levels;
            Ok(CgroupMemoryObservation::V2Limited(binding))
        } else {
            Ok(CgroupMemoryObservation::V2Unbounded {
                cgroup_path: leaf.display().to_string().into_boxed_str(),
                inspected_levels,
            })
        }
    }

    fn parse_limit(
        raw: &str,
        source: &Path,
    ) -> Result<CgroupMemoryLimit, CgroupMemoryProbeFailure> {
        let value = raw.trim();
        if value == "max" {
            return Ok(CgroupMemoryLimit::Unlimited);
        }
        let bytes = value.parse::<u64>().map_err(|_| {
            failure(
                CgroupMemoryProbeFailureKind::InvalidLimit,
                source,
                format!("expected literal max or an unsigned byte count, got {value:?}"),
            )
        })?;
        Ok(CgroupMemoryLimit::Finite(bytes))
    }

    fn parse_counter(raw: &str, source: &Path) -> Result<u64, CgroupMemoryProbeFailure> {
        let value = raw.trim();
        value.parse::<u64>().map_err(|_| {
            failure(
                CgroupMemoryProbeFailureKind::InvalidCounter,
                source,
                format!("expected an unsigned byte count, got {value:?}"),
            )
        })
    }

    fn parse_stat_counter(
        raw: &str,
        key: &str,
        source: &Path,
    ) -> Result<u64, CgroupMemoryProbeFailure> {
        for line in raw.lines() {
            let mut fields = line.split_whitespace();
            let Some(name) = fields.next() else {
                continue;
            };
            if name != key {
                continue;
            }
            let Some(value) = fields.next() else {
                return Err(failure(
                    CgroupMemoryProbeFailureKind::InvalidCounter,
                    source,
                    format!("memory.stat key {key:?} has no value"),
                ));
            };
            if fields.next().is_some() {
                return Err(failure(
                    CgroupMemoryProbeFailureKind::InvalidCounter,
                    source,
                    format!("memory.stat key {key:?} has trailing fields"),
                ));
            }
            return value.parse::<u64>().map_err(|_| {
                failure(
                    CgroupMemoryProbeFailureKind::InvalidCounter,
                    source,
                    format!("memory.stat key {key:?} is not an unsigned byte count"),
                )
            });
        }
        Err(failure(
            CgroupMemoryProbeFailureKind::MissingCounter,
            source,
            format!("memory.stat is missing required key {key:?}"),
        ))
    }

    fn read_required(path: &Path) -> Result<String, CgroupMemoryProbeFailure> {
        fs::read_to_string(path).map_err(|error| io_failure(path, error))
    }

    fn read_optional(path: &Path) -> Result<Option<String>, CgroupMemoryProbeFailure> {
        match fs::read_to_string(path) {
            Ok(value) => Ok(Some(value)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(io_failure(path, error)),
        }
    }

    fn io_failure(path: &Path, error: io::Error) -> CgroupMemoryProbeFailure {
        failure(
            CgroupMemoryProbeFailureKind::Io,
            path,
            error.to_string(),
        )
    }

    fn failure(
        kind: CgroupMemoryProbeFailureKind,
        path: &Path,
        detail: impl Into<Box<str>>,
    ) -> CgroupMemoryProbeFailure {
        CgroupMemoryProbeFailure::new(
            kind,
            path.display().to_string().into_boxed_str(),
            detail,
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::fs;
        use tempfile::TempDir;

        struct Fixture {
            _temp: TempDir,
            mount: PathBuf,
            cgroup_file: PathBuf,
            mountinfo_file: PathBuf,
        }

        impl Fixture {
            fn new(membership: &str) -> Self {
                let temp = TempDir::new().expect("fixture tempdir");
                let mount = temp.path().join("cgroup2");
                fs::create_dir_all(&mount).expect("fixture mount");
                let cgroup_file = temp.path().join("self.cgroup");
                fs::write(&cgroup_file, format!("0::{membership}\n"))
                    .expect("fixture membership");
                let mountinfo_file = temp.path().join("self.mountinfo");
                fs::write(
                    &mountinfo_file,
                    format!(
                        "29 23 0:26 / {} rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n",
                        mount.display()
                    ),
                )
                .expect("fixture mountinfo");
                Self {
                    _temp: temp,
                    mount,
                    cgroup_file,
                    mountinfo_file,
                }
            }

            fn level(&self, relative: &str, limit: &str, current: u64, inactive: u64) {
                let directory = self.mount.join(relative.trim_start_matches('/'));
                fs::create_dir_all(&directory).expect("fixture level");
                fs::write(directory.join("memory.max"), format!("{limit}\n"))
                    .expect("fixture memory.max");
                fs::write(directory.join("memory.current"), format!("{current}\n"))
                    .expect("fixture memory.current");
                fs::write(
                    directory.join("memory.stat"),
                    format!("anon 1\ninactive_file {inactive}\nactive_file 2\n"),
                )
                .expect("fixture memory.stat");
            }

            fn observe(&self) -> CgroupMemoryObservation {
                detect_from_proc_files(&self.cgroup_file, &self.mountinfo_file)
                    .unwrap_or_else(CgroupMemoryObservation::ProbeFailed)
            }
        }

        #[test]
        fn literal_max_is_typed_unbounded_even_when_current_is_large() {
            let fixture = Fixture::new("/tenant/leaf");
            fixture.level("tenant/leaf", "max", u64::MAX - 1, 0);
            assert!(matches!(
                fixture.observe(),
                CgroupMemoryObservation::V2Unbounded {
                    inspected_levels: 1,
                    ..
                }
            ));
        }

        #[test]
        fn cgroup_v2_root_accounting_without_memory_max_is_unbounded() {
            let fixture = Fixture::new("/");
            fs::write(fixture.mount.join("memory.current"), "1000\n")
                .expect("root current");
            fs::write(fixture.mount.join("memory.stat"), "inactive_file 700\n")
                .expect("root stat");
            assert!(matches!(
                fixture.observe(),
                CgroupMemoryObservation::V2Unbounded {
                    inspected_levels: 0,
                    ..
                }
            ));
        }

        #[test]
        fn finite_zero_remains_authoritative() {
            let fixture = Fixture::new("/tenant/leaf");
            fixture.level("tenant/leaf", "0", 0, 0);
            let CgroupMemoryObservation::V2Limited(observation) = fixture.observe() else {
                panic!("finite zero must be a real cgroup ceiling");
            };
            assert_eq!(observation.limit_bytes(), 0);
            assert_eq!(observation.available_bytes(), 0);
        }

        #[test]
        fn inactive_file_cache_is_conservatively_reclaimable() {
            let fixture = Fixture::new("/tenant/leaf");
            fixture.level("tenant/leaf", "1000", 1000, 700);
            let CgroupMemoryObservation::V2Limited(observation) = fixture.observe() else {
                panic!("finite ceiling must bind");
            };
            assert_eq!(observation.working_set_bytes(), 300);
            assert_eq!(observation.available_bytes(), 700);
        }

        #[test]
        fn finite_working_set_exhaustion_is_zero() {
            let fixture = Fixture::new("/tenant/leaf");
            fixture.level("tenant/leaf", "1000", 1200, 200);
            let CgroupMemoryObservation::V2Limited(observation) = fixture.observe() else {
                panic!("finite ceiling must bind");
            };
            assert_eq!(observation.working_set_bytes(), 1000);
            assert_eq!(observation.available_bytes(), 0);
        }

        #[test]
        fn binding_parent_accounts_for_sibling_pressure() {
            let fixture = Fixture::new("/tenant/leaf");
            fixture.level("tenant/leaf", "max", 200, 100);
            fixture.level("tenant", "1000", 950, 100);
            let CgroupMemoryObservation::V2Limited(observation) = fixture.observe() else {
                panic!("finite parent must bind an unlimited leaf");
            };
            assert_eq!(observation.available_bytes(), 150);
            assert!(observation.binding_path().ends_with("/tenant"));
            assert_eq!(observation.inspected_levels(), 2);
        }

        #[test]
        fn most_specific_covering_mount_resolves_non_root_membership() {
            let temp = TempDir::new().expect("fixture tempdir");
            let broad = temp.path().join("broad");
            let narrow = temp.path().join("narrow");
            fs::create_dir_all(&broad).expect("broad mount");
            fs::create_dir_all(narrow.join("leaf")).expect("narrow leaf");
            fs::write(narrow.join("leaf/memory.max"), "512\n").expect("max");
            fs::write(narrow.join("leaf/memory.current"), "256\n").expect("current");
            fs::write(narrow.join("leaf/memory.stat"), "inactive_file 64\n")
                .expect("stat");
            let cgroup_file = temp.path().join("self.cgroup");
            fs::write(&cgroup_file, "0::/tenant/leaf\n").expect("membership");
            let mountinfo_file = temp.path().join("self.mountinfo");
            fs::write(
                &mountinfo_file,
                format!(
                    "29 23 0:26 / {} rw - cgroup2 cgroup rw\n30 23 0:26 /tenant {} rw - cgroup2 cgroup rw\n",
                    broad.display(),
                    narrow.display()
                ),
            )
            .expect("mountinfo");
            let observation = detect_from_proc_files(&cgroup_file, &mountinfo_file)
                .expect("typed cgroup observation");
            let CgroupMemoryObservation::V2Limited(observation) = observation else {
                panic!("narrow finite mount must bind");
            };
            assert_eq!(observation.available_bytes(), 320);
            assert!(observation.binding_path().ends_with("/narrow/leaf"));
        }

        #[test]
        fn malformed_active_controller_fails_closed() {
            let fixture = Fixture::new("/tenant/leaf");
            fixture.level("tenant/leaf", "1000", 500, 100);
            fs::write(
                fixture.mount.join("tenant/leaf/memory.current"),
                "not-a-counter\n",
            )
            .expect("corrupt current");
            let CgroupMemoryObservation::ProbeFailed(failure) = fixture.observe() else {
                panic!("malformed active controller must fail closed");
            };
            assert_eq!(
                failure.kind(),
                CgroupMemoryProbeFailureKind::InvalidCounter
            );
        }

        #[test]
        fn inconsistent_cache_counter_fails_closed() {
            let fixture = Fixture::new("/tenant/leaf");
            fixture.level("tenant/leaf", "1000", 100, 101);
            let CgroupMemoryObservation::ProbeFailed(failure) = fixture.observe() else {
                panic!("inconsistent controller counters must fail closed");
            };
            assert_eq!(
                failure.kind(),
                CgroupMemoryProbeFailureKind::InconsistentCounters
            );
        }

        #[test]
        fn active_cgroup_v1_memory_controller_fails_closed() {
            let temp = TempDir::new().expect("fixture tempdir");
            let cgroup_file = temp.path().join("self.cgroup");
            fs::write(&cgroup_file, "4:memory:/legacy\n").expect("membership");
            let mountinfo_file = temp.path().join("self.mountinfo");
            fs::write(&mountinfo_file, "").expect("mountinfo");
            let failure = detect_from_proc_files(&cgroup_file, &mountinfo_file)
                .expect_err("v1 memory cannot inherit host availability");
            assert_eq!(
                failure.kind(),
                CgroupMemoryProbeFailureKind::UnsupportedCgroupV1
            );
        }

        #[test]
        fn hybrid_v1_memory_and_v2_membership_fails_closed() {
            let fixture = Fixture::new("/tenant/leaf");
            fs::write(
                &fixture.cgroup_file,
                "0::/tenant/leaf\n4:memory:/legacy\n",
            )
            .expect("hybrid membership");
            let failure = detect_from_proc_files(
                &fixture.cgroup_file,
                &fixture.mountinfo_file,
            )
            .expect_err("an active v1 memory controller cannot be ignored");
            assert_eq!(
                failure.kind(),
                CgroupMemoryProbeFailureKind::UnsupportedCgroupV1
            );
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn detect_cgroup_memory() -> CgroupMemoryObservation {
    linux::detect()
}

#[cfg(all(test, not(target_os = "linux")))]
mod non_linux_tests {
    use super::*;

    #[test]
    fn non_linux_platform_has_no_cgroup_controller() {
        assert_eq!(
            detect_cgroup_memory(),
            CgroupMemoryObservation::NotPresent
        );
    }
}
