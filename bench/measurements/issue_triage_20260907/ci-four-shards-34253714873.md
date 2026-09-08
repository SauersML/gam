# Issue 2627: four completed Rust shards

CI run 34253714873, commit 7c3c5b43b0e34823d2fd4c9d135d7cac63694676:

| Shard | Run | Passed | Failed | Timed out | Shard-filtered | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 971 | 954 | 12 | 5 | 8,630 | 4,203.898 |
| 7 | 950 | 926 | 21 | 3 | 8,651 | 4,595.737 |
| 8 | 943 | 915 | 24 | 4 | 8,658 | 3,779.472 |
| 9 | 939 | 914 | 23 | 2 | 8,662 | 2,414.331 |
| Completed subset | 3,803 | 3,709 | 80 | 14 | — | — |

The four shards select disjoint subsets of the same archived 9,601-test
workspace population. Their filtered counts must not be added as test skips.
Six Rust shards and both Python jobs were still active at this observation.
The separate gam-pyffi population finished with 120 passed and 4 failed of 124.
This is an older-head partial census, not current-main certification or evidence
that the complete issue has been fixed.

Exact failure identities are saved in the neighboring
`rust-shard{3,7,8,9}-summary-34253714873.txt` files. The newly downloaded shard 3
and 7 logs come from the workflow's uploaded `shard-logs-3` and `shard-logs-7`
artifacts and are retained under `ci-34253714873/`.

MSI validation of the local certificate-limit fix remains unavailable on this
second observation: `msi doctor` reports both acn112 and acn116 DOWN; `sinfo`
shows only invalid, drained, or unknown compute states. There are no owned
scheduler jobs. Formatting the two edited Rust files on the MSI login node
succeeded; this is syntax-formatting evidence only, not compilation or tests.
