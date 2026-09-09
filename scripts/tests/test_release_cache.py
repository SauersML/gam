"""Release cache policy contracts, executed without a compiler or cache service."""

import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HELPER = ROOT / ".github/scripts/release-cache.sh"


def shell(script, receipt=""):
    return subprocess.run(
        ["bash", "-c", 'set -euo pipefail; source "$1"; ' + script, "bash", str(HELPER)],
        input=receipt,
        text=True,
        capture_output=True,
        check=False,
    )


def receipt(**changes):
    counters = {
        "Compile requests": 1041,
        "Compile requests executed": 983,
        "Cache hits": 0,
        "Cache misses": 983,
        "Cache hits rate": "0.00 %",
        "Cache read errors": 0,
        "Cache write errors": 0,
        "Cache errors": 0,
    }
    counters.update(changes)
    return "".join(f"{key:36} {value}\n" for key, value in counters.items())


class ReleaseCacheTests(unittest.TestCase):
    def test_cold_release_is_valid_and_seeds_cache(self):
        result = shell("release_cache_verify_receipt", receipt())
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("::notice::Cold compiler cache", result.stdout)
        self.assertIn("requests=1041, hits=0, misses=983", result.stdout)

    def test_warm_receipt_accepts_crlf_and_language_counters(self):
        stats = receipt(**{"Cache hits": 983, "Cache misses": 0})
        stats += "Cache hits (Rust) 983\nCache hits rate (Rust) 100.00 %\n"
        result = shell("release_cache_verify_receipt", stats.replace("\n", "\r\n"))
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertNotIn("Cold compiler cache", result.stdout)

    def test_zero_requests_is_not_evidence_of_a_cold_build(self):
        result = shell(
            "release_cache_verify_receipt",
            receipt(**{"Compile requests": 0, "Cache misses": 0}),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No compiler requests", result.stdout)

    def test_each_read_error_counter_refuses_publication(self):
        for counter in ("Cache errors", "Cache read errors"):
            with self.subTest(counter=counter):
                result = shell("release_cache_verify_receipt", receipt(**{counter: 1}))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unhealthy cache receipt", result.stdout)

    def test_write_errors_are_reported_but_do_not_refuse_a_measured_build(self):
        """A write error is the backend refusing to store an object the compiler
        already produced (Publish to PyPI 34376153113: 80/120/387 on jobs whose
        every compile request executed). The artifact is unaffected; the receipt
        names the unseeded cache instead of refusing it."""
        result = shell("release_cache_verify_receipt", receipt(**{"Cache write errors": 387}))
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("::warning::Compiler cache refused 387 write(s)", result.stdout)
        self.assertIn("Measured release build: requests=1041", result.stdout)

    def test_missing_malformed_or_duplicate_counters_refuse_publication(self):
        for counter in (
            "Compile requests", "Cache hits", "Cache misses", "Cache errors",
            "Cache read errors", "Cache write errors",
        ):
            for invalid in (None, "invalid", -1, 1.5, "", "1 trailing"):
                with self.subTest(counter=counter, invalid=invalid):
                    stats = receipt(**{counter: invalid})
                    if invalid is None:
                        stats = "\n".join(
                            line for line in stats.splitlines()
                            if not line.startswith(f"{counter:36} ")
                        )
                    result = shell("release_cache_verify_receipt", stats)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Incomplete or invalid", result.stdout)
            result = shell("release_cache_verify_receipt", receipt() + f"{counter} 0\n")
            self.assertNotEqual(result.returncode, 0)

    def test_failed_stats_command_cannot_publish_a_valid_looking_receipt(self):
        result = shell(
            "sccache() { cat; return 23; }; release_cache_verify", receipt()
        )
        self.assertEqual(result.returncode, 23)

    def test_startup_retry_is_bounded_and_disables_idle_shutdown(self):
        result = shell("""
          attempts=0
          sccache() {
            [[ "$SCCACHE_IDLE_TIMEOUT" == 0 ]] || return 90
            if [[ "$1" == --start-server ]]; then
              attempts=$((attempts + 1))
              echo "start $attempts"
              (( attempts == 3 ))
            else
              echo "$1"
            fi
          }
          sleep() { echo "sleep $1"; }
          release_cache_start
          [[ "$RUSTC_WRAPPER" == sccache ]]
        """)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("sleep 20\n", result.stdout)
        self.assertIn("sleep 40\n", result.stdout)
        self.assertIn("--zero-stats\n", result.stdout)
        self.assertNotIn("start 4", result.stdout)

    def test_exhausted_startup_retries_never_reset_counters(self):
        result = shell("""
          sccache() { echo "$1"; return 7; }
          sleep() { echo "sleep $1"; }
          release_cache_start
        """)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout.count("--start-server\n"), 5)
        self.assertEqual(result.stdout.count("sleep "), 4)
        self.assertNotIn("--zero-stats", result.stdout)

    def test_counter_reset_failure_stops_before_build(self):
        result = shell("""
          sccache() { [[ "$1" == --start-server ]] || return 17; }
          release_cache_start
        """)
        self.assertEqual(result.returncode, 17)


if __name__ == "__main__":
    unittest.main()
