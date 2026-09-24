"""Regression tests for release-cache startup and date-only fallback evidence."""

import contextlib
from datetime import datetime, timezone
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

import citrixscan as scanner


def _stamp(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


def _release(version, day, source="document-history"):
    return {"full_version": version, "release_date": day,
            "variant": "ADC", "source": source}


def _gzip(stamp):
    return {"status": 200, "data": b"\x1f\x8b\x08\x00" + stamp.to_bytes(4, "little")
            + b"\x00" * 20}


class ReleaseFallbackTests(unittest.TestCase):
    def test_gzip_nearest_candidate_can_be_before_mtime(self):
        releases = [_release("12.1-1.1", "2025-09-08"),
                    _release("12.1-2.1", "2025-09-11"),
                    _release("14.1-3.1", "2025-10-10")]
        with patch.object(scanner, "_get_release_catalog", return_value=(releases, [])):
            candidates, _ = scanner.infer_release_candidates(_stamp("2025-09-12"))
        self.assertEqual([c["position"] for c in candidates],
                         ["previous", "best", "next"])
        self.assertEqual(candidates[1]["full_version"], "12.1-2.1")
        self.assertEqual(candidates[1]["lead_days"], -1)
        self.assertIn("-1d", scanner._format_release_candidates(candidates))
        self.assertAlmostEqual(sum(c["probability"] for c in candidates), 100.0)

    def test_only_older_releases_still_have_a_best_candidate(self):
        with patch.object(scanner, "_get_release_catalog", return_value=(
                [_release("14.1-1.1", "2025-09-08"),
                 _release("14.1-2.1", "2025-09-10")], [])):
            candidates, _ = scanner.infer_release_candidates(_stamp("2025-09-20"))
        self.assertEqual([c["position"] for c in candidates], ["previous", "best"])
        self.assertEqual(candidates[-1]["full_version"], "14.1-2.1")
        self.assertLess(candidates[-1]["lead_days"], 0)

    def test_distant_historical_releases_do_not_underflow_scores(self):
        with patch.object(scanner, "_get_release_catalog", return_value=(
                [_release("12.1-1.1", "2018-09-08"),
                 _release("12.1-2.1", "2018-09-10")], [])):
            candidates, _ = scanner.infer_release_candidates(_stamp("2032-09-20"))
        self.assertEqual(candidates[-1]["position"], "best")
        self.assertAlmostEqual(sum(c["probability"] for c in candidates), 100.0)

    def test_epa_lowercase_last_modified_is_only_a_fallback(self):
        head = {"status": 200, "headers": {
            "last-modified": "Thu, 11 Sep 2025 08:00:00 GMT",
            "content-type": "application/octet-stream"}}
        with patch.object(scanner, "http_get_binary", return_value=None), \
             patch.object(scanner, "http_get", return_value=head), \
             patch.object(scanner, "_get_release_catalog", return_value=(
                 [_release("12.1-1.1", "2025-09-10")], [])):
            raw, source, confidence, diagnostic = scanner.extract_version(
                [], [], {}, None, "127.0.0.1", 443, 5, deep_scan=False)
        self.assertEqual(raw, "")
        self.assertEqual(confidence, "LOW")
        self.assertIn("EPA file Last-Modified fallback indicator", source)
        self.assertIn("best=12.1-1.1", source)
        self.assertNotIn("min version", source)
        self.assertNotIn("relative score", source)
        self.assertIn("EPA Last-Modified=", diagnostic)

    def test_epa_date_survives_without_a_release_catalog(self):
        head = {"status": 200, "headers": {
            "last-modified": "Thu, 11 Sep 2025 08:00:00 GMT",
            "content-type": "application/octet-stream"}}
        with patch.object(scanner, "http_get_binary", return_value=None), \
             patch.object(scanner, "http_get", return_value=head), \
             patch.object(scanner, "_get_release_catalog", return_value=([], [])):
            raw, source, confidence, _ = scanner.extract_version(
                [], [], {}, None, "127.0.0.1", 443, 5, deep_scan=False)
        self.assertEqual(raw, "")
        self.assertEqual(confidence, "LOW")
        self.assertIn("no dated release candidates available", source)

    def test_unknown_gzip_does_not_hide_epa_header_or_override_gzip_candidates(self):
        stamp = _stamp("2025-09-12")
        head = {"status": 200, "headers": {
            "LAST-MODIFIED": "Thu, 11 Sep 2025 08:00:00 GMT",
            "CONTENT-TYPE": "application/octet-stream"}}

        def get_binary(host, port, path, ctx, timeout, max_bytes):
            return _gzip(stamp) if path.endswith("rdx_en.json.gz") else None

        with patch.object(scanner, "http_get_binary", side_effect=get_binary), \
             patch.object(scanner, "http_get", return_value=head), \
             patch.object(scanner, "_get_release_catalog", return_value=(
                 [_release("14.1-2.1", "2025-09-11")], [])):
            raw, source, confidence, diagnostic = scanner.extract_version(
                [], [], {}, None, "127.0.0.1", 443, 5, deep_scan=False)
        self.assertEqual(raw, "")
        self.assertTrue(source.startswith("GZIP MTIME fallback indicator"))
        self.assertEqual(confidence, "LOW")
        self.assertIn("EPA Last-Modified=", diagnostic)

    def test_epa_header_ignores_html_login_page(self):
        head = {"status": 200, "headers": {
            "last-modified": "Thu, 11 Sep 2025 08:00:00 GMT",
            "content-type": "text/html", "content-length": "2000000"}}
        with patch.object(scanner, "http_get_binary", return_value=None), \
             patch.object(scanner, "http_get", return_value=head):
            raw, source, _, _ = scanner.extract_version(
                [], [], {}, None, "127.0.0.1", 443, 5, deep_scan=False)
        self.assertEqual((raw, source), ("", ""))

    def test_inferred_epa_date_does_not_set_eol_or_firmware_version(self):
        head = {"status": 200, "headers": {
            "server": "NetScaler Gateway", "last-modified": "Thu, 11 Sep 2025 08:00:00 GMT",
            "content-type": "application/octet-stream"}, "body": ""}

        def get(host, port, path, ctx, timeout, method="GET", max_body=8192):
            if method == "HEAD":
                return head
            return {"status": 200 if path == "/" else 404,
                    "headers": head["headers"] if path == "/" else {},
                    "body": "", "url": path}

        tls = {"protocol": "TLSv1.3", "cipher": "TLS_AES_128_GCM_SHA256",
               "bits": 128, "cn": "", "san": "", "issuer": "", "not_after": ""}
        with patch.object(scanner.socket, "gethostbyname", return_value="127.0.0.1"), \
             patch.object(scanner.socket, "create_connection", return_value=MagicMock()), \
             patch.object(scanner, "create_ssl_context", return_value=None), \
             patch.object(scanner, "get_tls_info", return_value=tls), \
             patch.object(scanner, "http_get", side_effect=get), \
             patch.object(scanner, "http_get_binary", return_value=None), \
             patch.object(scanner, "_get_release_catalog", return_value=(
                 [_release("12.1-1.1", "2025-09-10")], [])):
            result = scanner.scan_target("127.0.0.1", modules="headers",
                                         deep_scan=False)
        self.assertTrue(result.is_netscaler)
        self.assertTrue(result.epa_available)
        self.assertEqual(result.version_raw, "")
        self.assertEqual(result.version_display, "")
        self.assertEqual(result.branch, "")
        self.assertFalse(result.eol)
        self.assertEqual(result.cve_results, [])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            scanner.print_result(result)
        self.assertIn("Version    : UNKNOWN", output.getvalue())
        self.assertIn("Fallback indicator (not a firmware version, LOW)", output.getvalue())

    def test_case_insensitive_security_headers(self):
        resp = {"status": 200, "headers": {
            "strict-transport-security": "max-age=1", "x-frame-options": "DENY",
            "x-content-type-options": "nosniff", "content-security-policy": "default-src 'self'"}}
        self.assertEqual(scanner.check_security_headers([resp]), [])

    def test_header_only_does_not_claim_cves_are_clear(self):
        result = scanner.ScanResult(
            target="example.test", ip="127.0.0.1", port=443, timestamp="now",
            is_netscaler=True, version_raw="14.1-66.59",
            version_parsed=(14, 1, 66, 59), version_display="14.1-66.59",
            modules_run=["headers"], risk_rating="UNKNOWN")
        self.assertEqual(scanner.calculate_risk(result), "UNKNOWN")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            scanner.print_result(result)
        self.assertIn("CVE assessment: not run", output.getvalue())
        self.assertNotIn("None found for", output.getvalue())

    def test_cve_module_requires_confirmed_version(self):
        result = scanner.ScanResult(
            target="example.test", ip="127.0.0.1", port=443, timestamp="now",
            is_netscaler=True, modules_run=["cve"], risk_rating="MEDIUM")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            scanner.print_result(result)
        self.assertIn("CVE assessment: unavailable (no confirmed firmware version)",
                      output.getvalue())


class StartupCacheTests(unittest.TestCase):
    def test_current_product_name_is_accepted_for_api_catalog(self):
        def api(path, params=None, timeout=10):
            if path == "/products":
                return [{"name": "NetScaler ADC (includes NetScaler Gateway)", "id": 1}]
            if path == "/versions":
                return [{"version": "NetScaler 14.1", "id": 2}]
            return [{"build_number": "66.59", "release_date": "2025-11-05"}]

        with tempfile.TemporaryDirectory() as folder, \
             patch.object(scanner, "RELEASE_CACHE_FILE", os.path.join(folder, "cache.json")), \
             patch.object(scanner, "_release_api_get", side_effect=api):
            releases, notes = scanner._fetch_and_cache_releases(3)
        self.assertEqual(notes, [])
        self.assertEqual(releases[0]["full_version"], "14.1-66.59")
        self.assertEqual(releases[0]["source"], "release-api")

    def test_manual_releases_written_even_when_api_offline(self):
        with tempfile.TemporaryDirectory() as folder:
            history_path = os.path.join(folder, "history.json")
            cache_path = os.path.join(folder, "cache.json")
            with open(history_path, "w", encoding="utf-8") as handle:
                json.dump({"releases": [_release("14.1-1.1", "2025-09-09")]}, handle)
            with patch.object(scanner, "RELEASE_CACHE_FILE", cache_path), \
                 patch.object(scanner, "DOCUMENT_HISTORY_FILE", history_path), \
                 patch.object(scanner, "_release_catalog", None), \
                 patch.object(scanner, "_release_catalog_notes", []), \
                 patch.object(scanner, "_release_api_get", side_effect=urllib.error.URLError("offline")):
                catalog, notes = scanner._get_release_catalog(3)
            self.assertEqual(len(catalog), 1)
            self.assertTrue(any("unavailable" in note for note in notes))
            with open(cache_path, encoding="utf-8") as handle:
                cached = json.load(handle)
            self.assertEqual(cached["releases"][0]["full_version"], "14.1-1.1")

    def test_cli_refreshes_catalog_with_headers_only_and_no_deep(self):
        with patch.object(sys, "argv", ["citrixscan.py", "127.0.0.1", "--no-deep"]), \
             patch.object(scanner, "_get_release_catalog", return_value=([], [])) as refresh, \
             patch.object(scanner, "scan_target", return_value=scanner.ScanResult(
                 target="127.0.0.1", ip="127.0.0.1", port=443, timestamp="now")) as scan, \
             patch.object(scanner, "print_result"), \
             patch.object(scanner, "print_summary"), \
             contextlib.redirect_stdout(io.StringIO()):
            scanner.main()
        refresh.assert_called_once_with(15)
        scan.assert_called_once()
        self.assertEqual(scan.call_args.args[-2:], ({"headers"}, False))


if __name__ == "__main__":
    unittest.main()
