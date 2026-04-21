//! Minimal Prometheus text-exposition registry (replaces
//! `metrics-exporter-prometheus`).
//!
//! We only need counters and histograms, keyed by a small set of fixed
//! labels. Rolling our own drops ~40 transitive crates from `Cargo.lock`
//! (the `metrics`/`metrics-util`/`indexmap`/`atomic-waker`/…  stack) and
//! keeps the `/metrics` contract entirely in-tree. The emitted text
//! matches the Prometheus 0.0.4 exposition format documented at
//! <https://prometheus.io/docs/instrumenting/exposition_formats/>.
//!
//! ## Concurrency
//! `RwLock<HashMap<..>>` is fine for our workload — counters/histograms
//! are hit on every HTTP request (per-handler middleware), but the scrape
//! endpoint is typically polled every 15 s so reader contention is low.
//! When we need lock-free update later, swap `RwLock` for a sharded map.
//!
//! ## Default histogram buckets
//! Matches `metrics-exporter-prometheus`'s defaults, which themselves come
//! from the Prometheus Go client library. Tweakable per-metric via
//! `register_histogram_with_buckets`.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::RwLock;

/// Default histogram bucket bounds (seconds-scaled). Upper bound `f64::INFINITY`
/// is appended implicitly when rendering — consumers do not need to supply it.
pub const DEFAULT_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Sorted label set keyed by ASCII name. Sorting keeps the serialised
/// label string stable regardless of the insertion order so the same
/// counter + label combination always maps to the same storage slot.
pub type Labels = Vec<(String, String)>;

fn sort_labels(mut labels: Labels) -> Labels {
    labels.sort_by(|a, b| a.0.cmp(&b.0));
    labels
}

fn format_labels(labels: &Labels) -> String {
    if labels.is_empty() {
        return String::new();
    }
    let mut out = String::from("{");
    for (i, (k, v)) in labels.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(k);
        out.push_str("=\"");
        // Escape the label value — per the Prometheus text format, the
        // characters `\`, `"`, and `\n` must be escaped. Rare in practice
        // but cheap to do and prevents crafted label values from breaking
        // the exposition output.
        for ch in v.chars() {
            match ch {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\n' => out.push_str("\\n"),
                c => out.push(c),
            }
        }
        out.push('"');
    }
    out.push('}');
    out
}

#[derive(Debug, Default)]
struct CounterFamily {
    help: String,
    values: HashMap<Labels, u64>,
}

#[derive(Debug, Default)]
struct HistogramFamily {
    help: String,
    buckets: Vec<f64>,
    series: HashMap<Labels, HistogramSeries>,
}

#[derive(Debug, Default, Clone)]
struct HistogramSeries {
    /// Cumulative bucket counts; index `i` is observations ≤ `buckets[i]`.
    /// Trailing `+Inf` bucket is the grand total (`count`), not stored here.
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

/// Prometheus-compatible registry used by the server. Typically wrapped in
/// an `Arc` and stashed on `AppState` so every handler can record into it.
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    counters: RwLock<HashMap<String, CounterFamily>>,
    histograms: RwLock<HashMap<String, HistogramFamily>>,
}

impl MetricsRegistry {
    /// Create an empty registry. Families are declared lazily on first use
    /// via `counter_inc` / `histogram_record` — separate `register_*`
    /// methods exist for setting help text ahead of time.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the `# HELP` text for a counter family. Called during startup;
    /// overwrites any previously registered help text for the same name.
    pub fn register_counter(&self, name: &str, help: &str) {
        let mut map = self.counters.write().expect("counters lock poisoned");
        map.entry(name.to_string())
            .or_default()
            .help = help.to_string();
    }

    /// Set the `# HELP` text and bucket bounds for a histogram family.
    /// Buckets are sorted and deduplicated; callers may pass
    /// [`DEFAULT_BUCKETS`] for the Prometheus client default.
    pub fn register_histogram(&self, name: &str, help: &str, buckets: &[f64]) {
        let mut normalised: Vec<f64> = buckets.iter().copied().collect();
        normalised.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        normalised.dedup();
        let mut map = self.histograms.write().expect("histograms lock poisoned");
        let family = map.entry(name.to_string()).or_default();
        family.help = help.to_string();
        family.buckets = normalised;
    }

    /// Increment a counter. Lazily creates the family if it didn't exist.
    pub fn counter_inc(&self, name: &str, labels: Labels, delta: u64) {
        let labels = sort_labels(labels);
        let mut map = self.counters.write().expect("counters lock poisoned");
        let family = map.entry(name.to_string()).or_default();
        *family.values.entry(labels).or_insert(0) += delta;
    }

    /// Record one observation into a histogram. Lazily creates the family
    /// with [`DEFAULT_BUCKETS`] if it didn't exist.
    pub fn histogram_record(&self, name: &str, labels: Labels, value: f64) {
        let labels = sort_labels(labels);
        let mut map = self.histograms.write().expect("histograms lock poisoned");
        let family = map.entry(name.to_string()).or_default();
        if family.buckets.is_empty() {
            family.buckets = DEFAULT_BUCKETS.to_vec();
        }
        let series = family.series.entry(labels).or_insert_with(|| HistogramSeries {
            counts: vec![0; family.buckets.len()],
            sum: 0.0,
            count: 0,
        });
        // Keep the cumulative-counts vector in sync with the (possibly
        // re-registered) bucket list. Extending with zeros is correct
        // because extra buckets haven't seen observations yet.
        if series.counts.len() < family.buckets.len() {
            series.counts.resize(family.buckets.len(), 0);
        }
        for (i, &upper) in family.buckets.iter().enumerate() {
            if value <= upper {
                series.counts[i] += 1;
            }
        }
        series.sum += value;
        series.count += 1;
    }

    /// Render the current snapshot as Prometheus text. Formatting follows
    /// the `0.0.4; charset=utf-8` content type: `# HELP` and `# TYPE`
    /// comments per family, then one sample per (name, labels) pair.
    pub fn render_prometheus(&self) -> String {
        let mut out = String::new();

        // Counters first — stable alphabetical order for reproducible
        // scrape output across invocations.
        let counters = self.counters.read().expect("counters lock poisoned");
        let mut names: Vec<&String> = counters.keys().collect();
        names.sort();
        for name in names {
            let family = &counters[name];
            if !family.help.is_empty() {
                let _ = writeln!(out, "# HELP {name} {}", family.help);
            }
            let _ = writeln!(out, "# TYPE {name} counter");
            let mut label_keys: Vec<&Labels> = family.values.keys().collect();
            label_keys.sort();
            for labels in label_keys {
                let _ = writeln!(
                    out,
                    "{name}{} {}",
                    format_labels(labels),
                    family.values[labels]
                );
            }
            out.push('\n');
        }
        drop(counters);

        let histograms = self.histograms.read().expect("histograms lock poisoned");
        let mut names: Vec<&String> = histograms.keys().collect();
        names.sort();
        for name in names {
            let family = &histograms[name];
            if !family.help.is_empty() {
                let _ = writeln!(out, "# HELP {name} {}", family.help);
            }
            let _ = writeln!(out, "# TYPE {name} histogram");
            let mut label_keys: Vec<&Labels> = family.series.keys().collect();
            label_keys.sort();
            for labels in label_keys {
                let series = &family.series[labels];
                // Emit one `_bucket{le="<upper>"}` line per boundary plus
                // the implicit `+Inf` line carrying the grand total. When
                // the series has pre-existing labels we splice `le=` in as
                // another comma-separated entry; when it doesn't we emit
                // the `le=` label alone.
                let base = format_labels(labels);
                let inner = trim_outer_braces(&base);
                let le_prefix: &str = if inner.is_empty() { "" } else { "," };
                for (i, &upper) in family.buckets.iter().enumerate() {
                    let _ = writeln!(
                        out,
                        "{name}_bucket{{{inner}{le_prefix}le=\"{}\"}} {}",
                        fmt_f64_prom(upper),
                        series.counts[i],
                    );
                }
                let _ = writeln!(
                    out,
                    "{name}_bucket{{{inner}{le_prefix}le=\"+Inf\"}} {}",
                    series.count
                );
                let _ = writeln!(
                    out,
                    "{name}_sum{} {}",
                    base,
                    fmt_f64_prom(series.sum),
                );
                let _ = writeln!(
                    out,
                    "{name}_count{} {}",
                    base,
                    series.count,
                );
            }
            out.push('\n');
        }

        out
    }
}

/// Strip the surrounding `{ … }` from a pre-formatted label block so we
/// can splice the `le=…` sample label into the same comma-separated
/// sequence. Returns `""` when the input has no labels.
fn trim_outer_braces(formatted: &str) -> &str {
    if formatted.is_empty() {
        return "";
    }
    let inner = formatted
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or(formatted);
    if inner.is_empty() { "" } else { inner }
}

/// Format a float the way the Prometheus go client does: `+Inf` for
/// infinity, fixed-point for integer values, and default `{}` otherwise.
fn fmt_f64_prom(v: f64) -> String {
    if v.is_infinite() {
        return if v.is_sign_positive() { "+Inf".into() } else { "-Inf".into() };
    }
    if v.is_nan() {
        return "NaN".into();
    }
    if v == v.trunc() && v.abs() < 1e15 {
        // Render bucket bounds like `1` rather than `1` (prost float default is `1`)
        // but avoid scientific notation for small fractional boundaries.
        // `.fract() == 0` floats print as "1" via `{}` in Rust, so use the
        // default format and we're done.
    }
    format!("{v}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> MetricsRegistry {
        let r = MetricsRegistry::new();
        r.register_counter("gigastt_http_requests_total", "Total HTTP requests processed");
        r.register_histogram(
            "gigastt_http_request_duration_seconds",
            "HTTP request duration",
            DEFAULT_BUCKETS,
        );
        r
    }

    #[test]
    fn test_render_empty_registry() {
        let r = MetricsRegistry::new();
        assert_eq!(r.render_prometheus(), "");
    }

    #[test]
    fn test_counter_increment_and_render() {
        let r = registry();
        r.counter_inc(
            "gigastt_http_requests_total",
            vec![
                ("method".into(), "GET".into()),
                ("path".into(), "/health".into()),
                ("status".into(), "200".into()),
            ],
            1,
        );
        r.counter_inc(
            "gigastt_http_requests_total",
            vec![
                ("method".into(), "GET".into()),
                ("path".into(), "/health".into()),
                ("status".into(), "200".into()),
            ],
            2,
        );
        let text = r.render_prometheus();
        assert!(text.contains("# HELP gigastt_http_requests_total Total HTTP requests processed"));
        assert!(text.contains("# TYPE gigastt_http_requests_total counter"));
        assert!(text.contains("gigastt_http_requests_total{method=\"GET\",path=\"/health\",status=\"200\"} 3"));
    }

    #[test]
    fn test_histogram_bucket_cumulative() {
        let r = registry();
        let labels = vec![("method".into(), "GET".into())];
        for v in [0.001, 0.03, 0.3, 1.5] {
            r.histogram_record(
                "gigastt_http_request_duration_seconds",
                labels.clone(),
                v,
            );
        }
        let text = r.render_prometheus();
        // 0.001 ≤ 0.005 → contributes to every bucket including 0.005+
        // 0.03  ≤ 0.05  → contributes to 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
        // 0.3   ≤ 0.5   → contributes to 0.5, 1.0, 2.5, 5.0, 10.0
        // 1.5   ≤ 2.5   → contributes to 2.5, 5.0, 10.0
        assert!(text.contains("gigastt_http_request_duration_seconds_bucket{method=\"GET\",le=\"0.005\"} 1"));
        assert!(text.contains("gigastt_http_request_duration_seconds_bucket{method=\"GET\",le=\"0.05\"} 2"));
        assert!(text.contains("gigastt_http_request_duration_seconds_bucket{method=\"GET\",le=\"0.5\"} 3"));
        assert!(text.contains("gigastt_http_request_duration_seconds_bucket{method=\"GET\",le=\"+Inf\"} 4"));
        assert!(text.contains("gigastt_http_request_duration_seconds_count{method=\"GET\"} 4"));
    }

    #[test]
    fn test_label_ordering_stable() {
        let r = MetricsRegistry::new();
        r.counter_inc("c", vec![("b".into(), "1".into()), ("a".into(), "2".into())], 1);
        r.counter_inc("c", vec![("a".into(), "2".into()), ("b".into(), "1".into())], 4);
        let text = r.render_prometheus();
        // Same counter despite different insert order — totals to 5.
        assert!(text.contains("c{a=\"2\",b=\"1\"} 5"));
    }

    #[test]
    fn test_label_escaping() {
        let r = MetricsRegistry::new();
        r.counter_inc("c", vec![("l".into(), "a\"b\\c\nd".into())], 1);
        let text = r.render_prometheus();
        assert!(text.contains("c{l=\"a\\\"b\\\\c\\nd\"} 1"), "escape failed: {text}");
    }

    #[test]
    fn test_empty_labels_render() {
        let r = MetricsRegistry::new();
        r.counter_inc("c", vec![], 7);
        let text = r.render_prometheus();
        assert!(text.contains("c 7"));
    }

    #[test]
    fn test_histogram_sum_tracks_observations() {
        let r = MetricsRegistry::new();
        r.register_histogram("h", "H", &[1.0, 2.0]);
        r.histogram_record("h", vec![], 0.5);
        r.histogram_record("h", vec![], 1.5);
        r.histogram_record("h", vec![], 2.5);
        let text = r.render_prometheus();
        assert!(text.contains("h_sum 4.5"));
        assert!(text.contains("h_count 3"));
    }
}
