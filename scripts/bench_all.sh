#!/usr/bin/env bash
#
# Run the signed-probe bench and print a paste-friendly report to stdout.
# Default grid (dense+stdenom × mid+q1 × N∈{16,32,64,128} × k=8 = 32 cells)
# finishes in ~5 s. Widen via env vars: BENCH_REGIMES, BENCH_TARGETS,
# BENCH_NS, BENCH_BLOCK_SIZES (comma-separated).

set -euo pipefail

cd "$(dirname "$0")/.."

env_label="$(hostname -s | tr '[:upper:]' '[:lower:]')"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

quick_tsv="$tmp_dir/quick.tsv"

rustc_version="$(rustc --version 2>/dev/null || echo 'rustc: unknown')"
os_info="$(uname -srm)"
commit_sha="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
commit_dirty=""
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    commit_dirty=" (dirty)"
fi
run_date="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
grid_regimes="${BENCH_REGIMES:-dense,stdenom}"
grid_targets="${BENCH_TARGETS:-mid,q1}"
grid_ns="${BENCH_NS:-16,32,64,128}"
grid_ks="${BENCH_BLOCK_SIZES:-8}"

# |regimes| × |targets| × |Ns| × (1 sasamoto + |ks| lookups)
count_csv() { echo "$1" | tr ',' '\n' | grep -c .; }
n_regimes=$(count_csv "$grid_regimes")
n_targets=$(count_csv "$grid_targets")
n_ns=$(count_csv "$grid_ns")
n_ks=$(count_csv "$grid_ks")
total_cells=$(( n_regimes * n_targets * n_ns * (1 + n_ks) ))

echo "==> building bench binary" >&2
cargo build --release --bench signed_probe 2>/dev/null >&2

echo "==> running signed-probe harness ($total_cells cells, ~5 s)" >&2
cargo bench --bench signed_probe -- --nocapture 2>/dev/null > "$quick_tsv"

cat <<EOF
=== dense-subset-sum bench report ===
env label:  $env_label
date (UTC): $run_date
commit:     $commit_sha$commit_dirty
rustc:      $rustc_version
uname:      $os_info
grid:       regimes=$grid_regimes targets=$grid_targets N=$grid_ns k=$grid_ks

--- signed-probe grid ---
EOF

awk -F'\t' '
function humanize_ns(ns) {
    if (ns < 1000)            return sprintf("%d ns", ns)
    if (ns < 1000000)         return sprintf("%.2f us", ns/1000)
    if (ns < 1000000000)      return sprintf("%.2f ms", ns/1000000)
    return sprintf("%.2f s", ns/1000000000)
}
function humanize_bytes(b) {
    if (b < 1024)             return sprintf("%d B", b)
    if (b < 1048576)          return sprintf("%.1f KiB", b/1024)
    if (b < 1073741824)       return sprintf("%.1f MiB", b/1048576)
    return sprintf("%.2f GiB", b/1073741824)
}
NR == 1 {
    ncols = NF
    for (i=1; i<=NF; i++) { header[i] = $i; width[i] = length($i) }
    next
}
{
    r = NR - 1
    for (i=1; i<=NF; i++) {
        v = $i
        if (header[i] == "median_ns")       v = humanize_ns(v+0)
        else if (header[i] == "peak_bytes") v = humanize_bytes(v+0)
        cell[r, i] = v
        if (length(v) > width[i]) width[i] = length(v)
    }
    nrows = r
}
END {
    for (i=1; i<=ncols; i++) {
        printf "%-*s", width[i], header[i]
        if (i < ncols) printf "  "
    }
    print ""
    for (i=1; i<=ncols; i++) {
        for (j=1; j<=width[i]; j++) printf "-"
        if (i < ncols) printf "  "
    }
    print ""
    for (r=1; r<=nrows; r++) {
        for (i=1; i<=ncols; i++) {
            printf "%-*s", width[i], cell[r, i]
            if (i < ncols) printf "  "
        }
        print ""
    }
}
' "$quick_tsv"

cat <<EOF

--- key comparisons ---
EOF

awk -F'\t' '
function humanize_ns(ns) {
    if (ns < 1000)            return sprintf("%d ns", ns)
    if (ns < 1000000)         return sprintf("%.2f us", ns/1000)
    if (ns < 1000000000)      return sprintf("%.2f ms", ns/1000000)
    return sprintf("%.2f s", ns/1000000000)
}
function humanize_bytes(b) {
    if (b < 1024)             return sprintf("%d B", b)
    if (b < 1048576)          return sprintf("%.1f KiB", b/1024)
    if (b < 1073741824)       return sprintf("%.1f MiB", b/1048576)
    return sprintf("%.2f GiB", b/1073741824)
}
function ratio(a, b) {
    if (b <= 0 || a <= 0) return "n/a"
    if (a >= b) return sprintf("%.0fx", a/b)
    return sprintf("1/%.0fx", b/a)
}
NR == 1 { next }
{
    r = NR - 1
    reg[r] = $1; tgt[r] = $2; n[r] = $3; meth[r] = $4
    ns_v[r] = $5 + 0; bytes_v[r] = $6 + 0; some[r] = ($7 == "true")
    nrows = r
    if (!some[r]) bail_count++
    # Index by (regime, target, N, method) for lookup below.
    key = $1 "|" $2 "|" $3 "|" $4
    idx_ns[key] = $5 + 0
    idx_bytes[key] = $6 + 0
    idx_some[key] = ($7 == "true")
    seen_sasamoto[$1 "|" $2 "|" $3] = 1
    # Remember the k of the first lookup method we see.
    if ($4 ~ /^lookup_k/) {
        first_lk[$1 "|" $2 "|" $3] = $4
    }
}
END {
    # Cell 1: realistic stdenom at q1 with largest N, if available.
    k1 = "stdenom|q1|" find_max_n("stdenom", "q1")
    key_s = k1 "|sasamoto"
    lk = first_lk[k1]
    if (lk != "" && key_s in idx_ns) {
        key_l = k1 "|" lk
        if (idx_some[key_s] && idx_some[key_l]) {
            printf "stdenom q1 N=%s:  sasamoto %s/%s  vs  %s %s/%s   (time %s, mem %s)\n", \
                n_of(k1), \
                humanize_ns(idx_ns[key_s]), humanize_bytes(idx_bytes[key_s]), \
                lk, humanize_ns(idx_ns[key_l]), humanize_bytes(idx_bytes[key_l]), \
                ratio(idx_ns[key_l], idx_ns[key_s]), \
                ratio(idx_bytes[key_l], idx_bytes[key_s])
        } else {
            printf "stdenom q1 N=%s:  sasamoto %s  lookup %s  (one of them bailed out)\n", \
                n_of(k1), \
                (idx_some[key_s] ? "ok" : "infeasible"), \
                (idx_some[key_l] ? "ok" : "infeasible")
        }
    }
    # Cell 2: any sasamoto bail-out where the matching lookup works.
    for (r=1; r<=nrows; r++) {
        if (meth[r] == "sasamoto" && !some[r]) {
            k = reg[r] "|" tgt[r] "|" n[r]
            lk = first_lk[k]
            if (lk != "" && idx_some[k "|" lk]) {
                printf "%s %s N=%s:  sasamoto infeasible  vs  %s %s/%s   (only lookup works)\n", \
                    reg[r], tgt[r], n[r], lk, \
                    humanize_ns(idx_ns[k "|" lk]), humanize_bytes(idx_bytes[k "|" lk])
                break
            }
        }
    }
    # Feasibility totals.
    sasamoto_total = 0; sasamoto_bail = 0
    for (r=1; r<=nrows; r++) {
        if (meth[r] == "sasamoto") {
            sasamoto_total++
            if (!some[r]) sasamoto_bail++
        }
    }
    if (sasamoto_total > 0) {
        printf "sasamoto returned None in %d/%d cells\n", sasamoto_bail, sasamoto_total
    }
}
function find_max_n(regime, target,   r, maxn) {
    maxn = 0
    for (r=1; r<=nrows; r++) {
        if (reg[r] == regime && tgt[r] == target && (n[r]+0) > maxn) maxn = n[r]+0
    }
    return maxn
}
function n_of(key,   parts) {
    split(key, parts, "|")
    return parts[3]
}
' "$quick_tsv"

cat <<EOF

--- end of report ---
Paste this block into a PR or issue comment to share the numbers.
EOF
