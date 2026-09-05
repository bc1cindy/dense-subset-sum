use crate::harness::vs_cja::per_coin_measurements_fee_aware;
use crate::harness::vs_cja::{
    Mapping, boltzmann_entropy, deterministic_links, enumerate_mappings_within,
    is_repeated_denomination_dense_case, non_derived_mappings_within, pairwise_input_output_prob,
};
use crate::{Ambiguity, KNEE, MAX_MONEY, Transaction, kappa};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

fn ambiguity_to_dict(py: Python<'_>, amb: Ambiguity) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    let kind = match amb {
        Ambiguity::Exact(_) => "exact",
        Ambiguity::LowerBound(_) => "lower_bound",
        // A count of denomination exchanges, not of W(E): it bounds the subset-sum count in
        // neither direction, so it must not share a kind with the tiers that do.
        Ambiguity::Diagnostic(_) => "diagnostic",
        Ambiguity::LogApprox(_) => "log_approx",
        Ambiguity::Unknown => "unknown",
    };
    d.set_item("kind", kind)?;
    d.set_item("count", amb.count())?; // Option<u128> -> int | None, no truncation
    d.set_item("log_w", amb.log())?;
    Ok(d.unbind())
}

/// Amount lists whose total leaves the Bitcoin domain are rejected at the boundary rather than
/// summed: the counting paths add them in `u64` and a debug build aborts on the overflow.
fn checked_amounts(label: &str, values: &[u64]) -> PyResult<()> {
    if valid_bitcoin_total(values) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{label} sum exceeds MAX_MONEY ({MAX_MONEY} sat)"
        )))
    }
}

fn checked_sides(inputs: &[u64], outputs: &[u64]) -> PyResult<()> {
    checked_amounts("inputs", inputs)?;
    checked_amounts("outputs", outputs)
}

#[pyfunction]
fn w_brute(
    py: Python<'_>,
    inputs: Vec<u64>,
    outputs: Vec<u64>,
    max_size: usize,
) -> PyResult<Py<PyDict>> {
    checked_sides(&inputs, &outputs)?;
    ambiguity_to_dict(py, crate::compute::w_brute(&inputs, &outputs, max_size))
}

#[pyfunction]
fn radix_mappings(py: Python<'_>, outputs: Vec<u64>, max_size: usize) -> PyResult<Py<PyDict>> {
    checked_amounts("outputs", &outputs)?;
    ambiguity_to_dict(py, crate::compute::radix_mappings(&outputs, max_size))
}

#[pyfunction]
#[pyo3(signature = (inputs, outputs, max_size, memory_budget=1_048_576))]
fn w_sparse(
    py: Python<'_>,
    inputs: Vec<u64>,
    outputs: Vec<u64>,
    max_size: usize,
    memory_budget: usize,
) -> PyResult<Py<PyDict>> {
    checked_sides(&inputs, &outputs)?;
    let mb = NonZeroUsize::new(memory_budget.max(1)).unwrap();
    ambiguity_to_dict(
        py,
        crate::compute::w_sparse(&inputs, &outputs, max_size, mb),
    )
}

#[pyfunction]
fn w_sasamoto(py: Python<'_>, inputs: Vec<u64>, outputs: Vec<u64>) -> PyResult<Py<PyDict>> {
    checked_sides(&inputs, &outputs)?;
    ambiguity_to_dict(py, crate::compute::w_sasamoto(&inputs, &outputs))
}

/// Feasibility-cascade dispatcher over the four W(E) paths: brute -> dp -> sparse -> sasamoto,
/// returning the accepted `Ambiguity` (kind/count/log_w) plus the `method` string that produced it
/// ("brute"|"dp"|"sparse"|"sasamoto"|"none"). Acceptance is Exact, then a Dense saddle-point
/// approximation, then a truncated lower bound — magnitude order, not guarantee order.
#[pyfunction]
fn w_count(py: Python<'_>, inputs: Vec<u64>, outputs: Vec<u64>) -> PyResult<Py<PyDict>> {
    use crate::compute::Method;
    checked_sides(&inputs, &outputs)?;
    let report = crate::compute::w_count(&inputs, &outputs);
    let d = ambiguity_to_dict(py, report.ambiguity)?;
    let method = match report.method {
        Method::Brute => "brute",
        Method::Dp => "dp",
        Method::Sparse => "sparse",
        Method::Sasamoto => "sasamoto",
        Method::None => "none",
    };
    d.bind(py).set_item("method", method)?;
    Ok(d)
}

#[pyfunction]
fn per_coin_density(py: Python<'_>, inputs: Vec<u64>, outputs: Vec<u64>) -> PyResult<Py<PyDict>> {
    checked_sides(&inputs, &outputs)?;
    let tx = Transaction::new(inputs.clone(), outputs);
    let n_in = inputs.len();
    let max_in = inputs.iter().copied().max().unwrap_or(0);
    let kap = kappa(max_in, n_in);

    let coins = PyList::empty(py);
    for m in per_coin_measurements_fee_aware(&tx, KNEE) {
        let d = PyDict::new(py);
        d.set_item("role", m.role.as_str())?;
        d.set_item("index", m.index)?;
        d.set_item("value", m.value)?;
        d.set_item("log_w", m.log_w_signed)?;
        d.set_item("kappa_c", m.kappa_c_at_value)?;
        coins.append(d)?;
    }
    let out = PyDict::new(py);
    out.set_item("kappa", kap)?;
    out.set_item("coins", coins)?;
    Ok(out.unbind())
}

/// Combined-coin cutoff used when no time budget is given: enumeration is exponential, so above
/// this we return None (the caller truncates rather than blocking). Tractability guard, not a
/// modelling claim — and a poor one, which is why `budget_ms` exists. Measured over one real
/// provenance walk, calls at or below this size still ranged from 54ms to 69s, and 64% of the
/// refusals it caused were between 25 and 32 coins, where the median call costs 641ms.
const LINK_GUARD: usize = 24;
const RADIX_MIN: usize = 15;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnalysisStop {
    SizeGuard,
    BudgetOverflow,
    AmountOutOfRange,
    NegativeFee,
    EnumerationDeadline,
    PruningDeadline,
}

impl AnalysisStop {
    const fn status(self) -> &'static str {
        match self {
            Self::SizeGuard | Self::EnumerationDeadline | Self::PruningDeadline => "refused",
            Self::BudgetOverflow | Self::AmountOutOfRange | Self::NegativeFee => "invalid",
        }
    }

    const fn reason(self) -> &'static str {
        match self {
            Self::SizeGuard => "size_guard",
            Self::BudgetOverflow => "budget_overflow",
            Self::AmountOutOfRange => "amount_out_of_range",
            Self::NegativeFee => "negative_fee",
            Self::EnumerationDeadline => "enumeration_deadline",
            Self::PruningDeadline => "pruning_deadline",
        }
    }
}

fn valid_bitcoin_total(values: &[u64]) -> bool {
    values
        .iter()
        .try_fold(0_u64, |total, value| total.checked_add(*value))
        .is_some_and(|total| total <= MAX_MONEY)
}

enum MappingAnalysis {
    DenseFastPath,
    Complete {
        transaction: Transaction,
        mappings: Vec<Mapping>,
        real_output_count: usize,
    },
    Stopped(AnalysisStop),
}

fn analyze_mappings(
    inputs: Vec<u64>,
    outputs: Vec<u64>,
    budget_ms: Option<u64>,
) -> MappingAnalysis {
    let deadline = match budget_ms {
        Some(ms) => match Instant::now().checked_add(Duration::from_millis(ms)) {
            Some(deadline) => Some(deadline),
            None => return MappingAnalysis::Stopped(AnalysisStop::BudgetOverflow),
        },
        None if inputs.len() + outputs.len() > LINK_GUARD => {
            return MappingAnalysis::Stopped(AnalysisStop::SizeGuard);
        }
        None => None,
    };

    if !valid_bitcoin_total(&inputs) || !valid_bitcoin_total(&outputs) {
        return MappingAnalysis::Stopped(AnalysisStop::AmountOutOfRange);
    }

    let real_output_count = outputs.len();
    let transaction = Transaction::new(inputs, outputs);
    let fee = transaction.fee();
    if fee < 0 {
        return MappingAnalysis::Stopped(AnalysisStop::NegativeFee);
    }
    if is_repeated_denomination_dense_case(&transaction.inputs, &transaction.outputs, RADIX_MIN) {
        return MappingAnalysis::DenseFastPath;
    }

    let balanced = if fee > 0 {
        let mut outputs = transaction.outputs.clone();
        outputs.push(fee as u64);
        Transaction::new(transaction.inputs.clone(), outputs)
    } else {
        transaction
    };
    let Some(all) = enumerate_mappings_within(&balanced, deadline) else {
        return MappingAnalysis::Stopped(AnalysisStop::EnumerationDeadline);
    };
    let Some(mappings) = non_derived_mappings_within(&all, deadline) else {
        return MappingAnalysis::Stopped(AnalysisStop::PruningDeadline);
    };
    MappingAnalysis::Complete {
        transaction: balanced,
        mappings,
        real_output_count,
    }
}

fn matrix_to_python(
    py: Python<'_>,
    matrix: Vec<Vec<f64>>,
    output_count: usize,
) -> PyResult<Py<PyList>> {
    let rows = PyList::empty(py);
    for row in matrix {
        let values = PyList::empty(py);
        for value in row.into_iter().take(output_count) {
            values.append(value)?;
        }
        rows.append(values)?;
    }
    Ok(rows.unbind())
}

#[pyfunction]
#[pyo3(signature = (inputs, outputs, budget_ms=None))]
fn pairwise_link_prob(
    py: Python<'_>,
    inputs: Vec<u64>,
    outputs: Vec<u64>,
    budget_ms: Option<u64>,
) -> PyResult<Option<Py<PyList>>> {
    match analyze_mappings(inputs, outputs, budget_ms) {
        MappingAnalysis::DenseFastPath => Ok(None),
        MappingAnalysis::Complete {
            transaction,
            mappings,
            real_output_count,
        } => matrix_to_python(
            py,
            pairwise_input_output_prob(&mappings, &transaction),
            real_output_count,
        )
        .map(Some),
        MappingAnalysis::Stopped(_) => Ok(None),
    }
}

/// Analyze the exact non-derived mapping family used by `pairwise_link_prob`.
///
/// The returned dictionary always contains `status`. A complete result also contains the mapping
/// count, entropy and deterministic input-output links. Refusals and invalid requests contain a
/// machine-readable `reason`; the dense fast path is reported separately because it deliberately
/// avoids enumeration and therefore has no exact mapping count.
#[pyfunction]
#[pyo3(signature = (inputs, outputs, budget_ms=None))]
fn mapping_analysis(
    py: Python<'_>,
    inputs: Vec<u64>,
    outputs: Vec<u64>,
    budget_ms: Option<u64>,
) -> PyResult<Py<PyDict>> {
    let result = PyDict::new(py);
    match analyze_mappings(inputs, outputs, budget_ms) {
        MappingAnalysis::DenseFastPath => result.set_item("status", "dense_fast_path")?,
        MappingAnalysis::Stopped(stop) => {
            result.set_item("status", stop.status())?;
            result.set_item("reason", stop.reason())?;
        }
        MappingAnalysis::Complete {
            transaction: _,
            mappings,
            real_output_count: _,
        } if mappings.is_empty() => {
            result.set_item("status", "infeasible")?;
            result.set_item("n_non_derived", 0)?;
        }
        MappingAnalysis::Complete {
            transaction,
            mappings,
            real_output_count,
        } => {
            let links = PyList::empty(py);
            for (input, output) in deterministic_links(&mappings, &transaction) {
                if output < real_output_count {
                    links.append((input, output))?;
                }
            }
            result.set_item("status", "complete")?;
            result.set_item("n_non_derived", mappings.len())?;
            result.set_item("entropy", boltzmann_entropy(mappings.len()))?;
            result.set_item("deterministic_links", links)?;
        }
    }
    Ok(result.unbind())
}

fn embedded_revision() -> Option<&'static str> {
    option_env!("DSS_GIT_REV").filter(|revision| is_valid_revision(revision))
}

fn is_valid_revision(revision: &str) -> bool {
    revision.len() == 40
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Require the immutable source revision used to build this extension.
///
/// Result-producing applications should call this before writing a publishable artifact. Normal
/// interactive use may continue to inspect `__rev__`, which is `None` for an unversioned build.
#[pyfunction]
#[pyo3(signature = (expected=None))]
fn require_build_revision(expected: Option<&str>) -> PyResult<String> {
    let revision = embedded_revision().ok_or_else(|| {
        PyRuntimeError::new_err(
            "this DSS build has no valid DSS_GIT_REV; published results require a 40-character Git revision",
        )
    })?;
    if let Some(expected) = expected
        && expected != revision
    {
        return Err(PyRuntimeError::new_err(format!(
            "DSS build revision {revision} does not match required revision {expected}"
        )));
    }
    Ok(revision.to_owned())
}

#[pymodule]
fn dss(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__rev__", embedded_revision())?;
    m.add_function(wrap_pyfunction!(per_coin_density, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_link_prob, m)?)?;
    m.add_function(wrap_pyfunction!(mapping_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(require_build_revision, m)?)?;
    m.add_function(wrap_pyfunction!(w_brute, m)?)?;
    m.add_function(wrap_pyfunction!(radix_mappings, m)?)?;
    m.add_function(wrap_pyfunction!(w_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(w_sasamoto, m)?)?;
    m.add_function(wrap_pyfunction!(w_count, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn build_revision_requires_canonical_lowercase_git_hash() {
        assert!(is_valid_revision(
            "0123456789abcdef0123456789abcdef01234567"
        ));
        assert!(!is_valid_revision(
            "0123456789ABCDEF0123456789ABCDEF01234567"
        ));
        assert!(!is_valid_revision("0123456789abcdef"));
        assert!(!is_valid_revision(
            "g123456789abcdef0123456789abcdef0123456"
        ));
    }

    #[test]
    fn mapping_analysis_rejects_amounts_outside_bitcoin_domain() {
        assert!(!valid_bitcoin_total(&[MAX_MONEY, 1]));
        assert!(!valid_bitcoin_total(&[u64::MAX, 1]));
        assert!(matches!(
            analyze_mappings(vec![MAX_MONEY + 1], vec![0], None),
            MappingAnalysis::Stopped(AnalysisStop::AmountOutOfRange)
        ));
    }

    proptest! {
        #[test]
        fn mapping_analysis_matches_direct_enumeration(
            inputs in prop::collection::vec(1_u64..20, 1..5),
        ) {
            let mut outputs = inputs.clone();
            outputs.reverse();
            let transaction = Transaction::new(inputs.clone(), outputs.clone());
            let all = enumerate_mappings_within(&transaction, None).expect("unbounded enumeration");
            let direct_mappings = non_derived_mappings_within(&all, None).expect("unbounded pruning");

            match analyze_mappings(inputs, outputs, None) {
                MappingAnalysis::Complete { transaction, mappings, real_output_count } => {
                    prop_assert_eq!(real_output_count, transaction.outputs.len());
                    prop_assert_eq!(&mappings, &direct_mappings);
                    let matrix = pairwise_input_output_prob(&mappings, &transaction);
                    let certain_from_matrix: Vec<_> = matrix.iter().enumerate().flat_map(|(i, row)| {
                        row.iter().enumerate().filter_map(move |(o, probability)| {
                            (*probability == 1.0).then_some((i, o))
                        })
                    }).collect();
                    prop_assert_eq!(deterministic_links(&mappings, &transaction), certain_from_matrix);
                }
                MappingAnalysis::DenseFastPath => {
                    prop_assert!(false, "small generated cases must not use the dense fast path");
                }
                MappingAnalysis::Stopped(stop) => {
                    prop_assert!(false, "unbounded valid case stopped: {stop:?}");
                }
            }
        }
    }
}
