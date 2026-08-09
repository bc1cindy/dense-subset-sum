use crate::harness::vs_cja::per_coin_measurements_fee_aware;
use crate::harness::vs_cja::{
    dense_uniform_matrix, enumerate_mappings_within, non_derived_mappings_within,
    pairwise_input_output_prob,
};
use crate::{Ambiguity, KNEE, Transaction, kappa};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::num::NonZeroUsize;

fn ambiguity_to_dict(py: Python<'_>, amb: Ambiguity) -> PyResult<Py<PyDict>> {
    let d = PyDict::new_bound(py);
    let kind = match amb {
        Ambiguity::Exact(_) => "exact",
        Ambiguity::LowerBound(_) => "lower_bound",
        Ambiguity::LogApprox(_) => "log_approx",
        Ambiguity::Unknown => "unknown",
    };
    d.set_item("kind", kind)?;
    d.set_item("count", amb.lower_bound_count())?; // Option<u128> -> int | None, no truncation
    d.set_item("log_w", amb.log())?;
    Ok(d.unbind())
}

#[pyfunction]
fn w_brute(
    py: Python<'_>,
    inputs: Vec<u64>,
    outputs: Vec<u64>,
    max_size: usize,
) -> PyResult<Py<PyDict>> {
    ambiguity_to_dict(py, crate::compute::w_brute(&inputs, &outputs, max_size))
}

#[pyfunction]
fn radix_mappings(py: Python<'_>, outputs: Vec<u64>, max_size: usize) -> PyResult<Py<PyDict>> {
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
    let mb = NonZeroUsize::new(memory_budget.max(1)).unwrap();
    ambiguity_to_dict(
        py,
        crate::compute::w_sparse(&inputs, &outputs, max_size, mb),
    )
}

#[pyfunction]
fn w_sasamoto(py: Python<'_>, inputs: Vec<u64>, outputs: Vec<u64>) -> PyResult<Py<PyDict>> {
    ambiguity_to_dict(py, crate::compute::w_sasamoto(&inputs, &outputs))
}

/// Feasibility-cascade dispatcher over the four W(E) paths: brute -> dp -> sparse -> sasamoto,
/// returning the best available `Ambiguity` (kind/count/log_w) plus the `method` string that produced
/// it ("brute"|"dp"|"sparse"|"sasamoto"|"none").
#[pyfunction]
fn w_count(py: Python<'_>, inputs: Vec<u64>, outputs: Vec<u64>) -> PyResult<Py<PyDict>> {
    use crate::compute::Method;
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
    let tx = Transaction::new(inputs.clone(), outputs);
    let n_in = inputs.len();
    let max_in = inputs.iter().copied().max().unwrap_or(0);
    let kap = kappa(max_in, n_in);

    let coins = PyList::empty_bound(py);
    for m in per_coin_measurements_fee_aware(&tx, KNEE) {
        let d = PyDict::new_bound(py);
        d.set_item("role", m.role.as_str())?;
        d.set_item("index", m.index)?;
        d.set_item("value", m.value)?;
        d.set_item("log_w", m.log_w_signed)?;
        d.set_item("kappa_c", m.kappa_c_at_value)?;
        coins.append(d)?;
    }
    let out = PyDict::new_bound(py);
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

#[pyfunction]
#[pyo3(signature = (inputs, outputs, budget_ms=None))]
fn pairwise_link_prob(
    py: Python<'_>,
    inputs: Vec<u64>,
    outputs: Vec<u64>,
    budget_ms: Option<u64>,
) -> PyResult<Option<Py<PyList>>> {
    // A budget replaces the size guard rather than joining it: the guard exists only
    // because there was no way to bound the cost directly, and applying both would
    // refuse the cheap large calls the budget was added to admit.
    let deadline = match budget_ms {
        Some(ms) => Some(std::time::Instant::now() + std::time::Duration::from_millis(ms)),
        None => {
            if inputs.len() + outputs.len() > LINK_GUARD {
                return Ok(None);
            }
            None
        }
    };
    let n_real_out = outputs.len();
    let tx = Transaction::new(inputs.clone(), outputs);
    let fee = tx.fee();
    if fee < 0 {
        return Ok(None);
    }
    // Dense-coinjoin fast path: a dense instance's link matrix is uniform (max ambiguity),
    // so emit it directly rather than running the exponential enumeration below.
    const RADIX_MIN: usize = 15;
    if let Some(m) = dense_uniform_matrix(&tx.inputs, &tx.outputs, RADIX_MIN) {
        let rows = PyList::empty_bound(py);
        for row in m {
            let r = PyList::empty_bound(py);
            for v in row {
                r.append(v)?;
            }
            rows.append(r)?;
        }
        return Ok(Some(rows.unbind()));
    }
    // Balance by appending the fee as an extra output (mirrors the fee-aware pipeline), enumerate,
    // then drop the fee column so the returned matrix is n_inputs x n_real_outputs.
    let balanced = if fee > 0 {
        let mut outs = tx.outputs.clone();
        outs.push(fee as u64); // fee > 0 here and fee: i64 from Σin−Σout, so the cast is exact
        Transaction::new(tx.inputs.clone(), outs)
    } else {
        tx.clone()
    };
    let Some(all) = enumerate_mappings_within(&balanced, deadline) else {
        return Ok(None);
    };
    let Some(non_derived) = non_derived_mappings_within(all.as_slice(), deadline) else {
        return Ok(None);
    };
    let matrix = pairwise_input_output_prob(&non_derived, &balanced);
    let rows = PyList::empty_bound(py);
    for row in matrix {
        let r = PyList::empty_bound(py);
        for v in row.into_iter().take(n_real_out) {
            r.append(v)?;
        }
        rows.append(r)?;
    }
    Ok(Some(rows.unbind()))
}

#[pymodule]
fn dss(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(per_coin_density, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_link_prob, m)?)?;
    m.add_function(wrap_pyfunction!(w_brute, m)?)?;
    m.add_function(wrap_pyfunction!(radix_mappings, m)?)?;
    m.add_function(wrap_pyfunction!(w_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(w_sasamoto, m)?)?;
    m.add_function(wrap_pyfunction!(w_count, m)?)?;
    Ok(())
}
