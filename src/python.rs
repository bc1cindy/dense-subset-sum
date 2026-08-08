use crate::harness::vs_cja::per_coin_measurements_fee_aware;
use crate::harness::vs_cja::{
    enumerate_mappings_within, non_derived_mappings_within, pairwise_input_output_prob,
};
use crate::{KNEE, Transaction, kappa};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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
    Ok(())
}
