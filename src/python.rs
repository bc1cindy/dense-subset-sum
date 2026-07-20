use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::{Transaction, kappa, KNEE};
use crate::harness::vs_cja::per_coin_measurements_fee_aware;
use crate::harness::vs_cja::{enumerate_mappings, non_derived_mappings, pairwise_input_output_prob};

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

/// Combined-coin cutoff: enumeration is exponential, so above this we return None (the caller
/// truncates rather than blocking). Tractability guard, not a modelling claim.
const LINK_GUARD: usize = 24;

#[pyfunction]
fn pairwise_link_prob(
    py: Python<'_>,
    inputs: Vec<u64>,
    outputs: Vec<u64>,
) -> PyResult<Option<Py<PyList>>> {
    if inputs.len() + outputs.len() > LINK_GUARD {
        return Ok(None);
    }
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
        outs.push(fee as u64);   // fee > 0 here and fee: i64 from Σin−Σout, so the cast is exact
        Transaction::new(tx.inputs.clone(), outs)
    } else {
        tx.clone()
    };
    let all = enumerate_mappings(&balanced);
    let non_derived = non_derived_mappings(&all);
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
