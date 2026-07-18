use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::{Transaction, kappa, KNEE};
use crate::harness::vs_cja::per_coin_measurements;

#[pyfunction]
fn per_coin_density(py: Python<'_>, inputs: Vec<u64>, outputs: Vec<u64>) -> PyResult<Py<PyDict>> {
    let tx = Transaction::new(inputs.clone(), outputs);
    let n_in = inputs.len();
    let max_in = inputs.iter().copied().max().unwrap_or(0);
    let kap = kappa(max_in, n_in);

    let coins = PyList::empty_bound(py);
    for m in per_coin_measurements(&tx, KNEE) {
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

#[pymodule]
fn dss(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(per_coin_density, m)?)?;
    Ok(())
}
