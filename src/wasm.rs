use ndarray::Array1;

use rusfun;
use crate::{sphere, cube, superball};

use wasm_bindgen::prelude::*;

/// Translates a string to a implemented function in rusfun to make them easily callable
pub fn get_function(function_name: &str) -> fn(&Array1<f64>, &Array1<f64>) -> Array1<f64> {
    match function_name {
        "sphere" => sphere::formfactor,
        "cube" => cube::formfactor,
        "superball" => superball::formfactor,
        _ => sphere::formfactor
    }
}

/// Calls Rust defined model functions by their function name for given parameters and a domain
#[wasm_bindgen]
pub fn superball_model(function_name: &str, p: Vec<f64>, x: Vec<f64>) -> Vec<f64> {
    rusfun::wasm::calculate_model(p, x, get_function(function_name))
}