//! Browser entry point and the few page elements the game talks to.

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
    crate::run();
}

fn element(id: &str) -> Option<web_sys::Element> {
    web_sys::window()?.document()?.get_element_by_id(id)
}

pub fn set_status(text: &str) {
    if let Some(el) = element("status") {
        el.set_text_content(Some(text));
    }
}

pub fn show_error(text: &str) {
    if let Some(el) = element("error") {
        el.set_text_content(Some(&format!(
            "Strata could not start: {text}. It needs a browser with WebGPU, such as a current Chrome, Edge, Safari or Firefox."
        )));
        let _ = el.set_attribute("style", "display:block");
    }
}

/// Whether the browser currently has the mouse locked to the game canvas.
pub fn pointer_locked() -> bool {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.pointer_lock_element())
        .is_some()
}

/// The value of `name` in the page URL's query string, if present.
pub fn url_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    search
        .trim_start_matches('?')
        .split('&')
        .find_map(|kv| kv.strip_prefix(name)?.strip_prefix('=').map(str::to_owned))
}

/// A fixed starting camera from the page URL, `?cam=x,y,z,yaw,pitch` (metres
/// and radians). Used to take the same screenshot before and after a change.
pub fn camera_from_url() -> Option<[f32; 5]> {
    let v: Vec<f32> = url_param("cam")?.split(',').filter_map(|x| x.parse().ok()).collect();
    v.try_into().ok()
}
