//! The WGSL is only compiled by the GPU driver at startup, so check it here.

#[test]
fn world_shader_validates() {
    let src = include_str!("../src/shaders/world.wgsl");
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
}
