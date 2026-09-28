
// ---------------------------------------------------------------- actors
// Appended to world.wgsl (see renderer.rs), so it shares its globals,
// lighting and materials. Actors are posed on the CPU each frame; see avatar.rs.

struct ActorOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex: vec3<f32>,
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) flags: u32,
};

@vertex
fn vs_actor(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex: vec3<f32>,
    @location(3) color: u32,
    @location(4) flags: u32,
) -> ActorOut {
    var o: ActorOut;
    o.clip = g.view_proj * vec4(pos, 1.0);
    // The first-person hand is squeezed into the nearest half of the depth
    // range (reverse Z: 1 is nearest), so it never pokes into walls.
    if ((flags & 1u) != 0u) {
        o.clip.z = o.clip.w * 0.5 + o.clip.z * 0.5;
    }
    o.world = pos;
    o.normal = normal;
    o.tex = tex;
    o.color = unpack4x8unorm(color);
    o.flags = flags;
    return o;
}

@vertex
fn vs_actor_shadow(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return g.sun_view_proj * vec4(pos, 1.0);
}

@fragment
fn fs_actor(i: ActorOut) -> @location(0) vec4<f32> {
    let n = normalize(i.normal);
    let mat = (i.flags >> 8u) & 255u;
    let pix = length(fwidth(i.tex)) * 0.7;
    var albedo = lin(i.color.rgb);
    var rough = 0.8;
    var f0 = 0.03;
    var sss = 0.0;
    if (mat != 0u) {
        // Held blocks show a voxel of terrain material. Its coordinates sit on a
        // fixed voxel near the camera, above the water line so nothing looks wet.
        let base = vec3(floor(g.camera_pos.x / 64.0) * 64.0, 64.0, floor(g.camera_pos.z / 64.0) * 64.0);
        let s = material(mat, base + i.tex, NORMALS[(i.flags >> 1u) & 7u], pix);
        albedo = s.albedo;
        rough = s.rough;
        f0 = s.f0;
        sss = s.sss;
    }

    let sun = normalize(g.sun_dir.xyz);
    let v = normalize(g.camera_pos.xyz - i.world);
    let ao = mix(0.55, 1.0, i.color.a);
    let wrap = sss * 0.6;
    let ndl = max((dot(n, sun) + wrap) / (1.0 + wrap), 0.0);
    // The first-person hand sits inside the player's own shadow volume; test
    // from a little towards the sun so only the world shades it.
    let own = select(0.0, 0.8, (i.flags & 1u) != 0u);
    let sh = sun_shadow(i.world + sun * own, n);
    let light = sun_light();
    let skylight = sky_fill();
    let ambient = fill_light(n, sh, sss) * ao;

    var c = albedo * (light * ndl * sh + ambient);
    c += light * sh * ggx_spec(n, v, sun, rough, f0);
    // Nearby lanterns light characters as they light the ground.
    c += albedo * lamp_light(i.world + n * 0.05, n);
    // A soft sky rim keeps the figure readable against dark ground.
    let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    c += skylight * albedo * rim * 0.35 * ao;
    return vec4(apply_fog(c, i.world), 1.0);
}
