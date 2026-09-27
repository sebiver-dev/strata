// Strata world shader. There are no textures: every surface colour is computed
// here from the material id and the world position.

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    // xyz: direction towards the sun, w: time in seconds
    sun_dir: vec4<f32>,
    // x: fog distance, y: 1.0 if the output needs manual sRGB encoding,
    // z: 1.0 if the camera is under water, w: unused
    params: vec4<f32>,
    // xyz: targeted voxel min corner in metres, w: 1.0 if a voxel is targeted
    highlight: vec4<f32>,
    screen: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
// One texel per 16 m chunk column, non-zero where voxel meshes are drawn.
@group(0) @binding(1) var near_mask: texture_2d<u32>;

const VOXEL: f32 = 0.5;
const CHUNK_M: f32 = 16.0;

fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(mix(hash3(i), hash3(i + vec3(1.0, 0.0, 0.0)), u.x),
                mix(hash3(i + vec3(0.0, 1.0, 0.0)), hash3(i + vec3(1.0, 1.0, 0.0)), u.x), u.y);
    let b = mix(mix(hash3(i + vec3(0.0, 0.0, 1.0)), hash3(i + vec3(1.0, 0.0, 1.0)), u.x),
                mix(hash3(i + vec3(0.0, 1.0, 1.0)), hash3(i + vec3(1.0, 1.0, 1.0)), u.x), u.y);
    return mix(a, b, u.z);
}

fn fbm(p: vec3<f32>) -> f32 {
    return 0.5 * vnoise(p) + 0.3 * vnoise(p * 2.7) + 0.2 * vnoise(p * 7.3);
}

const NORMALS = array<vec3<f32>, 6>(
    vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0),
    vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0),
);

// Palette colours below are written in sRGB; lighting happens in linear space.
fn lin(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3(2.2));
}

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let sun = normalize(g.sun_dir.xyz);
    let h = clamp(dir.y, -1.0, 1.0);
    let zenith = lin(vec3(0.30, 0.52, 0.86));
    let horizon = lin(vec3(0.74, 0.83, 0.92));
    let ground = lin(vec3(0.40, 0.44, 0.46));
    var c = mix(horizon, zenith, pow(max(h, 0.0), 0.55));
    c = mix(c, ground, smoothstep(0.0, -0.25, h));
    let s = max(dot(dir, sun), 0.0);
    c += vec3(1.0, 0.85, 0.6) * (pow(s, 8.0) * 0.25 + pow(s, 900.0) * 6.0);
    return c;
}

fn apply_fog(c: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let to = world - g.camera_pos.xyz;
    let dist = length(to);
    let dir = to / max(dist, 0.0001);
    var fog = 1.0 - exp(-pow(dist / g.params.x, 2.2));
    var fog_col = sky_color(normalize(vec3(dir.x, max(dir.y, 0.02), dir.z)));
    if (g.params.z > 0.5) {
        fog = 1.0 - exp(-dist / 9.0);
        fog_col = lin(vec3(0.10, 0.34, 0.40));
    }
    return mix(c, fog_col, clamp(fog, 0.0, 1.0));
}

fn finish(c: vec3<f32>) -> vec3<f32> {
    // Filmic tone curve (Narkowicz ACES fit), then optional sRGB encode.
    let x = c * 0.9;
    var m = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3(0.0), vec3(1.0));
    if (g.params.y > 0.5) {
        m = pow(m, vec3(1.0 / 2.2));
    }
    return m;
}

fn material_color(mat: u32, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let cell = floor(p / VOXEL + n * 0.01 - n * 0.5);
    // Per-voxel and fine detail shimmer at a distance, so it fades out there.
    let calm = smoothstep(120.0, 500.0, distance(p, g.camera_pos.xyz));
    let jitter = (hash3(cell) - 0.5) * (1.0 - calm);
    let fine = mix(fbm(p * 3.1), 0.5, calm);
    let broad = fbm(p * 0.21);
    var c = vec3(1.0, 0.0, 1.0);
    switch mat {
        case 1u: { // stone: layered strata
            let band = 0.5 + 0.5 * sin(p.y * 1.7 + broad * 6.0);
            c = mix(vec3(0.40, 0.40, 0.42), vec3(0.52, 0.50, 0.47), band) * (0.85 + 0.3 * fine);
        }
        case 2u: { c = vec3(0.42, 0.29, 0.19) * (0.8 + 0.4 * fine); }
        case 3u: { // grass: top is green, sides show dirt under a green lip
            let green = mix(vec3(0.27, 0.45, 0.18), vec3(0.44, 0.54, 0.24), broad) * (0.85 + 0.3 * fine);
            if (n.y > 0.5) {
                c = green;
            } else {
                let lip = fract(p.y / VOXEL);
                c = mix(vec3(0.42, 0.29, 0.19) * (0.8 + 0.4 * fine), green, smoothstep(0.62, 0.75, lip + fine * 0.2));
            }
        }
        case 4u: { c = vec3(0.80, 0.72, 0.52) * (0.9 + 0.2 * fine); }
        case 5u: { c = mix(vec3(0.43, 0.42, 0.40), vec3(0.62, 0.60, 0.56), step(0.55, vnoise(p * 9.0))); }
        case 6u: { c = vec3(0.93, 0.95, 0.98) * (0.94 + 0.08 * fine); }
        case 7u: { // bark
            let rings = 0.5 + 0.5 * sin((p.x + p.z) * 20.0 + fine * 4.0);
            c = mix(vec3(0.27, 0.19, 0.12), vec3(0.36, 0.26, 0.16), rings);
            if (n.y > 0.5 || n.y < -0.5) { c = vec3(0.55, 0.42, 0.27); }
        }
        case 8u: { c = mix(vec3(0.12, 0.30, 0.10), vec3(0.24, 0.42, 0.14), broad) * (0.7 + 0.5 * fine); }
        case 10u: { // planks
            let board = fract((p.y + p.x * 0.001) / VOXEL * 2.0);
            c = vec3(0.62, 0.46, 0.28) * (0.85 + 0.2 * fine) * (0.85 + 0.15 * smoothstep(0.0, 0.08, board));
        }
        default: {}
    }
    return lin(c * (1.0 + jitter * 0.12));
}

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) @interpolate(flat) info: u32,
    @location(2) ao: f32,
};

@vertex
fn vs_world(@location(0) pos: vec3<f32>, @location(1) data: u32) -> VOut {
    var o: VOut;
    o.clip = g.view_proj * vec4(pos, 1.0);
    o.world = pos;
    o.info = data;
    o.ao = f32((data >> 11u) & 3u) / 3.0;
    return o;
}

// True where a far-terrain fragment lies over a chunk column drawn in full voxels.
// Walls are tested just behind their face so they belong to the cell they bound.
fn under_near(p: vec3<f32>, n: vec3<f32>) -> bool {
    let c = vec2<i32>(floor((p.xz - n.xz * 0.01) / CHUNK_M));
    let size = vec2<i32>(textureDimensions(near_mask));
    if (any(c < vec2(0)) || any(c >= size)) {
        return false;
    }
    return textureLoad(near_mask, c, 0).r != 0u;
}

@fragment
fn fs_terrain(i: VOut) -> @location(0) vec4<f32> {
    return shade_terrain(i);
}

@fragment
fn fs_far_terrain(i: VOut) -> @location(0) vec4<f32> {
    if (under_near(i.world, NORMALS[i.info & 7u])) {
        discard;
    }
    return shade_terrain(i);
}

fn shade_terrain(i: VOut) -> vec4<f32> {
    let n = NORMALS[i.info & 7u];
    let mat = (i.info >> 3u) & 255u;
    let base = material_color(mat, i.world, n);
    let sun = normalize(g.sun_dir.xyz);
    let ao = mix(0.35, 1.0, i.ao);
    let diffuse = max(dot(n, sun), 0.0);
    let sky = 0.5 + 0.5 * n.y;
    var light = vec3(1.0, 0.92, 0.78) * diffuse * 1.5
              + vec3(0.40, 0.52, 0.72) * sky * 0.55 * ao
              + vec3(0.25, 0.22, 0.18) * (1.0 - sky) * 0.3 * ao;
    light *= mix(0.55, 1.0, ao);
    var c = base * light;

    // Outline the voxel the player is aiming at.
    if (g.highlight.w > 0.5) {
        let lo = g.highlight.xyz - 0.01;
        let hi = g.highlight.xyz + VOXEL + 0.01;
        if (all(i.world >= lo) && all(i.world <= hi)) {
            let f = fract(i.world / VOXEL);
            let e = min(min(f, 1.0 - f), vec3(1.0)) + abs(n) * 10.0;
            let edge = min(min(e.x, e.y), e.z);
            c = mix(c, vec3(1.6), (1.0 - smoothstep(0.02, 0.06, edge)) * 0.8);
        }
    }
    return vec4(finish(apply_fog(c, i.world)), 1.0);
}

@fragment
fn fs_water(i: VOut) -> @location(0) vec4<f32> {
    return shade_water(i);
}

@fragment
fn fs_far_water(i: VOut) -> @location(0) vec4<f32> {
    if (under_near(i.world, vec3(0.0))) {
        discard;
    }
    return shade_water(i);
}

fn shade_water(i: VOut) -> vec4<f32> {
    let t = g.sun_dir.w;
    let p = i.world;
    var n = NORMALS[i.info & 7u];
    if (n.y > 0.5) {
        let w1 = vnoise(vec3(p.x * 0.9 + t * 0.6, t * 0.3, p.z * 0.9));
        let w2 = vnoise(vec3(p.x * 2.3 - t * 0.4, t * 0.5, p.z * 2.3 + t * 0.2));
        n = normalize(vec3((w1 - 0.5) * 0.35 + (w2 - 0.5) * 0.15, 1.0, (w2 - 0.5) * 0.35 - (w1 - 0.5) * 0.15));
    }
    let view = normalize(p - g.camera_pos.xyz);
    let sun = normalize(g.sun_dir.xyz);
    let fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(-view, n), 0.0), 5.0);
    let refl = sky_color(reflect(view, n));
    let deep = lin(vec3(0.10, 0.30, 0.34));
    let spec = pow(max(dot(reflect(view, n), sun), 0.0), 180.0) * 4.0;
    var c = mix(deep, refl, fresnel) + vec3(1.0, 0.9, 0.7) * spec;
    let alpha = mix(0.72, 0.95, fresnel);
    return vec4(finish(apply_fog(c, p)), alpha);
}

// Sky: a single triangle covering the screen, drawn behind everything.
struct SkyOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> SkyOut {
    let uv = vec2(f32((vi << 1u) & 2u), f32(vi & 2u));
    var o: SkyOut;
    o.ndc = uv * 2.0 - 1.0;
    o.clip = vec4(o.ndc, 0.0, 1.0);
    return o;
}

@fragment
fn fs_sky(i: SkyOut) -> @location(0) vec4<f32> {
    let far = g.inv_view_proj * vec4(i.ndc, 0.5, 1.0);
    let dir = normalize(far.xyz / far.w - g.camera_pos.xyz);
    var c = sky_color(dir);
    if (g.params.z > 0.5) {
        c = lin(vec3(0.10, 0.34, 0.40));
    }
    return vec4(finish(c), 1.0);
}

// Crosshair drawn over the finished frame.
@fragment
fn fs_overlay(i: SkyOut) -> @location(0) vec4<f32> {
    let px = (i.ndc * 0.5) * g.screen.xy;
    let d = abs(px);
    let arm = (d.x < 1.0 && d.y < 9.0 && d.y > 3.0) || (d.y < 1.0 && d.x < 9.0 && d.x > 3.0);
    let centre = length(px) < 1.5;
    if (!(arm || centre)) {
        discard;
    }
    return vec4(0.97, 0.97, 0.95, 0.85);
}
