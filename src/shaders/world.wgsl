// Strata world shader. There are no textures: every surface colour is computed
// here from the material id and the world position.
//
// Frame layout (see renderer.rs):
//   1. shadow pass: terrain depth from the sun (vs_shadow)
//   2. scene pass: sky and terrain into an HDR target (fs_sky, fs_terrain)
//   3. water pass: reads a copy of the scene colour and depth for refraction,
//      depth tint and screen-space reflections (fs_water)
//   4. post pass: tone mapping and the crosshair onto the swapchain (fs_post)

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    sun_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    // xyz: direction towards the sun, w: time in seconds
    sun_dir: vec4<f32>,
    // x: fog distance, y: 1.0 if the output needs manual sRGB encoding,
    // z: 1.0 if the camera is under water, w: unused
    params: vec4<f32>,
    // xyz: targeted voxel min corner in metres, w: 1.0 if a voxel is targeted
    highlight: vec4<f32>,
    // xy: scene render size in pixels, z: shadow map size in texels,
    // w: scene render size / window size
    screen: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
// One texel per 16 m chunk column, non-zero where voxel meshes are drawn.
@group(0) @binding(1) var near_mask: texture_2d<u32>;
@group(1) @binding(0) var shadow_map: texture_depth_2d;
@group(1) @binding(1) var shadow_sampler: sampler_comparison;
@group(1) @binding(2) var scene_color: texture_2d<f32>;
@group(1) @binding(3) var scene_depth: texture_depth_2d;
@group(1) @binding(4) var lin_sampler: sampler;
@group(1) @binding(5) var hdr_input: texture_2d<f32>;
@group(1) @binding(6) var post_sampler: sampler;

const VOXEL: f32 = 0.5;
const CHUNK_M: f32 = 16.0;
// Matches terrain::WATER_LEVEL_M.
const WATER_LEVEL: f32 = 24.5;
const PI: f32 = 3.14159265;
const SUN_COLOR = vec3<f32>(1.0, 0.90, 0.74);

fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn hash2(p: vec2<f32>) -> f32 {
    var q = fract(vec3(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + 33.33);
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

fn vnoise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(i), hash2(i + vec2(1.0, 0.0)), u.x),
               mix(hash2(i + vec2(0.0, 1.0)), hash2(i + vec2(1.0, 1.0)), u.x), u.y);
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
    return pow(max(c, vec3(0.0)), vec3(2.2));
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

fn underwater_color() -> vec3<f32> {
    return lin(vec3(0.10, 0.34, 0.40));
}

fn apply_fog(c: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let to = world - g.camera_pos.xyz;
    let dist = length(to);
    let dir = to / max(dist, 0.0001);
    var fog = 1.0 - exp(-pow(dist / g.params.x, 2.2));
    var fog_col = sky_color(normalize(vec3(dir.x, max(dir.y, 0.02), dir.z)));
    if (g.params.z > 0.5) {
        fog = 1.0 - exp(-dist / 9.0);
        fog_col = underwater_color();
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

// ---------------------------------------------------------------- shadows

// 1.0 in full sun, 0.0 in shadow. Outside the shadow map everything is lit.
fn sun_shadow(world: vec3<f32>, n: vec3<f32>) -> f32 {
    let p = g.sun_view_proj * vec4(world + n * 0.05, 1.0);
    let ndc = p.xyz / p.w;
    let edge = max(abs(ndc.x), abs(ndc.y));
    if (edge > 1.0 || ndc.z > 1.0 || ndc.z < 0.0) {
        return 1.0;
    }
    let uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let texel = 1.0 / g.screen.z;
    // Four bilinear comparison taps give a 3x3 texel soft edge.
    var s = 0.0;
    for (var k = 0; k < 4; k++) {
        let o = (vec2(f32(k & 1), f32(k >> 1u)) - 0.5) * texel * 1.5;
        s += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + o, ndc.z - 0.0004);
    }
    return mix(s / 4.0, 1.0, smoothstep(0.85, 1.0, edge));
}

// ---------------------------------------------------------------- materials

struct Surface {
    albedo: vec3<f32>,
    rough: f32,
    // Specular reflectance at normal incidence.
    f0: f32,
    // Small-scale relief in metres, turned into a bumped normal.
    height: f32,
    // How much sunlight passes through thin layers (leaves, grass, snow).
    sss: f32,
};

// `pix` is the size of one pixel in metres at this point; detail smaller than
// a few pixels is faded out instead of shimmering.
fn material(mat: u32, p: vec3<f32>, n: vec3<f32>, pix: f32) -> Surface {
    // Fine patterns use coordinates wrapped every 64 m so f32 noise stays precise;
    // their frequencies are whole numbers per metre, so the wrap has no seam.
    let q = p - floor(p / 64.0) * 64.0;
    let cell = floor(p / VOXEL + n * 0.01 - n * 0.5);
    // Per-voxel and fine detail shimmer at a distance, so it fades out there.
    let calm = smoothstep(120.0, 500.0, distance(p, g.camera_pos.xyz));
    let jitter = (hash3(cell) - 0.5) * (1.0 - calm);
    var fine = 0.5;
    if (calm < 1.0) {
        fine = mix(fbm(p * 3.1), 0.5, calm);
    }
    let broad = 0.65 * vnoise(p * 0.21) + 0.35 * vnoise(p * 0.57);
    let d_cm = 1.0 - smoothstep(0.006, 0.02, pix);   // centimetre detail
    let d_dm = 1.0 - smoothstep(0.02, 0.08, pix);    // decimetre detail
    let top = n.y > 0.5;

    var s: Surface;
    s.albedo = vec3(1.0, 0.0, 1.0);
    s.rough = 0.9;
    s.f0 = 0.03;
    s.height = 0.0;
    s.sss = 0.0;
    var wettable = true;

    switch mat {
        case 1u: { // stone: layered strata, cracks and lichen on top
            let band = 0.5 + 0.5 * sin(p.y * 1.7 + broad * 6.0);
            var c = mix(vec3(0.40, 0.40, 0.42), vec3(0.54, 0.51, 0.47), band) * (0.8 + 0.35 * fine);
            let crack = 1.0 - abs(2.0 * vnoise(q * vec3(3.0, 5.0, 3.0)) - 1.0);
            let crack_m = smoothstep(0.90, 0.97, crack) * d_dm;
            c *= 1.0 - 0.55 * crack_m;
            if (top) {
                let lichen = smoothstep(0.62, 0.70, fbm(q * 2.0));
                c = mix(c, vec3(0.55, 0.58, 0.33), lichen * 0.7);
            }
            s.albedo = c;
            s.rough = 0.75;
            s.f0 = 0.04;
            s.height = fine * 0.04 - crack_m * 0.02 + vnoise(q * 17.0) * 0.006 * d_cm;
        }
        case 2u: { // dirt with pebbles
            var c = vec3(0.40, 0.28, 0.19) * (0.78 + 0.4 * fine);
            let pc = floor(q * 11.0);
            let peb = step(0.82, hash3(pc)) * d_dm;
            let pd = length(fract(q * 11.0) - 0.5);
            let pebble = peb * (1.0 - smoothstep(0.25, 0.4, pd));
            c = mix(c, vec3(0.52, 0.49, 0.45) * (0.8 + 0.4 * hash3(pc + 7.0)), pebble);
            s.albedo = c;
            s.rough = 0.95;
            s.height = fine * 0.02 + pebble * 0.012 + vnoise(q * 23.0) * 0.004 * d_cm;
        }
        case 3u: { // grass: meadow tints, blade speckle and tiny flowers
            let meadow = vnoise(p * 0.045);
            var green = mix(vec3(0.24, 0.42, 0.14), vec3(0.46, 0.55, 0.22), broad);
            green = mix(green, vec3(0.50, 0.52, 0.24), smoothstep(0.55, 0.75, meadow) * 0.6);
            let blades = vnoise2(q.xz * 41.0) * 0.6 + vnoise2(q.xz * 97.0) * 0.4;
            green *= 0.8 + 0.25 * fine + (blades - 0.5) * 0.45 * d_cm;
            if (top) {
                var c = green;
                // Daisies and buttercups, a couple of centimetres across.
                let fc = floor(q.xz * 7.0);
                let fh = hash2(fc);
                if (fh > 0.975 && d_dm > 0.0) {
                    let centre = (fc + 0.25 + 0.5 * vec2(hash2(fc + 3.1), hash2(fc + 5.7))) / 7.0;
                    let r = length(q.xz - centre);
                    let petal = (1.0 - smoothstep(0.012, 0.02, r)) * d_dm;
                    let kind = hash2(fc + 11.0);
                    var fl = vec3(0.95, 0.95, 0.90);
                    if (kind > 0.6) { fl = vec3(0.98, 0.84, 0.20); }
                    if (kind > 0.88) { fl = vec3(0.62, 0.48, 0.85); }
                    if (r < 0.005 && kind <= 0.6) { fl = vec3(0.98, 0.80, 0.25); }
                    c = mix(c, fl, petal);
                }
                s.albedo = c;
            } else {
                let lip = fract(p.y / VOXEL);
                let dirt = vec3(0.40, 0.28, 0.19) * (0.8 + 0.4 * fine);
                s.albedo = mix(dirt, green, smoothstep(0.62, 0.75, lip + fine * 0.2 + (blades - 0.5) * 0.15));
            }
            s.rough = 0.85;
            s.sss = 0.2;
            s.height = blades * 0.012 * d_cm + fine * 0.01;
        }
        case 4u: { // sand: wind ripples and grains
            let ripple = sin(dot(p.xz, vec2(0.8, 0.6)) * 11.0 + fbm(p * 0.7) * 6.0);
            let grain = vnoise(q * 61.0);
            var c = vec3(0.82, 0.74, 0.54) * (0.9 + 0.15 * fine) * (0.95 + 0.1 * grain * d_cm);
            s.albedo = c;
            s.rough = 0.8;
            s.f0 = 0.03 + step(0.985, hash3(floor(q * 60.0))) * 0.4 * d_cm;
            s.height = select(0.0, ripple * 0.006 * d_dm, top) + grain * 0.002 * d_cm;
        }
        case 5u: { // gravel: rounded stones with gaps
            let gc = floor(q * 6.0);
            let gd = length(fract(q * 6.0) - 0.5);
            let stone = 1.0 - smoothstep(0.3, 0.48, gd);
            let tint = hash3(gc);
            let c = mix(vec3(0.40, 0.39, 0.37), vec3(0.64, 0.61, 0.56), tint);
            s.albedo = c * mix(0.45, 1.0, mix(1.0, stone, d_dm));
            s.rough = 0.7;
            s.height = stone * 0.03 * d_dm;
        }
        case 6u: { // snow: soft, bluish in shade, glitters
            s.albedo = vec3(0.93, 0.95, 0.98) * (0.94 + 0.08 * fine);
            s.rough = 0.55;
            s.f0 = 0.02 + step(0.99, hash3(floor(q * 55.0))) * 0.8 * d_cm;
            s.sss = 0.3;
            s.height = fine * 0.02;
            wettable = false;
        }
        case 7u: { // bark: vertical grooves, rings on cut faces
            if (top || n.y < -0.5) {
                let r = length(fract(p.xz) - 0.5);
                let rings = 0.5 + 0.5 * sin(r * 90.0 + fine * 3.0);
                s.albedo = mix(vec3(0.50, 0.38, 0.24), vec3(0.62, 0.48, 0.30), rings);
                s.height = rings * 0.002;
            } else {
                let along = select(q.x, q.z, abs(n.x) > 0.5);
                let groove = vnoise(vec3(along * 13.0, q.y * 1.5, 0.5));
                let g2 = smoothstep(0.35, 0.75, groove);
                s.albedo = mix(vec3(0.20, 0.14, 0.09), vec3(0.38, 0.28, 0.18), g2) * (0.85 + 0.3 * fine);
                s.height = g2 * 0.02 * d_dm;
            }
            s.rough = 0.9;
            wettable = false;
        }
        case 8u: { // leaves: clumps of leaves, waxy and translucent
            let clump = vnoise(q * 4.0);
            let leaf = vnoise(q * 11.0 + 0.5);
            let shape = smoothstep(0.25, 0.6, clump * 0.6 + leaf * 0.4);
            let hue = vnoise(q * 1.7 + 3.0);
            var c = mix(vec3(0.11, 0.26, 0.08), vec3(0.25, 0.40, 0.12), hue);
            c = mix(c, vec3(0.36, 0.42, 0.12), smoothstep(0.6, 0.8, broad) * 0.5);
            // Dark gaps between clumps read as depth inside the crown.
            c *= mix(0.85, mix(0.68, 1.02, shape), d_dm);
            s.albedo = c;
            s.rough = 0.5;
            s.f0 = 0.04;
            s.sss = 0.7;
            s.height = shape * 0.015 * d_dm;
            wettable = false;
        }
        case 10u: { // planks
            let board = fract((p.y + p.x * 0.001) / VOXEL * 2.0);
            let grain = vnoise(vec3(q.x * 3.0, q.y * 40.0, q.z * 3.0));
            s.albedo = vec3(0.62, 0.46, 0.28) * (0.85 + 0.2 * fine) * (0.85 + 0.15 * smoothstep(0.0, 0.08, board))
                     * (0.92 + 0.12 * grain) * (1.0 + jitter * 0.15);
            s.rough = 0.6;
            s.height = smoothstep(0.0, 0.08, board) * 0.004;
            wettable = false;
        }
        default: {}
    }

    // Ground just above the waterline is darker and glossier.
    if (wettable) {
        let wet = 1.0 - smoothstep(WATER_LEVEL - 0.1, WATER_LEVEL + 0.5, p.y);
        s.albedo *= mix(1.0, 0.55, wet);
        s.rough = mix(s.rough, 0.2, wet);
    }
    s.albedo = lin(s.albedo);
    return s;
}

// Normal perturbed by a height field using screen-space derivatives
// (Mikkelsen's surface-gradient bump mapping). Needs uniform control flow.
fn bump_normal(n: vec3<f32>, world: vec3<f32>, h: f32) -> vec3<f32> {
    let dpx = dpdx(world);
    let dpy = dpdy(world);
    let dhx = dpdx(h);
    let dhy = dpdy(h);
    let r1 = cross(dpy, n);
    let r2 = cross(n, dpx);
    let det = dot(dpx, r1);
    if (abs(det) < 1e-12) {
        return n;
    }
    let grad = sign(det) * (dhx * r1 + dhy * r2);
    return normalize(abs(det) * n - grad);
}

fn ggx_spec(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32, f0: f32) -> f32 {
    let h = normalize(v + l);
    let nh = max(dot(n, h), 0.0);
    let nl = max(dot(n, l), 0.0);
    let a = rough * rough;
    let a2 = a * a;
    let d = a2 / (PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    let f = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
    return d * f * 0.25 * nl;
}

// ---------------------------------------------------------------- terrain

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) @interpolate(flat) info: u32,
    @location(2) ao: f32,
};

// Leaves sway a few centimetres in the wind. The offset depends only on the
// position, so neighbouring faces move together and no cracks open.
fn sway(pos: vec3<f32>, data: u32) -> vec3<f32> {
    if (((data >> 3u) & 255u) != 8u) {
        return pos;
    }
    let t = g.sun_dir.w;
    let ph = dot(pos, vec3(0.7, 0.3, 0.5));
    return pos + vec3(sin(t * 1.7 + ph), 0.5 * sin(t * 2.3 + ph * 1.3), cos(t * 1.3 + ph * 0.8)) * 0.025;
}

@vertex
fn vs_world(@location(0) pos: vec3<f32>, @location(1) data: u32) -> VOut {
    var o: VOut;
    o.clip = g.view_proj * vec4(sway(pos, data), 1.0);
    o.world = pos;
    o.info = data;
    o.ao = f32((data >> 11u) & 3u) / 3.0;
    return o;
}

@vertex
fn vs_shadow(@location(0) pos: vec3<f32>, @location(1) data: u32) -> @builtin(position) vec4<f32> {
    return g.sun_view_proj * vec4(sway(pos, data), 1.0);
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

// The in-plane axes of each face direction, matching `FACES` in mesh.rs.
const FACE_U = array<vec3<f32>, 6>(
    vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 1.0),
    vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0),
);
const FACE_V = array<vec3<f32>, 6>(
    vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0),
    vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0),
);

// Leaf faces on the outline of a crown lose a ragged band along their open
// edges, so crowns read as foliage rather than as solid cubes.
fn frayed(i: VOut, pix: f32) -> bool {
    let edges = (i.info >> 13u) & 15u;
    if (edges == 0u || pix > 0.06) {
        return false;
    }
    let fi = i.info & 7u;
    let f = fract(i.world / VOXEL);
    let fu = dot(f, FACE_U[fi]);
    let fv = dot(f, FACE_V[fi]);
    var d = 1.0;
    if ((edges & 1u) != 0u) { d = min(d, 1.0 - fu); }
    if ((edges & 2u) != 0u) { d = min(d, fu); }
    if ((edges & 4u) != 0u) { d = min(d, 1.0 - fv); }
    if ((edges & 8u) != 0u) { d = min(d, fv); }
    let q = i.world - floor(i.world / 64.0) * 64.0;
    let ragged = vnoise(q * 14.0) * 0.7 + vnoise(q * 31.0) * 0.3;
    // Narrower with distance, so far crowns do not sparkle.
    let width = 0.42 * (1.0 - smoothstep(0.02, 0.06, pix));
    return d < width * ragged;
}

@fragment
fn fs_terrain(i: VOut) -> @location(0) vec4<f32> {
    let pix = length(fwidth(i.world)) * 0.7;
    if (frayed(i, pix)) {
        discard;
    }
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
    let pix = length(fwidth(i.world)) * 0.7;
    let surf = material(mat, i.world, n, pix);
    let nb = bump_normal(n, i.world, surf.height);

    let sun = normalize(g.sun_dir.xyz);
    let to_cam = g.camera_pos.xyz - i.world;
    let v = normalize(to_cam);
    // Translucent materials (leaves most of all) get softer occlusion and a
    // light that wraps past the terminator instead of cutting off hard.
    let ao = mix(mix(0.35, 0.6, surf.sss), 1.0, i.ao);
    let facing = step(0.0, dot(n, sun));
    let sh = sun_shadow(i.world, n) * facing;
    let wrap = surf.sss * 0.6;
    let ndl = max((dot(nb, sun) + wrap) / (1.0 + wrap), 0.0);

    let sky = 0.5 + 0.5 * nb.y;
    // Skylight also reaches the undersides of thin layers through them.
    let under = 0.35 + 0.4 * surf.sss;
    let skylight = lin(vec3(0.62, 0.74, 0.90)) * 0.75;
    let ambient = (skylight * (sky + (1.0 - sky) * surf.sss * 0.6) + lin(vec3(0.50, 0.52, 0.40)) * (1.0 - sky) * under) * ao;
    // Sunlight scattered through leaves and grass, strongest when looking towards the sun.
    let through = pow(max(dot(-v, sun), 0.0), 4.0) * 1.6 + 0.35 * max(dot(-n, sun), 0.0) + 0.06;
    let trans = surf.sss * through * mix(0.25, 1.0, sh) * lin(vec3(0.85, 0.95, 0.45));

    var c = surf.albedo * (SUN_COLOR * 2.6 * ndl * sh + ambient * mix(0.55, 1.0, ao) + trans * ao);
    c += SUN_COLOR * 2.6 * sh * ggx_spec(nb, v, sun, surf.rough, surf.f0);
    // A hint of sky reflected on glossy (wet) surfaces.
    let fres = surf.f0 + (1.0 - surf.f0) * pow(1.0 - max(dot(nb, v), 0.0), 5.0);
    let gloss = 1.0 - surf.rough;
    c += sky_color(reflect(-v, nb)) * fres * gloss * gloss * gloss * 0.6 * ao;

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
    return vec4(apply_fog(c, i.world), 1.0);
}

// ---------------------------------------------------------------- water

// Slope of the water surface: a few travelling waves plus fine ripples that
// fade with distance so the far surface does not shimmer.
fn water_slope(xz: vec2<f32>, t: f32, dist: f32) -> vec2<f32> {
    var grad = vec2(0.0);
    var wavelength = 3.0;
    var amp = 0.022;
    var angle = 0.4;
    let fade = exp(-dist / 45.0);
    for (var k = 0; k < 6; k++) {
        let d = vec2(cos(angle), sin(angle));
        let w = 2.0 * PI / wavelength;
        let phase = dot(d, xz) * w + t * sqrt(9.8 * w);
        var a = amp;
        if (k >= 3) { a *= fade; }
        grad += d * w * a * cos(phase);
        wavelength *= 0.62;
        amp *= 0.58;
        angle += 2.39;
    }
    let r = vec2(vnoise2(xz * 5.0 + vec2(t * 0.9, t * 0.4)), vnoise2(xz * 5.0 - vec2(t * 0.5, t * 0.8)));
    grad += (r - 0.5) * 0.08 * fade;
    return grad;
}

fn world_from_depth(uv: vec2<f32>, d: f32) -> vec3<f32> {
    let w = g.inv_view_proj * vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    return w.xyz / w.w;
}

fn load_depth(uv: vec2<f32>) -> f32 {
    let px = clamp(vec2<i32>(uv * g.screen.xy), vec2<i32>(0), vec2<i32>(g.screen.xy) - 1);
    return textureLoad(scene_depth, px, 0);
}

// Distance the view ray travels through water behind the surface point `p`.
fn water_path(uv: vec2<f32>, d: f32, p: vec3<f32>) -> f32 {
    if (d <= 0.0) {
        return 200.0;
    }
    return min(distance(world_from_depth(uv, d), p), 200.0);
}

// Marches the reflected ray through the depth buffer. xyz: colour, w: confidence.
fn trace_reflection(origin: vec3<f32>, dir: vec3<f32>, dist: f32) -> vec4<f32> {
    var step_len = 0.3 + dist * 0.012;
    var prev = origin;
    var pos = origin;
    for (var s = 0; s < 28; s++) {
        pos += dir * step_len;
        let c = g.view_proj * vec4(pos, 1.0);
        if (c.w <= 0.0) {
            break;
        }
        let ndc = c.xyz / c.w;
        let uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (any(uv < vec2(0.0)) || any(uv > vec2(1.0))) {
            break;
        }
        let sd = load_depth(uv);
        // Reverse Z: a larger depth is nearer the camera, so the ray is now behind a surface.
        if (sd > ndc.z) {
            var lo = prev;
            var hi = pos;
            for (var b = 0; b < 5; b++) {
                let mid = (lo + hi) * 0.5;
                let mc = g.view_proj * vec4(mid, 1.0);
                let mn = mc.xyz / mc.w;
                let muv = vec2(mn.x * 0.5 + 0.5, 0.5 - mn.y * 0.5);
                if (load_depth(muv) > mn.z) { hi = mid; } else { lo = mid; }
            }
            let hc = g.view_proj * vec4(hi, 1.0);
            let hn = hc.xyz / hc.w;
            let huv = vec2(hn.x * 0.5 + 0.5, 0.5 - hn.y * 0.5);
            let surf = world_from_depth(huv, load_depth(huv));
            // Reject hits where the ray passed far behind a thin object.
            let gap = distance(surf, g.camera_pos.xyz) - distance(hi, g.camera_pos.xyz);
            if (abs(gap) > step_len * 2.0 + 0.3) {
                break;
            }
            let border = min(min(huv.x, 1.0 - huv.x), min(huv.y, 1.0 - huv.y));
            let conf = smoothstep(0.0, 0.08, border) * (1.0 - f32(s) / 28.0);
            return vec4(textureSampleLevel(scene_color, lin_sampler, huv, 0.0).rgb, conf);
        }
        prev = pos;
        step_len *= 1.14;
    }
    return vec4(0.0);
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
    let face_n = NORMALS[i.info & 7u];
    let to_cam = g.camera_pos.xyz - p;
    let dist = length(to_cam);
    let v = to_cam / dist;
    let uv = i.clip.xy / g.screen.xy;
    let sun = normalize(g.sun_dir.xyz);

    if (g.params.z > 0.5) {
        // Looking up at the surface from below: the world above, dimmed and tinted.
        let above = textureSampleLevel(scene_color, lin_sampler, uv, 0.0).rgb;
        return vec4(apply_fog(above * lin(vec3(0.55, 0.85, 0.85)), p), 1.0);
    }

    var n = face_n;
    if (face_n.y > 0.5) {
        let s = water_slope(p.xz, t, dist);
        n = normalize(vec3(-s.x, 1.0, -s.y));
    }

    // Refraction: look at the scene behind through a bent ray, unless that
    // would pick up something in front of the water.
    let d0 = load_depth(uv);
    let path0 = water_path(uv, d0, p);
    var ruv = uv + n.xz * 0.05 * clamp(path0 * 0.5, 0.0, 1.0) / (1.0 + dist * 0.05);
    var rd = load_depth(ruv);
    if (rd > i.clip.z) {
        ruv = uv;
        rd = d0;
    }
    let path = water_path(ruv, rd, p);
    let behind = textureSampleLevel(scene_color, lin_sampler, ruv, 0.0).rgb;

    let sh = sun_shadow(p, vec3(0.0, 1.0, 0.0));
    // Light absorbed per metre (red goes first), and light scattered back by the water body.
    let absorb = exp(-path * vec3(0.42, 0.11, 0.08));
    let scatter = lin(vec3(0.06, 0.24, 0.26)) * (0.45 + 0.9 * sh * max(sun.y, 0.0));
    var body = behind * absorb + scatter * (1.0 - absorb);

    // Shore foam where the water is shallow.
    let depth_below = select(10.0, p.y - world_from_depth(uv, d0).y, d0 > 0.0);
    let foam_n = vnoise2(p.xz * 4.0 + vec2(t * 0.3, -t * 0.2)) * 0.6 + vnoise2(p.xz * 11.0 - t * 0.5) * 0.4;
    let foam = (1.0 - smoothstep(0.0, 0.25, depth_below)) * smoothstep(0.35, 0.65, foam_n);
    body = mix(body, lin(vec3(0.92, 0.95, 0.95)) * (0.5 + 0.8 * sh), foam * 0.7);

    // Reflection: screen-space first, the sky where the screen has no answer.
    let r = reflect(-v, n);
    let sky_refl = sky_color(normalize(vec3(r.x, max(r.y, 0.0), r.z)));
    let ssr = trace_reflection(p + n * 0.02, r, dist);
    let refl = mix(sky_refl, ssr.rgb, ssr.w);

    let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    var c = mix(body, refl, fresnel * (1.0 - foam * 0.7));
    c += SUN_COLOR * 3.0 * sh * ggx_spec(n, v, sun, 0.07, 0.02) * (1.0 - foam);
    return vec4(apply_fog(c, p), 1.0);
}

// ---------------------------------------------------------------- full screen

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
        c = underwater_color();
    }
    return vec4(c, 1.0);
}

// Tone maps the HDR scene onto the screen.
@fragment
fn fs_post(i: SkyOut) -> @location(0) vec4<f32> {
    let uv = vec2(i.ndc.x * 0.5 + 0.5, 0.5 - i.ndc.y * 0.5);
    var c = textureSampleLevel(hdr_input, post_sampler, uv, 0.0).rgb;
    let q = i.ndc * 0.75;
    c *= 1.0 - 0.22 * dot(q, q);
    var m = finish(c);
    // Dither away banding in the sky gradient.
    m += (hash2(i.clip.xy) - 0.5) / 255.0;
    return vec4(m, 1.0);
}

// Crosshair drawn over the finished frame.
@fragment
fn fs_overlay(i: SkyOut) -> @location(0) vec4<f32> {
    let px = (i.ndc * 0.5) * g.screen.xy / g.screen.w;
    let d = abs(px);
    let arm = (d.x < 1.0 && d.y < 9.0 && d.y > 3.0) || (d.y < 1.0 && d.x < 9.0 && d.x > 3.0);
    let centre = length(px) < 1.5;
    if (!(arm || centre)) {
        discard;
    }
    return vec4(0.97, 0.97, 0.95, 0.85);
}
