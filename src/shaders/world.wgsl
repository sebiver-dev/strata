// Strata world shader. There are no textures: every surface colour is computed
// here from the material id and the world position.
//
// Frame layout (see renderer.rs):
//   1. shadow pass: terrain depth from the sun (vs_shadow)
//   2. scene pass: sky and terrain into an HDR target (fs_sky, fs_terrain)
//   3. water pass: reads a copy of the scene colour and depth for refraction,
//      depth tint and screen-space reflections (fs_water)
//   4. bloom: the scene halved down a chain of small targets and blurred
//      back up (fs_bloom_first, fs_bloom_down, fs_bloom_up)
//   5. post pass: tone mapping, bloom and the crosshair onto the swapchain (fs_post)

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    sun_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    // xyz: direction towards the sun, w: time in seconds
    sun_dir: vec4<f32>,
    // x: fog distance, y: 1.0 if the output needs manual sRGB encoding,
    // z: 1.0 if the camera is under water, w: brush shape (0 sphere, 1 cube, 2 block)
    params: vec4<f32>,
    // xyz: min corner in metres of the targeted voxel (of its grid block in
    // block mode), w: 0 when nothing is targeted,
    // otherwise 1 + the brush radius in voxels
    highlight: vec4<f32>,
    // xy: scene render size in pixels, z: shadow map size in texels,
    // w: scene render size / window size
    screen: vec4<f32>,
    // x: night (0 day, 1 night), y: number of lights in use
    sky: vec4<f32>,
    // Lanterns: xyz position in metres, w strength
    lights: array<vec4<f32>, 24>,
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
@group(1) @binding(7) var bloom_tex: texture_2d<f32>;

const VOXEL: f32 = 0.5;
const CHUNK_M: f32 = 16.0;
// Matches terrain::WATER_LEVEL_M: the water level along the home reach,
// between the falls either side of HOME_Z (terrain::HOME_Z_M).
const WATER_LEVEL: f32 = 24.5;
// Matches bridge::BANNER_W: banner edges lie on multiples of it along X.
const BANNER_W: f32 = 0.8;
const HOME_Z: f32 = 1024.0;
const FALL_COUNT: u32 = 13u;

// Matches terrain::FALLS: the Z of each fall's lip and its drop in metres.
fn fall(k: u32) -> vec2<f32> {
    var falls = array<vec2<f32>, 13>(
        vec2(380.0, 1.0), vec2(388.0, 1.0), vec2(396.0, 1.5), vec2(690.0, 8.0),
        vec2(925.0, 1.0), vec2(932.0, 1.0), vec2(939.0, 1.0),
        vec2(1290.0, 1.0), vec2(1297.0, 1.0), vec2(1304.0, 1.0), vec2(1540.0, 4.0),
        vec2(1790.0, 1.5), vec2(1798.0, 1.5),
    );
    return falls[k];
}

const CLIFF_FALL_COUNT: u32 = 3u;

// Matches vista::CLIFF_FALLS: where each waterfall off the castle bluff lands
// (CliffFall::foot, x and z in metres).
fn cliff_fall(k: u32) -> vec2<f32> {
    var feet = array<vec2<f32>, 3>(
        vec2(1062.6, 679.9), vec2(1045.8, 681.6), vec2(1019.7, 649.4),
    );
    return feet[k];
}

// Metres from the nearest spot where a cliff fall lands.
fn from_cliff_fall(xz: vec2<f32>) -> f32 {
    var best = 1e4;
    for (var k = 0u; k < CLIFF_FALL_COUNT; k++) {
        best = min(best, distance(xz, cliff_fall(k)));
    }
    return best;
}

// Matches terrain::fall_z: the lip bows a little across the river.
fn fall_z(z0: f32, x: f32) -> f32 {
    return z0 + 3.0 * sin(x * 0.21 + z0 * 0.01);
}

// Height of still water at a point, stepping down at each fall.
fn water_level_at(xz: vec2<f32>) -> f32 {
    var level = WATER_LEVEL;
    for (var k = 0u; k < FALL_COUNT; k++) {
        let f = fall(k);
        if (xz.y < fall_z(f.x, xz.x)) {
            level += f.y;
        }
        if (f.x > HOME_Z) {
            level -= f.y;
        }
    }
    return level;
}

// Metres downstream of the nearest fall's lip (large when none is near).
fn below_fall(xz: vec2<f32>) -> f32 {
    var best = 1e4;
    for (var k = 0u; k < FALL_COUNT; k++) {
        let d = xz.y - fall_z(fall(k).x, xz.x);
        if (d >= -2.0) {
            best = min(best, d);
        }
    }
    return best;
}
const PI: f32 = 3.14159265;

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

// ---------------------------------------------------------------- atmosphere
//
// A small analytic stand-in for a scattering sky. Sunlight loses blue on its
// long path through the air when the sun is low, so the same sun elevation
// drives the colour of direct light, the sky, the haze and the ambient light,
// and they always agree with each other.

// Relative extinction of sunlight per colour channel (blue is lost most).
const RAYLEIGH = vec3<f32>(0.17, 0.40, 1.0);
// Relative scattering that colours the sky itself.
const SKY_SCATTER = vec3<f32>(0.10, 0.32, 1.0);

// Air mass along a direction: 1 straight up, large towards the horizon.
fn air_mass(y: f32) -> f32 {
    return 1.0 / (max(y, 0.0) + 0.5 * exp(-max(y, 0.0) * 12.0) + 0.012);
}

// Colour and strength of direct sunlight at the ground.
fn day_light() -> vec3<f32> {
    let m = air_mass(g.sun_dir.y);
    let warm = 1.0 - smoothstep(0.2, 0.5, g.sun_dir.y);
    return vec3(1.0, 0.95, 0.88) * exp(-m * RAYLEIGH * 0.21) * mix(3.6, 2.9, warm);
}

const MOON_LIGHT = vec3<f32>(0.13, 0.16, 0.25);

// The main light: the sun by day, the moon at night (`sun_dir` follows it).
fn sun_light() -> vec3<f32> {
    return mix(day_light(), MOON_LIGHT, g.sky.x);
}

// Henyey-Greenstein phase function: how much light scatters towards the
// viewer at angle `mu` (cosine) from the sun. Larger g is more forward.
fn hg(mu: f32, gg: f32) -> f32 {
    let d = 1.0 + gg * gg - 2.0 * gg * mu;
    return (1.0 - gg * gg) / (4.0 * PI * pow(max(d, 1e-4), 1.5));
}

// Sky colour in a direction, without the sun disc or clouds.
fn sky_dome(dir: vec3<f32>) -> vec3<f32> {
    return sky_dome_base(dir);
}

fn sky_dome_base(dir: vec3<f32>) -> vec3<f32> {
    let night = g.sky.x;
    var day = vec3(0.0);
    if (night < 0.999) {
        day = day_dome(dir);
    }
    // Night: a deep blue dome, lighter at the horizon and around the moon.
    let y = max(dir.y, 0.0);
    let mu = dot(dir, normalize(g.sun_dir.xyz));
    var dark = mix(vec3(0.010, 0.015, 0.030), vec3(0.0025, 0.0045, 0.012), pow(y, 0.5));
    dark += MOON_LIGHT * (hg(mu, 0.8) * 0.05 + 0.01);
    return mix(day, dark, night);
}

fn day_dome(dir: vec3<f32>) -> vec3<f32> {
    let sun = normalize(g.sun_dir.xyz);
    let mu = dot(dir, sun);
    let y = max(dir.y, 0.0);
    let light = day_light();
    // Blue from above; towards the horizon the long path through the air
    // saturates the scattering and it turns pale and warm.
    let m = air_mass(y);
    let depth = 1.0 - exp(-m * SKY_SCATTER * 0.30);
    let rayleigh = depth * (0.75 * (1.0 + mu * mu));
    // Haze from larger particles: a broad bright glow around the sun.
    let haze = (hg(mu, 0.76) * 0.35 + hg(mu, 0.2) * 0.25) * (1.0 - exp(-m * 0.12));
    var c = light * (rayleigh + haze * vec3(1.0, 0.92, 0.80));
    // A little warmth soaks into the whole horizon band on the sunny side.
    c += light * vec3(0.10, 0.06, 0.03) * exp(-y * 9.0) * (0.5 + 0.5 * mu);
    // And the horizon itself is pale with dust and moisture.
    c += light * vec3(0.07, 0.068, 0.065) * exp(-y * 5.0);
    // Golden hour is art-directed rather than physical: a violet-blue high
    // sky, a lavender middle, and a horizon that burns orange towards the sun
    // and turns rose away from it.
    let golden = golden_hour();
    if (golden > 0.0) {
        let side = 0.5 + 0.5 * dot(normalize(vec2(dir.x, dir.z) + 1e-5), normalize(sun.xz + 1e-5));
        let zenith = vec3(0.07, 0.08, 0.26);
        let middle = mix(vec3(0.30, 0.20, 0.45), vec3(0.55, 0.30, 0.32), side);
        let horizon = mix(vec3(0.60, 0.30, 0.30), vec3(1.25, 0.52, 0.16), side * side);
        var p = mix(horizon, middle, smoothstep(0.0, 0.25, y));
        p = mix(p, zenith, smoothstep(0.22, 0.8, y));
        // The glow around the low sun.
        p += vec3(1.6, 0.75, 0.22) * pow(max(mu, 0.0), 16.0) + vec3(0.45, 0.2, 0.06) * pow(max(mu, 0.0), 4.0);
        c = mix(c, p, golden);
    }
    return c;
}

// 1 when the sun is low (golden hour), 0 when it is high; 0 at night.
fn golden_hour() -> f32 {
    return (1.0 - smoothstep(0.2, 0.5, g.sun_dir.y)) * (1.0 - g.sky.x);
}

fn fbm2(p: vec2<f32>) -> f32 {
    var a = 0.5;
    var f = 0.0;
    var q = p;
    for (var k = 0; k < 5; k++) {
        f += a * vnoise2(q);
        q = mat2x2<f32>(1.6, 1.2, -1.2, 1.6) * q + vec2(3.1, 1.7);
        a *= 0.5;
    }
    return f;
}

// A layer of drifting cumulus and streaks, lit by the sun: warm gold and pink
// where the light passes through them, violet-grey underneath. Returns the
// cloud colour and coverage.
fn clouds(dir: vec3<f32>) -> vec4<f32> {
    if (dir.y < 0.005) {
        return vec4(0.0);
    }
    // Where the ray meets a cloud deck far overhead; the horizon compresses.
    let uv = dir.xz / (dir.y + 0.06) * 1.6 + vec2(g.sun_dir.w * 0.004, g.sun_dir.w * 0.0015);
    let stretch = vec2(uv.x * 0.7 + uv.y * 0.3, uv.y * 1.6 - uv.x * 0.2);
    let shape = fbm2(stretch * 1.3);
    let detail = fbm2(stretch * 4.5 + 7.0);
    let dens = smoothstep(0.38, 0.66, shape * 0.8 + detail * 0.3);
    if (dens <= 0.0) {
        return vec4(0.0);
    }
    // Thickness towards the sun: denser behind means darker under here.
    let sun = normalize(g.sun_dir.xyz);
    let ahead = fbm2(stretch * 1.3 + sun.xz * 0.35);
    let thick = smoothstep(0.4, 0.85, ahead);
    let mu = dot(dir, sun);
    let light = sun_light();
    let golden = golden_hour();
    let lit = light * mix(vec3(1.0, 0.95, 0.9), vec3(1.0, 0.42, 0.16), golden)
        * (0.26 + 0.8 * hg(mu, 0.55)) * mix(1.0, 0.3, thick);
    // The shaded bodies take the colour of the sky around them, greyed.
    let shade = mix(sky_dome_base(vec3(0.0, 1.0, 0.0)) * 0.9, vec3(0.20, 0.13, 0.24), golden);
    // Edges glow where they are thin; the silver lining around the sun.
    let edge = (1.0 - dens) * pow(max(mu, 0.0), 6.0) * 1.5;
    var c = shade + lit * (0.55 + edge);
    // Clouds near the horizon fade into the haze.
    let fade = smoothstep(0.005, 0.12, dir.y);
    let night = g.sky.x;
    c = mix(c, vec3(0.012, 0.016, 0.03), night);
    return vec4(c, dens * fade * mix(0.9, 0.5, night));
}

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    var c = sky_dome(vec3(dir.x, max(dir.y, 0.0), dir.z));
    // Below the horizon: the haze over distant land, darker further down.
    c *= mix(1.0, 0.55, smoothstep(0.0, -0.3, dir.y));
    return c;
}

// The sun disc with a darker, warmer rim; at night the moon, with faint maria.
fn sun_disc(dir: vec3<f32>) -> vec3<f32> {
    let mu = dot(dir, normalize(g.sun_dir.xyz));
    let r = sqrt(max(1.0 - mu * mu, 0.0)) / mix(0.0085, 0.016, golden_hour());
    let disc = (1.0 - smoothstep(0.9, 1.0, r)) * step(0.0, mu);
    let limb = mix(vec3(1.0), vec3(1.0, 0.75, 0.5), r * r);
    let sun = day_light() * limb * mix(vec3(14.0), vec3(3.2, 1.9, 0.9), golden_hour());
    let maria = 0.8 + 0.2 * vnoise(dir * 900.0);
    let moon = vec3(0.85, 0.88, 0.95) * 1.4 * maria;
    return mix(sun, moon, g.sky.x) * disc;
}

// Pinpoint stars on a fixed grid of directions, twinkling a little.
fn stars(dir: vec3<f32>) -> vec3<f32> {
    if (g.sky.x < 0.01 || dir.y < 0.0) {
        return vec3(0.0);
    }
    let q = dir * 260.0;
    let cell = floor(q);
    let h = hash3(cell);
    if (h < 0.992) {
        return vec3(0.0);
    }
    let centre = cell + 0.5;
    let d = length(q - centre);
    let tw = 0.75 + 0.25 * sin(g.sun_dir.w * 3.0 + h * 400.0);
    let b = (1.0 - smoothstep(0.08, 0.35, d)) * (h - 0.992) * 90.0 * tw;
    return vec3(0.8, 0.85, 1.0) * b * g.sky.x * smoothstep(0.0, 0.25, dir.y);
}

// ---------------------------------------------------------------- lanterns

const LAMP_COLOR = vec3<f32>(1.0, 0.58, 0.26);
const LAMP_RANGE: f32 = 12.0;

// How much the lanterns count: they burn all the time but only matter at dusk.
fn lamp_on() -> f32 {
    return max(smoothstep(0.05, 0.8, g.sky.x), golden_hour() * 0.75);
}

// Light from nearby lanterns on a surface at `p` facing `n`.
fn lamp_light(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let on = lamp_on();
    if (on <= 0.0) {
        return vec3(0.0);
    }
    var sum = 0.0;
    let count = u32(g.sky.y);
    for (var k = 0u; k < count; k++) {
        let l = g.lights[k].xyz - p;
        let d2 = dot(l, l);
        if (d2 > LAMP_RANGE * LAMP_RANGE) {
            continue;
        }
        let window = 1.0 - d2 / (LAMP_RANGE * LAMP_RANGE);
        let ndl = max(dot(n, l * inverseSqrt(d2 + 1e-4)), 0.0) * 0.85 + 0.15;
        sum += ndl * window * window / (d2 + 0.35) * g.lights[k].w;
    }
    return LAMP_COLOR * 7.0 * sum * on;
}

// Light the air scatters towards the eye around each lantern along a view ray
// of length `len`: the closed-form integral of 1/r^2 along the ray.
fn lamp_glow(dir: vec3<f32>, len: f32) -> vec3<f32> {
    let on = lamp_on();
    if (on <= 0.0 || g.params.z > 0.5) {
        return vec3(0.0);
    }
    var sum = 0.0;
    let count = u32(g.sky.y);
    let cam = g.camera_pos.xyz;
    for (var k = 0u; k < count; k++) {
        let rel = g.lights[k].xyz - cam;
        let t0 = dot(rel, dir);
        let h2 = max(dot(rel, rel) - t0 * t0, 0.04);
        let h = sqrt(h2);
        sum += (atan((len - t0) / h) + atan(t0 / h)) / h * g.lights[k].w;
    }
    return LAMP_COLOR * 0.012 * sum * on;
}

fn underwater_color() -> vec3<f32> {
    return lin(vec3(0.10, 0.34, 0.40)) * (sun_light().g / 3.0);
}

// Aerial perspective: air between the camera and a point dims it and adds the
// light the air scatters towards the eye. The air is densest low in the
// valley and thins with height, so valleys fill with soft haze and distant
// hills turn blue, warm towards the sun.
fn apply_fog(c: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let to = world - g.camera_pos.xyz;
    let dist = length(to);
    let dir = to / max(dist, 0.0001);
    if (g.params.z > 0.5) {
        let f = 1.0 - exp(-dist / 9.0);
        return mix(c, underwater_color(), f);
    }
    // Density falls off exponentially above the water line; integrate it along the ray.
    let falloff = 1.0 / 38.0;
    let cam_water = water_level_at(g.camera_pos.xz);
    let h0 = g.camera_pos.y - cam_water;
    let dy = to.y * falloff;
    let base = mix(0.0016, 0.0011, golden_hour()) * exp(-max(h0, -20.0) * falloff);
    let integral = select((1.0 - exp(-dy)) / dy, 1.0 - 0.5 * dy, abs(dy) < 1e-3);
    let depth = base * dist * max(integral, 0.0);
    // Every clear day still fades the far distance into the horizon sky.
    let far = pow(dist / g.params.x, 2.2);
    let trans = exp(-(depth * mix(vec3(1.0), RAYLEIGH * 1.8, 0.45) + far));

    let horizon_dir = normalize(vec3(dir.x, max(dir.y, 0.0) * 0.5 + 0.02, dir.z));
    let sun = normalize(g.sun_dir.xyz);
    let light = sun_light();
    // In-scattered light: the horizon sky, plus a forward glow when looking towards the sun.
    let glow = light * hg(dot(dir, sun), 0.7) * mix(0.5, 0.18, golden_hour()) * vec3(1.0, 0.8, 0.55);
    let air = sky_dome(horizon_dir) * 0.95 + glow;
    var out = c * trans + air * (1.0 - trans);

    // Valley mist: a thin, patchy layer lying on the river and the low
    // meadows, thickest in the evening, lit warm on the side towards the sun.
    let mist_fall = 1.0 / 6.0;
    let hm = g.camera_pos.y - (cam_water + 1.5);
    let dym = to.y * mist_fall;
    let mist_integral = select((1.0 - exp(-dym)) / dym, 1.0 - 0.5 * dym, abs(dym) < 1e-3);
    let patchy = 0.3 + 1.4 * vnoise2((g.camera_pos.xz + to.xz * 0.6) / 70.0);
    let amount = mix(0.35, 1.0, golden_hour()) * (1.0 - 0.6 * g.sky.x);
    let mist = 0.004 * exp(-max(hm, -8.0) * mist_fall) * dist * max(mist_integral, 0.0) * patchy * amount;
    let mist_col = mix(air, air * vec3(0.9, 0.85, 1.05), golden_hour()) + glow * 0.4;
    let mt = exp(-mist);
    out = out * mt + mist_col * (1.0 - mt);
    return out + lamp_glow(dir, dist);
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
    // Light the surface gives off (lanterns).
    emit: vec3<f32>,
};

// `pix` is the size of one pixel in metres at this point; detail smaller than
// a few pixels is faded out instead of shimmering.
// Leaf clusters: distance to the nearest of one jittered point per cell, and
// a random value for that point's cluster.
fn leaf_cells(p: vec3<f32>) -> vec2<f32> {
    let c = floor(p);
    let f = p - c;
    var best = 8.0;
    var id = 0.0;
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            for (var z = -1; z <= 1; z++) {
                let o = vec3(f32(x), f32(y), f32(z));
                let k = c + o;
                let j = vec3(hash3(k), hash3(k + 17.3), hash3(k + 41.7)) * 0.8 + 0.1;
                let d = o + j - f;
                let dd = dot(d, d);
                if (dd < best) {
                    best = dd;
                    id = hash3(k + 5.1);
                }
            }
        }
    }
    return vec2(sqrt(best), id);
}

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
    s.emit = vec3(0.0);
    var wettable = true;

    switch mat {
        case 1u: { // stone: layered strata, cracks and lichen on top
            let band = 0.5 + 0.5 * sin(p.y * 1.7 + broad * 6.0);
            var c = mix(vec3(0.40, 0.40, 0.42), vec3(0.54, 0.51, 0.47), band) * (0.8 + 0.35 * fine);
            let crack = 1.0 - abs(2.0 * vnoise(q * vec3(3.0, 5.0, 3.0)) - 1.0);
            let crack_m = smoothstep(0.90, 0.97, crack) * d_dm;
            c *= 1.0 - 0.55 * crack_m;
            // Moss creeps over the tops of rocks and boulders in soft patches.
            let moss_m = smoothstep(0.35, 0.8, n.y + (fbm(q * 1.3) - 0.5) * 0.9);
            let moss = mix(vec3(0.18, 0.30, 0.09), vec3(0.34, 0.42, 0.14), vnoise(q * 9.0)) * (0.8 + 0.4 * fine);
            c = mix(c, moss, moss_m * 0.9);
            s.albedo = c;
            s.sss = moss_m * 0.15;
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
            {
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
                // Steep banks show the soil under the turf.
                let dirt = vec3(0.40, 0.28, 0.19) * (0.8 + 0.4 * fine);
                s.albedo = mix(dirt, c, smoothstep(0.42, 0.72, n.y + (fine - 0.5) * 0.25 + (blades - 0.5) * 0.1));
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
        case 5u: { // gravel: river pebbles of mixed size and tone
            let u = q * 7.0 + vec3(0.0, q.x * 0.37, 0.0);
            let gc = floor(u);
            let centre = gc + 0.25 + 0.5 * vec3(hash3(gc), hash3(gc + 3.0), hash3(gc + 7.0));
            let rad = 0.3 + 0.2 * hash3(gc + 11.0);
            let stone = 1.0 - smoothstep(rad - 0.12, rad, distance(u, centre));
            let tint = hash3(gc);
            let c = mix(vec3(0.42, 0.41, 0.39), vec3(0.66, 0.63, 0.58), tint);
            let bed = vec3(0.38, 0.35, 0.31) * (0.85 + 0.3 * fine);
            s.albedo = mix(bed, c, stone * d_dm + (1.0 - d_dm) * 0.5);
            s.rough = 0.7;
            s.height = stone * 0.02 * d_dm;
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
            // Rings only on flat cut ends, not on the flare where a trunk meets the ground.
            if (abs(n.y) > 0.97) {
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
        case 8u: { // leaves: overlapping leaves in soft clumps, waxy and translucent
            let clump = vnoise(q * 2.2);
            let leaf = vnoise(q * 9.0 + 0.5);
            let fine_leaf = vnoise(q * 23.0 + 1.7);
            let shape = smoothstep(0.3, 0.7, clump * 0.5 + leaf * 0.35 + fine_leaf * 0.15);
            let hue = vnoise(q * 0.9 + 3.0);
            var c = mix(vec3(0.10, 0.25, 0.07), vec3(0.24, 0.42, 0.11), hue);
            c = mix(c, vec3(0.38, 0.46, 0.13), smoothstep(0.6, 0.8, broad) * 0.5);
            // Leaves facing up catch the sky; the undersides and the gaps
            // between clumps fall into shade, so the crown reads as masses.
            c *= mix(0.7, 1.08, smoothstep(-0.4, 0.8, n.y));
            c *= mix(0.85, mix(0.6, 1.06, shape), d_dm);
            s.albedo = c;
            s.rough = 0.55;
            s.f0 = 0.04;
            s.sss = 0.75;
            s.height = (shape * 0.03 + fine_leaf * 0.008) * d_dm;
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
        case 11u: { // road: rounded cobblestones bedded in packed earth
            let earth = mix(vec3(0.36, 0.27, 0.19), vec3(0.46, 0.37, 0.27), fine);
            // Voronoi cells, about three stones per metre; the gap between the two
            // nearest centres is the mortar of earth between stones.
            let uv = q.xz * 2.3;
            let base = floor(uv);
            var f1 = 9.0;
            var f2 = 9.0;
            var id = vec2(0.0);
            for (var j = -1; j <= 1; j++) {
                for (var k = -1; k <= 1; k++) {
                    let c = base + vec2(f32(j), f32(k));
                    let o = c + 0.2 + 0.6 * vec2(hash2(c), hash2(c + 17.0));
                    let d = distance(uv, o);
                    if (d < f1) {
                        f2 = f1;
                        f1 = d;
                        id = c;
                    } else if (d < f2) {
                        f2 = d;
                    }
                }
            }
            let gap = f2 - f1;
            // Some stones are missing where the path is worn.
            let worn = smoothstep(0.35, 0.5, vnoise(p * 0.7) * 0.7 + hash2(id) * 0.3);
            let stone = smoothstep(0.05, 0.14, gap) * worn;
            let tone = hash2(id + 5.0);
            let grey = mix(vec3(0.44, 0.44, 0.45), vec3(0.64, 0.62, 0.60), tone) * (0.9 + 0.2 * fine);
            let c = mix(earth, grey, stone * mix(0.6, 1.0, d_dm));
            s.albedo = c * (0.92 + 0.12 * vnoise(q * 37.0) * d_cm);
            s.rough = mix(0.95, 0.7, stone);
            s.height = stone * smoothstep(0.0, 0.3, gap) * 0.03 * d_dm + fine * 0.008;
        }
        case 12u: { // lantern glass: warm light through faintly rippled panes
            let ripple = vnoise(q * 30.0);
            s.albedo = vec3(0.9, 0.7, 0.4);
            s.rough = 0.15;
            s.f0 = 0.04;
            let flicker = 0.9 + 0.1 * vnoise(vec3(g.sun_dir.w * 6.0, floor(p.x), floor(p.z)));
            s.emit = LAMP_COLOR * 4.5 * flicker * (0.85 + 0.3 * ripple);
            wettable = false;
        }
        case 26u: { // wrought iron: dark, a little glossy, rusty in places
            let rust = smoothstep(0.55, 0.8, vnoise(q * 9.0));
            s.albedo = mix(vec3(0.07, 0.065, 0.06), vec3(0.22, 0.12, 0.07), rust * 0.6);
            s.rough = mix(0.4, 0.8, rust);
            s.f0 = 0.25;
            wettable = false;
        }
        case 15u, 40u, 41u: { // lupin florets: violet, pink or white, each flower its own shade
            var lo = vec3(0.28, 0.16, 0.62);
            var hi = vec3(0.50, 0.36, 0.84);
            if (mat == 40u) { lo = vec3(0.70, 0.30, 0.50); hi = vec3(0.92, 0.58, 0.74); }
            if (mat == 41u) { lo = vec3(0.80, 0.78, 0.80); hi = vec3(0.96, 0.95, 0.92); }
            s.albedo = mix(lo, hi, vnoise(q * 40.0));
            s.rough = 0.7;
            s.sss = 0.5;
            wettable = false;
        }
        case 16u: { // daisy petals
            s.albedo = vec3(0.90, 0.90, 0.85);
            s.rough = 0.6;
            s.sss = 0.6;
            wettable = false;
        }
        case 17u: { // daisy heart
            s.albedo = vec3(0.95, 0.70, 0.12);
            s.rough = 0.7;
            wettable = false;
        }
        case 13u: { // tall grass blades
            let hue = vnoise(p * 0.15);
            s.albedo = mix(vec3(0.26, 0.44, 0.13), vec3(0.50, 0.56, 0.22), hue);
            s.rough = 0.6;
            s.f0 = 0.04;
            s.sss = 0.75;
            wettable = false;
        }
        case 18u: { // masonry: dressed stone blocks in courses with sunken mortar
            // Lay the courses on whichever wall plane the surface mostly faces.
            let an = abs(n);
            var u = select(p.x, p.z, an.x > an.z);
            var v = p.y;
            if (an.y > 0.7) { u = p.x; v = p.z; }
            let row = floor(v / 0.42);
            let col = floor(u / 0.7 + row * 0.5);
            let fu = fract(u / 0.7 + row * 0.5);
            let fv = fract(v / 0.42);
            let joint = min(min(fu, 1.0 - fu) * 0.7, min(fv, 1.0 - fv) * 0.42);
            let mortar = (1.0 - smoothstep(0.015, 0.035, joint)) * d_dm;
            let tint = hash2(vec2(col, row));
            // Cool grey field stone with the odd warm or dark block, as in the reference.
            let warm = hash2(vec2(col + 17.0, row - 5.0));
            var c = mix(vec3(0.43, 0.43, 0.42), vec3(0.60, 0.58, 0.54), tint);
            c = mix(c, vec3(0.58, 0.50, 0.40), smoothstep(0.7, 0.95, warm) * 0.6);
            c *= mix(1.0, 0.8, smoothstep(0.85, 0.97, hash2(vec2(row - 3.0, col + 9.0))));
            c *= 0.85 + 0.25 * fine;
            // Moss creeps into the lower courses and the tops of walls.
            let moss = smoothstep(0.55, 0.75, fbm(q * 0.9)) * (select(0.35, 0.9, top));
            c = mix(c, vec3(0.34, 0.40, 0.20), moss * 0.6);
            c = mix(c, vec3(0.30, 0.28, 0.25), mortar * 0.8);
            s.albedo = c;
            s.rough = 0.8;
            s.f0 = 0.04;
            s.height = (1.0 - mortar) * 0.02 + fine * 0.01 + vnoise(q * 19.0) * 0.004 * d_cm;
        }
        case 19u: { // lime plaster: warm, blotchy, weathered darker near the ground
            var c = vec3(0.86, 0.80, 0.67) * (0.9 + 0.12 * broad + 0.08 * fine);
            let stain = smoothstep(0.5, 0.8, vnoise(q * vec3(1.5, 0.6, 1.5)));
            c *= 1.0 - 0.12 * stain;
            s.albedo = c;
            s.rough = 0.9;
            s.height = fine * 0.008 + vnoise(q * 29.0) * 0.003 * d_cm;
        }
        case 20u: { // roof: overlapping rows of blue-grey slate
            let row = floor(p.y / 0.28);
            let along = p.x + p.z;
            let tile = floor(along / 0.4 + row * 0.5);
            let fv = fract(p.y / 0.28);
            let fu = fract(along / 0.4 + row * 0.5);
            let edge = (1.0 - smoothstep(0.0, 0.12, fv)) * d_dm;
            let gap = (1.0 - smoothstep(0.0, 0.06, min(fu, 1.0 - fu))) * d_dm;
            var base = mix(vec3(0.24, 0.29, 0.39), vec3(0.33, 0.36, 0.42), broad);
            base *= 0.8 + 0.35 * hash2(vec2(tile, row)) * (1.0 - calm);
            base *= 0.9 + 0.15 * broad;
            let moss = smoothstep(0.6, 0.8, fbm(q * 1.3)) * 0.35;
            var c = mix(base, vec3(0.35, 0.40, 0.22), moss);
            c *= 1.0 - 0.45 * max(edge, gap);
            s.albedo = c;
            s.rough = 0.55;
            s.f0 = 0.05;
            s.height = fv * 0.02 * d_dm - gap * 0.01;
            wettable = false;
        }
        case 21u: { // window: warm lamplight behind small panes in a dark frame
            let f = fract(p / VOXEL);
            let a = select(select(f.xy, f.zy, abs(n.x) > 0.5), f.xz, abs(n.y) > 0.5);
            let edge = min(min(a.x, 1.0 - a.x), min(a.y, 1.0 - a.y));
            let glass = smoothstep(0.08, 0.12, edge);
            let bars = 1.0 - smoothstep(0.02, 0.04, min(abs(a.x - 0.5), abs(a.y - 0.5)));
            let pane = glass * (1.0 - bars) * step(abs(n.y), 0.5);
            s.albedo = mix(vec3(0.16, 0.11, 0.07), vec3(0.95, 0.72, 0.42), pane);
            s.rough = mix(0.7, 0.1, pane);
            s.f0 = 0.04;
            // Lit all day but only bright once the light goes; each room its own warmth.
            let room = 0.75 + 0.5 * hash3(floor(p / 2.0));
            let flicker = 0.93 + 0.07 * vnoise(vec3(g.sun_dir.w * 3.0, floor(p.x / 2.0), floor(p.z / 2.0)));
            s.emit = LAMP_COLOR * pane * (0.6 + 3.4 * lamp_on()) * room * flicker;
            wettable = false;
        }
        case 22u: { // oak: dark planed timber with fine streaky grain
            let grain = vnoise(q * vec3(2.0, 16.0, 2.0)) + vnoise(q * vec3(16.0, 2.0, 2.0)) + vnoise(q * vec3(2.0, 2.0, 16.0));
            let g2 = smoothstep(0.9, 2.1, grain);
            s.albedo = mix(vec3(0.17, 0.11, 0.07), vec3(0.30, 0.20, 0.12), g2) * (0.9 + 0.15 * broad);
            s.rough = 0.75;
            s.height = g2 * 0.004 * d_cm;
            wettable = false;
        }
        case 23u: { // boards: warm floor planks with dark seams
            let row = p.z * 4.0;
            let seam = smoothstep(0.03, 0.07, min(fract(row), 1.0 - fract(row)));
            let tone = 0.85 + 0.25 * hash3(vec3(floor(row), floor(p.x / 1.7 + hash3(vec3(floor(row), 0.0, 1.0))), 0.0));
            let grain = vnoise(vec3(p.x * 3.0, p.z * 40.0, 0.5));
            s.albedo = vec3(0.46, 0.32, 0.19) * tone * (0.85 + 0.2 * grain) * mix(0.45, 1.0, seam);
            s.rough = 0.7;
            s.height = seam * 0.003;
            wettable = false;
        }
        case 24u: { // cloth: red wool blanket with a soft weave
            let weave = 0.5 + 0.25 * (sin(p.x * 180.0) + sin(p.z * 180.0 + p.y * 180.0));
            s.albedo = vec3(0.55, 0.12, 0.09) * (0.85 + 0.2 * weave) * (0.9 + 0.1 * fine);
            s.rough = 1.0;
            s.sss = 0.2;
            wettable = false;
        }
        case 55u: { // leaded glass lit from within: diamond panes glowing warm in narrow openings
            let an = abs(n);
            let u = select(p.x, p.z, an.x > an.z);
            let d1 = fract((u + p.y) / 0.15);
            let d2 = fract((u - p.y) / 0.15);
            let lead = (1.0 - smoothstep(0.04, 0.09, min(min(d1, 1.0 - d1), min(d2, 1.0 - d2)))) * d_dm;
            s.albedo = mix(vec3(0.95, 0.70, 0.40), vec3(0.07, 0.06, 0.05), lead);
            s.rough = mix(0.15, 0.6, lead);
            s.f0 = 0.04;
            let flicker = 0.92 + 0.08 * vnoise(vec3(g.sun_dir.w * 3.0, floor(p.x / 2.0), floor(p.z / 2.0)));
            s.emit = LAMP_COLOR * (1.0 - lead) * (0.5 + 3.4 * lamp_on()) * flicker;
            wettable = false;
        }
        case 53u: { // banner: deep blue-violet wool with a soft weave (trim in shade_terrain)
            let weave = 0.5 + 0.25 * (sin(p.x * 160.0) + sin(p.y * 160.0 + p.z * 160.0));
            s.albedo = vec3(0.10, 0.10, 0.34) * (0.85 + 0.2 * weave * d_cm) * (0.9 + 0.15 * fine);
            s.rough = 1.0;
            s.sss = 0.35;
            wettable = false;
        }
        case 54u: { // bridge stone: rough blocks, mottled, mossy on top, damp near the water
            let mottle = vnoise(q * 1.9) * 0.7 + broad * 0.3;
            var c = mix(vec3(0.44, 0.42, 0.39), vec3(0.62, 0.58, 0.52), mottle) * (0.84 + 0.28 * fine);
            // Pitted faces and a little lichen.
            let pit = smoothstep(0.62, 0.8, vnoise(q * 7.0)) * d_dm;
            c *= 1.0 - 0.08 * pit;
            let lichen = smoothstep(0.72, 0.85, vnoise(q * 3.3 + 11.0)) * d_dm;
            c = mix(c, vec3(0.62, 0.62, 0.50), lichen * 0.2);
            // Moss on the tops of stones and creeping into the lower courses.
            let level = water_level_at(p.xz);
            let low = 1.0 - smoothstep(level + 0.8, level + 3.5, p.y);
            let moss_m = smoothstep(0.55, 0.95, n.y + (fbm(q * 1.3) - 0.5) * 0.9) * 0.8
                + smoothstep(0.55, 0.75, fbm(q * 0.9)) * low * 0.7;
            let moss = mix(vec3(0.17, 0.26, 0.09), vec3(0.33, 0.40, 0.14), vnoise(q * 9.0)) * (0.8 + 0.4 * fine);
            c = mix(c, moss, clamp(moss_m, 0.0, 1.0) * 0.85);
            // A dark, damp band above the waterline with a ragged top edge.
            let edge = level + 0.9 + 0.5 * vnoise(vec3(q.x * 1.1, 0.0, q.z * 1.1));
            let damp = 1.0 - smoothstep(edge - 0.4, edge + 0.3, p.y);
            c = mix(c, c * vec3(0.42, 0.46, 0.40), damp);
            // Rain streaks down the faces.
            let streak = smoothstep(0.55, 0.9, vnoise(vec3(q.x * 5.0 + q.z * 5.0, q.y * 0.35, 0.5))) * (1.0 - abs(n.y));
            c *= 1.0 - 0.14 * streak;
            s.albedo = c;
            s.sss = clamp(moss_m, 0.0, 1.0) * 0.12;
            s.rough = mix(0.82, 0.45, damp);
            s.f0 = 0.04;
            s.height = fine * 0.03 - pit * 0.012 + vnoise(q * 17.0) * 0.006 * d_cm;
        }
        case 30u: { // dry grass blades: straw gold
            s.albedo = mix(vec3(0.52, 0.42, 0.18), vec3(0.72, 0.60, 0.30), vnoise(p * 0.3));
            s.rough = 0.65;
            s.f0 = 0.04;
            s.sss = 0.6;
            wettable = false;
        }
        case 31u: { // fresh grass blades of the verges: bright yellow-green
            s.albedo = mix(vec3(0.34, 0.54, 0.11), vec3(0.52, 0.64, 0.17), vnoise(p * 0.2));
            s.rough = 0.55;
            s.f0 = 0.04;
            s.sss = 0.8;
            wettable = false;
        }
        case 32u: { // wildflower stems and leaves: deep blue-green
            s.albedo = mix(vec3(0.14, 0.27, 0.09), vec3(0.22, 0.36, 0.13), vnoise(q * 7.0));
            s.rough = 0.6;
            s.sss = 0.5;
            wettable = false;
        }
        case 33u: { // deep purple lupin florets
            s.albedo = mix(vec3(0.16, 0.06, 0.36), vec3(0.30, 0.13, 0.54), vnoise(q * 40.0));
            s.rough = 0.7;
            s.sss = 0.45;
            wettable = false;
        }
        case 35u: { // boulder: speckled grey granite with hairline cracks under a thick cap of moss
            let tone = vnoise(q * 0.35);
            var c = mix(vec3(0.42, 0.42, 0.43), vec3(0.57, 0.55, 0.51), tone) * (0.85 + 0.25 * fine);
            // Dark mica and pale feldspar grains.
            let grain = hash3(floor(q * 38.0));
            c *= 1.0 + (0.28 * step(0.9, grain) - 0.3 * step(grain, 0.08)) * d_cm;
            // Hairline cracks and pale lichen rosettes.
            let cr = 1.0 - abs(2.0 * vnoise(q * vec3(2.2, 3.1, 2.2)) - 1.0);
            let crack = smoothstep(0.955, 0.99, cr) * smoothstep(0.45, 0.6, vnoise(q * 0.8 + 3.0)) * d_dm;
            c *= 1.0 - 0.5 * crack;
            let lichen = smoothstep(0.72, 0.8, vnoise(q * 4.0 + 7.0)) * d_dm;
            c = mix(c, vec3(0.70, 0.70, 0.60), lichen * 0.35);
            // Moss grows on whatever faces the sky, with a fuzzy, ragged edge,
            // but not below the waterline.
            let ragged = fbm(q * 1.7) - 0.5 + (vnoise(q * 11.0) - 0.5) * 0.35;
            let dry = smoothstep(water_level_at(p.xz), water_level_at(p.xz) + 0.3, p.y);
            let moss_m = smoothstep(0.42, 0.7, n.y + ragged * 0.8) * dry;
            let tuft = vnoise(q * 23.0);
            let moss = mix(vec3(0.20, 0.33, 0.08), vec3(0.42, 0.52, 0.16), vnoise(q * 5.0)) * (0.85 + 0.3 * tuft);
            c = mix(c, moss, moss_m);
            s.albedo = c;
            s.sss = moss_m * 0.3;
            s.rough = mix(0.7, 0.95, moss_m);
            s.f0 = 0.04;
            s.height = (1.0 - moss_m) * (fine * 0.03 - crack * 0.02) + moss_m * (0.03 + tuft * 0.012 * d_cm);
        }
        case 27u: { // bark: deep vertical fissures between plated ridges, moss on top
            let ridge = vnoise(vec3(q.x * 16.0, q.y * 0.9, q.z * 16.0)) * 0.6 + vnoise(vec3(q.x * 38.0, q.y * 2.5, q.z * 38.0)) * 0.4;
            // Long narrow fissures between plates of grey-brown bark.
            let plate = smoothstep(0.28, 0.55, ridge);
            let crack = smoothstep(0.55, 0.62, vnoise(vec3(q.x * 9.0, q.y * 7.0, q.z * 9.0))) * plate;
            var c = mix(vec3(0.10, 0.08, 0.065), mix(vec3(0.25, 0.21, 0.17), vec3(0.33, 0.28, 0.22), vnoise(q * 1.7)), plate);
            c = mix(c, c * 0.65, crack * d_dm);
            c *= 0.85 + 0.3 * fine;
            // Moss settles on the upper sides of roots and boughs.
            let moss_m = smoothstep(0.35, 0.85, n.y + (vnoise(q * 2.3) - 0.5) * 0.8);
            c = mix(c, mix(vec3(0.17, 0.27, 0.08), vec3(0.30, 0.38, 0.12), vnoise(q * 7.0)), moss_m * 0.85);
            s.albedo = c;
            s.rough = 0.9;
            s.sss = moss_m * 0.15;
            s.height = (plate * 0.03 - crack * 0.01) * d_dm + fine * 0.01;
            wettable = false;
        }
        case 28u: { // broadleaf foliage: rounded leaf clusters with sunlit tops, dark gaps
            let leaves = vnoise(q * 5.0) * 0.55 + vnoise(q * 13.0 + 0.5) * 0.45;
            let shape = smoothstep(0.3, 0.7, leaves);
            // Clusters about 0.7 m across, each a little dome that catches the
            // sun on its own, as in painted foliage. They fade out once a
            // cluster spans only a few pixels.
            let d_cl = 1.0 - smoothstep(0.06, 0.2, pix);
            let cl = leaf_cells(q * 1.5);
            let dome = max(1.0 - cl.x * cl.x * 1.3, 0.0);
            let hue = vnoise(p * 0.11 + 3.0) * 0.6 + cl.y * 0.4 * d_cl;
            var c = mix(vec3(0.12, 0.28, 0.07), vec3(0.30, 0.48, 0.12), hue);
            c = mix(c, vec3(0.38, 0.46, 0.14), smoothstep(0.62, 0.85, broad) * 0.45);
            // The gaps between clusters are deep green shade, not brown.
            let gap = mix(vec3(0.46, 0.68, 0.52), vec3(1.1), sqrt(dome));
            c *= mix(vec3(1.0), gap, d_cl);
            c *= mix(0.94, mix(0.8, 1.05, shape), d_dm);
            s.albedo = c;
            s.rough = 0.55;
            s.f0 = 0.04;
            s.sss = 1.0;
            s.height = dome * 0.16 * d_cl + shape * 0.02 * d_dm + leaves * 0.01;
            wettable = false;
        }
        case 29u: { // conifer needles: dark blue-green sprays
            let spray = vnoise(vec3(q.x * 9.0, q.y * 3.0, q.z * 9.0)) * 0.6 + vnoise(q * 23.0) * 0.4;
            let shape = smoothstep(0.3, 0.7, spray);
            let hue = vnoise(p * 0.09 + 7.0);
            var c = mix(vec3(0.05, 0.14, 0.09), vec3(0.13, 0.25, 0.12), hue);
            c *= mix(0.9, mix(0.65, 1.08, shape), d_dm);
            s.albedo = c;
            s.rough = 0.6;
            s.f0 = 0.04;
            s.sss = 0.45;
            s.height = shape * 0.04 * d_dm;
            wettable = false;
        }
        case 50u: { // castle banner: deep blue-violet wool hanging still in soft folds
            let fold = 0.5 + 0.5 * sin(dot(p.xz, vec2(7.0, 7.0)) + 0.8 * sin(p.y * 1.3));
            let weave = 0.5 + 0.25 * (sin(p.x * 160.0 + p.y * 160.0) + sin(p.z * 160.0 - p.y * 160.0));
            s.albedo = vec3(0.13, 0.11, 0.38) * (0.75 + 0.35 * fold) * (0.9 + 0.12 * weave * d_cm) * (0.92 + 0.12 * broad);
            s.rough = 0.95;
            s.sss = 0.3;
            wettable = false;
        }
        case 51u: { // ashlar: the castle's pale dressed limestone, in courses, streaked by rain
            let an = abs(n);
            var u = select(p.x, p.z, an.x > an.z);
            var v = p.y;
            if (an.y > 0.7) { u = p.x; v = p.z; }
            let row = floor(v / 0.5);
            let col = floor(u / 0.9 + row * 0.5);
            let fu = fract(u / 0.9 + row * 0.5);
            let fv = fract(v / 0.5);
            let joint = min(min(fu, 1.0 - fu) * 0.9, min(fv, 1.0 - fv) * 0.5);
            let mortar = (1.0 - smoothstep(0.01, 0.025, joint)) * d_dm;
            let tint = hash2(vec2(col, row));
            var c = mix(vec3(0.72, 0.68, 0.60), vec3(0.83, 0.79, 0.70), tint * (1.0 - calm * 0.7)) * (0.9 + 0.14 * fine);
            // Rain streaks run down from ledges; a little moss on the tops.
            let streak = smoothstep(0.5, 0.9, vnoise(q * vec3(2.5, 0.2, 2.5)));
            c *= 1.0 - 0.16 * streak - 0.06 * broad;
            let moss = smoothstep(0.62, 0.8, fbm(q * 0.9)) * select(0.15, 0.7, top);
            c = mix(c, vec3(0.38, 0.42, 0.24), moss * 0.5);
            c = mix(c, vec3(0.47, 0.44, 0.39), mortar * 0.7);
            s.albedo = c;
            s.rough = 0.8;
            s.f0 = 0.04;
            s.height = (1.0 - mortar) * 0.015 + fine * 0.008 + vnoise(q * 19.0) * 0.003 * d_cm;
        }
        case 52u: { // gilt: burnished gold leaf, a little worn
            let wear = smoothstep(0.55, 0.85, vnoise(q * 11.0));
            s.albedo = mix(vec3(0.85, 0.63, 0.24), vec3(0.55, 0.40, 0.18), wear * 0.5);
            s.rough = mix(0.3, 0.55, wear);
            s.f0 = 0.6;
            wettable = false;
        }
        default: {}
    }

    // High up, snow settles on whatever faces the sky and slides off steep
    // rock, with a soft, ragged snowline (the voxels carry the coarse cover).
    if (mat == 1u || mat == 2u || mat == 3u || mat == 5u || mat == 6u) {
        let ragged = vnoise(p * 0.08) * 16.0 + vnoise(p * 0.6) * 3.0;
        let line = 100.0 + ragged - 10.0 * smoothstep(0.75, 0.95, n.y);
        let cover = smoothstep(line - 3.0, line + 3.0, p.y) * smoothstep(0.5, 0.8, n.y);
        let snow = vec3(0.86, 0.89, 0.94) * (0.92 + 0.1 * vnoise(q * 3.0));
        s.albedo = mix(s.albedo, snow, cover);
        s.rough = mix(s.rough, 0.7, cover);
        s.sss = mix(s.sss, 0.4, cover);
    }

    // Ground just above the waterline is darker and glossier.
    if (wettable) {
        let level = water_level_at(p.xz);
        let wet = 1.0 - smoothstep(level - 0.1, level + 0.5, p.y);
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
    @location(3) normal: vec3<f32>,
};

// Face 7 marks a vertex of the smooth surface, whose normal is packed in
// octahedral form in bits 13..31 (see `smooth_data` in mesh.rs).
fn vertex_normal(info: u32) -> vec3<f32> {
    let face = info & 7u;
    if (face < 6u) {
        return NORMALS[face];
    }
    let q = vec2(f32((info >> 13u) & 511u), f32((info >> 22u) & 511u)) / 511.0 * 2.0 - 1.0;
    var n = vec3(q.x, 1.0 - abs(q.x) - abs(q.y), q.y);
    if (n.y < 0.0) {
        let sx = select(-1.0, 1.0, q.x >= 0.0);
        let sz = select(-1.0, 1.0, q.y >= 0.0);
        n = vec3((1.0 - abs(q.y)) * sx, n.y, (1.0 - abs(q.x)) * sz);
    }
    return normalize(n);
}

// Distance in metres beyond which grass blades and flowers are not drawn.
const PLANT_RANGE: f32 = 70.0;

// Grass blades and wildflowers: thin, wind-bent, left out of the shadow map.
fn is_plant(mat: u32) -> bool {
    return mat == 13u || (mat >= 15u && mat <= 17u) || (mat >= 30u && mat <= 33u) || mat == 40u || mat == 41u;
}

// Leaves sway a few centimetres in the wind. The offset depends only on the
// position, so neighbouring faces move together and no cracks open.
fn sway(pos: vec3<f32>, data: u32) -> vec3<f32> {
    let mat = (data >> 3u) & 255u;
    let t = g.sun_dir.w;
    if (is_plant(mat)) {
        // Grass bends from the root: gusts roll across the meadow as waves,
        // with a quicker flutter on top.
        let tip = f32((data >> 11u) & 3u) / 3.0;
        let wind = normalize(vec2(0.8, 0.6));
        let gust = vnoise2(pos.xz * 0.06 - wind * t * 0.9);
        let wave = 0.5 + 0.5 * sin(dot(pos.xz, wind) * 0.45 - t * 2.4);
        let flutter = sin(t * 5.3 + pos.x * 3.1 + pos.z * 2.7) * 0.25;
        let bend = (0.12 + 0.35 * gust * wave + flutter * 0.1) * tip;
        return pos + vec3(wind.x * bend, -abs(bend) * 0.25, wind.y * bend);
    }
    if (mat == 53u) {
        // Banners flap about their rod: the AO bits hold how far down the cloth
        // a vertex is. Both sides move the same way so the cloth stays whole.
        let tip = f32((data >> 11u) & 3u) / 3.0;
        var nn = vertex_normal(data);
        if (nn.x + nn.z < 0.0) {
            nn = -nn;
        }
        let ph = pos.x * 0.9 + pos.z * 0.7;
        let flap = 0.6 * sin(t * 1.4 + ph) + 0.4 * sin(t * 3.3 + ph * 1.7 + tip * 2.5);
        return pos + (nn * flap * 0.1 + vec3(0.06, 0.0, 0.03) * (0.5 + 0.5 * sin(t * 0.7 + ph))) * tip;
    }
    if (mat != 8u && mat != 28u && mat != 29u) {
        return pos;
    }
    let ph = dot(pos, vec3(0.7, 0.3, 0.5));
    return pos + vec3(sin(t * 1.7 + ph), 0.5 * sin(t * 2.3 + ph * 1.3), cos(t * 1.3 + ph * 0.8)) * 0.025;
}

@vertex
fn vs_world(@location(0) pos: vec3<f32>, @location(1) data: u32) -> VOut {
    var o: VOut;
    var p = sway(pos, data);
    // Far grass and flowers are too small to see: their upper vertices sink
    // into the ground there, so they cost almost no pixels (the ground's own
    // shading carries the meadow). The fade is continuous, so nothing pops.
    if (is_plant((data >> 3u) & 255u)) {
        let tip = f32((data >> 11u) & 3u) / 3.0;
        p.y -= smoothstep(PLANT_RANGE * 0.75, PLANT_RANGE, distance(pos, g.camera_pos.xyz)) * 2.5 * tip;
    }
    o.clip = g.view_proj * vec4(p, 1.0);
    o.world = pos;
    o.info = data;
    o.ao = f32((data >> 11u) & 3u) / 3.0;
    o.normal = vertex_normal(data);
    return o;
}

@vertex
fn vs_shadow(@location(0) pos: vec3<f32>, @location(1) data: u32) -> @builtin(position) vec4<f32> {
    // Grass blades are too thin to matter in the shadow map; collapse them.
    if (is_plant((data >> 3u) & 255u)) {
        return vec4(2.0, 2.0, 0.5, 1.0);
    }
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

@fragment
fn fs_terrain(i: VOut) -> @location(0) vec4<f32> {
    return shade_terrain(i);
}

@fragment
fn fs_far_terrain(i: VOut) -> @location(0) vec4<f32> {
    if (under_near(i.world, normalize(i.normal))) {
        discard;
    }
    return shade_terrain(i);
}

// Signed distance in metres from `p` to the edit brush (negative inside),
// drawn as a smooth shape rather than voxel steps. Shapes follow
// `Brush::contains` in game.rs: 2 is a 1 m block from the corner in
// `highlight.xyz`, 1 a cube and 0 a sphere around the aimed-at voxel.
fn brush_distance(p: vec3<f32>) -> f32 {
    let r = g.highlight.w - 1.0;
    if (g.params.w > 1.5) {
        // A little larger than the block so its own faces sit inside the glow
        // and only the rounded corners and the ground around it catch the rim.
        return sd_round_box(p - g.highlight.xyz - VOXEL, vec3(VOXEL + 0.08), 0.28);
    }
    let q = p - g.highlight.xyz - 0.5 * VOXEL;
    let size = (r + 0.5) * VOXEL;
    if (g.params.w > 0.5) {
        return sd_round_box(q, vec3(size), min(0.3 * size, 0.4));
    }
    return length(q) - (r + 0.35 + 0.15) * VOXEL;
}

fn sd_round_box(p: vec3<f32>, half: vec3<f32>, rad: f32) -> f32 {
    let q = abs(p) - half + rad;
    return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - rad;
}

fn shade_terrain(i: VOut) -> vec4<f32> {
    let n = normalize(i.normal);
    let mat = (i.info >> 3u) & 255u;
    let pix = length(fwidth(i.world)) * 0.7;
    var surf = material(mat, i.world, n, pix);
    if (mat == 13u || mat == 30u || mat == 31u) {
        // Blades darken towards the ground and dry out a little at the tips.
        surf.albedo *= mix(0.45, 1.1, i.ao);
        surf.albedo = mix(surf.albedo, surf.albedo * vec3(1.25, 1.1, 0.7), i.ao * i.ao * 0.5);
    }
    var ao_in = i.ao;
    if (mat == 53u) {
        // Gold trim down the sides and across the top, and a ring on the cloth.
        let fu = fract(i.world.x / BANNER_W);
        let side = 1.0 - smoothstep(0.05, 0.075, min(fu, 1.0 - fu));
        let top_band = 1.0 - smoothstep(0.04, 0.06, i.ao);
        let low_band = smoothstep(0.83, 0.85, i.ao) * (1.0 - smoothstep(0.87, 0.89, i.ao));
        let ring = 1.0 - smoothstep(0.02, 0.03, abs(length(vec2((fu - 0.5) * BANNER_W, (i.ao - 0.42) * 1.7)) - 0.15));
        surf.albedo = mix(surf.albedo, lin(vec3(0.85, 0.62, 0.20)), max(max(side, top_band), max(low_band, ring)));
        ao_in = 1.0;
    }
    let nb = bump_normal(n, i.world, surf.height);

    let sun = normalize(g.sun_dir.xyz);
    let to_cam = g.camera_pos.xyz - i.world;
    let v = normalize(to_cam);
    // Translucent materials (leaves most of all) get softer occlusion and a
    // light that wraps past the terminator instead of cutting off hard.
    let ao = mix(mix(0.35, 0.6, surf.sss), 1.0, ao_in);
    let facing = step(0.0, dot(n, sun));
    let sh = sun_shadow(i.world, n) * facing;
    let wrap = surf.sss * 0.6;
    let ndl = max((dot(nb, sun) + wrap) / (1.0 + wrap), 0.0);

    let light = sun_light();
    let sky = 0.5 + 0.5 * nb.y;
    // Skylight also reaches the undersides of thin layers through them.
    let under = 0.35 + 0.4 * surf.sss;
    // Light from the open sky: the dome's colour straight up, bluer in the
    // shade than the warm sun. Light bounced off the sunlit ground below is warm.
    var skylight = sky_dome(vec3(0.0, 1.0, 0.0)) * 0.8 + sky_dome(normalize(vec3(-sun.z, 0.02, sun.x))) * 0.3;
    // The painted golden-hour sky is more saturated than the light it really sheds.
    skylight = mix(skylight, vec3(dot(skylight, vec3(0.3, 0.5, 0.2))) * vec3(0.95, 0.95, 1.05), golden_hour() * 0.55);
    let bounce = light * lin(vec3(0.52, 0.47, 0.36)) * 0.16 * max(sun.y, 0.0);
    let ambient = (skylight * (sky + (1.0 - sky) * surf.sss * 0.6) + bounce * (1.0 - sky) * under) * ao;
    // Sunlight scattered through leaves and grass, strongest when looking towards the sun.
    let through = pow(max(dot(-v, sun), 0.0), 4.0) * 1.6 + 0.35 * max(dot(-n, sun), 0.0) + 0.06;
    let trans = surf.sss * through * mix(0.25, 1.0, sh) * lin(vec3(0.85, 0.95, 0.45)) * (light / 2.6);

    let lamps = lamp_light(i.world + n * 0.05, nb);
    var c = surf.albedo * (light * ndl * sh + ambient * mix(0.55, 1.0, ao) + trans * ao + lamps * ao);
    c += surf.emit * mix(0.25, 1.0, lamp_on());
    c += light * sh * ggx_spec(nb, v, sun, surf.rough, surf.f0);
    // A hint of sky reflected on glossy (wet) surfaces.
    let fres = surf.f0 + (1.0 - surf.f0) * pow(1.0 - max(dot(nb, v), 0.0), 5.0);
    let gloss = 1.0 - surf.rough;
    c += sky_color(reflect(-v, nb)) * fres * gloss * gloss * gloss * 0.6 * ao;

    // Preview what a click would edit: a soft glow over the brush shape with
    // a gentle bright rim where it meets the surface.
    if (g.highlight.w > 0.5) {
        let d = brush_distance(i.world);
        let pulse = 0.8 + 0.2 * sin(g.sun_dir.w * 4.0);
        let inside = 1.0 - smoothstep(-0.03, 0.03, d);
        let rim = exp(-abs(d) / max(0.035, pix * 1.5));
        c = mix(c, c * 1.2 + vec3(0.04, 0.045, 0.04), 0.4 * inside * pulse);
        c += vec3(1.0, 0.92, 0.66) * rim * 0.55 * pulse;
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
    // Around rocks in the current the white water is broken into streaks.
    let streak = vnoise2(vec2(p.x * 7.0, p.z * 1.5 - t * 1.4));
    let foam = (1.0 - smoothstep(0.0, 0.45, depth_below)) * smoothstep(0.35, 0.65, foam_n * 0.7 + streak * 0.3);
    // The pool at the foot of a fall churns white, calming downstream.
    let fall_d = below_fall(p.xz);
    var churn = (1.0 - smoothstep(0.0, 9.0, fall_d)) * smoothstep(-2.0, 0.0, fall_d);
    // So does the plunge pool where a fall off the castle bluff lands.
    churn = max(churn, 1.0 - smoothstep(1.0, 7.0, from_cliff_fall(p.xz)));
    let churn_n = vnoise2(vec2(p.x * 3.0, p.z * 2.0 - t * 2.2)) * 0.5 + vnoise2(p.xz * 7.0 + t * 0.7) * 0.5;
    var white = max(foam, churn * smoothstep(0.2, 0.55, churn_n * (0.6 + 0.5 * churn)));
    if (face_n.y < 0.5) {
        // A falling sheet: white streaks pouring down over a thin green body.
        let along = dot(p.xz, vec2(face_n.z, -face_n.x));
        let pour = vnoise2(vec2(along * 5.0, p.y * 0.8 + t * 2.6)) * 0.6
            + vnoise2(vec2(along * 13.0, p.y * 1.7 + t * 4.1)) * 0.4;
        white = 0.2 + 0.7 * smoothstep(0.35, 0.75, pour);
    }
    body = mix(body, lin(vec3(0.92, 0.95, 0.95)) * (0.5 + 0.8 * sh), white * 0.8);

    // Reflection: screen-space first, the sky where the screen has no answer.
    let r = reflect(-v, n);
    let sky_refl = sky_color(normalize(vec3(r.x, max(r.y, 0.0), r.z)));
    let ssr = trace_reflection(p + n * 0.02, r, dist);
    let refl = mix(sky_refl, ssr.rgb, ssr.w);

    let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    var c = mix(body, refl, fresnel * (1.0 - white * 0.8));
    c += sun_light() * 1.15 * sh * ggx_spec(n, v, sun, 0.07, 0.02) * (1.0 - white);
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

// Ridge height (radians above the horizon) of a far mountain range at a
// compass direction, sampled on a circle so it wraps without a seam.
fn range_height(flat: vec2<f32>, scale: f32, seed: f32, base: f32, top: f32) -> f32 {
    let q = flat * scale + vec2(seed, seed * 1.7);
    let r = 1.0 - abs(fbm2(q) * 2.0 - 1.0);
    return base + top * r * r;
}

// Snowy peaks beyond the edge of the world: two ranges of painted
// silhouettes seen through the haze. Their snow keeps the terrain's rules
// (column_info in terrain.rs and the snow cover in material()): it lies
// above a ragged snowline, which comes lower where the face is gentle, and
// slides off steep faces, which stay bare rock. rgb, coverage.
fn far_peaks(dir: vec3<f32>) -> vec4<f32> {
    let flat = normalize(dir.xz + vec2(1e-5, 0.0));
    let e = asin(clamp(dir.y, -1.0, 1.0));
    let sun = normalize(g.sun_dir.xyz);
    let sun_flat = normalize(sun.xz + vec2(1e-5, 0.0));
    let light = sun_light();
    let haze = sky_dome(normalize(vec3(dir.x, 0.03, dir.z)));
    let side = vec2(-flat.y, flat.x);
    let az = atan2(flat.y, flat.x);
    var col = vec3(0.0);
    var cover = 0.0;
    // Far range first, then the nearer, bolder one over it.
    for (var k = 0; k < 2; k++) {
        let near = f32(k);
        let scale = mix(2.2, 3.4, near);
        let seed = 11.0 + near * 7.0;
        let base = mix(0.035, 0.02, near);
        let top = mix(0.16, 0.12, near);
        let h = range_height(flat, scale, seed, base, top);
        if (e < h) {
            // The face's slope is the ridge's own gradient along the skyline,
            // taken over a span that widens below the crest, where the faces
            // are broader. Ribs and gullies break it into facets.
            let w = 0.006 + (h - e) * 0.5;
            let hl = range_height(normalize(flat - side * w), scale, seed, base, top);
            let hr = range_height(normalize(flat + side * w), scale, seed, base, top);
            let dh = (hr - hl) / (2.0 * w);
            let rib = vnoise2(vec2(az * 110.0 + e * 40.0 + near * 13.0, e * 7.0));
            let facet = vnoise2(vec2(az * 260.0, e * 70.0));
            let crest = 1.0 - smoothstep(0.0, 0.02, h - e);
            let steep = abs(dh) * (0.3 + 1.4 * rib) + 0.3 * crest * abs(dh) + (facet - 0.5) * 0.35 + 0.08;
            // Lit where the face turns towards the sun, and on the ribs.
            let facing = clamp(0.5 - dh * 0.6 * dot(side, sun_flat), 0.0, 1.0);
            let lit = clamp(facing * 0.8 + (rib - 0.5) * 0.5 + (facet - 0.5) * 0.2, 0.0, 1.0);
            // Snow by altitude above a ragged line, lower on gentle faces, and
            // none on steep ones.
            let ragged = vnoise2(vec2(az * 24.0 + near * 5.0, near)) * 0.7 + vnoise2(vec2(az * 90.0, 3.0 + near)) * 0.3;
            let gentle = 1.0 - smoothstep(0.25, 0.6, steep);
            let line = base + top * (0.5 + 0.3 * (ragged - 0.5) - 0.12 * gentle);
            let snow = smoothstep(line - 0.0025, line + 0.0025, e) * (1.0 - smoothstep(0.33, 0.5, steep));
            let snow_col = lin(vec3(0.95, 0.93, 0.98)) * (0.3 + 0.8 * lit) * light * 0.5;
            // Bare rock, with the air between tinting it blue-grey: more for
            // the far range and towards the foot, where the air is thicker.
            let rock_lit = lin(vec3(0.30, 0.30, 0.34)) * (0.2 + 0.6 * lit) * light * 0.3;
            let air = mix(0.45, 0.3, near) + 0.2 * (1.0 - smoothstep(0.0, top, e - base));
            let rock = mix(rock_lit, haze * lin(vec3(0.62, 0.70, 0.90)), air);
            let body = mix(rock, snow_col, snow);
            // Both ranges sink into the haze towards their feet.
            let fade = mix(0.5, 0.3, near) * (1.0 - smoothstep(0.0, 0.08, e - base) * 0.4);
            col = mix(body, haze, fade);
            cover = 1.0;
        }
    }
    return vec4(col, cover);
}

@fragment
fn fs_sky(i: SkyOut) -> @location(0) vec4<f32> {
    let far = g.inv_view_proj * vec4(i.ndc, 0.5, 1.0);
    let dir = normalize(far.xyz / far.w - g.camera_pos.xyz);
    let cl = clouds(dir);
    var c = mix(sky_color(dir) + sun_disc(dir) + stars(dir), cl.rgb, cl.a);
    let peaks = far_peaks(dir);
    c = mix(c, peaks.rgb, peaks.a) + lamp_glow(dir, 400.0);
    if (g.params.z > 0.5) {
        c = underwater_color();
    }
    return vec4(c, 1.0);
}

// ---------------------------------------------------------------- bloom

fn screen_uv(ndc: vec2<f32>) -> vec2<f32> {
    return vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return min(textureSampleLevel(hdr_input, post_sampler, uv, 0.0).rgb, vec3(64.0));
}

// Thirteen-tap downsample (as in Jimenez 2014): a wide, smooth halving
// without the blocky flicker of a plain box. `karis` weighs each group by
// its brightness so single very bright pixels do not flicker as sparks.
fn downsample(uv: vec2<f32>, karis: bool) -> vec3<f32> {
    let px = 1.0 / vec2<f32>(textureDimensions(hdr_input));
    let a = tap(uv + px * vec2(-2.0, -2.0));
    let b = tap(uv + px * vec2(0.0, -2.0));
    let c = tap(uv + px * vec2(2.0, -2.0));
    let d = tap(uv + px * vec2(-1.0, -1.0));
    let e = tap(uv + px * vec2(1.0, -1.0));
    let f = tap(uv + px * vec2(-2.0, 0.0));
    let m = tap(uv);
    let h = tap(uv + px * vec2(2.0, 0.0));
    let i = tap(uv + px * vec2(-1.0, 1.0));
    let j = tap(uv + px * vec2(1.0, 1.0));
    let k = tap(uv + px * vec2(-2.0, 2.0));
    let l = tap(uv + px * vec2(0.0, 2.0));
    let n = tap(uv + px * vec2(2.0, 2.0));
    var groups = array<vec3<f32>, 5>(
        (d + e + i + j) * 0.25,
        (a + b + f + m) * 0.25,
        (b + c + m + h) * 0.25,
        (f + m + k + l) * 0.25,
        (m + h + l + n) * 0.25,
    );
    let weights = array<f32, 5>(0.5, 0.125, 0.125, 0.125, 0.125);
    var sum = vec3(0.0);
    var wsum = 0.0;
    for (var q = 0; q < 5; q++) {
        var w = weights[q];
        if (karis) {
            w /= 1.0 + dot(groups[q], vec3(0.2126, 0.7152, 0.0722));
        }
        sum += groups[q] * w;
        wsum += w;
    }
    return sum / wsum;
}

@fragment
fn fs_bloom_first(i: SkyOut) -> @location(0) vec4<f32> {
    return vec4(downsample(screen_uv(i.ndc), true), 1.0);
}

@fragment
fn fs_bloom_down(i: SkyOut) -> @location(0) vec4<f32> {
    return vec4(downsample(screen_uv(i.ndc), false), 1.0);
}

// Nine-tap tent filter over the smaller level, added onto the larger one.
@fragment
fn fs_bloom_up(i: SkyOut) -> @location(0) vec4<f32> {
    let uv = screen_uv(i.ndc);
    let px = 1.0 / vec2<f32>(textureDimensions(hdr_input));
    var c = tap(uv) * 4.0;
    c += (tap(uv + vec2(px.x, 0.0)) + tap(uv - vec2(px.x, 0.0)) + tap(uv + vec2(0.0, px.y)) + tap(uv - vec2(0.0, px.y))) * 2.0;
    c += tap(uv + px) + tap(uv - px) + tap(uv + vec2(px.x, -px.y)) + tap(uv + vec2(-px.x, px.y));
    return vec4(c / 16.0, 1.0);
}

// Tone maps the HDR scene onto the screen.
@fragment
fn fs_post(i: SkyOut) -> @location(0) vec4<f32> {
    let uv = vec2(i.ndc.x * 0.5 + 0.5, 0.5 - i.ndc.y * 0.5);
    // Eyes adapt to the dark: expose nights brighter.
    var c = textureSampleLevel(hdr_input, post_sampler, uv, 0.0).rgb;
    // Bloom: a soft glow of everything, strongest around the brightest light.
    // The chain sums five blurred levels, hence the divide.
    let bloom = textureSampleLevel(bloom_tex, post_sampler, uv, 0.0).rgb / 5.0;
    c = mix(c, bloom, 0.07);
    c *= mix(1.0, 2.6, g.sky.x);
    let q = i.ndc * 0.75;
    c *= 1.0 - 0.22 * dot(q, q);
    var m = finish(c);
    // Grade: rich, painterly colour. More saturation, warm highlights and
    // shadows leaning violet-blue against the warm sun, and a gentle S-curve.
    let l = dot(m, vec3(0.2126, 0.7152, 0.0722));
    m = mix(vec3(l), m, 1.18);
    // Painterly greens: yellow-greens pulled towards a deeper, cooler green.
    let green = max(m.g - max(m.r, m.b), 0.0);
    m -= vec3(0.30, 0.22, -0.04) * green;
    m += vec3(-0.01, -0.012, 0.03) * (1.0 - l) * (1.0 - l);
    m += vec3(0.03, 0.012, -0.02) * l * l;
    m = mix(m, m * m * (3.0 - 2.0 * m), 0.35);
    m = clamp(m, vec3(0.0), vec3(1.0));
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
