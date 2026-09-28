//! Boulders: chunky granite rocks with broad, softly rounded facets, bedded
//! half into the ground in clusters (a big one with a few smaller ones leaning
//! on it). They crowd the river banks, the shallows and the foot of each fall,
//! stand now and then beside the roads and lie scattered over the meadows.
//!
//! Each boulder is an authored mesh, not sculpted voxels: a geodesic sphere
//! pushed out to a rounded polytope (the smooth minimum of a handful of
//! planes), stretched per rock into round, long, tall or slab-like shapes.
//! Its voxels are hidden `BUILT` blocks sized a little inside the model, so
//! the player collides with it but the surface mesher never draws it.

use crate::block::*;
use crate::chunk::{chunk_of, local_index, CHUNK, CHUNK_VOLUME};
use crate::mesh::{smooth_data, MeshData, Vertex};
use crate::noise::{hash2, unit, value3};
use crate::terrain::{below_fall, water_level, Terrain, ROAD_HALF_WIDTH_M};
use glam::{IVec2, IVec3, Vec2, Vec3};
use std::collections::HashMap;
use std::f32::consts::TAU;
use std::sync::OnceLock;

/// Grid cell (metres) that holds at most one boulder cluster.
pub const CELL_M: f32 = 6.0;
/// Largest radius of a cluster's main boulder in metres.
pub const MAX_R_M: f32 = 2.4;
/// No part of a cluster reaches further than this (metres) from its site.
pub const CLUSTER_REACH_M: f32 = 6.5;
/// Boulders keep this clear of a road's packed surface (metres).
const ROAD_CLEARANCE_M: f32 = 0.3;
/// Where the flat base sits, as a fraction of the vertical radius below the centre.
const BASE: f32 = 0.6;
/// How far the collision voxels stay inside the model's surface (metres).
const COLLISION_INSET_M: f32 = 0.2;

/// Where a cluster stands: its main boulder's spot on the ground.
pub struct Site {
    pub x: f32,
    pub z: f32,
    /// Ground height at the spot, metres.
    pub ground: f32,
    max_r: f32,
    hash: u32,
    composed: bool,
}

impl Site {
    /// The chunk whose mesh carries the whole cluster.
    pub fn chunk(&self) -> IVec3 {
        chunk_of(
            (Vec3::new(self.x, self.ground + 0.1, self.z) / VOXEL_SIZE)
                .floor()
                .as_ivec3(),
        )
    }
}

pub struct Boulder {
    /// Model origin in metres: the middle of the rock, below ground level.
    pub centre: Vec3,
    /// Half extents in metres along the rock's own axes (before `yaw`).
    pub radii: Vec3,
    /// Turn about the vertical, radians.
    pub yaw: f32,
    /// Ground height under the centre, metres, and its slope (rise per metre in X and Z).
    pub ground: f32,
    pub slope: Vec2,
    seed: u32,
}

/// A site for a cluster in grid cell (gx, gz), if one stands there.
pub fn site(t: &Terrain, gx: i32, gz: i32) -> Option<Site> {
    site_where(t, gx, gz, |_| true)
}

/// Like `site`, but first checks the spot (x, ground height, z in metres)
/// with `keep`, cheaply, before working out the rest.
fn site_where(t: &Terrain, gx: i32, gz: i32, keep: impl Fn(Vec3) -> bool) -> Option<Site> {
    let h = hash2(t.seed.wrapping_add(40), gx, gz);
    let mut x = (gx as f32 + 0.1 + 0.8 * unit(h)) * CELL_M;
    let mut z = (gz as f32 + 0.1 + 0.8 * unit(h.rotate_left(9))) * CELL_M;
    // A mossy group in the meadow ahead and left of the arrival spot, as in
    // the foreground of the reference view.
    let spot = vista_rocks();
    let composed = (spot / CELL_M).floor().as_ivec2() == IVec2::new(gx, gz);
    if composed {
        (x, z) = (spot.x, spot.y);
    }
    if !keep(Vec3::new(x, t.height_at(x, z).0, z)) {
        return None;
    }
    let info = t.column_info(x, z);
    let above_water = info.height_m - water_level(x, z);
    let road = t.road_distance(x, z);
    // Rocks crowd the foot of each fall, breaking the water into cascades.
    let (chance, max_r) = if below_fall(x, z, 12.0).is_some() && above_water < 2.5 {
        (0.9, MAX_R_M)
    } else if (-1.2..2.5).contains(&above_water) {
        (0.5, MAX_R_M)
    } else if (2.2..5.5).contains(&road) && above_water > 0.5 && info.height_m < 80.0 {
        // Now and then a rock sits at the edge of the cobbled road.
        (0.14, 1.2)
    } else if info.height_m > 36.0 && info.height_m < 120.0 && (0.4..0.8).contains(&slope(t, x, z)) {
        // Outcrops break through the grass where the valley walls steepen.
        (0.35, 1.9)
    } else if info.surface == GRASS && info.height_m < 60.0 {
        (0.05, MAX_R_M)
    } else {
        (0.0, 0.0)
    };
    let (chance, max_r) = if composed { (1.0, 1.7) } else { (chance, max_r) };
    if unit(h.rotate_left(19)) >= chance {
        return None;
    }
    Some(Site {
        x,
        z,
        ground: info.height_m,
        max_r,
        hash: h,
        composed,
    })
}

/// Rise per metre of the ground at (x, z), measured over a few metres.
fn slope(t: &Terrain, x: f32, z: f32) -> f32 {
    let d = 2.0;
    let dx = t.height_at(x + d, z).0 - t.height_at(x - d, z).0;
    let dz = t.height_at(x, z + d).0 - t.height_at(x, z - d).0;
    Vec2::new(dx, dz).length() / (2.0 * d)
}

/// Where the composed boulder group by the arrival spot stands (x, z metres).
fn vista_rocks() -> Vec2 {
    let (ax, az) = crate::vista::ARRIVAL;
    let yaw = crate::vista::SPAWN_YAW;
    let fwd = Vec2::new(yaw.cos(), yaw.sin());
    Vec2::new(ax, az) + fwd * 7.0 - fwd.perp() * 3.5
}

/// The boulders of a cluster: the main one first, then up to four smaller
/// ones huddled around it.
pub fn cluster(t: &Terrain, site: &Site) -> Vec<Boulder> {
    let hr = |k: u32| unit(site.hash.rotate_left(k));
    let size = if site.composed { 0.95 } else { hr(27) };
    let r = 0.55 + (site.max_r - 0.55) * size * size;
    let mut out = Vec::new();
    let Some(main) = boulder(t, site.x, site.z, r, site.hash) else {
        return out;
    };
    let reach = main.radii.x.max(main.radii.z);
    let count = if site.composed {
        4
    } else if r > 1.0 {
        1 + (hr(7) * 3.99) as u32
    } else {
        (hr(7) * 2.6) as u32
    };
    let a0 = hr(3) * TAU;
    out.push(main);
    for k in 0..count {
        let hk = hash2(site.hash, k as i32, 77);
        let hu = |s: u32| unit(hk.rotate_left(s));
        let sr = (r * (0.25 + 0.3 * hu(1))).max(0.35);
        let a = a0 + (k as f32 + 0.35 * (hu(2) - 0.5)) * TAU / count as f32;
        let d = (reach * (0.75 + 0.3 * hu(3)) + sr * 0.6).min(CLUSTER_REACH_M - sr * 1.3);
        if let Some(b) = boulder(t, site.x + d * a.cos(), site.z + d * a.sin(), sr, hk) {
            out.push(b);
        }
    }
    out
}

/// One boulder of radius `r` at (x, z), shaped and bedded from hash `h`, or
/// none where it would crowd a road or a building.
fn boulder(t: &Terrain, x: f32, z: f32, r: f32, h: u32) -> Option<Boulder> {
    let hr = |k: u32| unit(h.rotate_left(k));
    // Most are chunky and round; some long, some tall, some flat slabs.
    let kind = hr(13);
    let radii = if kind < 0.5 {
        Vec3::new(1.0 + 0.15 * hr(5), 0.9 + 0.15 * hr(6), 0.9 + 0.1 * hr(8))
    } else if kind < 0.72 {
        Vec3::new(1.3 + 0.2 * hr(5), 0.8 + 0.1 * hr(6), 0.85)
    } else if kind < 0.88 {
        Vec3::new(0.85, 1.15 + 0.2 * hr(6), 0.8 + 0.1 * hr(8))
    } else {
        Vec3::new(1.25, 0.6 + 0.1 * hr(6), 1.0 + 0.1 * hr(8))
    } * r;
    let reach = radii.x.max(radii.z);
    if t.road_distance(x, z) - reach < ROAD_HALF_WIDTH_M + ROAD_CLEARANCE_M || t.structures.path_at(t, x, z) {
        return None;
    }
    let g0 = t.height_at(x, z).0;
    let lo = Vec3::new(x - reach, g0 - radii.y * 2.0, z - reach);
    let hi = Vec3::new(x + reach, g0 + radii.y * 2.0, z + reach);
    if t.structures.touches(lo, hi) {
        return None;
    }
    // Fit the ground under the rock with a plane, and find its lowest rim.
    let s = reach * 0.9;
    let [px, nx, pz, nz] = [(s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)].map(|(dx, dz)| t.height_at(x + dx, z + dz).0);
    let slope = Vec2::new(px - nx, pz - nz) / (2.0 * s);
    let rim = px.min(nx).min(pz).min(nz).min(g0 - slope.length() * s);
    // Sink it a little past its waist, and always deep enough that no part of
    // the base shows on the downhill side.
    let sink = 0.2 * hr(21) - 0.05;
    let cy = (g0 - sink * radii.y).min(rim - 0.25 + BASE * radii.y);
    if cy + 0.7 * radii.y < g0 {
        return None;
    }
    Some(Boulder {
        centre: Vec3::new(x, cy, z),
        radii,
        yaw: hr(29) * TAU,
        ground: g0,
        slope,
        seed: h,
    })
}

/// The rock's form in its own unit space: the rounded intersection of a few
/// half-spaces (broad facets) and a sphere, with a flat base.
struct Shape {
    planes: Vec<(Vec3, f32)>,
    seed: u32,
}

impl Shape {
    fn new(seed: u32) -> Self {
        let hr = |k: u32| unit(hash2(seed, k as i32, 91));
        let mut planes = Vec::with_capacity(12);
        // A broad, slightly tilted crown where the moss settles.
        planes.push((
            Vec3::new(hr(0) - 0.5, 1.6, hr(1) - 0.5).normalize(),
            0.74 + 0.14 * hr(2),
        ));
        // Facets around the waist, some leaning up into shoulders.
        let n = 5 + (hr(3) * 2.99) as u32;
        let a0 = hr(4) * TAU;
        for i in 0..n {
            let a = a0 + (i as f32 + 0.6 * (hr(10 + i) - 0.5)) * TAU / n as f32;
            let y = -0.3 + 0.8 * hr(20 + i);
            planes.push((Vec3::new(a.cos(), y, a.sin()).normalize(), 0.74 + 0.2 * hr(30 + i)));
        }
        for i in 0..2 {
            let a = hr(40 + i) * TAU;
            let y = 0.7 + 0.6 * hr(45 + i);
            planes.push((Vec3::new(a.cos(), y, a.sin()).normalize(), 0.8 + 0.08 * hr(50 + i)));
        }
        planes.push((Vec3::NEG_Y, BASE));
        Self { planes, seed }
    }

    /// Distance from the centre to the surface along unit direction `d`.
    fn radius(&self, d: Vec3) -> f32 {
        // A smooth minimum over the distances to each plane rounds every edge.
        const K: f32 = 0.035;
        let mut sum = (-1.1f32 / K).exp();
        for &(n, off) in &self.planes {
            let dn = d.dot(n);
            if dn > 1e-3 {
                sum += (-(off / dn) / K).exp();
            }
        }
        let r = -K * sum.ln();
        // A gentle lumpiness so no two faces are quite flat.
        let q = d * 1.7;
        let lump = value3(self.seed, q.x + 5.0, q.y + 5.0, q.z + 5.0) - 0.5;
        r * (1.0 + 0.07 * lump)
    }
}

impl Shape {
    /// The surface normal along unit direction `d`, in the rock's unit
    /// space: the facing plane's normal, blended across rounded edges.
    fn facet_normal(&self, d: Vec3) -> Vec3 {
        const K: f32 = 0.035;
        let mut n = d * (-1.1f32 / K).exp();
        for &(pn, off) in &self.planes {
            let dn = d.dot(pn);
            if dn > 1e-3 {
                n += pn * (-(off / dn) / K).exp();
            }
        }
        n.normalize_or(d)
    }
}

impl Boulder {
    /// A normal from the rock's unit space into the world.
    fn normal_to_world(&self, n: Vec3) -> Vec3 {
        let v = n / self.radii;
        let (s, c) = self.yaw.sin_cos();
        Vec3::new(v.x * c - v.z * s, v.y, v.x * s + v.z * c).normalize_or_zero()
    }

    fn local_to_world(&self, l: Vec3) -> Vec3 {
        let v = l * self.radii;
        let (s, c) = self.yaw.sin_cos();
        self.centre + Vec3::new(v.x * c - v.z * s, v.y, v.x * s + v.z * c)
    }

    fn world_to_local(&self, p: Vec3) -> Vec3 {
        let v = p - self.centre;
        let (s, c) = self.yaw.sin_cos();
        Vec3::new(v.x * c + v.z * s, v.y, -v.x * s + v.z * c) / self.radii
    }

    /// Ground height (metres) under a point, from the fitted plane.
    pub fn ground_at(&self, x: f32, z: f32) -> f32 {
        self.ground + self.slope.dot(Vec2::new(x - self.centre.x, z - self.centre.z))
    }

    /// Appends the rock's triangles. With `cull`, triangles well under the
    /// ground are left out.
    pub fn model(&self, cull: bool, out: &mut MeshData) {
        let reach = self.radii.max_element();
        let sphere = geodesic(if reach < 0.8 {
            3
        } else if reach < 1.5 {
            4
        } else {
            5
        });
        let shape = Shape::new(self.seed);
        let pos: Vec<Vec3> = sphere
            .dirs
            .iter()
            .map(|&d| self.local_to_world(d * shape.radius(d)))
            .collect();
        // Flat-faced facets with rounded edges, rather than one smooth dome.
        let facets: Vec<Vec3> = sphere
            .dirs
            .iter()
            .map(|&d| self.normal_to_world(shape.facet_normal(d)))
            .collect();
        let mut normals = vec![Vec3::ZERO; pos.len()];
        for t in sphere.tris.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|k| t[k] as usize);
            let n = (pos[b] - pos[a]).cross(pos[c] - pos[a]);
            for i in [a, b, c] {
                normals[i] += n;
            }
        }
        let depth = |p: Vec3| self.ground_at(p.x, p.z) - p.y;
        let margin = 0.3 + 0.1 * reach;
        let mut remap = vec![u32::MAX; pos.len()];
        for t in sphere.tris.chunks(3) {
            if cull && t.iter().all(|&i| depth(pos[i as usize]) > margin) {
                continue;
            }
            for &i in t {
                let i = i as usize;
                if remap[i] == u32::MAX {
                    remap[i] = out.vertices.len() as u32;
                    // Darker where the rock meets the ground, and a touch on the underside.
                    let n = (normals[i].normalize_or_zero() * 0.25 + facets[i] * 0.75).normalize_or_zero();
                    let lift = -depth(pos[i]);
                    let ao = ((lift / 0.45 + 0.4) * 3.0).clamp(0.0, 3.0) as u32;
                    let ao = if n.y < -0.4 { ao.min(2) } else { ao };
                    out.vertices.push(Vertex {
                        pos: pos[i].to_array(),
                        data: smooth_data(BOULDER, ao, n),
                    });
                }
                out.indices.push(remap[i]);
            }
        }
    }

    /// Fills the voxels a little inside the rock with hidden collision, and
    /// clears grass that would poke through it. The ground is never touched.
    pub fn stamp(&self, origin: IVec3, data: &mut [Block; CHUNK_VOLUME]) {
        let reach = Vec3::splat(self.radii.max_element() * 1.05);
        let lo = ((self.centre - reach) / VOXEL_SIZE).floor().as_ivec3().max(origin);
        let hi = ((self.centre + reach) / VOXEL_SIZE)
            .ceil()
            .as_ivec3()
            .min(origin + IVec3::splat(CHUNK - 1));
        if lo.cmpgt(hi).any() {
            return;
        }
        let shape = Shape::new(self.seed);
        let inset = COLLISION_INSET_M / self.radii.min_element();
        for y in lo.y..=hi.y {
            for z in lo.z..=hi.z {
                for x in lo.x..=hi.x {
                    let v = IVec3::new(x, y, z);
                    let l = self.world_to_local((v.as_vec3() + 0.5) * VOXEL_SIZE);
                    let len = l.length();
                    let r = if len < 1e-4 { 1.0 } else { shape.radius(l / len) };
                    let i = {
                        let o = v - origin;
                        local_index(o.x, o.y, o.z)
                    };
                    if len < r - inset {
                        if matches!(data[i], AIR | TALL_GRASS | WATER) {
                            data[i] = BUILT;
                        }
                    } else if len < r + 0.1 && data[i] == TALL_GRASS {
                        data[i] = AIR;
                    }
                }
            }
        }
    }
}

/// Appends every boulder cluster whose site lies in chunk `cpos`; the whole
/// cluster goes into this one chunk's mesh.
pub fn append(t: &Terrain, cpos: IVec3, out: &mut MeshData) {
    let side = CHUNK as f32 * VOXEL_SIZE;
    let (ox, oz) = (cpos.x as f32 * side, cpos.z as f32 * side);
    let cell = |m: f32| (m / CELL_M).floor() as i32;
    for gz in cell(oz)..=cell(oz + side) {
        for gx in cell(ox)..=cell(ox + side) {
            let here = |p: Vec3| chunk_of(((p + Vec3::Y * 0.1) / VOXEL_SIZE).floor().as_ivec3()) == cpos;
            if let Some(s) = site_where(t, gx, gz, here) {
                for b in cluster(t, &s) {
                    b.model(true, out);
                }
            }
        }
    }
}

/// The footprints of every boulder that could stand within `margin` metres
/// of the chunk column at `cpos`: centre (x, z), horizontal radius, and top height.
pub fn footprints(t: &Terrain, cpos: IVec3, margin: f32) -> Vec<(Vec2, f32, f32)> {
    let side = CHUNK as f32 * VOXEL_SIZE;
    let (ox, oz) = (cpos.x as f32 * side, cpos.z as f32 * side);
    let cell = |m: f32| (m / CELL_M).floor() as i32;
    let reach = CLUSTER_REACH_M + margin;
    let mut out = Vec::new();
    for gz in cell(oz - reach)..=cell(oz + side + reach) {
        for gx in cell(ox - reach)..=cell(ox + side + reach) {
            if let Some(s) = site(t, gx, gz) {
                for b in cluster(t, &s) {
                    let r = b.radii.x.max(b.radii.z);
                    out.push((Vec2::new(b.centre.x, b.centre.z), r, b.centre.y + b.radii.y));
                }
            }
        }
    }
    out
}

/// Stamps the collision of every cluster that reaches into the chunk at `origin` (voxels).
pub fn stamp(t: &Terrain, origin: IVec3, data: &mut [Block; CHUNK_VOLUME]) {
    let side = CHUNK as f32 * VOXEL_SIZE;
    let lo = origin.as_vec3() * VOXEL_SIZE;
    let cell = |m: f32| (m / CELL_M).floor() as i32;
    let reach = CLUSTER_REACH_M;
    for gz in cell(lo.z - reach)..=cell(lo.z + side + reach) {
        for gx in cell(lo.x - reach)..=cell(lo.x + side + reach) {
            let near = |p: Vec3| p.y + reach >= lo.y && p.y - reach <= lo.y + side;
            let Some(s) = site_where(t, gx, gz, near) else { continue };
            for b in cluster(t, &s) {
                b.stamp(origin, data);
            }
        }
    }
}

/// A unit geodesic sphere: an icosahedron with each face cut into
/// `freq * freq` triangles, wound counter-clockwise seen from outside.
struct Sphere {
    dirs: Vec<Vec3>,
    tris: Vec<u32>,
}

fn geodesic(freq: usize) -> &'static Sphere {
    static CACHE: [OnceLock<Sphere>; 6] = [const { OnceLock::new() }; 6];
    CACHE[freq.min(5)].get_or_init(|| build_geodesic(freq.clamp(1, 5)))
}

fn build_geodesic(freq: usize) -> Sphere {
    let p = (1.0 + 5f32.sqrt()) * 0.5;
    let base = [
        Vec3::new(-1.0, p, 0.0),
        Vec3::new(1.0, p, 0.0),
        Vec3::new(-1.0, -p, 0.0),
        Vec3::new(1.0, -p, 0.0),
        Vec3::new(0.0, -1.0, p),
        Vec3::new(0.0, 1.0, p),
        Vec3::new(0.0, -1.0, -p),
        Vec3::new(0.0, 1.0, -p),
        Vec3::new(p, 0.0, -1.0),
        Vec3::new(p, 0.0, 1.0),
        Vec3::new(-p, 0.0, -1.0),
        Vec3::new(-p, 0.0, 1.0),
    ];
    const FACES: [[usize; 3]; 20] = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    let mut dirs = Vec::new();
    let mut tris = Vec::new();
    let mut seen: HashMap<[i32; 3], u32> = HashMap::new();
    let f = freq as f32;
    for face in FACES {
        let [a, b, c] = face.map(|i| base[i]);
        let mut id = |i: usize, j: usize| {
            let d = (a + (b - a) * (i as f32 / f) + (c - a) * (j as f32 / f)).normalize();
            let key = (d * 1e4).round().as_ivec3().to_array();
            *seen.entry(key).or_insert_with(|| {
                dirs.push(d);
                dirs.len() as u32 - 1
            })
        };
        for i in 0..freq {
            for j in 0..freq - i {
                let t = [id(i, j), id(i + 1, j), id(i, j + 1)];
                tris.extend_from_slice(&t);
                if i + j + 1 < freq {
                    let t = [id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)];
                    tris.extend_from_slice(&t);
                }
            }
        }
    }
    // Wind every triangle outward.
    for t in tris.chunks_mut(3) {
        let [a, b, c] = [0, 1, 2].map(|k| dirs[t[k] as usize]);
        if (b - a).cross(c - a).dot(a + b + c) < 0.0 {
            t.swap(1, 2);
        }
    }
    Sphere { dirs, tris }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_boulder(seed: u32, radii: Vec3) -> Boulder {
        Boulder {
            centre: Vec3::new(10.0, 30.0, 20.0),
            radii,
            yaw: unit(seed) * TAU,
            ground: 30.0 + radii.y * 0.3,
            slope: Vec2::new(0.1, -0.05),
            seed,
        }
    }

    #[test]
    fn boulder_models_are_closed_and_face_outward() {
        for (k, radii) in [
            Vec3::new(0.5, 0.4, 0.45),
            Vec3::new(1.2, 0.9, 1.0),
            Vec3::new(3.0, 1.2, 2.4),
        ]
        .into_iter()
        .enumerate()
        {
            let b = test_boulder(1234 + k as u32 * 77, radii);
            let mut out = MeshData::default();
            b.model(false, &mut out);
            assert!(out.indices.len() / 3 <= 500);
            // Closed: every edge is shared by exactly one other triangle, running the other way.
            let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
            for t in out.indices.chunks(3) {
                for (a, c) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    *edges.entry((a, c)).or_default() += 1;
                }
            }
            for (&(a, c), &n) in &edges {
                assert_eq!(n, 1);
                assert_eq!(edges.get(&(c, a)), Some(&1));
            }
            // Outward: seen from the rock's middle, every triangle turns its back.
            for t in out.indices.chunks(3) {
                let [p, q, r] = [0, 1, 2].map(|i| Vec3::from(out.vertices[t[i] as usize].pos) - b.centre);
                assert!((q - p).cross(r - p).dot(p) > 0.0);
            }
            // Bedded: the base lies below the ground, the crown well above it.
            let ys = out.vertices.iter().map(|v| v.pos[1]);
            let (lo, hi) = ys.fold((f32::MAX, f32::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)));
            assert!(lo < b.ground - radii.y * 0.2, "{lo} {}", b.ground);
            assert!(hi > b.ground + radii.y * 0.3);
        }
    }

    #[test]
    fn culling_drops_buried_triangles() {
        let b = test_boulder(99, Vec3::new(2.0, 1.4, 1.8));
        let (mut all, mut culled) = (MeshData::default(), MeshData::default());
        b.model(false, &mut all);
        b.model(true, &mut culled);
        assert!(culled.indices.len() < all.indices.len());
        assert!(culled.vertices.len() < all.vertices.len());
    }

    #[test]
    fn river_banks_have_boulders_and_roads_do_not() {
        let t = Terrain::new(20260927);
        let (mut sites, mut bank, mut rocks, mut groups) = (0, 0, 0, 0);
        for gz in 150..190 {
            for gx in 120..180 {
                let Some(s) = site(&t, gx, gz) else { continue };
                sites += 1;
                let c = cluster(&t, &s);
                groups += (c.len() > 1) as i32;
                for b in &c {
                    rocks += 1;
                    let (x, z) = (b.centre.x, b.centre.z);
                    bank += (b.ground - water_level(x, z) < 2.5) as i32;
                    let reach = b.radii.x.max(b.radii.z);
                    assert!(t.road_distance(x, z) - reach >= ROAD_HALF_WIDTH_M);
                    assert!(b.radii.max_element() <= MAX_R_M * 1.5);
                    assert!(Vec2::new(x - s.x, z - s.z).length() + reach <= CLUSTER_REACH_M + 0.5);
                }
            }
        }
        assert!(bank > 20, "{bank} boulders by the river");
        assert!(groups * 3 > sites, "{groups} clusters of {sites}");
        assert!(rocks < sites * 4);
    }

    #[test]
    fn boulders_stamp_hidden_collision_above_ground() {
        let mut t = Terrain::new(20260927);
        let mut found = 0;
        'outer: for gz in 150..190 {
            for gx in 120..180 {
                let Some(s) = site(&t, gx, gz) else { continue };
                let c = s.chunk();
                let chunk = t.generate(c);
                let rocks = cluster(&t, &s);
                // Outcrops sink into steep slopes; this checks rocks bedded on gentle ground.
                let Some(b) = rocks
                    .first()
                    .filter(|b| b.radii.min_element() >= 0.8 && b.slope.length() < 0.35)
                else {
                    continue;
                };
                // The column under the rock's middle turns solid just above
                // the ground, but never drawn; the ground itself is untouched.
                let v = (Vec3::new(b.centre.x, b.ground, b.centre.z) / VOXEL_SIZE)
                    .floor()
                    .as_ivec3();
                if (-1..4).any(|dy| crate::chunk::chunk_of(v + IVec3::Y * dy) != c) {
                    continue;
                }
                let at = |dy: i32| {
                    let l = crate::chunk::local_of(v + IVec3::Y * dy);
                    chunk.get(l.x, l.y, l.z)
                };
                assert!((0..4).any(|dy| at(dy) == BUILT), "{:?} {:?}", b.centre, b.radii);
                assert!(!matches!(at(-1), AIR | BUILT));
                found += 1;
                if found > 5 {
                    break 'outer;
                }
            }
        }
        assert!(found > 0);
    }

    /// Writes a few clusters near spawn, with the ground around them, for a
    /// visual check: `STRATA_ROCKS_OUT=/tmp/rocks.txt cargo test dump_rocks -- --ignored`.
    #[test]
    #[ignore]
    fn dump_rocks() {
        use std::fmt::Write;
        let path = std::env::var("STRATA_ROCKS_OUT").unwrap_or("/tmp/rocks.txt".into());
        let t = Terrain::new(20260927);
        let mut s = String::new();
        let (mut n, mut gz) = (0, 150);
        while n < 6 && gz < 250 {
            for gx in 120..180 {
                let Some(site) = site(&t, gx, gz) else { continue };
                let c = cluster(&t, &site);
                if c.len() < 3 || n >= 6 {
                    continue;
                }
                n += 1;
                writeln!(
                    s,
                    "cluster {} {} {} {}",
                    site.x,
                    site.ground,
                    site.z,
                    water_level(site.x, site.z)
                )
                .unwrap();
                for k in 0..=24 {
                    for j in 0..=24 {
                        let (x, z) = (site.x - 6.0 + j as f32 * 0.5, site.z - 6.0 + k as f32 * 0.5);
                        writeln!(s, "g {x} {} {z}", t.height_at(x, z).0).unwrap();
                    }
                }
                let mut m = MeshData::default();
                for b in &c {
                    b.model(true, &mut m);
                }
                for v in &m.vertices {
                    let q = |a: u32| ((v.data >> a) & 511) as f32 / 511.0 * 2.0 - 1.0;
                    let (u, w) = (q(13), q(22));
                    let mut nv = Vec3::new(u, 1.0 - u.abs() - w.abs(), w);
                    if nv.y < 0.0 {
                        nv = Vec3::new((1.0 - w.abs()) * u.signum(), nv.y, (1.0 - u.abs()) * w.signum());
                    }
                    let nv = nv.normalize();
                    let [x, y, z] = v.pos;
                    writeln!(s, "v {x} {y} {z} {} {} {} {}", nv.x, nv.y, nv.z, (v.data >> 11) & 3).unwrap();
                }
                for f in m.indices.chunks(3) {
                    writeln!(s, "f {} {} {}", f[0], f[1], f[2]).unwrap();
                }
                println!("cluster of {} rocks, {} triangles", c.len(), m.indices.len() / 3);
            }
            gz += 1;
        }
        std::fs::write(path, s).unwrap();
    }
}
