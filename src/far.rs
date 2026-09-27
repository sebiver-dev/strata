//! Distant terrain. Beyond the voxel view radius the valley is drawn straight
//! from the terrain's height function as a smooth height field, with rounded
//! crowns and trunks for its forests. The world is cut into 64 m tiles, and a
//! tile's grid spacing grows from 2 m to 8 m with distance. Near the player the
//! full voxel meshes take over, and the shader hides far tiles under any chunk
//! column that has its voxel mesh.

use crate::block::*;
use crate::budget::{Cost, Deadline};
use crate::mesh::{smooth_data, MeshData, Vertex, FACES};
use crate::terrain::{water_level, Terrain, TREE_CELL_M, WORLD_SIZE_M};
use glam::{IVec2, Vec2, Vec3};
use std::collections::HashMap;

/// Edge length of one far tile in metres.
pub const TILE_M: f32 = 64.0;
/// Tiles along each side of the world.
pub const TILES: i32 = (WORLD_SIZE_M / TILE_M) as i32;
/// Cell size in metres for each level of detail, finest first.
pub const LEVEL_CELL_M: [f32; 3] = [2.0, 4.0, 8.0];
/// Tiles nearer than these distances use the finer levels.
const LEVEL_DIST_M: [f32; 2] = [320.0, 768.0];
/// How far past a threshold the camera must move before a tile switches level,
/// so walking along a boundary does not rebuild the same tiles over and over.
const HYSTERESIS_M: f32 = 24.0;
/// Walls hang this far below the tile edge so neighbouring tiles at a different
/// level never leave a crack.
const SKIRT_M: f32 = 3.0;
/// Forests are drawn on tiles up to this level (4 m cells).
const TREE_MAX_LEVEL: u8 = 1;

/// Level of detail for a tile at a given horizontal distance.
pub fn level_for(dist_m: f32) -> u8 {
    LEVEL_DIST_M.iter().filter(|&&d| dist_m >= d).count() as u8
}

fn wanted_level(current: Option<u8>, dist_m: f32) -> u8 {
    let l = level_for(dist_m);
    match current {
        Some(c) if l > c && level_for(dist_m - HYSTERESIS_M) <= c => c,
        Some(c) if l < c && level_for(dist_m + HYSTERESIS_M) >= c => c,
        _ => l,
    }
}

/// Horizontal distance from a point to the nearest point of a tile.
pub fn tile_distance(tile: IVec2, p: Vec3) -> f32 {
    let lo = tile.as_vec2() * TILE_M;
    let q = Vec2::new(p.x, p.z);
    (q.clamp(lo, lo + TILE_M) - q).length()
}

/// Which far tiles exist and at what level.
#[derive(Default)]
pub struct FarField {
    levels: HashMap<IVec2, u8>,
    /// How long building and uploading one tile takes, per level.
    costs: [Cost; LEVEL_CELL_M.len()],
}

impl FarField {
    pub fn tile_count(&self) -> usize {
        self.levels.len()
    }

    /// Builds tiles that are missing or whose level no longer suits the camera,
    /// nearest first, until the time budget runs out. Each finished mesh goes to
    /// `upload`. Returns the number of tiles built.
    pub fn update(
        &mut self,
        terrain: &Terrain,
        camera: Vec3,
        deadline: &Deadline,
        share: f64,
        mut upload: impl FnMut(IVec2, &MeshData),
    ) -> usize {
        let mut todo = Vec::new();
        for z in 0..TILES {
            for x in 0..TILES {
                let t = IVec2::new(x, z);
                let d = tile_distance(t, camera);
                let cur = self.levels.get(&t).copied();
                let want = wanted_level(cur, d);
                if cur != Some(want) {
                    todo.push((d, t, want));
                }
            }
        }
        todo.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut built = 0;
        for (_, t, level) in todo {
            let cost = &mut self.costs[level as usize];
            if !deadline.fits(cost, share, built == 0) {
                break;
            }
            let start = web_time::Instant::now();
            let mesh = build_tile(terrain, t, level);
            upload(t, &mesh);
            cost.record(start.elapsed().as_secs_f64() * 1000.0);
            self.levels.insert(t, level);
            built += 1;
        }
        built
    }
}

/// Emits one face of an axis-aligned box (used for flat water).
pub(crate) fn face(verts: &mut Vec<Vertex>, idx: &mut Vec<u32>, lo: Vec3, hi: Vec3, fi: usize, mat: Block) {
    let f = &FACES[fi];
    let size = hi - lo;
    let (n, u, v) = (f.n.as_vec3(), f.u.as_vec3(), f.v.as_vec3());
    let base = lo + n.max(Vec3::ZERO) * size;
    let start = verts.len() as u32;
    for (cu, cv) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
        let pos = base + u * size * cu + v * size * cv;
        verts.push(Vertex {
            pos: pos.to_array(),
            data: fi as u32 | ((mat as u32) << 3) | (3 << 11),
        });
    }
    idx.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
}

/// A closed surface of revolution around a vertical axis through `centre`
/// (x and z), following `profile` as (height, radius) pairs from bottom to top.
/// `wobble` gives each ring a slightly irregular outline so crowns do not look
/// turned on a lathe.
pub(crate) fn revolve(
    out: &mut MeshData,
    centre: Vec2,
    profile: &[(f32, f32)],
    segments: u32,
    mat: Block,
    wobble: u32,
) {
    let rings = profile.len();
    let start = out.vertices.len() as u32;
    for (r, &(y, radius)) in profile.iter().enumerate() {
        // Slope of the outline, for normals that lean up or down with it.
        let (ya, ra) = profile[r.saturating_sub(1)];
        let (yb, rb) = profile[(r + 1).min(rings - 1)];
        let slope = if (yb - ya).abs() > 1e-4 {
            (rb - ra) / (yb - ya)
        } else {
            0.0
        };
        for k in 0..segments {
            let a = k as f32 / segments as f32 * std::f32::consts::TAU;
            let bump = if wobble != 0 {
                let h = crate::noise::hash3(wobble, k as i32, r as i32, 0);
                0.82 + 0.3 * crate::noise::unit(h)
            } else {
                1.0
            };
            let (c, s) = (a.cos(), a.sin());
            let pos = Vec3::new(centre.x + c * radius * bump, y, centre.y + s * radius * bump);
            // Top and bottom caps close to a point; their normals point along the axis.
            let n = if radius < 1e-3 {
                Vec3::new(0.0, if r == 0 { -1.0 } else { 1.0 }, 0.0)
            } else {
                Vec3::new(c, -slope, s).normalize()
            };
            out.vertices.push(Vertex {
                pos: pos.to_array(),
                data: smooth_data(mat, 3, n),
            });
        }
    }
    for r in 0..rings as u32 - 1 {
        for k in 0..segments {
            let k1 = (k + 1) % segments;
            let a = start + r * segments + k;
            let b = start + r * segments + k1;
            let c = start + (r + 1) * segments + k1;
            let d = start + (r + 1) * segments + k;
            out.indices.extend_from_slice(&[a, c, b, a, d, c]);
        }
    }
}

/// A rounded crown for a tree, from its bounding box: a lumpy ellipsoid for
/// broadleaf trees and a tiered spire for tall, narrow conifers.
fn crown(out: &mut MeshData, lo: Vec3, hi: Vec3, seed: u32, level: u8) {
    let c = Vec2::new((lo.x + hi.x) * 0.5, (lo.z + hi.z) * 0.5);
    let r = (hi.x - lo.x) * 0.5;
    let (y0, y1) = (lo.y, hi.y);
    let segs = if level == 0 { 9 } else { 6 };
    let mut profile = Vec::new();
    if y1 - y0 > 1.5 * (hi.x - lo.x) {
        // Three drooping tiers narrowing to a point.
        let h = y1 - y0;
        profile.push((y0, 0.0));
        for (t, w) in [
            (0.02, 0.95),
            (0.22, 0.62),
            (0.3, 0.8),
            (0.5, 0.45),
            (0.58, 0.6),
            (0.8, 0.25),
        ] {
            profile.push((y0 + h * t, r * w));
        }
        profile.push((y1, 0.0));
    } else {
        // A cloud of rounded clumps: one in the middle, the rest around it.
        let rings = if level == 0 { 5 } else { 4 };
        let around = if level == 0 { 5 } else { 3 };
        let h = y1 - y0;
        let ball = |out: &mut MeshData, at: Vec3, rad: f32, k: u32| {
            let mut profile = Vec::new();
            for i in 0..=rings {
                let a = i as f32 / rings as f32 * std::f32::consts::PI;
                profile.push((at.y - rad * 0.85 * a.cos(), rad * a.sin()));
            }
            revolve(
                out,
                Vec2::new(at.x, at.z),
                &profile,
                segs.min(7),
                LEAVES,
                (seed | 1).wrapping_add(k),
            );
        };
        ball(out, Vec3::new(c.x, y0 + h * 0.5, c.y), r * 0.64, 0);
        for k in 0..around {
            let u = crate::noise::unit(crate::noise::hash3(seed, k as i32, 3, 9));
            let a = (k as f32 + 0.4 * u) * std::f32::consts::TAU / around as f32;
            let at = Vec3::new(
                c.x + a.cos() * r * 0.5,
                y0 + h * (0.3 + 0.25 * u),
                c.y + a.sin() * r * 0.5,
            );
            ball(out, at, r * (0.42 + 0.1 * u), k + 1);
        }
        return;
    }
    revolve(out, c, &profile, segs, LEAVES, seed | 1);
}

/// Meshes one far tile at a level of detail.
pub fn build_tile(terrain: &Terrain, tile: IVec2, level: u8) -> MeshData {
    let cell = LEVEL_CELL_M[level as usize];
    let n = (TILE_M / cell) as i32;
    let origin = tile.as_vec2() * TILE_M;

    // Heights on the grid corners plus a one-cell border for normals.
    let w = n + 3;
    let mut heights = Vec::with_capacity((w * w) as usize);
    for j in -1..=n + 1 {
        for i in -1..=n + 1 {
            let x = origin.x + i as f32 * cell;
            let z = origin.y + j as f32 * cell;
            heights.push(terrain.height_at(x, z).0);
        }
    }
    let h = |i: i32, j: i32| heights[((j + 1) * w + i + 1) as usize];

    let mut out = MeshData::default();
    let corner = |i: i32, j: i32| Vec3::new(origin.x + i as f32 * cell, h(i, j), origin.y + j as f32 * cell);
    let normal = |i: i32, j: i32| {
        let dx = (h(i + 1, j) - h(i - 1, j)) / (2.0 * cell);
        let dz = (h(i, j + 1) - h(i, j - 1)) / (2.0 * cell);
        Vec3::new(-dx, 1.0, -dz).normalize()
    };
    for j in 0..=n {
        for i in 0..=n {
            let p = corner(i, j);
            let mat = terrain.column_info(p.x, p.z).surface;
            out.vertices.push(Vertex {
                pos: p.to_array(),
                data: smooth_data(mat, 3, normal(i, j)),
            });
        }
    }
    let vi = |i: i32, j: i32| (j * (n + 1) + i) as u32;
    for j in 0..n {
        for i in 0..n {
            let (a, b, c, d) = (vi(i, j), vi(i + 1, j), vi(i + 1, j + 1), vi(i, j + 1));
            // Counter-clockwise seen from above.
            out.indices.extend_from_slice(&[a, c, b, a, d, c]);
            let top = h(i, j).min(h(i + 1, j)).min(h(i, j + 1)).min(h(i + 1, j + 1));
            let (cx, cz) = (origin.x + (i as f32 + 0.5) * cell, origin.y + (j as f32 + 0.5) * cell);
            let level = water_level(cx, cz);
            if top < level {
                let lo = Vec3::new(origin.x + i as f32 * cell, top - 1.0, origin.y + j as f32 * cell);
                let hi = Vec3::new(lo.x + cell, level - 0.06, lo.z + cell);
                face(&mut out.water_vertices, &mut out.water_indices, lo, hi, 2, WATER);
                // Where the next cell downstream sits a fall lower, the water pours over.
                let below = water_level(cx, cz + cell);
                if below < level && h(i, j + 1).min(h(i + 1, j + 1)) < below {
                    let lo = Vec3::new(lo.x, below - 0.06, hi.z);
                    face(&mut out.water_vertices, &mut out.water_indices, lo, hi, 4, WATER);
                }
            }
        }
    }
    // Skirts hang from the tile's edges so neighbours at another level never
    // leave a crack. They are drawn from both sides.
    let edge: Vec<(i32, i32)> = (0..=n)
        .map(|i| (i, 0))
        .chain((0..=n).map(|j| (n, j)))
        .chain((0..=n).rev().map(|i| (i, n)))
        .chain((0..=n).rev().map(|j| (0, j)))
        .collect();
    for pair in edge.windows(2) {
        let [(i0, j0), (i1, j1)] = [pair[0], pair[1]];
        if (i0, j0) == (i1, j1) {
            continue;
        }
        let base = out.vertices.len() as u32;
        for (i, j) in [(i0, j0), (i1, j1)] {
            let mut p = corner(i, j);
            p.y -= SKIRT_M + cell;
            let data = out.vertices[vi(i, j) as usize].data;
            out.vertices.push(Vertex {
                pos: p.to_array(),
                data,
            });
        }
        let (a, b) = (vi(i0, j0), vi(i1, j1));
        let (c, d) = (base + 1, base);
        out.indices.extend_from_slice(&[a, b, c, a, c, d, a, c, b, a, d, c]);
    }

    if level <= TREE_MAX_LEVEL {
        let first = (origin / TREE_CELL_M).ceil().as_ivec2();
        let last = ((origin + TILE_M) / TREE_CELL_M).ceil().as_ivec2();
        for gz in first.y..last.y {
            for gx in first.x..last.x {
                let Some([canopy, trunk]) = terrain.tree_boxes(gx, gz) else {
                    continue;
                };
                let seed = crate::noise::hash3(terrain.seed, gx, gz, 77);
                crown(&mut out, canopy.0, canopy.1, seed, level);
                if level == 0 {
                    let c = Vec2::new((trunk.0.x + trunk.1.x) * 0.5, (trunk.0.z + trunk.1.z) * 0.5);
                    let top = (trunk.1.y + 0.5).max(canopy.0.y + (canopy.1.y - canopy.0.y) * 0.5);
                    let profile = [
                        (trunk.0.y - 0.5, 0.0),
                        (trunk.0.y - 0.5, 0.55),
                        (trunk.0.y + 0.6, 0.42),
                        // Up into the crown, so no gap shows between its clumps.
                        (top, 0.3),
                        (top, 0.0),
                    ];
                    revolve(&mut out, c, &profile, 6, WOOD, 0);
                }
            }
        }
    }
    terrain.structures.far(&mut out, origin, origin + TILE_M);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_grow_with_distance_and_resist_flicker() {
        assert_eq!(level_for(0.0), 0);
        assert_eq!(level_for(400.0), 1);
        assert_eq!(level_for(2000.0), 2);
        // Just past a threshold, a tile keeps its level until it is clearly past.
        assert_eq!(wanted_level(Some(0), 330.0), 0);
        assert_eq!(wanted_level(Some(0), 360.0), 1);
        assert_eq!(wanted_level(Some(1), 310.0), 1);
        assert_eq!(wanted_level(Some(1), 280.0), 0);
        assert_eq!(wanted_level(None, 330.0), 1);
    }

    #[test]
    fn tile_distance_is_zero_inside() {
        let t = IVec2::new(2, 3);
        assert_eq!(tile_distance(t, Vec3::new(150.0, 40.0, 200.0)), 0.0);
        assert_eq!(tile_distance(t, Vec3::new(100.0, 40.0, 200.0)), 28.0);
    }

    #[test]
    fn ground_covers_the_tile_and_faces_up() {
        let terrain = Terrain::new(7);
        for level in 0..3u8 {
            let m = build_tile(&terrain, IVec2::new(12, 15), level);
            let mut area = 0.0;
            for tri in m.indices.chunks(3) {
                let v = [0, 1, 2].map(|k| m.vertices[tri[k] as usize]);
                let [a, b, c] = v.map(|v| Vec3::from(v.pos));
                let normal = (b - a).cross(c - a);
                let mat = (v[0].data >> 3) & 255;
                // Ground triangles (not trees or skirts) wind counter-clockwise from above.
                if mat != LEAVES as u32 && mat != WOOD as u32 && normal.y.abs() > 1e-4 {
                    assert!(normal.y > 0.0, "level {level}: a ground triangle faces down");
                    area += normal.y * 0.5;
                }
            }
            assert!((area - TILE_M * TILE_M).abs() < 1.0, "level {level} covers {area} m²");
        }
    }

    #[test]
    fn crowns_are_closed_and_face_outward() {
        let mut out = MeshData::default();
        // A broadleaf crown of clumps and a conifer spire.
        crown(&mut out, Vec3::new(0.0, 10.0, 0.0), Vec3::new(8.0, 16.0, 8.0), 5, 0);
        crown(&mut out, Vec3::new(20.0, 10.0, 0.0), Vec3::new(24.0, 22.0, 4.0), 5, 0);
        // Every triangle winds the same way as the outward normals its vertices carry.
        let normal_of = |d: u32| {
            let q = |s: u32| ((d >> s) & 511) as f32 / 511.0 * 2.0 - 1.0;
            let (u, v) = (q(13), q(22));
            let mut m = Vec3::new(u, 1.0 - u.abs() - v.abs(), v);
            if m.y < 0.0 {
                let sign = |a: f32| if a >= 0.0 { 1.0 } else { -1.0 };
                (m.x, m.z) = ((1.0 - v.abs()) * sign(u), (1.0 - u.abs()) * sign(v));
            }
            m.normalize()
        };
        for tri in out.indices.chunks(3) {
            let verts = [0, 1, 2].map(|k| out.vertices[tri[k] as usize]);
            let [a, b, c] = verts.map(|v| Vec3::from(v.pos));
            let normal = (b - a).cross(c - a);
            if normal.length() > 1e-5 {
                let n = verts.iter().map(|v| normal_of(v.data)).sum::<Vec3>();
                assert!(normal.dot(n) > 0.0);
            }
        }
    }

    #[test]
    fn river_tiles_have_water() {
        let terrain = Terrain::new(42);
        let z = 1000.0;
        let x = terrain.river_x(z);
        let tile = IVec2::new((x / TILE_M) as i32, (z / TILE_M) as i32);
        let m = build_tile(&terrain, tile, 1);
        assert!(!m.water_indices.is_empty());
    }

    #[test]
    fn far_tiles_are_much_lighter_than_voxels() {
        let terrain = Terrain::new(20260927);
        let tile = IVec2::new(TILES / 2, TILES / 2);
        let tris: Vec<usize> = (0..3)
            .map(|l| build_tile(&terrain, tile, l).indices.len() / 3)
            .collect();
        assert!(tris[0] > tris[1] && tris[1] > tris[2], "{tris:?}");
        assert!(tris[0] < 40_000, "{tris:?}");
    }
}
