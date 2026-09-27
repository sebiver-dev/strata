//! Distant terrain. Beyond the voxel view radius the valley is drawn straight
//! from the terrain's height function as blocky columns on a coarser grid. The
//! world is cut into 64 m tiles, and a tile's cells grow from 2 m to 8 m with
//! distance. Near the player the full voxel meshes take over, and the shader
//! hides far tiles under any chunk column that has its voxel mesh.

use crate::block::*;
use crate::budget::{Cost, Deadline};
use crate::mesh::{MeshData, Vertex, FACES};
use crate::terrain::{Terrain, TREE_CELL_M, WATER_LEVEL_M, WORLD_SIZE_M};
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

/// Emits one face of an axis-aligned box. Corners at or below `shade_below`
/// are darkened a little, which grounds tall walls.
fn face(
    verts: &mut Vec<Vertex>,
    idx: &mut Vec<u32>,
    lo: Vec3,
    hi: Vec3,
    fi: usize,
    mat: Block,
    shade_below: Option<f32>,
) {
    let f = &FACES[fi];
    let size = hi - lo;
    let (n, u, v) = (f.n.as_vec3(), f.u.as_vec3(), f.v.as_vec3());
    let base = lo + n.max(Vec3::ZERO) * size;
    let start = verts.len() as u32;
    for (cu, cv) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
        let pos = base + u * size * cu + v * size * cv;
        let ao = match shade_below {
            Some(b) if pos.y <= b + 1e-3 => 2,
            _ => 3,
        };
        verts.push(Vertex {
            pos: pos.to_array(),
            data: fi as u32 | ((mat as u32) << 3) | (ao << 11),
        });
    }
    idx.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
}

fn solid_box(out: &mut MeshData, lo: Vec3, hi: Vec3, mat: Block, faces: &[usize]) {
    for &fi in faces {
        face(&mut out.vertices, &mut out.indices, lo, hi, fi, mat, None);
    }
}

/// Three stacked boxes that follow a tree crown's shape more closely than its
/// bounding box: a tapering stack for tall (conifer) crowns, and a wide middle
/// with narrower top and bottom for round ones. Used for the nearest far tiles,
/// where the swap to the real voxel tree is easiest to notice.
fn rounded_canopy(lo: Vec3, hi: Vec3) -> [(Vec3, Vec3); 3] {
    let c = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    let slab = |y0: f32, y1: f32, w: f32| {
        let h = Vec3::new(half.x * w, 0.0, half.z * w);
        (Vec3::new(c.x - h.x, y0, c.z - h.z), Vec3::new(c.x + h.x, y1, c.z + h.z))
    };
    let y = |t: f32| lo.y + (hi.y - lo.y) * t;
    if hi.y - lo.y > 1.5 * (hi.x - lo.x) {
        [
            slab(y(0.0), y(0.4), 1.0),
            slab(y(0.4), y(0.72), 0.68),
            slab(y(0.72), y(1.0), 0.36),
        ]
    } else {
        [
            slab(y(0.0), y(0.25), 0.7),
            slab(y(0.25), y(0.8), 1.0),
            slab(y(0.8), y(1.0), 0.62),
        ]
    }
}

/// Meshes one far tile at a level of detail.
pub fn build_tile(terrain: &Terrain, tile: IVec2, level: u8) -> MeshData {
    let cell = LEVEL_CELL_M[level as usize];
    let n = (TILE_M / cell) as i32;
    let origin = tile.as_vec2() * TILE_M;

    // Column tops for the tile plus a one-cell border, snapped to the voxel grid.
    let w = n + 2;
    let mut cols = Vec::with_capacity((w * w) as usize);
    for j in -1..=n {
        for i in -1..=n {
            let x = origin.x + (i as f32 + 0.5) * cell;
            let z = origin.y + (j as f32 + 0.5) * cell;
            let c = terrain.column_info(x, z);
            let top = (c.height_m / VOXEL_SIZE).round() * VOXEL_SIZE;
            cols.push((top, c));
        }
    }
    let at = |i: i32, j: i32| cols[((j + 1) * w + i + 1) as usize];

    let mut out = MeshData::default();
    // +X, -X, +Z, -Z walls and the neighbour each one faces.
    const SIDES: [(usize, i32, i32); 4] = [(0, 1, 0), (1, -1, 0), (4, 0, 1), (5, 0, -1)];
    for j in 0..n {
        for i in 0..n {
            let (top, c) = at(i, j);
            let lo = Vec3::new(origin.x + i as f32 * cell, 0.0, origin.y + j as f32 * cell);
            let hi = Vec3::new(lo.x + cell, top, lo.z + cell);
            face(&mut out.vertices, &mut out.indices, lo, hi, 2, c.surface, None);
            if top < WATER_LEVEL_M {
                let (wlo, whi) = (Vec3::new(lo.x, top, lo.z), Vec3::new(hi.x, WATER_LEVEL_M - 0.06, hi.z));
                face(
                    &mut out.water_vertices,
                    &mut out.water_indices,
                    wlo,
                    whi,
                    2,
                    WATER,
                    None,
                );
            }
            for (fi, di, dj) in SIDES {
                let (ni, nj) = (i + di, j + dj);
                let mut bottom = at(ni, nj).0;
                if ni < 0 || nj < 0 || ni >= n || nj >= n {
                    bottom = bottom.min(top - SKIRT_M - cell);
                }
                if bottom < top {
                    let mat = if top - bottom <= cell { c.surface } else { c.subsurface };
                    let wall_lo = Vec3::new(lo.x, bottom, lo.z);
                    face(&mut out.vertices, &mut out.indices, wall_lo, hi, fi, mat, Some(bottom));
                }
            }
        }
    }

    if level <= TREE_MAX_LEVEL {
        let first = (origin / TREE_CELL_M).ceil().as_ivec2();
        let last = ((origin + TILE_M) / TREE_CELL_M).ceil().as_ivec2();
        for gz in first.y..last.y {
            for gx in first.x..last.x {
                let Some([canopy, trunk]) = terrain.tree_boxes(gx, gz) else {
                    continue;
                };
                if level == 0 {
                    for (lo, hi) in rounded_canopy(canopy.0, canopy.1) {
                        solid_box(&mut out, lo, hi, LEAVES, &[0, 1, 2, 3, 4, 5]);
                    }
                } else {
                    solid_box(&mut out, canopy.0, canopy.1, LEAVES, &[0, 1, 2, 3, 4, 5]);
                }
                if level == 0 {
                    solid_box(&mut out, trunk.0, trunk.1, WOOD, &[0, 1, 4, 5]);
                }
            }
        }
    }
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
    fn tops_cover_the_tile_and_faces_point_outward() {
        let terrain = Terrain::new(7);
        for level in 0..3u8 {
            let m = build_tile(&terrain, IVec2::new(12, 15), level);
            let mut top_area = 0.0;
            for tri in m.indices.chunks(3) {
                let v = [0, 1, 2].map(|k| m.vertices[tri[k] as usize]);
                let [a, b, c] = v.map(|v| Vec3::from(v.pos));
                let normal = (b - a).cross(c - a);
                let fi = (v[0].data & 7) as usize;
                assert!(
                    normal.normalize().dot(FACES[fi].n.as_vec3()) > 0.99,
                    "face {fi} winds inward"
                );
                if fi == 2 && (v[0].data >> 3) & 255 != LEAVES as u32 {
                    top_area += normal.length() * 0.5;
                }
            }
            assert!(
                (top_area - TILE_M * TILE_M).abs() < 1.0,
                "level {level} covers {top_area} m²"
            );
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
