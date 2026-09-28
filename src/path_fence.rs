//! A rustic rail fence along the cobbled path from the spawn rise down to the
//! bridge: chunky, slightly crooked posts of weathered timber with two split
//! rails that sag a little between them, following the slope of the ground.
//!
//! The fence is laid out from the path's centre line in `vista`, so it
//! follows the path wherever that is tuned to run. It keeps off the paving,
//! the bridge, the path's lanterns, the road and the cottages, and breaks here
//! and there as an old fence does.

use crate::block::*;
use crate::mesh::MeshData;
use crate::model::{self, soft_box};
use crate::noise::{hash2, unit};
use crate::terrain::{water_level, Terrain};
use crate::vista;
use glam::{Vec2, Vec3};

/// The fence stands this far from the path's centre line.
pub const OFFSET_M: f32 = 2.0;
/// No post stands closer than this to the path's centre line.
pub const CLEARANCE_M: f32 = vista::PATH_HALF_WIDTH_M + 0.5;
/// No post or rail comes closer than this to a path lantern's post.
pub const LANTERN_CLEAR_M: f32 = 1.2;
/// Distance between posts along the path.
const SPACING_M: f32 = 2.3;
/// The fence stops this far short of where the path ends on the rise, leaving
/// the arrival spot open.
const RISE_CLEAR_M: f32 = 5.0;
/// And starts this far from the bridge's side, off its deck and abutments.
const BRIDGE_CLEAR_M: f32 = 2.5;
/// How much of a post is sunk into the ground.
pub const POST_SINK_M: f32 = 0.45;

/// One fence post: where it meets the ground and how it stands.
#[derive(Clone, Debug)]
pub struct PathPost {
    /// Where the post's axis meets the ground (metres).
    pub foot: Vec3,
    /// Height of its top above the foot.
    pub height: f32,
    /// Its half thickness.
    pub half: f32,
    /// How far its top leans off plumb, sideways.
    lean: Vec2,
    /// Which way the path lies from the post (unit, across the fence).
    towards_path: Vec2,
}

impl PathPost {
    /// Top of the post's axis.
    pub fn top(&self) -> Vec3 {
        self.foot + Vec3::new(self.lean.x, self.height, self.lean.y)
    }

    /// Point on the post's axis at a height above its foot.
    fn at(&self, y: f32) -> Vec3 {
        let k = y / self.height;
        self.foot + Vec3::new(self.lean.x * k, y, self.lean.y * k)
    }
}

/// An unbroken run of path fence: posts with rails between each pair.
#[derive(Clone, Debug)]
pub struct PathFence {
    pub posts: Vec<PathPost>,
    seed: u32,
    lo: Vec3,
    hi: Vec3,
}

/// The path's centre at a Z and the unit normal across it (pointing to +X).
fn path_frame(z: f32) -> Option<(Vec2, Vec2)> {
    let x = vista::path_x(z)?;
    let (za, zb) = (z - 0.25, z + 0.25);
    let (xa, xb) = match (vista::path_x(za), vista::path_x(zb)) {
        (Some(a), Some(b)) => (a, b),
        (Some(a), None) => (a, x),
        (None, Some(b)) => (x, b),
        (None, None) => return None,
    };
    let dz = if vista::path_x(za).is_some() && vista::path_x(zb).is_some() {
        0.5
    } else {
        0.25
    };
    let tangent = Vec2::new(xb - xa, dz).normalize();
    Some((Vec2::new(x, z), Vec2::new(tangent.y, -tangent.x)))
}

/// The Z range the path covers, found from `vista::path_x` so it follows the path.
fn path_span() -> Option<(f32, f32)> {
    let z0 = (0..800)
        .map(|k| vista::BRIDGE_Z - 20.0 + k as f32 * 0.25)
        .find(|&z| vista::path_x(z).is_some())?;
    let mut z1 = z0;
    while vista::path_x(z1 + 0.25).is_some() {
        z1 += 0.25;
    }
    Some((z0, z1))
}

/// Whether a post at `p` would stand on or crowd the path or one of its lanterns.
pub fn clear_of_path(p: Vec2) -> bool {
    vista::path_distance(p.x, p.y) >= CLEARANCE_M
        && vista::path_lanterns().all(|(x, z, _)| Vec2::new(x, z).distance(p) >= LANTERN_CLEAR_M)
}

/// Distance from `p` to the segment `a`..`b`.
fn segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let d = b - a;
    let k = ((p - a).dot(d) / d.length_squared().max(1e-6)).clamp(0.0, 1.0);
    p.distance(a + d * k)
}

impl PathFence {
    /// Lays out the fence on both sides of the path, starting clear of the
    /// bridge at `bridge_z`. `blocked(p)` says where something else stands
    /// (buildings, doorways, the road) and the fence must break.
    pub fn plan(t: &Terrain, bridge_z: f32, seed: u32, blocked: impl Fn(Vec2) -> bool) -> Vec<PathFence> {
        let Some((z0, z1)) = path_span() else {
            return Vec::new();
        };
        let start = z0.max(bridge_z + crate::bridge::BRIDGE_HALF_W + BRIDGE_CLEAR_M);
        let end = z1 - RISE_CLEAR_M;
        let lanterns: Vec<Vec2> = vista::path_lanterns().map(|(x, z, _)| Vec2::new(x, z)).collect();
        let mut out = Vec::new();
        for (si, side) in [-1.0f32, 1.0].into_iter().enumerate() {
            let r = |k: i32, j: i32| unit(hash2(seed ^ (si as u32 * 0x9e37), k, j));
            let mut run: Vec<PathPost> = Vec::new();
            let flush = |run: &mut Vec<PathPost>, out: &mut Vec<PathFence>| {
                if run.len() >= 2 {
                    out.push(PathFence::new(t, std::mem::take(run), seed ^ out.len() as u32));
                }
                run.clear();
            };
            let mut z = start;
            let mut k = 0;
            while z <= end {
                let Some((c, n)) = path_frame(z) else { break };
                // Posts go in a little unevenly, as they would by hand.
                let p = c + n * side * (OFFSET_M + r(k, 0) * 0.25);
                let (h, _) = t.height_at(p.x, p.y);
                let steep = (t.height_at(p.x + 0.5, p.y).0 - h)
                    .abs()
                    .max((t.height_at(p.x, p.y + 0.5).0 - h).abs());
                // Now and then a post has rotted away and left a gap.
                let fallen = r(k, 1) < 0.08;
                let ok = !fallen && clear_of_path(p) && h > water_level(p.x, p.y) + 0.6 && steep < 0.6 && !blocked(p);
                let post = ok.then(|| PathPost {
                    foot: Vec3::new(p.x, h, p.y),
                    height: 0.95 + r(k, 2) * 0.4,
                    half: 0.085 + r(k, 3) * 0.04,
                    lean: Vec2::new(r(k, 4) - 0.5, r(k, 5) - 0.5) * 0.2,
                    towards_path: -n * side,
                });
                match post {
                    Some(post) => {
                        // A rail must not pass through a lantern or over the paving.
                        let joins = run.last().is_some_and(|last: &PathPost| {
                            let (a, b) = (Vec2::new(last.foot.x, last.foot.z), p);
                            let mid = (a + b) * 0.5;
                            lanterns.iter().all(|&l| segment_distance(l, a, b) >= LANTERN_CLEAR_M)
                                && vista::path_distance(mid.x, mid.y) >= CLEARANCE_M
                                && a.distance(b) < SPACING_M * 1.6
                                && !blocked(mid)
                        });
                        if !joins {
                            flush(&mut run, &mut out);
                        }
                        run.push(post);
                    }
                    None => flush(&mut run, &mut out),
                }
                // Step on by about one spacing along the path, not along Z.
                let step = SPACING_M * (0.9 + 0.2 * r(k, 6));
                z += step * n.x.abs().max(0.3);
                k += 1;
            }
            flush(&mut run, &mut out);
        }
        out
    }

    fn new(t: &Terrain, posts: Vec<PathPost>, seed: u32) -> Self {
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in &posts {
            lo = lo.min(p.foot - Vec3::new(0.6, 1.0, 0.6));
            hi = hi.max(p.foot + Vec3::new(0.6, p.height + 0.5, 0.6));
        }
        // Rails follow the ground between posts; make room for any dip under them.
        for w in posts.windows(2) {
            let m = (w[0].foot + w[1].foot) * 0.5;
            let h = t.height_at(m.x, m.z).0;
            lo.y = lo.y.min(h - 1.0);
        }
        PathFence { posts, seed, lo, hi }
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.lo, self.hi)
    }

    /// Invisible collision along the fence line; the model draws it.
    pub fn block(&self, p: Vec3, ground: f32) -> Option<Block> {
        if !(0.0..1.2).contains(&(p.y - ground)) {
            return None;
        }
        let q = Vec2::new(p.x, p.z);
        let near = self.posts.windows(2).any(|w| {
            let (a, b) = (Vec2::new(w[0].foot.x, w[0].foot.z), Vec2::new(w[1].foot.x, w[1].foot.z));
            segment_distance(q, a, b) < 0.3
        });
        near.then_some(BUILT)
    }

    /// Crooked, roughly round posts and split rails sagging between them.
    pub fn model(&self, t: &Terrain, out: &mut MeshData) {
        let r = |k: usize, j: i32| unit(hash2(self.seed ^ 0x51f1, k as i32, j)) - 0.5;
        for (k, p) in self.posts.iter().enumerate() {
            post(out, p, r(k, 0) * 0.8);
        }
        for (k, w) in self.posts.windows(2).enumerate() {
            let (a, b) = (&w[0], &w[1]);
            for (j, y) in [0.42f32, 0.82].into_iter().enumerate() {
                let rr = |i: i32| r(k, 10 + i + j as i32 * 7);
                // Here and there a rail has come down.
                if rr(9) < -0.44 {
                    continue;
                }
                // Nailed to the path side of the posts, a little uneven in height.
                let face = |p: &PathPost, y: f32| {
                    let s = p.towards_path * (p.half + 0.035);
                    p.at(y) + Vec3::new(s.x, 0.0, s.y)
                };
                let ya = (y + rr(0) * 0.08).min(a.height - 0.12);
                let yb = (y + rr(1) * 0.08).min(b.height - 0.12);
                let (pa, pb) = (face(a, ya), face(b, yb));
                let along = (pb - pa).normalize_or_zero();
                // Rails run a little past the posts, and split rails are thicker one way.
                let (pa, pb) = (
                    pa - along * (0.08 + rr(2).abs() * 0.1),
                    pb + along * (0.08 + rr(3).abs() * 0.1),
                );
                let sag = 0.04 + (rr(4) + 0.5) * 0.08;
                let mid = (pa + pb) * 0.5 - Vec3::Y * sag;
                // Keep a sagging rail out of a hump in the ground.
                let ground = t.height_at(mid.x, mid.z).0;
                let mid = Vec3::new(mid.x, mid.y.max(ground + 0.18), mid.z);
                let half = Vec2::new(0.07 + rr(5) * 0.02, 0.048 + rr(6) * 0.012);
                let roll = rr(7) * 0.7;
                rail(out, pa, mid, half, roll);
                rail(out, mid, pb, half, roll + rr(8) * 0.2);
            }
        }
    }
}

/// A roughly round post: two square beams turned an eighth apart make an
/// eight-sided section, sunk into the ground below its foot.
fn post(out: &mut MeshData, p: &PathPost, twist: f32) {
    let a = p.foot - Vec3::Y * POST_SINK_M;
    let b = p.top();
    let x = (b - a).normalize();
    let side = x.cross(Vec3::X).normalize();
    let up = side.cross(x).normalize();
    let len = (b - a).length();
    let c = (a + b) * 0.5;
    for (k, turn) in [twist, twist + std::f32::consts::FRAC_PI_4].into_iter().enumerate() {
        let (s, co) = turn.sin_cos();
        let u = side * co + up * s;
        // One of the pair is a touch thinner, so the section is not a perfect octagon.
        let h = p.half * if k == 0 { 1.0 } else { 0.92 };
        // Axes (u, x, v) are right-handed: u x x = v.
        soft_box(
            out,
            c,
            [u, x, u.cross(x).normalize()],
            Vec3::new(h, len * 0.5, h),
            FENCE_WOOD,
        );
    }
    // A rounded, weathered top: a smaller cap a little below the cut.
    model::soft_box(
        out,
        b - x * 0.03,
        [side, x, side.cross(x).normalize()],
        Vec3::new(p.half * 0.8, 0.04, p.half * 0.8),
        FENCE_WOOD,
    );
}

/// A split rail from `a` to `b` with a flattened section rolled by `roll`.
fn rail(out: &mut MeshData, a: Vec3, b: Vec3, half: Vec2, roll: f32) {
    let d = b - a;
    let len = d.length();
    if len < 1e-3 {
        return;
    }
    let x = d / len;
    let side = x.cross(Vec3::Y).normalize_or_zero();
    let up = side.cross(x);
    let (s, c) = roll.sin_cos();
    let y = up * c + side * s;
    let z = x.cross(y);
    soft_box(
        out,
        (a + b) * 0.5,
        [x, y, z],
        Vec3::new(len * 0.5 + 0.03, half.x, half.y),
        FENCE_WOOD,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::Structure;

    fn world() -> Terrain {
        Terrain::new(20260927)
    }

    fn fences(t: &Terrain) -> Vec<&PathFence> {
        t.structures
            .iter()
            .filter_map(|s| match s {
                Structure::PathFence(f) => Some(f),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_path_has_a_fence_on_both_sides() {
        let t = world();
        let f = fences(&t);
        let posts: Vec<&PathPost> = f.iter().flat_map(|f| f.posts.iter()).collect();
        assert!(posts.len() >= 16, "{} posts", posts.len());
        // Both sides of the path have some.
        let side = |p: &PathPost| p.foot.x - vista::path_x(p.foot.z).unwrap();
        assert!(posts.iter().any(|p| side(p) < 0.0) && posts.iter().any(|p| side(p) > 0.0));
        // It breaks here and there rather than running unbroken.
        assert!(f.len() >= 3, "{} runs", f.len());
    }

    #[test]
    fn posts_reach_the_ground() {
        let t = world();
        for f in fences(&t) {
            for p in &f.posts {
                let (x, z) = (p.foot.x, p.foot.z);
                let h = t.height_at(x, z).0;
                assert!((p.foot.y - h).abs() < 1e-3, "post at {x},{z} floats");
                // Its buried end lies under the ground all round its footprint.
                for (dx, dz) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    let g = t.height_at(x + dx * p.half, z + dz * p.half).0;
                    assert!(p.foot.y - POST_SINK_M < g - 0.2, "post at {x},{z} shows its end");
                }
                assert!(p.top().y > h + 0.9);
            }
        }
    }

    #[test]
    fn posts_keep_off_the_paving_the_bridge_and_the_lanterns() {
        let t = world();
        let bridge = t
            .structures
            .iter()
            .find_map(|s| match s {
                Structure::Bridge(b) => Some(b.bounds()),
                _ => None,
            })
            .unwrap();
        for f in fences(&t) {
            for p in &f.posts {
                let (x, z) = (p.foot.x, p.foot.z);
                assert!(vista::path_distance(x, z) >= CLEARANCE_M, "post at {x},{z} on the path");
                let on_bridge = x > bridge.0.x && x < bridge.1.x && z > bridge.0.z && z < bridge.1.z;
                assert!(!on_bridge, "post at {x},{z} on the bridge");
                for (lx, lz, _) in vista::path_lanterns() {
                    assert!(
                        Vec2::new(lx, lz).distance(Vec2::new(x, z)) >= LANTERN_CLEAR_M,
                        "post at {x},{z} on a lantern"
                    );
                }
            }
            // No rail passes through a lantern either.
            for w in f.posts.windows(2) {
                let (a, b) = (Vec2::new(w[0].foot.x, w[0].foot.z), Vec2::new(w[1].foot.x, w[1].foot.z));
                for (lx, lz, _) in vista::path_lanterns() {
                    assert!(segment_distance(Vec2::new(lx, lz), a, b) >= LANTERN_CLEAR_M);
                }
            }
        }
    }

    #[test]
    fn the_fence_collides_but_leaves_the_path_open() {
        let t = world();
        let f = fences(&t)[0];
        let (a, b) = (&f.posts[0], &f.posts[1]);
        let mid = (a.foot + b.foot) * 0.5;
        let g = t.height_at(mid.x, mid.z).0;
        assert_eq!(t.structures.block_at(mid + Vec3::Y * 0.5, g), Some(BUILT));
        // The path's centre line is free.
        let z = mid.z;
        let x = vista::path_x(z).unwrap();
        assert_ne!(t.structures.block_at(Vec3::new(x, g + 0.5, z), g), Some(BUILT));
    }

    #[test]
    fn dbg_dump() {
        let t = world();
        for f in fences(&t) {
            let v: Vec<String> = f
                .posts
                .iter()
                .map(|p| format!("({:.1},{:.1},{:.1})", p.foot.x, p.foot.y, p.foot.z))
                .collect();
            println!("RUN {}", v.join(" "));
        }
        for l in vista::path_lanterns() {
            println!("LANTERN {:?}", l);
        }
        let mut z = 950.0;
        while z < 1024.0 {
            println!(
                "PATH z={z} x={:?} road={:.1} river={:.1}",
                vista::path_x(z),
                t.road_x(z),
                t.river_x(z)
            );
            z += 6.0;
        }
        for s in t.structures.iter() {
            if let Structure::Cottage(c) = s {
                println!("COTTAGE {:?}", c.bounds());
            }
            if let Structure::Bridge(c) = s {
                println!("BRIDGE {:?}", c.bounds());
            }
        }
    }
}
