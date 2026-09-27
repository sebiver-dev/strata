//! The player's body, built every frame from smooth rounded forms (tapered
//! ellipsoids and softened boxes): a procedurally posed figure with a walk cycle, a head that follows the view, and whatever tool
//! is selected held in the right hand. First person shows only the right arm
//! and the tool at the edge of the view; the whole body still casts a shadow.

use crate::block::{Block, VOXEL_SIZE};
use crate::game::{Brush, Shape, BLOCK_VOXELS};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use std::ops::Range;

/// Vertex of an actor mesh (the player, later other characters).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ActorVertex {
    /// World position in metres.
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    /// Position inside a textured part, in metres of a single voxel (0..0.5),
    /// so held blocks show the same surface as the terrain.
    pub tex: [f32; 3],
    /// sRGB colour in the low three bytes, ambient occlusion (255 = open) in the top byte.
    pub color: u32,
    /// Bit 0: drawn in front of the world (first-person hand); bits 1..4: the
    /// part's local face (terrain face order); bits 8..16: terrain material, 0 for plain colour.
    pub flags: u32,
}

/// One frame of actor geometry. `scene` indexes what the camera sees, `shadow`
/// what casts sun shadows; both are ranges of `indices`.
#[derive(Default)]
pub struct ActorDraw {
    pub vertices: Vec<ActorVertex>,
    pub indices: Vec<u32>,
    pub scene: Range<u32>,
    pub shadow: Range<u32>,
}

/// What the player is holding.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tool {
    /// A building block of a material, placed one grid block at a time.
    Block(Block),
    /// A shaping brush: a handle with a head the size and shape of the brush.
    Brush(Block, Brush),
}

impl Tool {
    pub fn from_selection(material: Block, brush: Brush) -> Self {
        match brush.shape {
            Shape::Block => Tool::Block(material),
            Shape::Sphere | Shape::Cube => Tool::Brush(material, brush),
        }
    }
}

/// Everything about the player the body's pose depends on.
pub struct PoseInput {
    /// Feet position.
    pub pos: Vec3,
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub on_ground: bool,
    pub flying: bool,
    pub in_water: bool,
    pub tool: Tool,
}

// Palette, sRGB.
const SKIN: [u8; 3] = [214, 160, 122];
const HAIR: [u8; 3] = [74, 48, 30];
const TUNIC: [u8; 3] = [46, 92, 110];
const TUNIC_TRIM: [u8; 3] = [196, 162, 88];
const BELT: [u8; 3] = [70, 46, 28];
const TROUSERS: [u8; 3] = [92, 78, 62];
const BOOTS: [u8; 3] = [58, 40, 28];
const EYES: [u8; 3] = [30, 26, 24];
const HANDLE: [u8; 3] = [132, 94, 58];
const METAL: [u8; 3] = [150, 150, 156];

/// Animation state carried between frames.
#[derive(Default)]
pub struct Rig {
    /// Walk cycle phase in radians.
    phase: f32,
    /// How far into the walk the pose is, 0 standing to 1 full stride.
    stride: f32,
    /// Seconds left of the current dig or build swing.
    swing: f32,
    time: f32,
}

const SWING_S: f32 = 0.28;

impl Rig {
    /// Starts a dig or build swing of the tool arm.
    pub fn act(&mut self) {
        if self.swing <= SWING_S * 0.35 {
            self.swing = SWING_S;
        }
    }

    pub fn update(&mut self, p: &PoseInput, dt: f32) {
        self.time += dt;
        self.swing = (self.swing - dt).max(0.0);
        let speed = Vec3::new(p.vel.x, 0.0, p.vel.z).length();
        let walking = if p.flying || !(p.on_ground || p.in_water) {
            0.0
        } else {
            (speed / 4.6).min(1.4)
        };
        self.stride += (walking - self.stride) * (dt * 10.0).min(1.0);
        // About 1.6 m per full step cycle at walking pace.
        self.phase = (self.phase + speed * dt * std::f32::consts::TAU / 1.6) % std::f32::consts::TAU;
    }

    /// Swing progress 0..1 (0 when idle) shaped as a quick down-and-back chop.
    fn chop(&self) -> f32 {
        if self.swing <= 0.0 {
            return 0.0;
        }
        let t = 1.0 - self.swing / SWING_S;
        (t * std::f32::consts::PI).sin()
    }

    /// Builds this frame's body. `third_person` draws the full body; otherwise
    /// only the view-model arm is visible and the body just casts its shadow.
    pub fn build(&self, p: &PoseInput, eye: Vec3, third_person: bool) -> ActorDraw {
        let mut d = ActorDraw::default();
        if !third_person {
            self.view_model(&mut d, p, eye);
        }
        let body_start = d.indices.len() as u32;
        self.body(&mut d, p);
        let end = d.indices.len() as u32;
        d.shadow = body_start..end;
        d.scene = if third_person { body_start..end } else { 0..body_start };
        d
    }

    fn body(&self, d: &mut ActorDraw, p: &PoseInput) {
        use std::f32::consts::{FRAC_PI_2, PI};
        // Local frame: +Z forward, +Y up, +X the figure's left.
        let root = Mat4::from_translation(p.pos) * Mat4::from_rotation_y(FRAC_PI_2 - p.yaw);
        let s = self.stride;
        let swing = self.phase.sin() * 0.55 * s;
        let bob = (self.phase * 2.0).cos().abs() * 0.03 * s + (self.time * 1.8).sin() * 0.004;
        let air = !p.on_ground && !p.in_water && !p.flying;
        let hips = root * Mat4::from_translation(Vec3::new(0.0, bob, 0.0));

        // Legs: thigh, knee, shin and boot. The knee bends as the foot swings forward.
        for (side, leg_phase) in [(1.0, self.phase), (-1.0, self.phase + PI)] {
            let (mut a, mut bend) = (leg_phase.sin() * 0.55 * s, s * (0.12 + 0.75 * leg_phase.cos().max(0.0)));
            if air {
                (a, bend) = (side * 0.35, 0.5);
            }
            if p.flying {
                (a, bend) = (-0.25 + side * 0.05, 0.35);
            }
            let hip = hips * Mat4::from_translation(Vec3::new(0.095 * side, 0.86, 0.0)) * Mat4::from_rotation_x(-a);
            blob(
                d,
                hip,
                Vec3::new(0.0, -0.2, 0.0),
                Vec3::new(0.078, 0.23, 0.085),
                ROUND,
                1.2,
                TROUSERS,
            );
            let knee = hip * Mat4::from_translation(Vec3::new(0.0, -0.4, 0.0)) * Mat4::from_rotation_x(bend);
            blob(
                d,
                knee,
                Vec3::new(0.0, -0.18, 0.0),
                Vec3::new(0.06, 0.21, 0.066),
                ROUND,
                1.15,
                TROUSERS,
            );
            let ankle = knee * Mat4::from_translation(Vec3::new(0.0, -0.38, 0.0)) * Mat4::from_rotation_x(-bend * 0.5);
            blob(
                d,
                ankle,
                Vec3::new(0.0, -0.03, 0.035),
                Vec3::new(0.066, 0.056, 0.115),
                (2.6, 2.2),
                1.0,
                BOOTS,
            );
            blob(
                d,
                ankle,
                Vec3::new(0.0, 0.05, 0.0),
                Vec3::new(0.064, 0.06, 0.07),
                ROUND,
                1.0,
                BOOTS,
            );
        }

        // Torso leans a little into the walk and while flying.
        let lean = s * 0.06 + if p.flying { 0.25 } else { 0.0 };
        let torso = hips * Mat4::from_translation(Vec3::new(0.0, 0.84, 0.0)) * Mat4::from_rotation_x(lean);
        blob(
            d,
            torso,
            Vec3::new(0.0, 0.03, 0.0),
            Vec3::new(0.165, 0.1, 0.11),
            ROUND,
            1.0,
            TROUSERS,
        );
        blob(
            d,
            torso,
            Vec3::new(0.0, 0.3, 0.0),
            Vec3::new(0.19, 0.27, 0.12),
            (2.6, 2.3),
            1.12,
            TUNIC,
        );
        blob(
            d,
            torso,
            Vec3::new(0.0, 0.06, 0.0),
            Vec3::new(0.2, 0.08, 0.132),
            ROUND,
            0.95,
            TUNIC,
        );
        blob(
            d,
            torso,
            Vec3::new(0.0, 0.11, 0.0),
            Vec3::new(0.196, 0.03, 0.128),
            (5.0, 2.2),
            1.0,
            BELT,
        );
        blob(
            d,
            torso,
            Vec3::new(0.0, 0.11, 0.128),
            Vec3::new(0.028, 0.024, 0.012),
            ROUND,
            1.0,
            METAL,
        );
        blob(
            d,
            torso,
            Vec3::new(0.0, 0.545, 0.0),
            Vec3::new(0.12, 0.035, 0.092),
            (3.0, 2.0),
            1.0,
            TUNIC_TRIM,
        );
        for x in [0.2, -0.2] {
            blob(d, torso, Vec3::new(x, 0.49, 0.0), Vec3::splat(0.075), ROUND, 1.0, TUNIC);
        }

        // Head follows the view pitch.
        let neck =
            torso * Mat4::from_translation(Vec3::new(0.0, 0.56, 0.0)) * Mat4::from_rotation_x(-p.pitch * 0.6 - lean);
        blob(
            d,
            neck,
            Vec3::new(0.0, 0.04, 0.0),
            Vec3::new(0.05, 0.06, 0.05),
            ROUND,
            1.0,
            SKIN,
        );
        let head = neck * Mat4::from_translation(Vec3::new(0.0, 0.18, 0.0));
        blob(d, head, Vec3::ZERO, Vec3::new(0.105, 0.125, 0.115), ROUND, 1.0, SKIN);
        blob(
            d,
            head,
            Vec3::new(0.0, -0.06, 0.02),
            Vec3::new(0.08, 0.07, 0.09),
            ROUND,
            1.0,
            SKIN,
        );
        blob(
            d,
            head,
            Vec3::new(0.0, 0.045, -0.028),
            Vec3::new(0.116, 0.11, 0.12),
            ROUND,
            1.0,
            HAIR,
        );
        for x in [0.042, -0.042] {
            blob(
                d,
                head,
                Vec3::new(x, 0.012, 0.103),
                Vec3::new(0.015, 0.017, 0.01),
                ROUND,
                1.0,
                EYES,
            );
        }
        for x in [0.104, -0.104] {
            blob(
                d,
                head,
                Vec3::new(x, -0.005, -0.005),
                Vec3::new(0.018, 0.032, 0.024),
                ROUND,
                1.0,
                SKIN,
            );
        }
        blob(
            d,
            head,
            Vec3::new(0.0, -0.025, 0.118),
            Vec3::new(0.016, 0.024, 0.02),
            ROUND,
            1.0,
            SKIN,
        );

        // Arms swing against the legs. The right arm holds the tool out front
        // and chops with it on each edit.
        let shoulder_y = 0.49;
        let left = torso
            * Mat4::from_translation(Vec3::new(0.23, shoulder_y, 0.0))
            * Mat4::from_rotation_x(swing * 0.8)
            * Mat4::from_rotation_z(0.08 + if air { 0.3 } else { 0.0 });
        let left_hand = arm(d, left, 0.25 + s * 0.15);
        blob(
            d,
            left_hand,
            Vec3::ZERO,
            Vec3::new(0.042, 0.05, 0.038),
            ROUND,
            1.0,
            SKIN,
        );
        let raise = 0.45 + p.pitch.clamp(-0.8, 0.8) * 0.5 - self.chop() * 0.7 - swing * 0.2;
        let right = torso
            * Mat4::from_translation(Vec3::new(-0.23, shoulder_y, 0.0))
            * Mat4::from_rotation_x(-raise)
            * Mat4::from_rotation_z(-0.08);
        let right_hand = arm(d, right, 0.55 + self.chop() * 0.3);
        blob(
            d,
            right_hand,
            Vec3::ZERO,
            Vec3::new(0.045, 0.05, 0.042),
            ROUND,
            1.0,
            SKIN,
        );
        let grip = right_hand * Mat4::from_rotation_x(FRAC_PI_2 - 0.2);
        tool(d, grip, p.tool, 0);
    }

    /// The right arm and tool as seen from the eye, fixed to the camera.
    fn view_model(&self, d: &mut ActorDraw, p: &PoseInput, eye: Vec3) {
        let s = self.stride;
        let bob_x = (self.phase).sin() * 0.012 * s;
        let bob_y = (self.phase * 2.0).cos().abs() * 0.012 * s + (self.time * 1.8).sin() * 0.003;
        let chop = self.chop();
        // Camera frame: +Z forward, +Y up, +X left, like the body.
        let cam = Mat4::from_translation(eye)
            * Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2 - p.yaw)
            * Mat4::from_rotation_x(-p.pitch);
        let hand = cam
            * Mat4::from_translation(Vec3::new(-0.27 + bob_x, -0.25 + bob_y - chop * 0.05, 0.5 + chop * 0.04))
            * Mat4::from_rotation_x(0.55 + chop * 0.7)
            * Mat4::from_rotation_y(-0.35);
        let start = d.vertices.len();
        // Forearm reaching in from below the view, then the tool in the fist.
        let arm = hand * Mat4::from_rotation_x(0.6);
        blob(
            d,
            arm,
            Vec3::new(0.0, -0.2, 0.0),
            Vec3::new(0.048, 0.2, 0.052),
            ROUND,
            1.2,
            TUNIC,
        );
        blob(
            d,
            arm,
            Vec3::new(0.0, -0.06, 0.0),
            Vec3::new(0.052, 0.022, 0.056),
            (3.0, 2.0),
            1.0,
            TUNIC_TRIM,
        );
        blob(d, arm, Vec3::ZERO, Vec3::new(0.045, 0.05, 0.043), ROUND, 1.0, SKIN);
        // Slightly smaller than life so the tool does not fill the view.
        let grip = hand * Mat4::from_translation(Vec3::new(0.0, 0.0, 0.02)) * Mat4::from_scale(Vec3::splat(0.8));
        tool(d, grip, p.tool, 0);
        for v in &mut d.vertices[start..] {
            v.flags |= 1;
        }
    }
}

/// Upper arm, elbow bent by `elbow` radians, forearm and cuff; returns the wrist frame.
fn arm(d: &mut ActorDraw, shoulder: Mat4, elbow: f32) -> Mat4 {
    blob(
        d,
        shoulder,
        Vec3::new(0.0, -0.15, 0.0),
        Vec3::new(0.055, 0.17, 0.058),
        ROUND,
        1.1,
        TUNIC,
    );
    let fore = shoulder * Mat4::from_translation(Vec3::new(0.0, -0.3, 0.0)) * Mat4::from_rotation_x(-elbow);
    blob(
        d,
        fore,
        Vec3::new(0.0, -0.12, 0.0),
        Vec3::new(0.047, 0.15, 0.05),
        ROUND,
        1.15,
        TUNIC,
    );
    blob(
        d,
        fore,
        Vec3::new(0.0, -0.23, 0.0),
        Vec3::new(0.05, 0.022, 0.053),
        (3.0, 2.0),
        1.0,
        TUNIC_TRIM,
    );
    fore * Mat4::from_translation(Vec3::new(0.0, -0.29, 0.0))
}

/// The held tool, gripped at `grip` with its working end along local +Y.
fn tool(d: &mut ActorDraw, grip: Mat4, tool: Tool, flags: u32) {
    match tool {
        Tool::Block(m) => {
            // A softened block the size of a fist, tilted so three sides show.
            let at = grip
                * Mat4::from_translation(Vec3::new(0.0, 0.1, 0.02))
                * Mat4::from_quat(Quat::from_euler(glam::EulerRot::YXZ, 0.6, 0.35, 0.0));
            // Textured like a whole building block.
            stuff(
                d,
                at,
                Vec3::splat(0.085),
                (4.0, 4.0),
                m,
                flags,
                BLOCK_VOXELS as f32 * VOXEL_SIZE,
            );
        }
        Tool::Brush(m, b) => {
            // A turned wooden handle with a head of the material, bigger for bigger brushes.
            blob(
                d,
                grip,
                Vec3::new(0.0, 0.12, 0.0),
                Vec3::new(0.017, 0.2, 0.017),
                (6.0, 2.0),
                1.15,
                HANDLE,
            );
            blob(
                d,
                grip,
                Vec3::new(0.0, 0.31, 0.0),
                Vec3::new(0.027, 0.02, 0.027),
                (3.0, 2.0),
                1.0,
                METAL,
            );
            let r = 0.055 + 0.012 * b.radius as f32;
            let head = grip * Mat4::from_translation(Vec3::new(0.0, 0.32 + r, 0.0));
            let round = if b.shape == Shape::Cube { (4.0, 4.0) } else { ROUND };
            stuff(d, head, Vec3::splat(r), round, m, flags, VOXEL_SIZE);
        }
    }
}

/// Exponents of a plain ellipsoid; larger values square a shape off.
const ROUND: (f32, f32) = (2.0, 2.0);

/// A smooth coloured part: an ellipsoid with radii `radii` about `center`,
/// squared off by `round` (vertical, around) and scaled by `taper` towards its top.
fn blob(d: &mut ActorDraw, at: Mat4, center: Vec3, radii: Vec3, round: (f32, f32), taper: f32, color: [u8; 3]) {
    surface(d, at, center, radii, round, taper, color, 0, (0.0, 0.0));
}

/// A smooth part made of terrain material whose width shows `span` metres of it.
fn stuff(d: &mut ActorDraw, at: Mat4, radii: Vec3, round: (f32, f32), m: Block, flags: u32, span: f32) {
    let scale = span * 0.5 / radii.max_element();
    let tex = (scale, span * 0.5);
    surface(
        d,
        at,
        Vec3::ZERO,
        radii,
        round,
        1.0,
        [255, 255, 255],
        flags | (m as u32) << 8,
        tex,
    );
}

/// `w` raised to `e`, keeping its sign.
fn spow(w: f32, e: f32) -> f32 {
    w.signum() * w.abs().powf(e)
}

/// Appends a superellipsoid with smooth normals. Lower parts get a little
/// occlusion so forms read as solid. Texture coordinates are the local
/// position times `tex.0`, plus `tex.1`.
#[allow(clippy::too_many_arguments)]
fn surface(
    d: &mut ActorDraw,
    at: Mat4,
    center: Vec3,
    radii: Vec3,
    round: (f32, f32),
    taper: f32,
    color: [u8; 3],
    flags: u32,
    tex: (f32, f32),
) {
    use std::f32::consts::{PI, TAU};
    // Fewer facets on small parts; they cover a few pixels.
    let (rings, segs) = if radii.max_element() < 0.035 { (5, 8) } else { (9, 16) };
    let (e1, e2) = (2.0 / round.0, 2.0 / round.1);
    let base = d.vertices.len() as u32;
    for i in 0..=rings {
        let th = -PI / 2.0 + PI * i as f32 / rings as f32;
        let (st, ct) = th.sin_cos();
        let y = spow(st, e1);
        // Width at this height: 1 at the bottom, `taper` at the top.
        let k = 1.0 + (taper - 1.0) * (y + 1.0) * 0.5;
        for j in 0..=segs {
            let ph = TAU * j as f32 / segs as f32;
            let (sp, cp) = ph.sin_cos();
            let unit = Vec3::new(spow(ct, e1) * spow(cp, e2) * k, y, spow(ct, e1) * spow(sp, e2) * k);
            let grad = Vec3::new(
                spow(ct, 2.0 - e1) * spow(cp, 2.0 - e2) / (radii.x * k),
                spow(st, 2.0 - e1) / radii.y,
                spow(ct, 2.0 - e1) * spow(sp, 2.0 - e2) / (radii.z * k),
            );
            let local_n = grad.normalize_or(Vec3::new(0.0, st.signum(), 0.0));
            let pos = center + unit * radii;
            let axis = local_n.abs().max_position();
            let face = axis as u32 * 2 + (local_n[axis] < 0.0) as u32;
            let ao = 190 + (65.0 * (y + 1.0) * 0.5) as u32;
            d.vertices.push(ActorVertex {
                pos: at.transform_point3(pos).to_array(),
                normal: at.transform_vector3(local_n).normalize().to_array(),
                tex: (pos * tex.0 + Vec3::splat(tex.1)).to_array(),
                color: color[0] as u32 | (color[1] as u32) << 8 | (color[2] as u32) << 16 | ao << 24,
                flags: flags | face << 1,
            });
        }
    }
    let row = segs + 1;
    for i in 0..rings {
        for j in 0..segs {
            let a = base + i * row + j;
            let (b, c) = (a + row, a + 1);
            // Counter-clockwise seen from outside.
            d.indices.extend_from_slice(&[a, b, c, c, b, b + 1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::STONE;

    fn pose(tool: Tool) -> PoseInput {
        PoseInput {
            pos: Vec3::new(10.0, 30.0, 10.0),
            vel: Vec3::ZERO,
            yaw: 0.3,
            pitch: 0.1,
            on_ground: true,
            flying: false,
            in_water: false,
            tool,
        }
    }

    #[test]
    fn surfaces_wind_outwards_with_smooth_normals() {
        for round in [ROUND, (4.0, 4.0), (6.0, 2.0)] {
            let mut d = ActorDraw::default();
            let radii = Vec3::new(0.2, 0.3, 0.1);
            surface(
                &mut d,
                Mat4::IDENTITY,
                Vec3::ZERO,
                radii,
                round,
                1.0,
                [0; 3],
                0,
                (0.0, 0.0),
            );
            for t in d.indices.chunks(3) {
                let [a, b, c] = [t[0], t[1], t[2]].map(|i| Vec3::from(d.vertices[i as usize].pos));
                let n = (b - a).cross(c - a);
                // Slivers at the poles, where a ring shrinks to a point, have no direction.
                let shortest = [(a, b), (b, c), (c, a)]
                    .map(|(p, q)| p.distance(q))
                    .into_iter()
                    .fold(1.0, f32::min);
                if shortest > 1e-3 {
                    assert!(
                        n.dot(a + b + c) > 0.0,
                        "inward triangle in {round:?}: {a} {b} {c} n={n}"
                    );
                }
            }
            for v in &d.vertices {
                let (p, n) = (Vec3::from(v.pos), Vec3::from(v.normal));
                assert!((n.length() - 1.0).abs() < 1e-3);
                assert!(n.dot(p) >= -1e-4, "normal points in at {p}");
            }
        }
    }

    #[test]
    fn body_stands_on_the_feet_and_is_player_sized() {
        let p = pose(Tool::Block(STONE));
        let d = Rig::default().build(&p, p.pos + Vec3::Y * 1.6, true);
        let ys: Vec<f32> = d.vertices.iter().map(|v| v.pos[1] - p.pos.y).collect();
        let lo = ys.iter().cloned().fold(f32::MAX, f32::min);
        let hi = ys.iter().cloned().fold(f32::MIN, f32::max);
        assert!(lo.abs() < 0.03, "feet at {lo}");
        assert!((1.6..1.9).contains(&hi), "top at {hi}");
        assert_eq!(d.scene, d.shadow);
    }

    #[test]
    fn first_person_shows_only_the_hand() {
        let p = pose(Tool::Brush(
            STONE,
            Brush {
                shape: Shape::Sphere,
                radius: 2,
            },
        ));
        let d = Rig::default().build(&p, p.pos + Vec3::Y * 1.6, false);
        assert!(!d.scene.is_empty());
        assert_eq!(d.scene.end, d.shadow.start);
        let hand = &d.indices[d.scene.start as usize..d.scene.end as usize];
        assert!(hand.iter().all(|&i| d.vertices[i as usize].flags & 1 == 1));
        let body = &d.indices[d.shadow.start as usize..d.shadow.end as usize];
        assert!(body.iter().all(|&i| d.vertices[i as usize].flags & 1 == 0));
    }

    #[test]
    fn tool_follows_the_brush() {
        let m = STONE;
        let brush = |shape| Brush { shape, radius: 1 };
        assert_eq!(Tool::from_selection(m, brush(Shape::Block)), Tool::Block(m));
        assert!(matches!(Tool::from_selection(m, brush(Shape::Cube)), Tool::Brush(..)));
        assert!(matches!(Tool::from_selection(m, brush(Shape::Sphere)), Tool::Brush(..)));
    }
}
