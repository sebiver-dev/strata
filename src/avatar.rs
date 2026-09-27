//! The player's body, built from boxes every frame: a procedurally posed
//! figure with a walk cycle, a head that follows the view, and whatever tool
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
        // Local frame: +Z forward, +Y up, +X the figure's left.
        let root = Mat4::from_translation(p.pos) * Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2 - p.yaw);
        let s = self.stride;
        let swing = self.phase.sin() * 0.55 * s;
        let bob = (self.phase * 2.0).cos().abs() * 0.03 * s + (self.time * 1.8).sin() * 0.004;
        let air = !p.on_ground && !p.in_water && !p.flying;
        let hips = root * Mat4::from_translation(Vec3::new(0.0, bob, 0.0));

        // Legs, hinged at the hips; knees are implied by the boots.
        for (side, phase) in [(1.0, swing), (-1.0, -swing)] {
            let mut a = phase;
            if air {
                a = side * 0.35;
            }
            if p.flying {
                a = -0.25 + side * 0.05;
            }
            let hip = hips * Mat4::from_translation(Vec3::new(0.095 * side, 0.86, 0.0)) * Mat4::from_rotation_x(-a);
            part(
                d,
                hip,
                Vec3::new(0.0, -0.33, 0.0),
                Vec3::new(0.075, 0.33, 0.085),
                TROUSERS,
            );
            part(d, hip, Vec3::new(0.0, -0.76, 0.025), Vec3::new(0.08, 0.1, 0.11), BOOTS);
        }

        // Torso leans a little into the walk and while flying.
        let lean = s * 0.06 + if p.flying { 0.25 } else { 0.0 };
        let torso = hips * Mat4::from_translation(Vec3::new(0.0, 0.84, 0.0)) * Mat4::from_rotation_x(lean);
        part(d, torso, Vec3::new(0.0, 0.29, 0.0), Vec3::new(0.19, 0.28, 0.11), TUNIC);
        part(d, torso, Vec3::new(0.0, 0.05, 0.0), Vec3::new(0.2, 0.09, 0.12), TUNIC);
        part(
            d,
            torso,
            Vec3::new(0.0, 0.09, 0.0),
            Vec3::new(0.205, 0.035, 0.125),
            BELT,
        );
        part(
            d,
            torso,
            Vec3::new(0.0, 0.09, 0.126),
            Vec3::new(0.035, 0.03, 0.006),
            METAL,
        );
        part(
            d,
            torso,
            Vec3::new(0.0, 0.555, 0.0),
            Vec3::new(0.2, 0.025, 0.12),
            TUNIC_TRIM,
        );

        // Head follows the view pitch.
        let neck =
            torso * Mat4::from_translation(Vec3::new(0.0, 0.58, 0.0)) * Mat4::from_rotation_x(-p.pitch * 0.6 - lean);
        part(d, neck, Vec3::new(0.0, 0.02, 0.0), Vec3::new(0.06, 0.04, 0.06), SKIN);
        let head = neck * Mat4::from_translation(Vec3::new(0.0, 0.17, 0.0));
        part(d, head, Vec3::ZERO, Vec3::new(0.13, 0.14, 0.13), SKIN);
        part(d, head, Vec3::new(0.0, 0.12, -0.01), Vec3::new(0.14, 0.04, 0.145), HAIR);
        part(d, head, Vec3::new(0.0, 0.02, -0.12), Vec3::new(0.14, 0.12, 0.025), HAIR);
        for x in [0.13, -0.13] {
            part(d, head, Vec3::new(x, 0.05, -0.04), Vec3::new(0.012, 0.07, 0.09), HAIR);
        }
        for x in [0.05, -0.05] {
            part(d, head, Vec3::new(x, 0.02, 0.131), Vec3::new(0.018, 0.016, 0.004), EYES);
        }
        part(
            d,
            head,
            Vec3::new(0.0, -0.02, 0.14),
            Vec3::new(0.015, 0.03, 0.012),
            SKIN,
        );

        // Arms swing against the legs. The right arm holds the tool out front
        // and chops with it on each edit.
        let shoulder_y = 0.54;
        let left = torso
            * Mat4::from_translation(Vec3::new(0.25, shoulder_y, 0.0))
            * Mat4::from_rotation_x(swing * 0.8)
            * Mat4::from_rotation_z(0.06 + if air { 0.3 } else { 0.0 });
        arm(d, left);
        let raise = 0.75 + p.pitch.clamp(-0.8, 0.8) * 0.6 - self.chop() * 0.9 - swing * 0.2;
        let right = torso
            * Mat4::from_translation(Vec3::new(-0.25, shoulder_y, 0.0))
            * Mat4::from_rotation_x(-raise)
            * Mat4::from_rotation_z(-0.06);
        arm(d, right);
        let grip = right
            * Mat4::from_translation(Vec3::new(0.0, -0.56, 0.02))
            * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2 - 0.3);
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
        part(d, arm, Vec3::new(0.0, -0.2, 0.0), Vec3::new(0.05, 0.2, 0.055), TUNIC);
        part(
            d,
            arm,
            Vec3::new(0.0, -0.395, 0.0),
            Vec3::new(0.052, 0.02, 0.057),
            TUNIC_TRIM,
        );
        part(d, arm, Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.045, 0.05, 0.05), SKIN);
        // Slightly smaller than life so the tool does not fill the view.
        let grip = hand * Mat4::from_translation(Vec3::new(0.0, 0.0, 0.02)) * Mat4::from_scale(Vec3::splat(0.8));
        tool(d, grip, p.tool, 0);
        for v in &mut d.vertices[start..] {
            v.flags |= 1;
        }
    }
}

fn arm(d: &mut ActorDraw, shoulder: Mat4) {
    part(
        d,
        shoulder,
        Vec3::new(0.0, -0.2, 0.0),
        Vec3::new(0.06, 0.2, 0.065),
        TUNIC,
    );
    part(
        d,
        shoulder,
        Vec3::new(0.0, -0.405, 0.0),
        Vec3::new(0.062, 0.025, 0.067),
        TUNIC_TRIM,
    );
    part(
        d,
        shoulder,
        Vec3::new(0.0, -0.49, 0.0),
        Vec3::new(0.05, 0.065, 0.055),
        SKIN,
    );
}

/// The held tool, gripped at `grip` with its working end along local +Y.
fn tool(d: &mut ActorDraw, grip: Mat4, tool: Tool, flags: u32) {
    match tool {
        Tool::Block(m) => {
            // A block the size of a fist, tilted so three faces show.
            let at = grip
                * Mat4::from_translation(Vec3::new(0.0, 0.09, 0.02))
                * Mat4::from_quat(Quat::from_euler(glam::EulerRot::YXZ, 0.6, 0.35, 0.0));
            // Textured like a whole building block, so its faces show the grid.
            block_box(d, at, Vec3::splat(0.085), m, flags, BLOCK_VOXELS as f32 * VOXEL_SIZE);
        }
        Tool::Brush(m, b) => {
            // A wooden handle with a shaped head of the material, bigger for bigger brushes.
            part(d, grip, Vec3::new(0.0, 0.12, 0.0), Vec3::new(0.018, 0.2, 0.018), HANDLE);
            part(d, grip, Vec3::new(0.0, 0.31, 0.0), Vec3::new(0.03, 0.015, 0.03), METAL);
            let r = 0.055 + 0.012 * b.radius as f32;
            let head = grip * Mat4::from_translation(Vec3::new(0.0, 0.33 + r, 0.0));
            if b.shape == Shape::Cube {
                block_box(d, head, Vec3::splat(r), m, flags, VOXEL_SIZE);
            } else {
                // A rounded head: three crossed slabs read as a ball at hand size.
                let t = r * 0.62;
                block_box(d, head, Vec3::new(r, t, t), m, flags, VOXEL_SIZE);
                block_box(d, head, Vec3::new(t, r, t), m, flags, VOXEL_SIZE);
                block_box(d, head, Vec3::new(t, t, r), m, flags, VOXEL_SIZE);
            }
        }
    }
}

/// A box of terrain material whose largest side shows `span` metres of it.
fn block_box(d: &mut ActorDraw, at: Mat4, half: Vec3, m: Block, flags: u32, span: f32) {
    let scale = span * 0.5 / half.max_element();
    cuboid(
        d,
        at,
        Vec3::ZERO,
        half,
        [255, 255, 255],
        flags | (m as u32) << 8,
        (scale, span * 0.5),
    );
}

fn part(d: &mut ActorDraw, at: Mat4, center: Vec3, half: Vec3, color: [u8; 3]) {
    cuboid(d, at, center, half, color, 0, (0.0, 0.0));
}

/// Unit face directions and in-face axes, in the terrain's face order.
const FACES: [(Vec3, Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::Y, Vec3::Z),
    (Vec3::NEG_X, Vec3::Z, Vec3::Y),
    (Vec3::Y, Vec3::Z, Vec3::X),
    (Vec3::NEG_Y, Vec3::X, Vec3::Z),
    (Vec3::Z, Vec3::X, Vec3::Y),
    (Vec3::NEG_Z, Vec3::Y, Vec3::X),
];

/// Appends a box with half extents `half` centred at `center` in `at`'s frame.
/// Lower corners get a little occlusion so parts read as solid forms.
/// Texture coordinates are the local position times `tex.0`, plus `tex.1`.
fn cuboid(d: &mut ActorDraw, at: Mat4, center: Vec3, half: Vec3, color: [u8; 3], flags: u32, tex: (f32, f32)) {
    for (face, (n, u, v)) in FACES.iter().enumerate() {
        let base = d.vertices.len() as u32;
        let normal = at.transform_vector3(*n).normalize();
        for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let local = *n + *u * a + *v * b;
            let p = center + local * half;
            // Occlusion from the corner's height within the part.
            let ao = if local.y < 0.0 { 200u32 } else { 255 };
            let uvw = (center + local * half) * tex.0 + Vec3::splat(tex.1);
            d.vertices.push(ActorVertex {
                pos: at.transform_point3(p).to_array(),
                normal: normal.to_array(),
                tex: uvw.to_array(),
                color: color[0] as u32 | (color[1] as u32) << 8 | (color[2] as u32) << 16 | ao << 24,
                flags: flags | (face as u32) << 1,
            });
        }
        // u x v = n for every face, so (0, 1, 2) winds counter-clockwise seen from outside.
        d.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
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
    fn faces_wind_outwards() {
        for (n, u, v) in FACES {
            assert!(u.cross(v).abs_diff_eq(n, 1e-6), "{n}");
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
