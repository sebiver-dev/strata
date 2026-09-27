//! GPU side: one wgpu device, a sky pass, opaque terrain, blended water and a
//! crosshair overlay. Runs on Vulkan, Metal, DirectX 12 and browser WebGPU.

use crate::mesh::{MeshData, Vertex};
use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Mat4, Vec3, Vec4};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    pub params: [f32; 4],
    pub highlight: [f32; 4],
    pub screen: [f32; 4],
}

struct GpuMesh {
    buffers: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    water: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    min: Vec3,
    max: Vec3,
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    manual_srgb: bool,
    depth: wgpu::TextureView,
    sky: wgpu::RenderPipeline,
    terrain: wgpu::RenderPipeline,
    water: wgpu::RenderPipeline,
    overlay: wgpu::RenderPipeline,
    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    meshes: HashMap<IVec3, GpuMesh>,
    pub adapter_name: String,
    pub triangles_drawn: u32,
}

fn create_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Result<Self, String> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).map_err(|e| e.to_string())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("No suitable GPU adapter: {e}"))?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("strata"),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|e| e.to_string())?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or("Surface is not supported by this GPU")?;
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or("Surface is not supported by this GPU")?;
        config.format = format;
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/world.wgsl").into()),
        });

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32],
        };

        let pipeline = |label: &str,
                        vs: &str,
                        fs: &str,
                        buffers: &[Option<wgpu::VertexBufferLayout>],
                        blend: Option<wgpu::BlendState>,
                        depth_write: bool,
                        depth_compare: wgpu::CompareFunction,
                        cull: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: cull,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(depth_compare),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        use wgpu::CompareFunction::{Always, Greater, GreaterEqual};
        let sky = pipeline("sky", "vs_fullscreen", "fs_sky", &[], None, false, Always, None);
        let terrain = pipeline(
            "terrain",
            "vs_world",
            "fs_terrain",
            &[Some(vertex_layout.clone())],
            None,
            true,
            Greater,
            Some(wgpu::Face::Back),
        );
        let water = pipeline(
            "water",
            "vs_world",
            "fs_water",
            &[Some(vertex_layout)],
            Some(wgpu::BlendState::ALPHA_BLENDING),
            false,
            GreaterEqual,
            None,
        );
        let overlay = pipeline(
            "overlay",
            "vs_fullscreen",
            "fs_overlay",
            &[],
            Some(wgpu::BlendState::ALPHA_BLENDING),
            false,
            Always,
            None,
        );

        let depth = create_depth(&device, config.width, config.height);
        Ok(Self {
            surface,
            device,
            queue,
            manual_srgb: !format.is_srgb(),
            config,
            depth,
            sky,
            terrain,
            water,
            overlay,
            globals,
            bind_group,
            meshes: HashMap::new(),
            adapter_name,
            triangles_drawn: 0,
        })
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth = create_depth(&self.device, w, h);
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn manual_srgb(&self) -> bool {
        self.manual_srgb
    }

    pub fn upload(&mut self, cpos: IVec3, mesh: &MeshData) {
        if mesh.is_empty() {
            self.meshes.remove(&cpos);
            return;
        }
        let make = |v: &[Vertex], i: &[u32]| {
            if i.is_empty() {
                return None;
            }
            let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chunk vertices"),
                contents: bytemuck::cast_slice(v),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chunk indices"),
                contents: bytemuck::cast_slice(i),
                usage: wgpu::BufferUsages::INDEX,
            });
            Some((vb, ib, i.len() as u32))
        };
        let size = crate::chunk::CHUNK as f32 * crate::block::VOXEL_SIZE;
        let min = cpos.as_vec3() * size;
        self.meshes.insert(
            cpos,
            GpuMesh {
                buffers: make(&mesh.vertices, &mesh.indices),
                water: make(&mesh.water_vertices, &mesh.water_indices),
                min,
                max: min + Vec3::splat(size),
            },
        );
    }

    pub fn remove(&mut self, cpos: IVec3) {
        self.meshes.remove(&cpos);
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    pub fn render(&mut self, globals: &Globals) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(globals));

        let planes = frustum_planes(Mat4::from_cols_array_2d(&globals.view_proj));
        let visible: Vec<&GpuMesh> = self
            .meshes
            .values()
            .filter(|m| aabb_visible(&planes, m.min, m.max))
            .collect();

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut triangles = 0;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    // Reverse Z: 0 is infinitely far away.
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.sky);
            pass.draw(0..3, 0..1);

            pass.set_pipeline(&self.terrain);
            for m in &visible {
                if let Some((vb, ib, n)) = &m.buffers {
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..*n, 0, 0..1);
                    triangles += n / 3;
                }
            }
            pass.set_pipeline(&self.water);
            for m in &visible {
                if let Some((vb, ib, n)) = &m.water {
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..*n, 0, 0..1);
                    triangles += n / 3;
                }
            }
            pass.set_pipeline(&self.overlay);
            pass.draw(0..3, 0..1);
        }
        self.triangles_drawn = triangles;
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }
}

/// Left, right, bottom and top clip planes. Near and far are skipped: the
/// projection has an infinite far plane and chunks behind the camera fail the
/// side planes anyway.
fn frustum_planes(m: Mat4) -> [Vec4; 4] {
    let (r0, r1, r3) = (m.row(0), m.row(1), m.row(3));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1]
}

fn aabb_visible(planes: &[Vec4; 4], min: Vec3, max: Vec3) -> bool {
    planes.iter().all(|p| {
        let v = Vec3::new(
            if p.x >= 0.0 { max.x } else { min.x },
            if p.y >= 0.0 { max.y } else { min.y },
            if p.z >= 0.0 { max.z } else { min.z },
        );
        p.truncate().dot(v) + p.w >= 0.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_keeps_what_is_in_front() {
        let proj = Mat4::perspective_infinite_reverse_rh(1.2, 1.5, 0.05);
        let view = Mat4::look_to_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        let planes = frustum_planes(proj * view);
        assert!(aabb_visible(
            &planes,
            Vec3::new(-1.0, -1.0, -20.0),
            Vec3::new(1.0, 1.0, -10.0)
        ));
        assert!(!aabb_visible(
            &planes,
            Vec3::new(-1.0, -1.0, 10.0),
            Vec3::new(1.0, 1.0, 20.0)
        ));
        assert!(!aabb_visible(
            &planes,
            Vec3::new(100.0, -1.0, -20.0),
            Vec3::new(110.0, 1.0, -10.0)
        ));
    }
}
