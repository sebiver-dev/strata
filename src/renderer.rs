//! GPU side: one wgpu device and a small frame graph. A sun shadow pass,
//! then sky and terrain into an HDR target, then water (which reads a copy of
//! that target for refraction and reflections), then tone mapping and the
//! crosshair onto the screen. Runs on Vulkan, Metal, DirectX 12 and browser WebGPU.

use crate::avatar::{ActorDraw, ActorVertex};
use crate::mesh::{MeshData, Vertex};
use crate::terrain::WORLD_CHUNKS_XZ;
use bytemuck::{Pod, Zeroable};
use glam::{IVec2, IVec3, Mat4, Vec3, Vec4};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Shadow map resolution in texels per side.
pub const SHADOW_SIZE: u32 = 2048;
/// Half the width of the square area around the player that casts sun shadows.
pub const SHADOW_RANGE_M: f32 = 72.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub sun_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub sun_dir: [f32; 4],
    pub params: [f32; 4],
    pub highlight: [f32; 4],
    pub screen: [f32; 4],
}

/// Orthographic view-projection for the sun's shadow map, centred on `center`.
/// The centre is snapped to whole texels so shadow edges do not crawl as the
/// player moves.
pub fn sun_view_proj(center: Vec3, sun: Vec3) -> Mat4 {
    let up = if sun.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let view = Mat4::look_to_rh(Vec3::ZERO, -sun, up);
    let texel = 2.0 * SHADOW_RANGE_M / SHADOW_SIZE as f32;
    let c = view.transform_point3(center);
    let (x, y) = ((c.x / texel).floor() * texel, (c.y / texel).floor() * texel);
    let r = SHADOW_RANGE_M;
    let proj = Mat4::orthographic_rh(x - r, x + r, y - r, y + r, -c.z - 250.0, -c.z + 250.0);
    proj * view
}

struct GpuMesh {
    buffers: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    water: Option<(wgpu::Buffer, wgpu::Buffer, u32)>,
    min: Vec3,
    max: Vec3,
}

/// Screen-sized render targets, rebuilt on resize.
struct Targets {
    hdr: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    scene_copy: wgpu::Texture,
    depth_copy: wgpu::Texture,
    scene_bind: wgpu::BindGroup,
    post_bind: wgpu::BindGroup,
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    manual_srgb: bool,
    shadow_view: wgpu::TextureView,
    shadow_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    scene_layout: wgpu::BindGroupLayout,
    post_layout: wgpu::BindGroupLayout,
    targets: Targets,
    shadow: wgpu::RenderPipeline,
    sky: wgpu::RenderPipeline,
    terrain: wgpu::RenderPipeline,
    water: wgpu::RenderPipeline,
    post: wgpu::RenderPipeline,
    far_terrain: wgpu::RenderPipeline,
    far_water: wgpu::RenderPipeline,
    overlay: wgpu::RenderPipeline,
    actor: wgpu::RenderPipeline,
    actor_shadow: wgpu::RenderPipeline,
    actors: ActorBuffers,
    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    meshes: HashMap<IVec3, GpuMesh>,
    far_meshes: HashMap<IVec2, GpuMesh>,
    /// One texel per chunk column of the world, non-zero where the column's
    /// voxel meshes are drawn so far tiles must stay hidden there.
    near_mask: wgpu::Texture,
    pub adapter_name: String,
    pub triangles_drawn: u32,
    pub far_triangles_drawn: u32,
    /// Fraction of the window's pixels the 3D scene is rendered at.
    render_scale: f32,
    /// Forced render scale from the page URL (`?scale=`), for testing.
    pub scale_override: Option<f32>,
    timer: Option<GpuTimer>,
    /// Smoothed GPU time per pass in milliseconds: shadow, scene, water, post.
    pub gpu_ms: Option<[f32; 4]>,
}

/// GPU pass timings from timestamp queries, read back without stalling: a new
/// readback starts only once the previous one has arrived.
struct GpuTimer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// 0: idle, 1: waiting for the map, 2: mapped and ready to read.
    state: Arc<std::sync::atomic::AtomicU8>,
    period_ns: f32,
}

const TIMED_PASSES: u32 = 4;

/// This frame's actor geometry on the GPU, rewritten every frame and grown as needed.
#[derive(Default)]
struct ActorBuffers {
    buffers: Option<(wgpu::Buffer, wgpu::Buffer)>,
    scene: std::ops::Range<u32>,
    shadow: std::ops::Range<u32>,
}

impl GpuTimer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let size = TIMED_PASSES as u64 * 2 * 8;
        Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pass timings"),
                ty: wgpu::QueryType::Timestamp,
                count: TIMED_PASSES * 2,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timings resolve"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timings readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: Arc::new(std::sync::atomic::AtomicU8::new(0)),
            period_ns: queue.get_timestamp_period(),
        }
    }

    fn writes(&self, pass: u32) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(pass * 2),
            end_of_pass_write_index: Some(pass * 2 + 1),
        })
    }
}

fn make_mesh(device: &wgpu::Device, mesh: &MeshData, min: Vec3, max: Vec3) -> GpuMesh {
    let make = |v: &[Vertex], i: &[u32]| {
        if i.is_empty() {
            return None;
        }
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertices"),
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: bytemuck::cast_slice(i),
            usage: wgpu::BufferUsages::INDEX,
        });
        Some((vb, ib, i.len() as u32))
    };
    GpuMesh {
        buffers: make(&mesh.vertices, &mesh.indices),
        water: make(&mesh.water_vertices, &mesh.water_indices),
        min,
        max,
    }
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn pass_desc<'a>(
    label: &'a str,
    color: &'a [Option<wgpu::RenderPassColorAttachment<'a>>],
    depth: Option<wgpu::RenderPassDepthStencilAttachment<'a>>,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'a>>,
) -> wgpu::RenderPassDescriptor<'a> {
    wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: color,
        depth_stencil_attachment: depth,
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    }
}

fn view(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor::default())
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32, ty: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(ty),
        count: None,
    }
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
        let info = adapter.get_info();
        let adapter_name = match info.name.as_str() {
            "" => format!("{:?} adapter", info.device_type),
            name => format!("{name} ({:?})", info.device_type),
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("strata"),
                required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|e| e.to_string())?;

        device.set_device_lost_callback(|reason, message| log::error!("GPU device lost ({reason:?}): {message}"));

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
            source: wgpu::ShaderSource::Wgsl(
                concat!(include_str!("shaders/world.wgsl"), include_str!("shaders/actor.wgsl")).into(),
            ),
        });

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let side = WORLD_CHUNKS_XZ as u32;
        let near_mask = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("near mask"),
            size: wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &near_mask.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
            ],
        });
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene inputs"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Depth),
                sampler_entry(1, wgpu::SamplerBindingType::Comparison),
                texture_entry(2, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(3, wgpu::TextureSampleType::Depth),
                sampler_entry(4, wgpu::SamplerBindingType::Filtering),
            ],
        });
        let post_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post inputs"),
            entries: &[
                texture_entry(5, wgpu::TextureSampleType::Float { filterable: true }),
                sampler_entry(6, wgpu::SamplerBindingType::Filtering),
            ],
        });
        let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let world_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world"),
            bind_group_layouts: &[Some(&bgl), Some(&scene_layout)],
            immediate_size: 0,
        });
        let post_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post"),
            bind_group_layouts: &[Some(&bgl), Some(&post_layout)],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32],
        };

        struct Desc<'a> {
            label: &'a str,
            layout: &'a wgpu::PipelineLayout,
            vs: &'a str,
            fs: Option<&'a str>,
            buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
            target: Option<wgpu::TextureFormat>,
            blend: Option<wgpu::BlendState>,
            depth: Option<(bool, wgpu::CompareFunction)>,
            bias: wgpu::DepthBiasState,
            cull: Option<wgpu::Face>,
        }
        let pipeline = |d: Desc| {
            let targets = [d.target.map(|format| wgpu::ColorTargetState {
                format,
                blend: d.blend,
                write_mask: wgpu::ColorWrites::ALL,
            })];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(d.label),
                layout: Some(d.layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(d.vs),
                    compilation_options: Default::default(),
                    buffers: d.buffers,
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: d.cull,
                    ..Default::default()
                },
                depth_stencil: d.depth.map(|(write, compare)| wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(write),
                    depth_compare: Some(compare),
                    stencil: Default::default(),
                    bias: d.bias,
                }),
                multisample: Default::default(),
                fragment: d.fs.map(|fs| wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        use wgpu::CompareFunction::{Always, Greater, GreaterEqual, LessEqual};
        let world_buffers = [Some(vertex_layout)];
        let shadow = pipeline(Desc {
            label: "shadow",
            layout: &shadow_pipeline_layout,
            vs: "vs_shadow",
            fs: None,
            buffers: &world_buffers,
            target: None,
            blend: None,
            depth: Some((true, LessEqual)),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
            cull: None,
        });
        let sky = pipeline(Desc {
            label: "sky",
            layout: &world_pipeline_layout,
            vs: "vs_fullscreen",
            fs: Some("fs_sky"),
            buffers: &[],
            target: Some(HDR_FORMAT),
            blend: None,
            depth: Some((false, Always)),
            bias: Default::default(),
            cull: None,
        });
        let terrain = pipeline(Desc {
            label: "terrain",
            layout: &world_pipeline_layout,
            vs: "vs_world",
            fs: Some("fs_terrain"),
            buffers: &world_buffers,
            target: Some(HDR_FORMAT),
            blend: None,
            depth: Some((true, Greater)),
            bias: Default::default(),
            cull: Some(wgpu::Face::Back),
        });
        // Water is opaque: it composites the scene behind it itself.
        let water = pipeline(Desc {
            label: "water",
            layout: &world_pipeline_layout,
            vs: "vs_world",
            fs: Some("fs_water"),
            buffers: &world_buffers,
            target: Some(HDR_FORMAT),
            blend: None,
            depth: Some((true, GreaterEqual)),
            bias: Default::default(),
            cull: None,
        });
        let far_terrain = pipeline(Desc {
            label: "far terrain",
            layout: &world_pipeline_layout,
            vs: "vs_world",
            fs: Some("fs_far_terrain"),
            buffers: &world_buffers,
            target: Some(HDR_FORMAT),
            blend: None,
            depth: Some((true, Greater)),
            bias: Default::default(),
            cull: Some(wgpu::Face::Back),
        });
        let far_water = pipeline(Desc {
            label: "far water",
            layout: &world_pipeline_layout,
            vs: "vs_world",
            fs: Some("fs_far_water"),
            buffers: &world_buffers,
            target: Some(HDR_FORMAT),
            blend: None,
            depth: Some((true, GreaterEqual)),
            bias: Default::default(),
            cull: None,
        });
        let post = pipeline(Desc {
            label: "post",
            layout: &post_pipeline_layout,
            vs: "vs_fullscreen",
            fs: Some("fs_post"),
            buffers: &[],
            target: Some(format),
            blend: None,
            depth: None,
            bias: Default::default(),
            cull: None,
        });
        let actor_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ActorVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Uint32, 4 => Uint32],
        };
        let actor_buffers = [Some(actor_layout)];
        let actor = pipeline(Desc {
            label: "actor",
            layout: &world_pipeline_layout,
            vs: "vs_actor",
            fs: Some("fs_actor"),
            buffers: &actor_buffers,
            target: Some(HDR_FORMAT),
            blend: None,
            depth: Some((true, Greater)),
            bias: Default::default(),
            cull: Some(wgpu::Face::Back),
        });
        let actor_shadow = pipeline(Desc {
            label: "actor shadow",
            layout: &shadow_pipeline_layout,
            vs: "vs_actor_shadow",
            fs: None,
            buffers: &actor_buffers,
            target: None,
            blend: None,
            depth: Some((true, LessEqual)),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 2.0,
                clamp: 0.0,
            },
            cull: None,
        });
        let overlay = pipeline(Desc {
            label: "overlay",
            layout: &post_pipeline_layout,
            vs: "vs_fullscreen",
            fs: Some("fs_overlay"),
            buffers: &[],
            target: Some(format),
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            depth: None,
            bias: Default::default(),
            cull: None,
        });

        let shadow_view = view(&texture(
            &device,
            "sun shadow map",
            SHADOW_SIZE,
            SHADOW_SIZE,
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        ));
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let timer = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| GpuTimer::new(&device, &queue));
        let targets = Self::create_targets(
            &device,
            &scene_layout,
            &post_layout,
            &shadow_view,
            &shadow_sampler,
            &linear_sampler,
            config.width,
            config.height,
        );
        Ok(Self {
            surface,
            device,
            queue,
            manual_srgb: !format.is_srgb(),
            config,
            shadow_view,
            shadow_sampler,
            linear_sampler,
            scene_layout,
            post_layout,
            targets,
            shadow,
            sky,
            terrain,
            water,
            post,
            far_terrain,
            far_water,
            overlay,
            actor,
            actor_shadow,
            actors: ActorBuffers::default(),
            globals,
            bind_group,
            meshes: HashMap::new(),
            far_meshes: HashMap::new(),
            near_mask,
            adapter_name,
            triangles_drawn: 0,
            far_triangles_drawn: 0,
            render_scale: 1.0,
            scale_override: None,
            timer,
            gpu_ms: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_targets(
        device: &wgpu::Device,
        scene_layout: &wgpu::BindGroupLayout,
        post_layout: &wgpu::BindGroupLayout,
        shadow_view: &wgpu::TextureView,
        shadow_sampler: &wgpu::Sampler,
        linear_sampler: &wgpu::Sampler,
        w: u32,
        h: u32,
    ) -> Targets {
        use wgpu::TextureUsages as U;
        let hdr = texture(
            device,
            "hdr",
            w,
            h,
            HDR_FORMAT,
            U::RENDER_ATTACHMENT | U::TEXTURE_BINDING | U::COPY_SRC,
        );
        let depth = texture(device, "depth", w, h, DEPTH_FORMAT, U::RENDER_ATTACHMENT | U::COPY_SRC);
        let scene_copy = texture(device, "scene copy", w, h, HDR_FORMAT, U::TEXTURE_BINDING | U::COPY_DST);
        let depth_copy = texture(
            device,
            "depth copy",
            w,
            h,
            DEPTH_FORMAT,
            U::TEXTURE_BINDING | U::COPY_DST,
        );
        let (hdr_view, depth_view) = (view(&hdr), view(&depth));
        let scene_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene inputs"),
            layout: scene_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view(&scene_copy)),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&view(&depth_copy)),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
                },
            ],
        });
        let post_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post inputs"),
            layout: post_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(linear_sampler),
                },
            ],
        });
        Targets {
            hdr,
            hdr_view,
            depth,
            depth_view,
            scene_copy,
            depth_copy,
            scene_bind,
            post_bind,
        }
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        // The scene shaders cost the same per pixel however dense the screen is,
        // so very high resolutions (Retina, 4K) render the scene with fewer
        // pixels and scale it up. The crosshair stays sharp.
        let budget = if cfg!(target_arch = "wasm32") { 2.1e6 } else { 3.7e6 };
        let fit = (budget / (w as f32 * h as f32)).sqrt().min(1.0);
        self.render_scale = self.scale_override.unwrap_or(fit).clamp(0.25, 1.0);
        let (rw, rh) = self.render_size();
        self.targets = Self::create_targets(
            &self.device,
            &self.scene_layout,
            &self.post_layout,
            &self.shadow_view,
            &self.shadow_sampler,
            &self.linear_sampler,
            rw,
            rh,
        );
    }

    /// Size of the window in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Size the 3D scene is rendered at before it is scaled to the window.
    pub fn render_size(&self) -> (u32, u32) {
        let s = self.render_scale;
        (
            ((self.config.width as f32 * s).round() as u32).max(1),
            ((self.config.height as f32 * s).round() as u32).max(1),
        )
    }

    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    pub fn manual_srgb(&self) -> bool {
        self.manual_srgb
    }

    pub fn upload(&mut self, cpos: IVec3, mesh: &MeshData) {
        if mesh.is_empty() {
            self.meshes.remove(&cpos);
            return;
        }
        let size = crate::chunk::CHUNK as f32 * crate::block::VOXEL_SIZE;
        let min = cpos.as_vec3() * size;
        let gpu = make_mesh(&self.device, mesh, min, min + Vec3::splat(size));
        self.meshes.insert(cpos, gpu);
    }

    /// Replaces the mesh of one far tile.
    pub fn upload_far(&mut self, tile: IVec2, mesh: &MeshData) {
        let size = crate::far::TILE_M;
        let top = crate::terrain::WORLD_CHUNKS_Y as f32 * crate::chunk::CHUNK as f32 * crate::block::VOXEL_SIZE;
        let min = Vec3::new(tile.x as f32 * size, 0.0, tile.y as f32 * size);
        let gpu = make_mesh(&self.device, mesh, min, min + Vec3::new(size, top, size));
        self.far_meshes.insert(tile, gpu);
    }

    /// Marks which chunk columns are covered by voxel meshes (one byte per column, row-major in z).
    pub fn set_near_mask(&mut self, mask: &[u8]) {
        let side = WORLD_CHUNKS_XZ as u32;
        self.queue.write_texture(
            self.near_mask.as_image_copy(),
            mask,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(side),
                rows_per_image: Some(side),
            },
            wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Replaces the posed characters drawn this frame.
    pub fn set_actors(&mut self, draw: &ActorDraw) {
        let a = &mut self.actors;
        let vbytes = std::mem::size_of_val(draw.vertices.as_slice()) as u64;
        let ibytes = std::mem::size_of_val(draw.indices.as_slice()) as u64;
        let fits = a
            .buffers
            .as_ref()
            .is_some_and(|(v, i)| v.size() >= vbytes && i.size() >= ibytes);
        if !fits {
            let make = |label, size: u64, usage| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: size.next_power_of_two().max(4096),
                    usage: usage | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
            a.buffers = Some((
                make("actor vertices", vbytes, wgpu::BufferUsages::VERTEX),
                make("actor indices", ibytes, wgpu::BufferUsages::INDEX),
            ));
        }
        if let Some((vb, ib)) = &a.buffers {
            self.queue.write_buffer(vb, 0, bytemuck::cast_slice(&draw.vertices));
            self.queue.write_buffer(ib, 0, bytemuck::cast_slice(&draw.indices));
        }
        a.scene = draw.scene.clone();
        a.shadow = draw.shadow.clone();
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

        self.read_timings();
        let planes = frustum_planes(Mat4::from_cols_array_2d(&globals.view_proj));
        let eye = Vec3::from_slice(&globals.camera_pos[..3]);
        // Near to far, so the depth test rejects hidden pixels before the
        // (expensive) material shader runs on them.
        let by_distance = |a: &&GpuMesh, b: &&GpuMesh| {
            let da = eye.distance_squared(eye.clamp(a.min, a.max));
            let db = eye.distance_squared(eye.clamp(b.min, b.max));
            da.total_cmp(&db)
        };
        let mut visible: Vec<&GpuMesh> = self
            .meshes
            .values()
            .filter(|m| aabb_visible(&planes, m.min, m.max))
            .collect();
        visible.sort_unstable_by(by_distance);
        let sun_planes = frustum_planes(Mat4::from_cols_array_2d(&globals.sun_view_proj));
        let casters: Vec<&GpuMesh> = self
            .meshes
            .values()
            .filter(|m| aabb_visible(&sun_planes, m.min, m.max))
            .collect();
        let mut far_visible: Vec<&GpuMesh> = self
            .far_meshes
            .values()
            .filter(|m| aabb_visible(&planes, m.min, m.max))
            .collect();
        far_visible.sort_unstable_by(by_distance);

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut triangles = 0;
        let mut far_triangles = 0;
        let depth_attachment = |view, load| {
            Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            })
        };
        let color_attachment = |view, load| {
            [Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })]
        };
        let t = &self.targets;
        let timer = self.timer.as_ref();

        // 1. Sun shadow map.
        {
            let mut pass = encoder.begin_render_pass(&pass_desc(
                "shadow",
                &[],
                depth_attachment(&self.shadow_view, wgpu::LoadOp::Clear(1.0)),
                timer.and_then(|t| t.writes(0)),
            ));
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.shadow);
            for m in &casters {
                if let Some((vb, ib, n)) = &m.buffers {
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..*n, 0, 0..1);
                }
            }
            if let (Some((vb, ib)), false) = (&self.actors.buffers, self.actors.shadow.is_empty()) {
                pass.set_pipeline(&self.actor_shadow);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(self.actors.shadow.clone(), 0, 0..1);
            }
        }

        // 2. Sky, characters and terrain. Reverse Z: depth 0 is infinitely far away.
        {
            let color = color_attachment(&t.hdr_view, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
            let mut pass = encoder.begin_render_pass(&pass_desc(
                "scene",
                &color,
                depth_attachment(&t.depth_view, wgpu::LoadOp::Clear(0.0)),
                timer.and_then(|t| t.writes(1)),
            ));
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, &t.scene_bind, &[]);
            pass.set_pipeline(&self.sky);
            pass.draw(0..3, 0..1);
            // Characters first: they are near the camera and hide terrain behind them.
            if let (Some((vb, ib)), false) = (&self.actors.buffers, self.actors.scene.is_empty()) {
                pass.set_pipeline(&self.actor);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(self.actors.scene.clone(), 0, 0..1);
            }
            pass.set_pipeline(&self.terrain);
            for m in &visible {
                if let Some((vb, ib, n)) = &m.buffers {
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..*n, 0, 0..1);
                    triangles += n / 3;
                }
            }
            pass.set_pipeline(&self.far_terrain);
            for m in &far_visible {
                if let Some((vb, ib, n)) = &m.buffers {
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..*n, 0, 0..1);
                    far_triangles += n / 3;
                }
            }
        }

        // 3. Water, reading a snapshot of what is behind it. The pass always
        // runs (empty without water) so its timing slot is always written.
        let has_water = visible.iter().chain(&far_visible).any(|m| m.water.is_some());
        if has_water {
            let size = t.hdr.size();
            encoder.copy_texture_to_texture(t.hdr.as_image_copy(), t.scene_copy.as_image_copy(), size);
            encoder.copy_texture_to_texture(t.depth.as_image_copy(), t.depth_copy.as_image_copy(), size);
        }
        {
            let color = color_attachment(&t.hdr_view, wgpu::LoadOp::Load);
            let mut pass = encoder.begin_render_pass(&pass_desc(
                "water",
                &color,
                depth_attachment(&t.depth_view, wgpu::LoadOp::Load),
                timer.and_then(|t| t.writes(2)),
            ));
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, &t.scene_bind, &[]);
            pass.set_pipeline(&self.far_water);
            for m in &far_visible {
                if let Some((vb, ib, n)) = &m.water {
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..*n, 0, 0..1);
                    far_triangles += n / 3;
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
        }

        // 4. Tone mapping and the crosshair onto the screen.
        {
            let color = color_attachment(&view, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
            let mut pass = encoder.begin_render_pass(&pass_desc("post", &color, None, timer.and_then(|t| t.writes(3))));
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, &t.post_bind, &[]);
            pass.set_pipeline(&self.post);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&self.overlay);
            pass.draw(0..3, 0..1);
        }
        self.triangles_drawn = triangles;
        self.far_triangles_drawn = far_triangles;
        let start_readback = timer.is_some_and(|t| t.state.load(Ordering::Acquire) == 0);
        if let (Some(t), true) = (timer, start_readback) {
            encoder.resolve_query_set(&t.queries, 0..TIMED_PASSES * 2, &t.resolve, 0);
            encoder.copy_buffer_to_buffer(&t.resolve, 0, &t.readback, 0, t.resolve.size());
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
        if let (Some(t), true) = (&self.timer, start_readback) {
            t.state.store(1, Ordering::Release);
            let state = t.state.clone();
            t.readback.map_async(wgpu::MapMode::Read, .., move |r| {
                state.store(if r.is_ok() { 2 } else { 0 }, Ordering::Release);
            });
        }
    }

    /// Picks up the latest pass timings if they have arrived.
    fn read_timings(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Poll);
        let Some(t) = &self.timer else { return };
        if t.state.load(Ordering::Acquire) != 2 {
            return;
        }
        let ticks: Option<Vec<u64>> = t
            .readback
            .get_mapped_range(..)
            .ok()
            .map(|data| data.as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b)).collect());
        let ms: [f32; 4] = match &ticks {
            Some(ticks) if ticks.len() >= TIMED_PASSES as usize * 2 => std::array::from_fn(|i| {
                let (a, b) = (ticks[i * 2], ticks[i * 2 + 1]);
                b.saturating_sub(a) as f32 * t.period_ns / 1e6
            }),
            _ => [f32::NAN; 4],
        };
        t.readback.unmap();
        t.state.store(0, Ordering::Release);
        // Ignore nonsense from drivers that reset or quantise the counters.
        if ms.iter().any(|m| !m.is_finite() || *m > 1000.0) {
            return;
        }
        self.gpu_ms = Some(match self.gpu_ms {
            Some(old) => std::array::from_fn(|i| old[i] * 0.9 + ms[i] * 0.1),
            None => ms,
        });
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
