use rayon::prelude::*;
use csg::{render::{distance::Sdf, RenderConfiguration}, scene::Scene};
use glam::*;
use wgpu::util::DeviceExt;
use crate::{compute::ComputeContext, util::*};



pub fn render_image_cpu_multithreaded<C, S>(
    screen_width: usize, screen_height: usize, get_color: C,
    distance: S, cfg: &RenderConfiguration,
) -> Vec<u8>
where
    S: Sdf,
    C: Fn(Vec2, usize, usize, S, &RenderConfiguration) -> Vec3 + Send + Sync,
{
    let mut image
        = Vec::with_capacity(screen_width * screen_height);

    (0..screen_width * screen_height)
        .into_par_iter()
        .map(|i| (i % screen_width, i / screen_width))
        .map(|(x, y)| (
            ((2 * x) as f32 + 0.5) / (screen_width  - 1) as f32 - 1.0,
            ((2 * y) as f32 + 0.5) / (screen_height - 1) as f32 - 1.0,
        ))
        .map(|(x, y)| get_color(
            vec2(x, y), screen_width, screen_height, distance, cfg,
        ))
        .map(compact_color)
        .collect_into_vec(&mut image);

    let image = image.into_boxed_slice();
    let image: Box<[u8]> = bytemuck::allocation::cast_slice_box(image);
    
    image.into_vec()
}

pub fn render_image_cpu_singlethreaded<C, S>(
    screen_width: usize, screen_height: usize, get_color: C,
    distance: S, cfg: &RenderConfiguration,
) -> Vec<u8>
where
    S: Sdf,
    C: Fn(Vec2, usize, usize, S, &RenderConfiguration) -> Vec3 + Send + Sync,
{
    (0..screen_width * screen_height)
        .map(|i| (i % screen_width, i / screen_width))
        .map(|(x, y)| (
            (2 * x as i32 - screen_width as i32 + 1) as f32 / screen_width as f32,
            (2 * y as i32 - screen_height as i32 + 1) as f32 / screen_height as f32,
        ))
        .map(|(x, y)| get_color(
            vec2(x, y), screen_width, screen_height, distance, cfg,
        ))
        .flat_map(compact_color)
        .collect()
}

pub fn render_image_gpu(
    screen_width: usize, screen_height: usize, scene: &Scene,
    cfg: &RenderConfiguration, context: &ComputeContext,
) -> Vec<u8> {
    use wgpu::include_wgsl;

    const COMPUTE_BLOCK_SIZE: u64 = 512;
    const COMPUTE_WORKGROUP_SIZE: u64 = 8;

    let shader = context.device.create_shader_module(include_wgsl!("shaders/compute_image.wgsl"));

    let render_uniform_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("render_configuration"),
        contents: bytemuck::bytes_of(cfg),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct CallUniform {
        screen_width: u32,
        screen_height: u32,
        y_offset: u32,
        index: u32,
    }

    let call_data_uniform_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("call_data"),
        size: std::mem::size_of::<CallUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let scene_bytes = scene.as_bytes();
    let scene_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scene_buffer"),
        contents: bytemuck::cast_slice(&scene_bytes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let image_buffer_len = screen_width * screen_height;
    let image_buffer_elem_size = std::mem::size_of::<u32>();
    let gpu_image_buffer_desc = wgpu::BufferDescriptor {
        label: Some("image_buffer"),
        size: image_buffer_elem_size as u64 * COMPUTE_BLOCK_SIZE.pow(2),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    };

    let gpu_image_buffer = context.device.create_buffer(&gpu_image_buffer_desc);
    let cpu_image_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        size: (image_buffer_elem_size * image_buffer_len) as u64,
        ..gpu_image_buffer_desc
    });

    let group_layout = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute_data_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_data"),
        layout: &group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu_image_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scene_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: render_uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: call_data_uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute_pipeline_layout_descriptor"),
        bind_group_layouts: &[&group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline_descriptor"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "compute_image",
    });

    let n_compute_blocks = image_buffer_len as u64
        / (COMPUTE_BLOCK_SIZE * COMPUTE_BLOCK_SIZE);

    let compute_block_height = screen_height as u64 / n_compute_blocks;

    for i in 0..n_compute_blocks {
        let mut encoder = context.device.create_command_encoder(&default());

        context.queue.write_buffer(
            &call_data_uniform_buffer, 0, bytemuck::bytes_of(&CallUniform {
                screen_width: screen_width as u32,
                screen_height: screen_height as u32,
                y_offset: (i * compute_block_height) as u32,
                index: i as u32,
            }),
        );

        context.device.poll(wgpu::Maintain::Wait);

        {
            let mut pass = encoder.begin_compute_pass(&default());

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(
                COMPUTE_BLOCK_SIZE as u32 / COMPUTE_WORKGROUP_SIZE as u32,
                COMPUTE_BLOCK_SIZE as u32 / COMPUTE_WORKGROUP_SIZE as u32,
                1,
            );
        }

        let compute_volume = COMPUTE_BLOCK_SIZE * COMPUTE_BLOCK_SIZE;

        encoder.copy_buffer_to_buffer(
            &gpu_image_buffer,
            0,
            &cpu_image_buffer,
            i * compute_volume * image_buffer_elem_size as u64,
            compute_volume * image_buffer_elem_size as u64,
        );

        context.queue.submit([encoder.finish()]);
    }

    cpu_image_buffer.slice(..).map_async(wgpu::MapMode::Read, Result::unwrap);

    context.device.poll(wgpu::Maintain::Wait);

    let view = cpu_image_buffer.slice(..).get_mapped_range();

    view.to_vec()
}

pub fn compact_color(mut color: Vec3) -> [u8; 4] {
    color *= 255.0;

    [
        color.x as u8,
        color.y as u8,
        color.z as u8,
        0xFF,
    ]
}