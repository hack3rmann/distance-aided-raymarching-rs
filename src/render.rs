use rayon::prelude::*;
use csg::{render::{distance::Sdf, RenderConfiguration}, scene::Scene};
use glam::*;
use wgpu::util::DeviceExt;
use crate::{compute::ComputeContext, benchmark::Benchmark, util::default};
use std::time::{Duration, Instant};



pub fn render_image_cpu_multithreaded<C, S>(
    screen_width: usize, screen_height: usize, get_color: C,
    distance: S, cfg: &RenderConfiguration, bench: Option<&mut Benchmark>,
) -> Vec<u8>
where
    S: Sdf,
    C: Fn(Vec2, usize, usize, S, &RenderConfiguration) -> Vec3 + Send + Sync,
{
    let mut image
        = Vec::with_capacity(screen_width * screen_height);

    let render_start = Instant::now();

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

    let render_time = Instant::now().duration_since(render_start);

    if let Some(bench) = bench {
        bench.render = render_time;
        bench.copy = Duration::from_nanos(0);
    }

    let image = image.into_boxed_slice();
    let image: Box<[u8]> = bytemuck::allocation::cast_slice_box(image);
    
    image.into_vec()
}

pub fn render_image_cpu_singlethreaded<C, S>(
    screen_width: usize, screen_height: usize, get_color: C,
    distance: S, cfg: &RenderConfiguration, bench: Option<&mut Benchmark>,
) -> Vec<u8>
where
    S: Sdf,
    C: Fn(Vec2, usize, usize, S, &RenderConfiguration) -> Vec3 + Send + Sync,
{
    let render_start = Instant::now();

    let image = (0..screen_width * screen_height)
        .map(|i| (i % screen_width, i / screen_width))
        .map(|(x, y)| (
            (2 * x as i32 - screen_width as i32 + 1) as f32 / screen_width as f32,
            (2 * y as i32 - screen_height as i32 + 1) as f32 / screen_height as f32,
        ))
        .map(|(x, y)| get_color(
            vec2(x, y), screen_width, screen_height, distance, cfg,
        ))
        .flat_map(compact_color)
        .collect();

    let render_time = Instant::now().duration_since(render_start);

    if let Some(bench) = bench {
        bench.render = render_time;
        bench.copy = Duration::from_nanos(0);
    }

    image
}

pub fn render_image_gpu(
    screen_width: usize, screen_height: usize, scene: &Scene,
    cfg: &RenderConfiguration, context: &ComputeContext,
    bench: Option<&mut Benchmark>,
) -> Vec<u8> {
    use wgpu::include_wgsl;

    const COMPUTE_BLOCK_SIZE: u64 = 512;
    const COMPUTE_WORKGROUP_SIZE: u64 = 8;
    const COMPUTE_VOLUME: u64 = COMPUTE_BLOCK_SIZE * COMPUTE_BLOCK_SIZE;

    let shader = context.device.create_shader_module(include_wgsl!("shaders/compute_image.wgsl"));

    let cfg_copy_time = Instant::now();

    let render_uniform_buffer = context.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("render_configuration"),
            contents: bytemuck::bytes_of(cfg),
            usage: wgpu::BufferUsages::UNIFORM,
        },
    );

    let cfg_copy_time = Instant::now().duration_since(cfg_copy_time);

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
    struct CallUniform {
        screen_width: u32,
        screen_height: u32,
        y_offset: u32,
        index: u32,
    }

    let call_data_uniform_buffer_gpu = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("call_data"),
        size: std::mem::size_of::<CallUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let call_data_uniform_buffer_cpu = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("call_data_cpu"),
        size: std::mem::size_of::<CallUniform>() as u64,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let scene_copy_time = Instant::now();

    let scene_bytes = scene.as_bytes();

    let scene_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("scene_buffer"),
        contents: bytemuck::cast_slice(&scene_bytes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let scene_copy_time = Instant::now().duration_since(scene_copy_time);

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
                resource: call_data_uniform_buffer_gpu.as_entire_binding(),
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

    const TIMESTAMP_SIZE: u64 = wgpu::QUERY_SIZE as u64;
    const QUERY_SIZE: u64 = 4;

    let query = context.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: QUERY_SIZE as u32,
    });

    let query_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamps"),
        size: wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT * n_compute_blocks,
        usage: wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });

    let cpu_query_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cpu_timestamps"),
        size: TIMESTAMP_SIZE * QUERY_SIZE * n_compute_blocks,
        usage: wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let full_compute_time = Instant::now();

    for i in 0..n_compute_blocks {
        call_data_uniform_buffer_cpu.slice(..)
            .map_async(wgpu::MapMode::Write, Result::unwrap);

        context.device.poll(wgpu::Maintain::Wait);

        call_data_uniform_buffer_cpu.slice(..)
            .get_mapped_range_mut()
            .swap_with_slice(bytemuck::bytes_of_mut(&mut CallUniform {
                screen_width: screen_width as u32,
                screen_height: screen_height as u32,
                y_offset: (i * compute_block_height) as u32,
                index: i as u32,
            }));

        call_data_uniform_buffer_cpu.unmap();

        let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{i}_compute_pass_encoder")),
        });

        encoder.write_timestamp(&query, 0);

        encoder.copy_buffer_to_buffer(
            &call_data_uniform_buffer_cpu,
            0,
            &call_data_uniform_buffer_gpu,
            0,
            std::mem::size_of::<CallUniform>() as u64,
        );

        encoder.write_timestamp(&query, 1);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{i}_compute_pass")),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.insert_debug_marker(&format!("{i}_compute_pass"));
            pass.dispatch_workgroups(
                COMPUTE_BLOCK_SIZE as u32 / COMPUTE_WORKGROUP_SIZE as u32,
                COMPUTE_BLOCK_SIZE as u32 / COMPUTE_WORKGROUP_SIZE as u32,
                1,
            );
        }

        encoder.write_timestamp(&query, 2);

        encoder.copy_buffer_to_buffer(
            &gpu_image_buffer,
            0,
            &cpu_image_buffer,
            i * COMPUTE_VOLUME * image_buffer_elem_size as u64,
            COMPUTE_VOLUME * image_buffer_elem_size as u64,
        );

        encoder.write_timestamp(&query, 3);

        encoder.resolve_query_set(
            &query,
            0..QUERY_SIZE as u32,
            &query_buffer,
            wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT * i,
        );

        context.device.poll(wgpu::Maintain::wait_for(
            context.queue.submit([encoder.finish()])
        ));
    }

    let full_compute_time = Instant::now().duration_since(full_compute_time);

    {
        let mut encoder = context.device.create_command_encoder(&default());

        for i in 0..n_compute_blocks {
            encoder.copy_buffer_to_buffer(
                &query_buffer,
                i * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,
                &cpu_query_buffer,
                i * TIMESTAMP_SIZE * QUERY_SIZE,
                TIMESTAMP_SIZE * QUERY_SIZE,
            );
        }

        context.device.poll(wgpu::Maintain::wait_for(
            context.queue.submit([encoder.finish()])
        ));
    }

    let buffer_map_time = Instant::now();

    cpu_query_buffer.slice(..).map_async(wgpu::MapMode::Read, Result::unwrap);
    cpu_image_buffer.slice(..).map_async(wgpu::MapMode::Read, Result::unwrap);

    context.device.poll(wgpu::Maintain::Wait);

    let buffer_map_time = Instant::now().duration_since(buffer_map_time);

    let compute_bench = {
        let buf_range = cpu_query_buffer.slice(..).get_mapped_range();

        let timestamps: &[u64] = bytemuck::cast_slice(&buf_range[..]);

        let period = context.queue.get_timestamp_period();

        timestamps.chunks(4)
            .map(|pass| {
                let &[start, call_copy, compute, image_copy] = pass else {
                    panic!("failed to parse timestamp query result")
                };

                let copy = call_copy - start + image_copy - compute;
                let render = compute - call_copy;

                Benchmark {
                    render: Duration::from_nanos((render as f32 * period) as u64),
                    copy: Duration::from_nanos((copy as f32 * period) as u64),
                }
            })
            .reduce(|accum, elem| Benchmark {
                render: accum.render + elem.render,
                copy: accum.copy + elem.copy,
            })
            .expect("failed to sum benches")
    };

    context.device.poll(wgpu::Maintain::Wait);

    let view = cpu_image_buffer.slice(..).get_mapped_range();

    let vec_copy_time = Instant::now();
    let result = view.to_vec();
    let vec_copy_time = Instant::now().duration_since(vec_copy_time);

    if let Some(bench) = bench {
        bench.render = compute_bench.render;
        bench.copy = full_compute_time - compute_bench.render
            + compute_bench.copy
            + vec_copy_time
            + buffer_map_time
            + cfg_copy_time
            + scene_copy_time;
    }

    result
}

pub fn compact_color(mut color: Vec3) -> [u8; 4] {
    color *= 255.0;

    color.x = csg::geometry::clamp(color.x, 0.0, 255.0);
    color.y = csg::geometry::clamp(color.y, 0.0, 255.0);
    color.z = csg::geometry::clamp(color.z, 0.0, 255.0);

    [
        color.x as u8,
        color.y as u8,
        color.z as u8,
        0xFF,
    ]
}