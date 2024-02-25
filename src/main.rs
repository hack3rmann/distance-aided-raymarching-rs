#![allow(unused)]



pub mod compute;
pub mod render;
pub mod util;

use compute::{ComputeContext, ComputeContextMode};
use render::*;

use std::{error::Error, io::Read, sync::Arc, path::Path};
use itertools::Itertools;
use wgpu::{include_wgsl, BindGroupDescriptor};
use clap::Parser;
use csg::{geometry::{self, *}, scene::Scene};
use glam::*;
use csg::render::{get_color, RenderConfiguration};



fn default<T: Default>() -> T {
    T::default()
}



const DEFAULT_SCREEN_WIDTH: usize = 2520;
const DEFAULT_SCREEN_HEIGHT: usize = 1680;



#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let cli_args = CliArgs::parse();

    let result_name = cli_args.out.unwrap_or_else(|| String::from("out.png"));

    let screen_width = cli_args.scale
        * cli_args.width.unwrap_or(DEFAULT_SCREEN_WIDTH) / cli_args.low;
    
    let screen_height = cli_args.scale
        * cli_args.height.unwrap_or(DEFAULT_SCREEN_HEIGHT) / cli_args.low;

    let geometry = csg::union! {
        Geometry::intersect(
            Ball::new(Vec3::X, 0.4 * Vec3::X, 0.8),
            Ball::new(Vec3::X, 0.0 * Vec3::X, 0.6),
        ),
        Geometry::smooth_union(
            Ball::new(Vec3::Y, -0.6 * Vec3::X, 0.4),
            Ball::new(Vec3::Z, -1.0 * Vec3::X + 0.4 * Vec3::Y, 0.2),
            0.25,
        ),
        Geometry::subtract(
            Ball::new(Vec3::X, 1.5 * Vec3::X + 0.5 * Vec3::Y, 0.5),
            Ball::new(Vec3::Y, 1.5 * Vec3::X, 0.5),
        ),
    };

    let render_cfg = RenderConfiguration::default();

    let image = match cli_args.r#type {
        ComputationType::Gpu => {
            let context = ComputeContext::new(ComputeContextMode::ReleaseSilent).await?;
            let scene = Scene::from(&geometry);
        
            render_image_gpu(
                screen_width, screen_height, &scene, &render_cfg, &context,
            )
        },
        ComputationType::MultiCpu => {
            render::render_image_cpu_multithreaded(
                screen_width, screen_height, get_color, geometry.sdf(), &render_cfg,
            )
        },
        ComputationType::SingleCpu => {
            render::render_image_cpu_singlethreaded(
                screen_width, screen_height, get_color, geometry.sdf(), &render_cfg,
            )
        },
    };

    let file = std::fs::File::create(&result_name)?;

    let mut buf_writer = std::io::BufWriter::new(file);

    let mut encoder = png::Encoder::new(
        buf_writer, screen_width as u32, screen_height as u32,
    );

    encoder.set_color(png::ColorType::Rgba);

    let mut writer = encoder.write_header()?;
    writer.write_image_data(bytemuck::cast_slice(&image))?;

    Ok(())
}

#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct CliArgs {
    /// Name of the output image file
    #[arg(short, long)]
    out: Option<String>,

    /// Width of the output image
    #[arg(long)]
    width: Option<usize>,

    /// Height of the output image
    #[arg(long)]
    height: Option<usize>,

    /// Do parallel image rendering
    #[arg(long, default_value_t = false)]
    do_parallel: bool,

    /// Decreases resolution by <LOW> times
    #[arg(long, short, default_value_t = 1)]
    low: usize,

    /// Increases resolution <SCALE> times
    #[arg(long, short, default_value_t = 1)]
    scale: usize,

    /// Determines computing method, valid values are 'gpu' (for computation on GPU), 
    /// 'multicpu' (for multithreaded CPU computation), 
    /// 'singlecpu' (for computation on single thread on CPU)
    #[arg(long, short, default_value_t = ComputationType::Gpu)]
    r#type: ComputationType,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ComputationType {
    SingleCpu,
    MultiCpu,
    Gpu,
}

impl std::fmt::Display for ComputationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Gpu => "gpu",
            Self::MultiCpu => "multicpu",
            Self::SingleCpu => "singlecpu",
        })
    }
}

impl std::str::FromStr for ComputationType {
    type Err = ParseComputationTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gpu" => Ok(Self::Gpu),
            "multicpu" => Ok(Self::MultiCpu),
            "singlecpu" => Ok(Self::SingleCpu),
            _ => Err(ParseComputationTypeError::FailedToParse(s.to_owned())),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ParseComputationTypeError {
    #[error("unresolved computation type '{0}', allowed values are: 'gpu', 'multicpu', 'singlecpu'")]
    FailedToParse(String),
}

#[cfg(test)]
mod tests {
    use csg::{geometry::*, scene::*};
    use super::*;

    #[test]
    fn scene_to_bytes() {
        let geometry = Geometry::union(
            StraightPrism::new(Vec3::ONE, Vec3::ZERO, Vec3::ONE),
            Ball::new(Vec3::ONE, Vec3::ZERO, 0.1),
        );

        let scene = Scene::from(&geometry);

        let bytes = scene.as_bytes();

        eprintln!("{:?}", &bytes);
    }
}