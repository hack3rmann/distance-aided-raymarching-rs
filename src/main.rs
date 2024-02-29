pub mod compute;
pub mod render;
pub mod util;
pub mod benchmark;

use compute::{ComputeContext, ComputeContextMode};
use render::*;

use std::{error::Error, path::Path};
use clap::Parser;
use csg::{geometry::*, scene::Scene, render::{get_color, RenderConfiguration}};
use glam::*;
use thiserror::Error;
use util::*;
use benchmark::Benchmark;



const DEFAULT_SCREEN_WIDTH: usize = 2 * 512;
const DEFAULT_SCREEN_HEIGHT: usize = 512;
const RENDER_CFG_FILE: &str = "render_config.toml";
const DEFAULT_OUT_FILE_NAME: &str = "out.png";



#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let cli_args = CliArgs::parse();

    let result_name = cli_args.out;

    let screen_width = cli_args.scale * cli_args.width;    
    let screen_height = cli_args.scale * cli_args.height;

    assert!(cli_args.scale.is_power_of_two(), "image scale should be a power of 2");
    assert!(screen_width % 512 == 0, "image width should be a multiple of 512");
    assert!(screen_height % 512 == 0, "image height should be a multiple of 512");

    let geometry = geometry();

    let render_cfg = match fetch_render_cfg(&cli_args.cfg).await {
        Ok(cfg) => cfg,
        Err(FetchRenderCfgError::IoError(..)) => {
            log::error!(
                "{} not found, using default configuration",
                &cli_args.cfg,
            );
            default()
        },
        Err(err) => panic!("failed to configurate rendering: {err}"),
    };

    let mut bench = Benchmark::new();

    let image = match cli_args.r#type {
        ComputationType::Gpu => {
            let context = ComputeContext::new(cli_args.mode).await?;
            let scene = Scene::from(&geometry);
        
            render_image_gpu(
                screen_width, screen_height, &scene,
                &render_cfg, &context, Some(&mut bench),
            )
        },
        ComputationType::MultiCpu => {
            render::render_image_cpu_multithreaded(
                screen_width, screen_height, get_color,
                geometry.sdf(), &render_cfg, Some(&mut bench),
            )
        },
        ComputationType::SingleCpu => {
            render::render_image_cpu_singlethreaded(
                screen_width, screen_height, get_color,
                geometry.sdf(), &render_cfg, Some(&mut bench),
            )
        },
    };

    if cli_args.bench {
        println!("{bench}");
    }

    let file = std::fs::File::create(&result_name)?;

    let buf_writer = std::io::BufWriter::new(file);

    let mut encoder = png::Encoder::new(
        buf_writer, screen_width as u32, screen_height as u32,
    );

    encoder.set_color(png::ColorType::Rgba);

    let mut writer = encoder.write_header()?;
    writer.write_image_data(bytemuck::cast_slice(&image))?;

    Ok(())
}

pub fn geometry() -> impl Geometry {
    csg::union! {
        Geometry::smooth_union(
            Geometry::smooth_union(
                Geometry::subtract(
                    Mandelbulb::new(Vec3::ONE, 1.2 * Vec3::Y, 20, 4.0),
                    csg::union! {
                        Geometry::subtract(
                            Torus::new(Vec3::X, vec3(0.0, 0.0, 0.0), 0.8, 0.2),
                            Torus::new(Vec3::X, vec3(0.0, 0.0, 0.0), 0.6, 0.2),
                        ),
                        Geometry::smooth_union(
                            Ball::new(Vec3::ONE, vec3(0.0, 0.4, 0.0), 0.3),
                            Ball::new(0.8 * Vec3::ONE, vec3(0.0, 1.0, 0.0), 0.6),
                            0.4,
                        ),
                    },
                ),
                StraightPrism::new(vec3(0.56, 1.0, 0.59), Vec3::ZERO, 0.3 * Vec3::ONE),
                0.25,
            ),
            Geometry::smooth_union(
                Ball::new(Vec3::ONE, vec3(0.0, -0.4, 0.0), 0.3),
                Ball::new(vec3(0.71, 0.56, 0.79), vec3(0.0, -1.0, 0.0), 0.5),
                0.4,
            ),
            0.25,
        ),
        Geometry::subtract(
            Ball::new(vec3(0.23, 0.72, 1.0), Vec3::ZERO, 1.5),
            StraightPrism::new(0.8 * Vec3::ONE, Vec3::ZERO, 1.2 * Vec3::ONE),
        ),
    }
}

pub async fn fetch_render_cfg(path: impl AsRef<Path>)
    -> Result<RenderConfiguration, FetchRenderCfgError>
{
    let cfg_string = tokio::fs::read_to_string(path).await?;
    let cfg = toml::from_str(&cfg_string)?;
    Ok(cfg)
}

#[derive(Debug, Error)]
pub enum FetchRenderCfgError {
    #[error("failed to read configuraion file")]
    IoError(#[from] tokio::io::Error),

    #[error("failed to parse configuration file")]
    ParseError(#[from] toml::de::Error),
}

#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
#[allow(rustdoc::invalid_html_tags)]
struct CliArgs {
    /// Name of the output image file
    #[arg(short, long, default_value_t = DEFAULT_OUT_FILE_NAME.into())]
    out: String,

    /// Width of the output image, must be a multiple of 512
    #[arg(long, default_value_t = DEFAULT_SCREEN_WIDTH)]
    width: usize,

    /// Height of the output image, must be a multiple of 512
    #[arg(long, default_value_t = DEFAULT_SCREEN_HEIGHT)]
    height: usize,

    /// Increases resolution by <SCALE> times, should be a power of 2
    #[arg(long, short, default_value_t = 4)]
    scale: usize,

    /// Determines computation method, valid values are 
    /// 'gpu' (for computation on GPU), 
    /// 'multicpu' (for multithreaded CPU computation), 
    /// 'singlecpu' (for computation on single thread on CPU)
    #[arg(long, short, default_value_t = ComputationType::Gpu)]
    r#type: ComputationType,

    /// The name of configuraion TOML file
    #[arg(long, default_value_t = RENDER_CFG_FILE.into())]
    cfg: String,

    /// Enables or disables debug and validation backend. 
    /// Valid values are: 'debug', 'validation', 'silent'
    #[arg(long, short, default_value_t = ComputeContextMode::ReleaseSilent)]
    mode: ComputeContextMode,

    /// Enables benchmarking
    #[arg(long, short)]
    bench: bool,
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

    #[tokio::test]
    async fn default_render_config() {
        let cfg = RenderConfiguration::default();
        let cfg_string = toml::ser::to_string(&cfg).unwrap();

        tokio::fs::write("render_config.toml", &cfg_string).await.unwrap();
    }
}
