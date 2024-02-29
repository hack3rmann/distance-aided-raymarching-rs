#![macro_use]



pub mod distance;

use serde::{Serialize, Deserialize};
use glam::*;
use distance::*;



pub fn spherical_to_cartesian(radius: f32, theta: f32, phi: f32) -> Vec3 {
    radius * Vec3::new(
        phi.sin() * theta.sin(),
        phi.cos(),
        phi.sin() * theta.cos(),
    )
}

pub fn compute_normal(pos: Vec3, sdf: impl Sdf, epsilon: f32) -> Vec3 {
    let eps = vec2(epsilon, 0.0);

    vec3(
        (sdf(pos + eps.xyy()).distance - sdf(pos - eps.xyy()).distance) / epsilon,
        (sdf(pos + eps.yxy()).distance - sdf(pos - eps.yxy()).distance) / epsilon,
        (sdf(pos + eps.yyx()).distance - sdf(pos - eps.yyx()).distance) / epsilon,
    ).normalize()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitInfo {
    pub pos: Option<Vec3>,
    pub color: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[derive(Serialize, Deserialize)]
pub struct HitConfig {
    pub raymarch_settings: RayMarchSettings,
    pub ao_config: AoConfig,
    pub normal_eps: f32,
    pub normal_lift: f32,
    pub ambient: f32,
    pub fresnel_power: f32,
}

impl Default for HitConfig {
    fn default() -> Self {
        Self {
            raymarch_settings: RayMarchSettings {
                max_n_steps: 1000,
                max_distance: 1000.0,
                epsilon: 0.0001,
            },
            ao_config: AoConfig::default(),
            normal_eps: 0.001,
            normal_lift: 0.001,
            ambient: 0.1,
            fresnel_power: 1.8,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[derive(Serialize, Deserialize)]
pub struct AoConfig {
    pub len: f32,
    pub n_steps: u32,
    pub min_value: f32,
    pub power: f32,
}

impl Default for AoConfig {
    fn default() -> Self {
        Self {
            len: 0.2,
            n_steps: 20,
            min_value: 0.3,
            power: 0.3,
        }
    }
}

pub fn hit(
    mut ro: Vec3, mut rd: Vec3, to_light: Vec3,
    n_steps: usize, sdf: impl Sdf, cfg: &HitConfig,
) -> HitInfo {
    let mut stack = smallvec::SmallVec::<[_; 16]>::new();
    let mut hit_pos = ro;

    for i in 0..n_steps + 1 {
        let hit = raymarch(ro, rd, sdf, cfg.raymarch_settings);

        let Some(RaymarchHitInfo { pos, .. }) = hit else {
            stack.push((sky_color(rd, to_light), 0.0));
            continue;
        };

        if i == 0 {
            hit_pos = pos;
        }

        let normal = compute_normal(pos, sdf, cfg.normal_eps);
        
        let is_shadow = raymarch(
            pos + cfg.normal_lift * normal, to_light, sdf, cfg.raymarch_settings
        ).is_some();
        
        let albedo = sdf(pos).color;
        let brightness = if !is_shadow {
            f32::max(cfg.ambient, normal.dot(to_light))
        } else { cfg.ambient };

        let fresnel_factor = (1.0 - normal.dot(rd).abs())
            .clamp(0.0, 1.0)
            .powf(cfg.fresnel_power);
        
        let ao = compute_ambient_occlusion(
            pos, normal, cfg.ao_config.len / cfg.ao_config.n_steps as f32,
            cfg.ao_config.n_steps as usize, cfg.ao_config.min_value, cfg.ao_config.power, sdf,
        );

        stack.push((ao * brightness * albedo, fresnel_factor));

        ro = pos + cfg.normal_lift * normal;
        rd = reflect(rd, normal);
    }

    let (color, _) = stack.into_iter()
        .rev()
        .reduce(|(color, _), (diffuse, cur_factor)| (
            mix(diffuse, color, cur_factor),
            cur_factor,
        ))
        .unwrap();
    
    HitInfo {
        pos: Some(hit_pos),
        color,
    }
}

pub fn compute_ambient_occlusion(
    pos: Vec3, normal: Vec3, step_size: f32, n_steps: usize,
    min_value: f32, power: f32, sdf: impl Sdf,
) -> f32 {
    let unoccluded_value = step_size * (n_steps * (n_steps + 1) / 2) as f32;
    let occluded_value = (1..=n_steps)
        .map(|idx| pos + idx as f32 * step_size * normal)
        .map(sdf)
        .map(|dst| dst.distance.max(0.0))
        .sum::<f32>();

    min_value + (1.0 - min_value) * f32::powf(occluded_value / unoccluded_value, power)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[derive(Serialize, Deserialize)]
pub struct Camera {
    pub distance: f32,
    pub phi: f32,
    pub theta: f32,
    pub vfov: f32,
    pub target_pos: Vec3,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            distance: 2.0,
            phi: std::f32::consts::FRAC_PI_2,
            theta: 0.0,
            target_pos: Vec3::ZERO,
            vfov: 1.5 * std::f32::consts::FRAC_PI_3,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[derive(Serialize, Deserialize)]
pub struct RenderConfiguration {
    pub to_light: Vec3,
    pub n_bounces: u32,
    pub camera: Camera,
    pub hit_cfg: HitConfig,
    pub super_sample_angle: f32,
    pub division_eps: f32,
}

impl Default for RenderConfiguration {
    fn default() -> Self {
        Self {
            to_light: vec3(1.0, 0.5, -0.2).normalize(),
            n_bounces: 5,
            camera: Camera::default(),
            hit_cfg: HitConfig::default(),
            super_sample_angle: std::f32::consts::FRAC_PI_6,
            division_eps: 0.001,
        }
    }
}

pub fn get_color(
    screen_coord: Vec2, screen_width: usize, screen_height: usize,
    sdf: impl Sdf, cfg: &RenderConfiguration,
) -> Vec3 {
    let aspect_ratio = screen_height as f32 / screen_width as f32;
    
    let camera_pos = cfg.camera.target_pos + spherical_to_cartesian(
        cfg.camera.distance, cfg.camera.theta, cfg.camera.phi,
    );
    let camera_direction = Vec3::normalize(cfg.camera.target_pos - camera_pos);
    let camera_tangent = -cfg.camera.theta.sin() * Vec3::Z + cfg.camera.theta.cos() * Vec3::X;
    let camera_bitangent = Vec3::cross(camera_direction, camera_tangent);

    let fov_tan = f32::tan(0.5 * cfg.camera.vfov);
    let ray_direction = Vec3::normalize(camera_direction
        + (screen_coord.x / aspect_ratio) * fov_tan * camera_tangent
        + screen_coord.y * fov_tan * camera_bitangent
    );
    let ray_origin = camera_pos;

    let rotation = Mat3::from_axis_angle(ray_direction, cfg.super_sample_angle);

    let pixel_size = 2.0 / screen_height as f32;
    let offset_x = 0.25 * pixel_size * rotation * camera_tangent;
    let offset_y = 0.25 * pixel_size * rotation * camera_bitangent;

    let n_bounces = cfg.n_bounces as usize;
    
    let mut color
           = hit(ray_origin, ray_direction + offset_x, cfg.to_light, n_bounces, sdf, &cfg.hit_cfg).color;
    color += hit(ray_origin, ray_direction - offset_x, cfg.to_light, n_bounces, sdf, &cfg.hit_cfg).color;
    color += hit(ray_origin, ray_direction + offset_y, cfg.to_light, n_bounces, sdf, &cfg.hit_cfg).color;
    color += hit(ray_origin, ray_direction - offset_y, cfg.to_light, n_bounces, sdf, &cfg.hit_cfg).color;

    0.25 * color
}

pub fn reflect(dir: Vec3, normal: Vec3) -> Vec3 {
    dir - 2.0 * dir.dot(normal) * normal
}

pub fn mix<V>(lhs: V, rhs: V, param: f32) -> V
where
    V: std::ops::Mul<f32, Output = V>,
    V: std::ops::Add<V, Output = V>,
{
    rhs * param + lhs * (1.0 - param)
}

pub fn sky_color(direction: Vec3, to_light: Vec3) -> Vec3 {
    let sunness = direction.dot(to_light);

    let mid_color = vec3(0.61, 0.72, 0.82);
    let top_color = vec3(0.2, 0.49, 0.75);

    let sky_mix = 1.0 - (1.0 - direction.y.abs()).powi(10);
    let sky_color = mix(mid_color, top_color, sky_mix);
    let sun_param = 0.01;
    
    if sunness >= 1.0 - sun_param {
        let sun_color = vec3(1.0, 1.0, 0.88);
        let amount = (sunness + sun_param - 1.0) / sun_param;
        return mix(sky_color, sun_color, amount.powf(1.0 - sun_param))
    }

    if 0.0 <= direction.y {
        sky_color
    } else {
        mix(mid_color, 0.2 * Vec3::ONE, sky_mix)
    }
}