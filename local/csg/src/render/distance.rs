#![macro_use]



use glam::*;



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceInfo {
    pub distance: f32,
    pub color: Vec3
}

impl PartialOrd for DistanceInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}



pub trait Sdf: Fn(Vec3) -> DistanceInfo + Copy + Send + Sync { }
impl<F: Fn(Vec3) -> DistanceInfo + Copy + Send + Sync> Sdf for F { }



pub fn mandelbulb_sdf(color: Vec3, n_steps: usize, power: f32) -> impl Sdf {
    move |pos| {
        let mut z = pos;
        let mut dr = 1.0;
        let mut r = 0.0;

        for _ in 0..n_steps {
            r = z.length();

            if r > 4.0 {
                break;
            }

            let mut theta = f32::acos(z.z / r);
            let mut phi = f32::atan2(z.y, z.x);
            dr = r.powf(power - 1.0) * power * dr + 1.0;

            let zr = r.powf(power);
            theta *= power;
            phi *= power;

            z = zr * vec3(
                theta.sin() * phi.cos(),
                phi.sin() * theta.sin(),
                theta.cos(),
            );

            z += pos;
        }

        DistanceInfo {
            distance: 0.5 * r.ln() * r / dr,
            color,
        }
    }
}



#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RayMarchSettings {
    pub max_n_steps: u32,
    pub max_distance: f32,
    pub epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaymarchHitInfo {
    pub n_steps: u32,
    pub distance: f32,
    pub pos: Vec3,
}

pub fn raymarch(
    ro: Vec3, rd: Vec3, sdf: impl Sdf, settings: RayMarchSettings,
) -> Option<RaymarchHitInfo> {
    let mut pos = ro;
    let mut distance = 0.0;

    for i in 0..settings.max_n_steps {
        let displacement = sdf(pos).distance;
        let next_pos = pos + rd * displacement;

        if displacement < settings.epsilon {
            return Some(RaymarchHitInfo {
                n_steps: i + 1,
                pos: next_pos,
                distance: distance + displacement,
            });
        }

        pos = next_pos;
        distance += displacement;

        if distance > settings.max_distance {
            return None;
        }
    }

    Some(RaymarchHitInfo { n_steps: settings.max_n_steps, pos, distance })
}