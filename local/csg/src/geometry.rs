#![macro_use]



use crate::render::distance::{Sdf, DistanceInfo};
use crate::render::mix;
use crate::scene::{Scene, SceneNode, ObjectData, CsgOp};
use glam::*;



pub trait Geometry: Sized + Send + Sync {
    fn sdf(&self) -> impl Sdf;

    fn union<G: Geometry>(self, other: G) -> Union<Self, G> {
        Union { first: self, second: other }
    }

    fn intersect<G: Geometry>(self, other: G) -> Intersection<Self, G> {
        Intersection { first: self, second: other }
    }

    fn smooth_union<G: Geometry>(self, other: G, param: f32) -> SmoothUnion<Self, G> {
        SmoothUnion { first: self, second: other, param }
    }

    fn smooth_intersection<G: Geometry>(self, other: G, param: f32) -> SmoothIntersection<Self, G> {
        SmoothIntersection { first: self, second: other, param }
    }

    fn compliment(self) -> Compliment<Self> {
        Compliment { geometry: self }
    }

    fn subtract<G: Geometry>(self, other: G) -> Difference<Self, G> {
        Difference { left: self, right: other }
    }

    fn smooth_subtract<G: Geometry>(self, other: G, param: f32) -> SmoothDifference<Self, G> {
        SmoothDifference { left: self, right: other, param }
    }

    fn insert_to_scene(&self, scene: &mut Scene);
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ball {
    pub color: Vec3,
    pub centre: Vec3,
    pub radius: f32,
}

impl Ball {
    pub const fn new(color: Vec3, centre: Vec3, radius: f32) -> Self {
        Self { color, radius, centre }
    }
}

impl Geometry for Ball {
    fn sdf(&self) -> impl Sdf {
        move |pos| DistanceInfo {
            distance: Vec3::length(self.centre - pos) - self.radius,
            color: self.color,
        }
    }

    fn insert_to_scene(&self, scene: &mut Scene) {
        scene.nodes.push(SceneNode::Object {
            color: self.color,
            offset: self.centre,
            data: ObjectData::Ball { radius: self.radius },
        });
    }
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Semispace {
    pub color: Vec3,
    pub normal: Vec3,
    pub dist: f32,
}

impl Semispace {
    pub const fn new(color: Vec3, normal: Vec3, dist: f32) -> Self {
        Self { color, normal, dist }
    }
}

impl Geometry for Semispace {
    fn sdf(&self) -> impl Sdf {
        move |pos| DistanceInfo {
            distance: pos.dot(self.normal) - self.dist,
            color: self.color,
        }
    }

    fn insert_to_scene(&self, scene: &mut Scene) {
        scene.nodes.push(SceneNode::Object {
            color: self.color,
            offset: Vec3::ZERO,
            data: ObjectData::Semispace {
                normal: self.normal,
                distance: self.dist,
            },
        });
    }
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StraightPrism {
    pub color: Vec3,
    pub offset: Vec3,
    pub sizes: Vec3,
}

impl StraightPrism {
    pub const fn new(color: Vec3, offset: Vec3, sizes: Vec3) -> Self {
        Self { color, offset, sizes }
    }
}

impl Geometry for StraightPrism {
    fn sdf(&self) -> impl Sdf {
        move |pos| {
            let q = Vec3::abs(pos - self.offset) - self.sizes;
            let distance = q.max(Vec3::ZERO).length()
                + q.x.max(q.y.max(q.z)).min(0.0);

            DistanceInfo {
                distance, color: self.color,
            }
        }
    }

    fn insert_to_scene(&self, scene: &mut Scene) {
        scene.nodes.push(SceneNode::Object {
            color: self.color,
            offset: self.offset,
            data: ObjectData::StraightPrism { sizes: self.sizes },
        });
    }
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Torus {
    pub color: Vec3,
    pub offset: Vec3,
    pub inner_radius: f32,
    pub outer_radius: f32,
}

impl Torus {
    pub const fn new(color: Vec3, offset: Vec3, inner_radius: f32, outer_radius: f32) -> Self {
        Self { color, offset, inner_radius, outer_radius }
    }
}

impl Geometry for Torus {
    fn sdf(&self) -> impl Sdf {
        move |pos| {
            let q = vec2(
                (pos - self.offset).xz().length() - self.outer_radius,
                pos.y - self.offset.y,
            );

            DistanceInfo {
                distance: q.length() - self.inner_radius,
                color: self.color,
            }
        }
    }

    fn insert_to_scene(&self, scene: &mut Scene) {
        scene.nodes.push(SceneNode::Object {
            color: self.color,
            offset: self.offset,
            data: ObjectData::Torus {
                inner_radius: self.inner_radius,
                outer_radius: self.outer_radius,
            },
        });
    }
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mandelbulb {
    pub color: Vec3,
    pub offset: Vec3,
    pub n_steps: usize,
    pub power: f32,
}

impl Mandelbulb {
    pub const fn new(color: Vec3, offset: Vec3, n_steps: usize, power: f32) -> Self {
        Self { color, offset, n_steps, power }
    }
}

impl Geometry for Mandelbulb {
    fn sdf(&self) -> impl Sdf {
        move |pos| {
            let mut z = pos - self.offset;
            let mut dr = 1.0;
            let mut r = 0.0;

            for _ in 0..self.n_steps {
                r = z.length();

                if r > 4.0 {
                    break;
                }

                let mut theta = f32::acos(z.z / r);
                let mut phi = f32::atan2(z.y, z.x);
                dr = r.powf(self.power - 1.0) * self.power * dr + 1.0;

                let zr = r.powf(self.power);
                theta *= self.power;
                phi *= self.power;

                z = zr * vec3(
                    theta.sin() * phi.cos(),
                    phi.sin() * theta.sin(),
                    theta.cos(),
                );

                z += pos - self.offset;
            }

            DistanceInfo {
                distance: 0.5 * r.ln() * r / dr,
                color: self.color,
            }
        }
    }

    fn insert_to_scene(&self, scene: &mut Scene) {
        scene.nodes.push(SceneNode::Object {
            color: self.color,
            offset: self.offset,
            data: ObjectData::Mandelbulb {
                n_steps: self.n_steps,
                power: self.power,
            },
        });
    }
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Union<G1, G2> {
    pub first: G1,
    pub second: G2,
}

impl<G1: Geometry, G2: Geometry> Geometry for Union<G1, G2> {
    fn sdf(&self) -> impl Sdf {
        move |pos| {
            let left = self.first.sdf()(pos);
            let right = self.second.sdf()(pos);

            if left <= right { left } else { right }
        }
    }

    fn insert_to_scene(&self, scene: &mut Scene) {
        let self_index = scene.nodes.len();

        scene.nodes.push(SceneNode::Transform(CsgOp::Union {
            lhs: self_index + 1,
            rhs: usize::MAX,
        }));

        self.first.insert_to_scene(scene);

        let right_index = scene.nodes.len();

        self.second.insert_to_scene(scene);

        unsafe {
            match scene.nodes.get_unchecked_mut(self_index) {
                SceneNode::Transform(CsgOp::Union { rhs, .. })
                    => *rhs = right_index,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intersection<G1, G2> {
    pub first: G1,
    pub second: G2,
}

impl<G1: Geometry, G2: Geometry> Geometry for Intersection<G1, G2> {
    fn sdf(&self) -> impl Sdf {
        move |pos| {
            let left = self.first.sdf()(pos);
            let right = self.second.sdf()(pos);

            if left <= right { right } else { left }
        }
    }

    fn insert_to_scene(&self, scene: &mut Scene) {
        let self_index = scene.nodes.len();

        scene.nodes.push(SceneNode::Transform(CsgOp::Intersection {
            lhs: self_index + 1,
            rhs: usize::MAX,
        }));

        self.first.insert_to_scene(scene);
        
        let right_index = scene.nodes.len();

        self.second.insert_to_scene(scene);

        unsafe {
            match scene.nodes.get_unchecked_mut(self_index) {
                SceneNode::Transform(CsgOp::Intersection { rhs, .. })
                    => *rhs = right_index,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothUnion<G1, G2> {
    pub first: G1,
    pub second: G2,
    pub param: f32,
}

impl<G1: Geometry, G2: Geometry> Geometry for SmoothUnion<G1, G2> {
    fn sdf(&self) -> impl Sdf {
        move |pos| {
            let left = self.first.sdf()(pos);
            let right = self.second.sdf()(pos);
            let distance = smooth_min(left.distance, right.distance, self.param);
            let color = (right.distance * left.color + left.distance * right.color)
                / (left.distance + right.distance);
            DistanceInfo { distance, color }
        }
    }

    fn insert_to_scene(&self, scene: &mut Scene) {
        let self_index = scene.nodes.len();

        scene.nodes.push(SceneNode::Transform(CsgOp::SmoothUnion {
            lhs: self_index + 1,
            rhs: usize::MAX,
            param: self.param,
        }));

        self.first.insert_to_scene(scene);
        
        let right_index = scene.nodes.len();

        self.second.insert_to_scene(scene);

        unsafe {
            match scene.nodes.get_unchecked_mut(self_index) {
                SceneNode::Transform(CsgOp::SmoothUnion { rhs, .. })
                    => *rhs = right_index,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothIntersection<G1, G2> {
    pub first: G1,
    pub second: G2,
    pub param: f32,
}

impl<G1: Geometry, G2: Geometry> Geometry for SmoothIntersection<G1, G2> {
    fn sdf(&self) -> impl Sdf {
        move |pos| {
            let left = self.first.sdf()(pos);
            let right = self.second.sdf()(pos);
            let distance = smooth_max(left.distance, right.distance, self.param);
            let color = (right.distance * left.color + left.distance * right.color)
                / (left.distance + right.distance);
            DistanceInfo { distance, color }
        }
    }

    fn insert_to_scene(&self, scene: &mut Scene) {
        let self_index = scene.nodes.len();

        scene.nodes.push(SceneNode::Transform(CsgOp::SmoothIntersection {
            lhs: self_index + 1,
            rhs: usize::MAX,
            param: self.param,
        }));

        self.first.insert_to_scene(scene);
        
        let right_index = scene.nodes.len();

        self.second.insert_to_scene(scene);

        unsafe {
            match scene.nodes.get_unchecked_mut(self_index) {
                SceneNode::Transform(CsgOp::SmoothIntersection { rhs, .. })
                    => *rhs = right_index,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Compliment<G> {
    pub geometry: G,
}

impl<G: Geometry> Geometry for Compliment<G> {
    fn sdf(&self) -> impl Sdf {
        move |pos| {
            let info = self.geometry.sdf()(pos);
            DistanceInfo { distance: -info.distance, ..info }
        }
    }

    fn insert_to_scene(&self, scene: &mut Scene) {
        let self_index = scene.nodes.len();

        scene.nodes.push(SceneNode::Transform(
            CsgOp::Compliment { from: self_index + 1 }
        ));

        self.geometry.insert_to_scene(scene);
    }
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Difference<G1, G2> {
    pub left: G1,
    pub right: G2,
}

impl<G1: Geometry, G2: Geometry> Geometry for Difference<G1, G2> {
    fn sdf(&self) -> impl Sdf {
        move |pos| {
            let left = self.left.sdf()(pos);
            let right = self.right.sdf()(pos);

            if -left.distance <= right.distance { right } else {
                DistanceInfo {
                    distance: -left.distance,
                    ..left
                }
            }
        }
    }

    fn insert_to_scene(&self, scene: &mut Scene) {
        let self_index = scene.nodes.len();

        scene.nodes.push(SceneNode::Transform(CsgOp::Difference {
            lhs: self_index + 1,
            rhs: usize::MAX,
        }));

        self.left.insert_to_scene(scene);
        
        let right_index = scene.nodes.len();

        self.right.insert_to_scene(scene);

        unsafe {
            match scene.nodes.get_unchecked_mut(self_index) {
                SceneNode::Transform(CsgOp::Difference { rhs, .. })
                    => *rhs = right_index,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothDifference<G1, G2> {
    pub left: G1,
    pub right: G2,
    pub param: f32,
}

impl<G1: Geometry, G2: Geometry> Geometry for SmoothDifference<G1, G2> {
    fn sdf(&self) -> impl Sdf {
        move |pos| {
            let left = self.left.sdf()(pos);
            let right = self.right.sdf()(pos);
            let distance = smooth_diff(left.distance, right.distance, self.param);
            let color = (right.distance * left.color + left.distance * right.color)
                / (left.distance + right.distance);
            DistanceInfo { distance, color }
        }
    }

    fn insert_to_scene(&self, scene: &mut Scene) {
        let self_index = scene.nodes.len();

        scene.nodes.push(SceneNode::Transform(CsgOp::SmoothDifference {
            lhs: self_index + 1,
            rhs: usize::MAX,
            param: self.param,
        }));

        self.left.insert_to_scene(scene);
        
        let right_index = scene.nodes.len();

        self.right.insert_to_scene(scene);

        unsafe {
            match scene.nodes.get_unchecked_mut(self_index) {
                SceneNode::Transform(CsgOp::SmoothDifference { rhs, .. })
                    => *rhs = right_index,
                _ => std::hint::unreachable_unchecked(),
            }
        }
    }
}



pub fn smooth_min(lhs: f32, rhs: f32, param: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (rhs - lhs) / param, 0.0, 1.0);
    mix(rhs, lhs, h) - param * h * (1.0 - h)
}

pub fn smooth_max(lhs: f32, rhs: f32, param: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (rhs - lhs) / param, 0.0, 1.0);
    mix(rhs, lhs, h) + param * h * (1.0 - h)
}

pub fn smooth_diff(lhs: f32, rhs: f32, param: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (rhs + lhs) / param, 0.0, 1.0);
    mix(rhs, -lhs, h) + param * h * (1.0 - h)
}

fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}



#[macro_export]
#[allow(unused)]
macro_rules! union {
    ($once:expr $(,)?) => {
        $once
    };
    ($first:expr, $($objs:expr),+ $(,)?) => {
        $first $(.union($objs))+
    };
}