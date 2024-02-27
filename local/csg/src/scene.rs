use glam::*;
use serde::{Deserialize, Serialize};
use crate::geometry::*;



#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CsgOp {
    Union { lhs: usize, rhs: usize },
    SmoothUnion { lhs: usize, rhs: usize, param: f32 },
    Intersection { lhs: usize, rhs: usize },
    SmoothIntersection { lhs: usize, rhs: usize, param: f32 },
    Difference { lhs: usize, rhs: usize },
    SmoothDifference { lhs: usize, rhs: usize, param: f32 },
    Compliment { from: usize },
}



#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SceneNode {
    Transform(CsgOp),
    Object {
        color: Vec3,
        offset: Vec3,
        data: ObjectData,
    },
}



#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ObjectData {
    Ball { radius: f32 },
    Semispace { normal: Vec3, distance: f32 },
    Mandelbulb { n_steps: usize, power: f32 },
    StraightPrism { sizes: Vec3 },
    Torus { inner_radius: f32, outer_radius: f32 },
}



#[derive(Clone, Debug)]
pub struct Scene {
    pub nodes: Vec<SceneNode>,
}

impl Scene {
    pub const fn new() -> Self {
        Self { nodes: vec![] }
    }

    pub fn as_bytes(&self) -> Vec<u32> {
        self.nodes.iter()
            .flat_map(|node| -> [u32; 12] {
                let flags = (Self::node_flags(node) as u32).to_le_bytes();

                match node {
                    SceneNode::Object { color, offset, data } => unsafe {
                        let data = Self::object_bytes(*color, *offset, data);
                        std::mem::transmute((0_u32, flags, data))
                    },
                    SceneNode::Transform(op) => unsafe {
                        let data = Self::operation_bytes(op);
                        std::mem::transmute((0_u32, flags, [0_u32; 7], data))
                    },
                }
            })
            .collect()
    }

    fn node_flags(node: &SceneNode) -> u8 {
        match node {
            SceneNode::Object { data, .. } => match data {
                ObjectData::Ball { .. }          => 0x0 << 3,
                ObjectData::StraightPrism { .. } => 0x1 << 3,
                ObjectData::Mandelbulb { .. }    => 0x2 << 3,
                ObjectData::Semispace { .. }     => 0x3 << 3,
                ObjectData::Torus { .. }         => 0x4 << 3,
            },
            SceneNode::Transform(CsgOp::Union { .. }) => 0x1,
            SceneNode::Transform(CsgOp::SmoothUnion { .. }) => 0x2,
            SceneNode::Transform(CsgOp::Intersection { .. }) => 0x3,
            SceneNode::Transform(CsgOp::SmoothIntersection { .. }) => 0x4,
            SceneNode::Transform(CsgOp::Difference { .. }) => 0x5,
            SceneNode::Transform(CsgOp::SmoothDifference { .. }) => 0x6,
            SceneNode::Transform(CsgOp::Compliment { .. }) => 0x7,
        }
    }

    fn operation_bytes(op: &CsgOp) -> [u8; 12] {
        match op {
            CsgOp::Difference { lhs, rhs }
                | CsgOp::Intersection { lhs, rhs }
                | CsgOp::Union { lhs, rhs } =>
            {
                let lhs_bytes = (*lhs as u32).to_le_bytes();
                let rhs_bytes = (*rhs as u32).to_le_bytes();
                let param = 0_u32.to_le_bytes();

                bytemuck::cast([lhs_bytes, rhs_bytes, param])
            },
            CsgOp::SmoothDifference { lhs, rhs, param }
                | CsgOp::SmoothIntersection { lhs, rhs, param }
                | CsgOp::SmoothUnion { lhs, rhs, param } =>
            {
                let lhs_bytes = (*lhs as u32).to_le_bytes();
                let rhs_bytes = (*rhs as u32).to_le_bytes();
                let param = param.to_le_bytes();

                bytemuck::cast([lhs_bytes, rhs_bytes, param])
            },
            CsgOp::Compliment { from } => {
                let lhs_bytes = (*from as u32).to_le_bytes();
                let rhs_bytes = 0_u32.to_le_bytes();
                let param = 0_u32.to_le_bytes();

                bytemuck::cast([lhs_bytes, rhs_bytes, param])
            },
        }
    }

    fn object_data_bytes(object: &ObjectData) -> [u8; 16] {
        use std::mem::transmute;

        match object {
            ObjectData::Ball { radius } => unsafe {
                transmute(([0_u8; 12], radius.to_le_bytes()))
            },
            ObjectData::StraightPrism { sizes } => unsafe {
                transmute([
                    [0; 4],
                    sizes.x.to_le_bytes(),
                    sizes.y.to_le_bytes(),
                    sizes.z.to_le_bytes(),
                ])
            },
            ObjectData::Mandelbulb { n_steps, power } => unsafe {
                transmute((
                    [0_u8; 8],
                    (*n_steps as u32).to_le_bytes(),
                    power.to_le_bytes(),
                ))
            },
            ObjectData::Semispace { normal, distance } => unsafe {
                transmute([
                    normal.x.to_le_bytes(),
                    normal.y.to_le_bytes(),
                    normal.z.to_le_bytes(),
                    distance.to_le_bytes(),
                ])
            },
            ObjectData::Torus { inner_radius, outer_radius } => unsafe {
                transmute((
                    [0_u8; 8],
                    inner_radius.to_le_bytes(),
                    outer_radius.to_le_bytes(),
                ))
            },
        }
    }

    fn object_bytes(color: Vec3, offset: Vec3, data: &ObjectData) -> [u8; 40] {
        unsafe {
            std::mem::transmute((
                color.x.to_le_bytes(),
                color.y.to_le_bytes(),
                color.z.to_le_bytes(),
                offset.x.to_le_bytes(),
                offset.y.to_le_bytes(),
                offset.z.to_le_bytes(),
                Self::object_data_bytes(data),
            ))
        }
    }
}

impl<G: Geometry> From<&G> for Scene {
    fn from(value: &G) -> Self {
        let mut result = Self::new();
        value.insert_to_scene(&mut result);
        result
    }
}



#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SerializeableSceneNode {
    Union(Vec<Self>),
    SmoothUnion { param: f32, nodes: Vec<Self> },
    Intersection(Vec<Self>),
    SmoothIntersection { param: f32, nodes: Vec<Self> },
    Difference { left: Box<Self>, right: Box<Self> },
    SmoothDifference { left: Box<Self>, right: Box<Self>, param: f32 },
    Compliment(Box<Self>),
    Object(ObjectData),
}