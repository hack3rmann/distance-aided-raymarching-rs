use crate::util::*;



#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum ComputeContextMode {
    Debug,
    ReleaseValidation,
    ReleaseSilent,
}

impl From<ComputeContextMode> for wgpu::InstanceFlags {
    fn from(value: ComputeContextMode) -> Self {
        use ComputeContextMode::*;

        match value {
            Debug => Self::DEBUG | Self::VALIDATION,
            ReleaseValidation => Self::VALIDATION,
            ReleaseSilent => Self::empty(),
        }
    }
}



#[derive(Debug)]
pub struct ComputeContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl ComputeContext {
    pub async fn new(mode: ComputeContextMode)
        -> Result<Self, wgpu::RequestDeviceError>
    {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: mode.into(),
            ..default()
        });

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }).await.expect("failed to request the adapter");

        let limits = adapter.limits();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                label: None,
                required_limits: default(),
            },
            None,
        ).await?;

        device.set_device_lost_callback(|reason, msg| {
            if msg != "Device dropped." {
                eprintln!("the device is lost: '{msg}', because: {reason:?}");
            }
        });

        Ok(Self { instance, adapter, device, queue })
    }
}