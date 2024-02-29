use crate::util::*;



#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum ComputeContextMode {
    Debug,
    ReleaseValidation,
    ReleaseSilent,
}

impl std::fmt::Display for ComputeContextMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Debug => "debug",
            Self::ReleaseValidation => "validation",
            Self::ReleaseSilent => "silent",
        })
    }
}

impl std::str::FromStr for ComputeContextMode {
    type Err = ParseComputeContextModeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "debug" => Self::Debug,
            "validation" => Self::ReleaseValidation,
            "silent" => Self::ReleaseSilent,
            _ => return Err(ParseComputeContextModeError::InvalidArg(s.to_owned()))
        })
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum ParseComputeContextModeError {
    #[error("invalid compute context mode '{0}', valid values are: 'debug', 'validation', 'silent'")]
    InvalidArg(String),
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

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                label: None,
                required_limits: adapter.limits(),
            },
            None,
        ).await?;

        device.set_device_lost_callback(|reason, msg| {
            if msg != "Device dropped." {
                log::error!("the device is lost: '{msg}', because: {reason:?}");
            }
        });

        Ok(Self { instance, adapter, device, queue })
    }
}