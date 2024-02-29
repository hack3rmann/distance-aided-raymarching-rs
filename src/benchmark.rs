#![macro_use]

use std::time::Duration;



#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Benchmark {
    pub render: Duration,
    pub copy: Duration,
}

impl Benchmark {
    pub const fn new() -> Self {
        Self {
            render: Duration::from_nanos(0),
            copy: Duration::from_nanos(0),
        }
    }
}

impl std::fmt::Display for Benchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "time spent on rendering: {:?}, time spent on data copying: {:?}", self.render, self.copy)
    }
}