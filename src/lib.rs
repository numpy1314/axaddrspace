//! [ArceOS-Hypervisor](https://github.com/arceos-hypervisor/) guest VM address space management module.

#![no_std]
#![feature(const_trait_impl)]

#[macro_use]
extern crate log;
extern crate alloc;

mod addr;
mod address_space;
pub mod device;
mod frame;
mod hal;
mod npt;

pub use addr::*;
pub use address_space::*;

pub use frame::PhysFrame;
pub use hal::AxMmHal;
pub use npt::NestedPageTable;

use axerrno::AxError;
use memory_set::MappingError;

/// Information about nested page faults.
#[derive(Debug)]
pub struct NestedPageFaultInfo {
    /// Access type that caused the nested page fault.
    pub access_flags: MappingFlags,
    /// Guest physical address that caused the nested page fault.
    pub fault_guest_paddr: GuestPhysAddr,
}

fn mapping_err_to_ax_err(err: MappingError) -> AxError {
    warn!("Mapping error: {err:?}");
    match err {
        MappingError::InvalidParam => AxError::InvalidInput,
        MappingError::AlreadyExists => AxError::AlreadyExists,
        MappingError::BadState => AxError::BadState,
    }
}

#[cfg(test)]
pub(crate) mod test_utils;
