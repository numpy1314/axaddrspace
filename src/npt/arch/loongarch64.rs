use core::arch::asm;
use core::fmt;
use page_table_entry::{GenericPTE, MappingFlags};
use page_table_multiarch::{PageTable64, PagingMetaData};
// use memory_addr::HostPhysAddr;
use crate::{GuestPhysAddr, HostPhysAddr};

bitflags::bitflags! {
    /// LoongArch TLB entry attribute field
    #[derive(Debug)]
    pub struct LoongArchDescriptorAttr: u64 {
        /// Entry valid bit
        const VALID =       1 << 0;
        /// Global mapping bit (ASID not used for matching when G=1)
        const GLOBAL =      1 << 1;
        /// Privilege level field (PLV0-PLV3)
        const PLV =         0b11 << 2;
        /// No-execute bit
        const NX =          1 << 4;
        /// No-read bit
        const NR =          1 << 5;
        /// No-write bit
        const NW =          1 << 6;
        /// Memory access type (MAT)
        const MAT =         0b11 << 8;
        /// Dirty bit
        const D =           1 << 10;
        /// Modified bit
        const V =           1 << 11;
    }
}

#[repr(u64)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum MemType {
    StrongUncached = 0,  // 强序非缓存
    CoherentCached = 1,  // 一致可缓存
    WeakUncached = 2,    // 弱序非缓存
}

impl LoongArchDescriptorAttr {
    const MAT_MASK: u64 = 0b11 << 8;

    const fn from_mem_type(mem_type: MemType) -> Self {
        let bits = match mem_type {
            MemType::StrongUncached => 0,
            MemType::CoherentCached => 1 << 8,
            MemType::WeakUncached => 2 << 8,
        };
        Self::from_bits_retain(bits)
    }

    fn mem_type(&self) -> MemType {
        match (self.bits() & Self::MAT_MASK) >> 8 {
            0 => MemType::StrongUncached,
            1 => MemType::CoherentCached,
            2 => MemType::WeakUncached,
            _ => unreachable!(),
        }
    }
}

impl From<LoongArchDescriptorAttr> for MappingFlags {
    fn from(attr: LoongArchDescriptorAttr) -> Self {
        let mut flags = Self::empty();
        if attr.contains(LoongArchDescriptorAttr::VALID) {
            flags |= Self::PRESENT;
        }
        if !attr.contains(LoongArchDescriptorAttr::NR) {
            flags |= Self::READ;
        }
        if !attr.contains(LoongArchDescriptorAttr::NW) {
            flags |= Self::WRITE;
        }
        if !attr.contains(LoongArchDescriptorAttr::NX) {
            flags |= Self::EXECUTE;
        }
        match attr.mem_type() {
            MemType::StrongUncached => flags |= Self::UNCACHED | Self::DEVICE,
            MemType::WeakUncached => flags |= Self::UNCACHED,
            _ => {}
        }
        flags
    }
}

impl From<MappingFlags> for LoongArchDescriptorAttr {
    fn from(flags: MappingFlags) -> Self {
        let mut attr = Self::empty();
        if flags.contains(MappingFlags::PRESENT) {
            attr |= Self::VALID;
        }
        if flags.contains(MappingFlags::READ) {
            attr |= Self::NR;
        }
        if flags.contains(MappingFlags::WRITE) {
            attr |= Self::NW;
        }
        if flags.contains(MappingFlags::EXECUTE) {
            attr |= Self::NX;
        }
        if flags.contains(MappingFlags::DEVICE) {
            attr |= Self::from_mem_type(MemType::StrongUncached);
        } else if flags.contains(MappingFlags::UNCACHED) {
            attr |= Self::from_mem_type(MemType::WeakUncached);
        } else {
            attr |= Self::from_mem_type(MemType::CoherentCached);
        }
        attr
    }
}

/// LoongArch TLB entry structure
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct LoongArchPTE(u64);

impl LoongArchPTE {
    const PHYS_ADDR_MASK: u64 = 0x0000_ffff_ffff_f000; // 物理地址掩码 (48位)
    const PS_MASK: u64 = 0b11111 << 12; // 页大小字段掩码

    /// Create empty page table entry
    pub const fn empty() -> Self {
        Self(0)
    }
}

impl GenericPTE for LoongArchPTE {
    fn bits(self) -> usize {
        self.0 as usize
    }

    fn new_page(paddr: HostPhysAddr, flags: MappingFlags, is_huge: bool) -> Self {
        let mut attr = LoongArchDescriptorAttr::from(flags);
        let ps = if is_huge {
            // 大页标志 (2MB/1GB)
            0b11000 // PS字段设置为大页标识
        } else {
            0 // 默认4KB页
        };
        Self(attr.bits() | (ps << 12) | (paddr.as_usize() as u64 & Self::PHYS_ADDR_MASK))
    }

    fn new_table(paddr: HostPhysAddr) -> Self {
        let attr = LoongArchDescriptorAttr::VALID;
        Self(attr.bits() | (paddr.as_usize() as u64 & Self::PHYS_ADDR_MASK))
    }

    fn paddr(&self) -> HostPhysAddr {
        HostPhysAddr::from((self.0 & Self::PHYS_ADDR_MASK) as usize)
    }

    fn flags(&self) -> MappingFlags {
        LoongArchDescriptorAttr::from_bits_truncate(self.0).into()
    }

    fn set_paddr(&mut self, paddr: HostPhysAddr) {
        self.0 = (self.0 & !Self::PHYS_ADDR_MASK) | (paddr.as_usize() as u64 & Self::PHYS_ADDR_MASK);
    }

    fn set_flags(&mut self, flags: MappingFlags, is_huge: bool) {
        let mut attr = LoongArchDescriptorAttr::from(flags);
        let ps = if is_huge { 0b11000 } else { 0 };
        self.0 = (self.0 & !(Self::PS_MASK | LoongArchDescriptorAttr::all().bits())) 
            | attr.bits() 
            | (ps << 12);
    }

    fn is_unused(&self) -> bool {
        self.0 == 0
    }

    fn is_present(&self) -> bool {
        LoongArchDescriptorAttr::from_bits_truncate(self.0).contains(LoongArchDescriptorAttr::VALID)
    }

    fn is_huge(&self) -> bool {
        (self.0 & Self::PS_MASK) == (0b11000 << 12)
    }

    fn clear(&mut self) {
        self.0 = 0;
    }
}

impl fmt::Debug for LoongArchPTE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LoongArchPTE")
            .field("raw", &self.0)
            .field("paddr", &self.paddr())
            .field("attr", &LoongArchDescriptorAttr::from_bits_truncate(self.0))
            .field("flags", &self.flags())
            .finish()
    }
}

/// LoongArch nested paging metadata
#[derive(Copy, Clone)]
pub struct LoongArchPagingMetaData;

impl PagingMetaData for LoongArchPagingMetaData {
    const LEVELS: usize = 4; // 4-level page table
    const PA_MAX_BITS: usize = 48; // 48-bit physical address
    const VA_MAX_BITS: usize = 48; // 48-bit virtual address (GPA)

    type VirtAddr = GuestPhysAddr;

    fn flush_tlb(vaddr: Option<Self::VirtAddr>) {
        unsafe {
            if let Some(vaddr) = vaddr {
                // Invalidate individual GPA using INVTLB instruction
                asm!(
                    "invtlb 0, {}, {}",
                    in(reg) 0, // GID=0 for current Guest
                    in(reg) vaddr.as_usize()
                );
            } else {
                // Invalidate entire GTLB
                asm!("invtlb 0, $r0, $r0");
            }
            // Memory barrier to ensure TLB invalidation completion
            asm!("dbar 0");
        }
    }
}

/// LoongArch nested page table type
pub type NestedPageTable = PageTable64<LoongArchPagingMetaData, LoongArchPTE>;