use core::arch::asm;
use core::fmt;

use page_table_entry::{GenericPTE, MappingFlags};
use page_table_multiarch::{PageTable64, PagingMetaData};

use crate::{HostPhysAddr, GuestPhysAddr};

bitflags::bitflags! {
    /// 使用 u64，和 PTE 宽度保持一致，避免 usize/u64 来回转换
    #[derive(Clone, Copy, Debug)]
    pub struct LaPteAttr: u64 {
        const V   = 1 << 0;        // Valid
        const D   = 1 << 1;        // Dirty
        const PLV = 0b11 << 2;     // Privilege Level mask
        const PLV0 = 0b00 << 2;
        const PLV1 = 0b01 << 2;
        const PLV2 = 0b10 << 2;
        const PLV3 = 0b11 << 2;

        const MAT = 0b11 << 4;     // Memory Access Type mask
        const MAT_SUC = 0b00 << 4; // Strongly-Ordered Uncached (Device)
        const MAT_CC  = 0b01 << 4; // Coherent Cached (Normal)
        const MAT_WB  = 0b10 << 4; // Weakly-Ordered Uncached (NormalNonCache)

        const G = 1 << 6;          // Global
        const P = 1 << 7;          // Present 
        const W = 1 << 8;          // Writable

        const NR = 1 << 61;        // Not Readable
        const NX = 1 << 62;        // Not Executable
        const RPLV = 1 << 63;      // Relative Privilege Level Check
    }
}

/// Memory type enum for LoongArch mappings.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaMemType {
    Device = 0,          // MAT_SUC
    Normal = 1,          // MAT_CC
    NormalNonCache = 2,  // MAT_WB
}

impl LaPteAttr {
    /// 从内存类型生成属性
    pub const fn from_mem_type(mem: LaMemType) -> Self {
        match mem {
            LaMemType::Device => Self::MAT_SUC,
            LaMemType::Normal => Self::MAT_CC | Self::G,
            LaMemType::NormalNonCache => Self::MAT_WB | Self::G,
        }
    }

    /// 从属性反推内存类型
    pub fn mem_type(&self) -> LaMemType {
        let mat = LaPteAttr::from_bits_truncate(self.bits() & LaPteAttr::MAT.bits());
        if mat == LaPteAttr::MAT_SUC {
            LaMemType::Device
        } else if mat == LaPteAttr::MAT_CC {
            LaMemType::Normal
        } else if mat == LaPteAttr::MAT_WB {
            LaMemType::NormalNonCache
        } else {
            LaMemType::Device
        }
    }

    /// 设定 PLV（0~3）
    pub fn with_plv(mut self, plv: u8) -> Self {
        self &= !LaPteAttr::PLV;
        let plv_bits = ((plv as u64) & 0b11) << 2;
        self |= LaPteAttr::from_bits_truncate(plv_bits);
        self
    }
}

/// LaPteAttr ↔ MappingFlags 的互转
impl From<LaPteAttr> for MappingFlags {
    fn from(attr: LaPteAttr) -> Self {
        let mut flags = MappingFlags::empty();

        // READ: 只要不是 NR 且 V 置位，就当作可读
        if attr.contains(LaPteAttr::V) && !attr.contains(LaPteAttr::NR) {
            flags |= MappingFlags::READ;
        }

        if attr.contains(LaPteAttr::W) {
            flags |= MappingFlags::WRITE;
        }

        if !attr.contains(LaPteAttr::NX) {
            flags |= MappingFlags::EXECUTE;
        }

        if attr.mem_type() == LaMemType::Device {
            flags |= MappingFlags::DEVICE;
        }

        flags
    }
}

impl From<MappingFlags> for LaPteAttr {
    fn from(flags: MappingFlags) -> Self {
        let mut attr = LaPteAttr::empty();

        if flags.contains(MappingFlags::DEVICE) {
            attr |= LaPteAttr::from_mem_type(LaMemType::Device);
        } else {
            attr |= LaPteAttr::from_mem_type(LaMemType::Normal);
        }

        // READ：通常与 V 关联；WRITE：置 W；EXECUTE：控制 NX
        if flags.contains(MappingFlags::READ) {
            attr |= LaPteAttr::V;
        }
        if flags.contains(MappingFlags::WRITE) {
            attr |= LaPteAttr::W;
        }
        if flags.contains(MappingFlags::EXECUTE) {
            attr &= !LaPteAttr::NX;
        } else {
            attr |= LaPteAttr::NX;
        }

        attr
    }
}

/// PTE 容器
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct LaPTE(u64);

impl LaPTE {
    const PHYS_ADDR_MASK: u64 = 0x0000_ffff_ffff_f000;

    pub const fn empty() -> Self {
        Self(0)
    }

    pub fn raw(&self) -> u64 {
        self.0
    }

    fn attr(&self) -> LaPteAttr {
        LaPteAttr::from_bits_truncate(self.0)
    }
}

/// 实现 GenericPTE
impl GenericPTE for LaPTE {
    fn bits(self) -> usize {
        self.0 as usize
    }

    fn paddr(&self) -> HostPhysAddr {
        HostPhysAddr::from((self.0 & Self::PHYS_ADDR_MASK) as usize)
    }

    fn flags(&self) -> MappingFlags {
        MappingFlags::from(self.attr())
    }

    fn is_unused(&self) -> bool {
        self.0 == 0
    }

    fn is_present(&self) -> bool {
        self.attr().contains(LaPteAttr::V)
    }

    fn is_huge(&self) -> bool {
        false
    }

    fn clear(&mut self) {
        self.0 = 0;
    }

    fn new_page(paddr: HostPhysAddr, flags: MappingFlags, _is_huge: bool) -> Self {
        let mut attr = LaPteAttr::from(flags);

        if flags.contains(MappingFlags::READ) || flags.contains(MappingFlags::WRITE) {
            attr |= LaPteAttr::V;
        }

        Self((paddr.as_usize() as u64 & Self::PHYS_ADDR_MASK) | attr.bits())
    }

    fn new_table(paddr: HostPhysAddr) -> Self {
        Self((paddr.as_usize() as u64 & Self::PHYS_ADDR_MASK) | LaPteAttr::V.bits())
    }

    fn set_paddr(&mut self, paddr: HostPhysAddr) {
        self.0 = (self.0 & !Self::PHYS_ADDR_MASK) | (paddr.as_usize() as u64 & Self::PHYS_ADDR_MASK);
    }

    fn set_flags(&mut self, flags: MappingFlags, _is_huge: bool) {
        let attr = LaPteAttr::from(flags) | LaPteAttr::V; // 一般需要保持 V=1
        self.0 = (self.0 & Self::PHYS_ADDR_MASK) | attr.bits();
    }
}

/// Paging metadata
#[derive(Copy, Clone)]
pub struct LaPagingMetaData;

impl PagingMetaData for LaPagingMetaData {
    const LEVELS: usize = 4;
    const PA_MAX_BITS: usize = 48;
    const VA_MAX_BITS: usize = 48;
    type VirtAddr = GuestPhysAddr;

    unsafe fn flush_tlb(vaddr: Option<Self::VirtAddr>) {
        if let Some(va) = vaddr {
            let va_usize = va.as_usize();
            asm!("dbar 0", "invtlb 0x05, $r0, {va}", va = in(reg) va_usize);
        } else {
            asm!("dbar 0", "invtlb 0, $r0, $r0");
        }
    }
}

/// Page table type alias
pub type NestedPageTable<H> = PageTable64<LaPagingMetaData, LaPTE, H>;
