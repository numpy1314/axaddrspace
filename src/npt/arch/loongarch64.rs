use core::arch::asm;
use core::fmt;
use page_table_entry::{GenericPTE, MappingFlags};
use page_table_multiarch::{PageTable64, PagingMetaData};
use memory_addr::{HostPhysAddr, GuestPhysAddr};

// LoongArch PTE attribute bits (based on LoongArch Vol1 v1.10 §5.4 / 表 7-38)
bitflags::bitflags! {
    /// Memory attribute fields in the LoongArch64 translation table format descriptors.
    #[derive(Clone, Copy, Debug)]
    pub struct LaPteAttr: usize {
        const V = 1 << 0; // Valid
        const D = 1 << 1; // Dirty

        const PLV = 0b11 << 2; // Privilege Level Range
        const PLV0 = 0b00 << 2; // PLV0
        const PLV1 = 0b01 << 2; // PLV1
        const PLV2 = 0b10 << 2; // PLV2
        const PLV3 = 0b11 << 2; // PLV3

        const MAT = 0b11 << 4; // Memory Access Type
        const MAT_SUC = 0b00 << 4; // Strongly-Ordered Uncached (Device)
        const MAT_CC = 0b01 << 4; // Coherent Cached (Normal)
        const MAT_WB = 0b10 << 4; // Weakly-Ordered Uncached (NormalNonCache)

        const G = 1 << 6; // Global
        const P = 1 << 7; // Present
        const W = 1 << 8; // Writable
        const NR = 1 << 61; // Not Readable
        const NX = 1 << 62; // Not Executable
        const RPLV = 1 << 63; // Relative Privilege Level Check
    }
}

/// Memory type enum for LoongArch mappings.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaMemType {
    Device = 0,      // MAT_SUC
    Normal = 1,      // MAT_CC
    NormalNonCache = 2, // MAT_WB
}

impl LaPteAttr {
    // PPN / physical address mask: 48-bit physical addresses
    pub const PHYS_ADDR_MASK: u64 = 0x0000_ffff_ffff_f000;

    // MAT encodings
    const MAT_SUC: u64 = Self::MAT_SUC.bits();
    const MAT_CC: u64 = Self::MAT_CC.bits();
    const MAT_WB: u64 = Self::MAT_WB.bits();

    /// Create LaPteAttr from memory type
    pub const fn from_mem_type(mem: LaMemType) -> Self {
        let mut attr = Self::empty();
        attr |= match mem {
            LaMemType::Device => Self::MAT_SUC,
            LaMemType::Normal => Self::MAT_CC | Self::G,
            LaMemType::NormalNonCache => Self::MAT_WB | Self::G,
        };
        attr
    }

    /// Infer LaMemType from PTE attributes
    pub fn mem_type(&self) -> LaMemType {
        match self.bits() & Self::MAT {
            Self::MAT_SUC => LaMemType::Device,
            Self::MAT_CC => LaMemType::Normal,
            Self::MAT_WB => LaMemType::NormalNonCache,
            _ => LaMemType::Device, // Default to device for unknown types
        }
    }

    /// Set privilege level
    pub fn with_plv(mut self, plv: u8) -> Self {
        self = self & !Self::PLV; // Clear existing PLV
        self |= ((plv as u64) & 0b11) << 2;
        self
    }
}

/// Mapping between LaPteAttr and generic flags
impl From<LaPteAttr> for MappingFlags {
    fn from(attr: LaPteAttr) -> Self {
        let mut flags = MappingFlags::empty();
        if attr.contains(LaPteAttr::V) {
            flags |= if !attr.contains(LaPteAttr::NR) { MappingFlags::READ } else { MappingFlags::empty() };
        }
        if attr.contains(LaPteAttr::W) {
            flags |= MappingFlags::WRITE;
            flags |= MappingFlags::DIRTY;
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
        let mut attr = Self::empty();
        if flags.contains(MappingFlags::DEVICE) {
            attr |= LaPteAttr::from_mem_type(LaMemType::Device);
        } else {
            attr |= LaPteAttr::from_mem_type(LaMemType::Normal);
        }

        if flags.contains(MappingFlags::READ) {
            attr |= LaPteAttr::V;
        }
        if flags.contains(MappingFlags::WRITE) {
            attr |= LaPteAttr::W | LaPteAttr::D;
        }
        if flags.contains(MappingFlags::EXECUTE) {
            attr &= !LaPteAttr::NX;
        } else {
            attr |= LaPteAttr::NX;
        }

        attr
    }
}

/// LaPTE container
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct LaPTE(u64);

impl LaPTE {
    pub const fn empty() -> Self {
        Self(0)
    }

    pub fn raw(&self) -> u64 {
        self.0
    }

    pub fn paddr(&self) -> HostPhysAddr {
        HostPhysAddr::from((self.0 & Self::PHYS_ADDR_MASK) as usize)
    }
}

impl GenericPTE for LaPTE {
    fn bits(self) -> usize {
        self.0 as usize
    }

    fn new_page(paddr: HostPhysAddr, flags: MappingFlags, _is_huge: bool) -> Self {
        let mut attr = LaPteAttr::from(flags);
        attr |= if flags.contains(MappingFlags::READ) || flags.contains(MappingFlags::WRITE) {
            LaPteAttr::V
        } else {
            LaPteAttr::empty()
        };
        
        Self((paddr.as_usize() as u64 & Self::PHYS_ADDR_MASK) | attr.bits())
    }

    fn new_table(paddr: HostPhysAddr) -> Self {
        Self((paddr.as_usize() as u64 & Self::PHYS_ADDR_MASK) | LaPteAttr::V.bits())
    }

    fn set_paddr(&mut self, paddr: HostPhysAddr) {
        self.0 = (self.0 & !Self::PHYS_ADDR_MASK) | (paddr.as_usize() as u64 & Self::PHYS_ADDR_MASK);
    }

    fn set_flags(&mut self, flags: MappingFlags, _is_huge: bool) {
        let attr = LaPteAttr::from(flags);
        self.0 = (self.0 & Self::PHYS_ADDR_MASK) | attr.bits();
    }
}

impl fmt::Debug for LaPTE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LaPTE")
            .field("raw", &self.0)
            .field("paddr", &self.paddr())
            .field("mem_type", &self.attr().mem_type())
            .finish()
    }
}

impl LaPTE {
    fn attr(&self) -> LaPteAttr {
        LaPteAttr::from_bits_truncate(self.0)
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
            asm!("dbar 0", "invtlb 0x05, $r0, {va}", va = in(reg) va_usize,);
        } else {
            asm!("dbar 0", "invtlb 0, $r0, $r0",);
        }
    }
}

/// Page table type alias
pub type NestedPageTable<H> = PageTable64<LaPagingMetaData, LaPTE, H>;