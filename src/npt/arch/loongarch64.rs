use core::arch::asm;
use core::fmt;
use page_table_entry::{GenericPTE, MappingFlags};
use page_table_multiarch::{PageTable64, PagingMetaData};
// use memory_addr::HostPhysAddr;
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

        const MAT = 0b11 << 4; // Memory Access Type Range
        const MAT_SUC = 0b00 << 4; // Strongly-Ordered Uncached
        const MAT_CC = 0b01 << 4; // Coherent Cached
        const MAT_WB = 0b10 << 4; // Weakly-Ordered Uncached

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
enum LaMemType {
    MAT_SUC = 0,
    MAT_CC = 1,
    MAT_WB = 2,
}

impl LaPteAttr {
    // PPN / physical address mask: assume max 48-bit physical addresses, page granularity 4K
    // PPN occupies bits [12..PALEN-1] in the PTE; using conservative mask up to 48 bits here.
    pub const PHYS_ADDR_MASK: u64 = 0x0000_ffff_ffff_f000;

    // MAT encodings (LoongArch uses MAT[1:0] in bits 5:4 for simple encodings)
    // We'll adopt a small mapping for common types; these encodings must match platform's MAT->memory model.
    const MAT_SUC: u64 = 0b00 << 4;
    const MAT_CC: u64 = 0b01 << 4;  // treat as cacheable write-back
    const MAT_WB: u64 = 0b10 << 4;  // outer non-cache (example)

    /// Create LaPteAttr representing a LoongArch memory type
    pub const fn from_mem_type(mem: LaMemType) -> Self {
        let bits = match mem {
            LaMemType::Device => Self::MAT_SUC,
            LaMemType::Normal => Self::MAT_CC | Self::G_BASIC.bits(),
            LaMemType::NormalNonCache => Self::MAT_WB | Self::G_BASIC.bits(),
        };
        // SAFETY: bits are within u64 bitfield
        Self::from_bits_truncate(bits)
    }

    /// Try to infer LaMemType from the MAT bits
    pub fn mem_type(&self) -> LaMemType {
        let mat = self.bits() & Self::MAT_MASK.bits();
        match mat {
            Self::MAT_NORMAL_WB => LaMemType::Normal,
            Self::MAT_NORMAL_NC => LaMemType::NormalNonCache,
            _ => LaMemType::Device,
        }
    }

    /// Set PLV (2-bit) into the attribute
    pub fn with_plv(self, plv: u8) -> Self {
        let plv_val = ((plv as u64) & 0b11) << 2;
        let bits = (self.bits() & !Self::PLV_MASK.bits()) | plv_val;
        Self::from_bits_retain(bits)
    }
}

/// Mapping between LaPteAttr and generic MappingFlags
impl From<LaPteAttr> for MappingFlags {
    fn from(attr: LaPteAttr) -> Self {
        let mut flags = MappingFlags::empty();
        if attr.contains(LaPteAttr::V) {
            // treat valid as readable unless NR set
            if !attr.contains(LaPteAttr::NR) {
                flags |= MappingFlags::READ;
            }
        }
        if attr.contains(LaPteAttr::W) {
            flags |= MappingFlags::WRITE;
        }
        if !attr.contains(LaPteAttr::NX) {
            // NX==1 means not executable; invert
            flags |= MappingFlags::EXECUTE;
        }
        // device detection: if MAT==MAT_DEVICE treat as DEVICE
        if attr.mem_type() == LaMemType::Device {
            flags |= MappingFlags::DEVICE;
        }
        flags
    }
}

impl From<MappingFlags> for LaPteAttr {
    fn from(flags: MappingFlags) -> Self {
        let mut attr = if flags.contains(MappingFlags::DEVICE) {
            if flags.contains(MappingFlags::UNCACHED) {
                LaPteAttr::from_mem_type(LaMemType::NormalNonCache)
            } else {
                LaPteAttr::from_mem_type(LaMemType::Device)
            }
        } else {
            LaPteAttr::from_mem_type(LaMemType::Normal)
        };

        // set basic accessibility bits:
        if flags.contains(MappingFlags::READ) {
            attr |= LaPteAttr::V;
        }
        if flags.contains(MappingFlags::WRITE) {
            attr |= LaPteAttr::W | LaPteAttr::D;
        }
        // EXECUTE -> clear NX; otherwise set NX
        if flags.contains(MappingFlags::EXECUTE) {
            // ensure NX is cleared
            attr = LaPteAttr::from_bits_truncate(attr.bits() & !LaPteAttr::NX.bits());
        } else {
            attr |= LaPteAttr::NX;
        }

        attr
    }
}

/// LaPTE: concrete 64-bit PTE container for LoongArch
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
        HostPhysAddr::from((self.0 & LaPteAttr::PHYS_ADDR_MASK) as usize)
    }
}

impl GenericPTE for LaPTE {
    fn bits(self) -> usize {
        self.0 as usize
    }

    fn new_page(paddr: HostPhysAddr, flags: MappingFlags, _is_huge: bool) -> Self {
        let mut attr = LaPteAttr::from(flags);
        // set valid bit if readable or requested
        if flags.contains(MappingFlags::READ) || flags.contains(MappingFlags::WRITE) {
            attr |= LaPteAttr::V;
        }
        // set dirty if writable
        if flags.contains(MappingFlags::WRITE) {
            attr |= LaPteAttr::D;
        }
        // combine attr bits with physical address
        Self(attr.bits() | (paddr.as_usize() as u64 & LaPteAttr::PHYS_ADDR_MASK))
    }

    fn new_table(paddr: HostPhysAddr) -> Self {
        // For a table descriptor, typically we mark not-present as V may be 0;
        // but to point to next-level we set P (or other convention). We'll set V=1 here.
        let attr = LaPteAttr::V; // at minimum mark valid pointer
        Self(attr.bits() | (paddr.as_usize() as u64 & LaPteAttr::PHYS_ADDR_MASK))
    }

    fn paddr(&self) -> HostPhysAddr {
        HostPhysAddr::from((self.0 & LaPteAttr::PHYS_ADDR_MASK) as usize)
    }

    fn flags(&self) -> MappingFlags {
        LaPteAttr::from_bits_truncate(self.0).into()
    }

    fn set_paddr(&mut self, paddr: HostPhysAddr) {
        self.0 = (self.0 & !LaPteAttr::PHYS_ADDR_MASK)
            | (paddr.as_usize() as u64 & LaPteAttr::PHYS_ADDR_MASK);
    }

    fn set_flags(&mut self, flags: MappingFlags, _is_huge: bool) {
        let mut attr = LaPteAttr::from(flags);
        if flags.contains(MappingFlags::READ) {
            attr |= LaPteAttr::V;
        }
        if flags.contains(MappingFlags::WRITE) {
            attr |= LaPteAttr::D | LaPteAttr::W;
        }
        // preserve physical address
        self.0 = (self.0 & LaPteAttr::PHYS_ADDR_MASK) | attr.bits();
    }

    fn is_unused(&self) -> bool {
        self.0 == 0
    }

    fn is_present(&self) -> bool {
        LaPteAttr::from_bits_truncate(self.0).contains(LaPteAttr::V)
    }

    fn is_huge(&self) -> bool {
        // For LoongArch, huge page is indicated by directory H bit or alignment; here we conservatively:
        // if PPN encodes a large mapping, detection depends on table level. We return false => not huge.
        false
    }

    fn clear(&mut self) {
        self.0 = 0;
    }
}

impl fmt::Debug for LaPTE {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LaPTE")
            .field("raw", &self.0)
            .field("paddr", &self.paddr())
            .field("attr", &LaPteAttr::from_bits_truncate(self.0))
            .field("flags", &self.flags())
            .finish()
    }
}

/// Paging metadata for LoongArch
#[derive(Copy, Clone)]
pub struct LaPagingMetaData;

impl PagingMetaData for LaPagingMetaData {
    const LEVELS: usize = 4; // depends on PALEN/VALEN configuration in CSR.PWCL/PWCH
    const PA_MAX_BITS: usize = 48;
    const VA_MAX_BITS: usize = 48;
    type VirtAddr = GuestPhysAddr; // or GuestVirtAddr as appropriate
    // TODO: based on loongarch64 lvz, page: 15
    fn flush_tlb(vaddr: Option<Self::VirtAddr>) {
        unsafe {
            if let Some(va) = vaddr {
                // flush single vaddr
                let va_usize = va.as_usize();
                asm!(
                    "dbar 0",
                    "invtlb 0x05, $r0, {va}",
                    va = in(reg) va_usize,
                );
            } else {
                // flush all
                asm!("dbar 0", "invtlb 0, $r0, $r0",);
            }
        }
    }
}

/// Convenience alias: a page table type for LoongArch using our PTE & metadata.
pub type NestedPageTable<H> = PageTable64<LaPagingMetaData, LaPTE, H>;
