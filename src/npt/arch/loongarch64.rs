use core::arch::asm;
use core::fmt;
use page_table_entry::{GenericPTE, MappingFlags};
use page_table_multiarch::{PageTable64, PagingMetaData};
// use memory_addr::HostPhysAddr;
use crate::{GuestPhysAddr, HostPhysAddr};

/// LoongArch PTE attribute bits (based on LoongArch Vol1 v1.10 §5.4 / 表 7-38)
bitflags::bitflags! {
    #[derive(Debug)]
    pub struct LaPteAttr: u64 {
        /// bit 0: V - valid
        const V = 1 << 0;
        /// bit 1: D - dirty (written)
        const D = 1 << 1;

        /// bits 3:2: PLV (privilege level)
        const PLV_MASK = 0b11 << 2;

        /// bits 5:4: MAT (memory attribute/type)
        const MAT_MASK = 0b11 << 4;

        /// bit 6 (basic page): G (global)
        const G_BASIC = 1 << 6;

        /// bit 7: P (page present indicator used by software semantics in the manual)
        const P = 1 << 7;

        /// bit 8: W (writable)
        const W = 1 << 8;

        // High bits for LA64 extensions (TLBELO layout / TLB registers)
        /// bit 61: NR (not readable) - LA64 only
        const NR = 1u64 << 61;
        /// bit 62: NX (not executable) - LA64 only
        const NX = 1u64 << 62;
        /// bit 63: RPLV (restricted privilege level enable) - LA64 only
        const RPLV = 1u64 << 63;

        // For convenience, define masks/combined forms:
        /// mask to extract low flag bits (0..12)
        const LOW_MASK = (1 << 12) - 1;
    }
}

/// Memory type enum for LoongArch mappings.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaMemType {
    Device = 0,
    Normal = 1,
    NormalNonCache = 2,
}

impl LaPteAttr {
    // PPN / physical address mask: assume max 48-bit physical addresses, page granularity 4K
    // PPN occupies bits [12..PALEN-1] in the PTE; using conservative mask up to 48 bits here.
    pub const PHYS_ADDR_MASK: u64 = 0x0000_ffff_ffff_f000;

    // MAT encodings (LoongArch uses MAT[1:0] in bits 5:4 for simple encodings)
    // We'll adopt a small mapping for common types; these encodings must match platform's MAT->memory model.
    const MAT_DEVICE: u64 = 0b00 << 4;
    const MAT_NORMAL_WB: u64 = 0b01 << 4;  // treat as cacheable write-back
    const MAT_NORMAL_NC: u64 = 0b10 << 4;  // outer non-cache (example)

    /// Create LaPteAttr representing a LoongArch memory type
    pub const fn from_mem_type(mem: LaMemType) -> Self {
        let bits = match mem {
            LaMemType::Device => (Self::MAT_DEVICE),
            LaMemType::Normal => (Self::MAT_NORMAL_WB | Self::G_BASIC.bits()),
            LaMemType::NormalNonCache => (Self::MAT_NORMAL_NC | Self::G_BASIC.bits()),
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
    pub fn with_plv(mut self, plv: u8) -> Self {
        let plv_val = ((plv as u64) & 0b11) << 2;
        self.bits = (self.bits & !Self::PLV_MASK.bits()) | plv_val;
        self
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
            attr.bits &= !LaPteAttr::NX.bits();
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
    pub const fn empty() -> Self { Self(0) }
    pub fn raw(&self) -> u64 { self.0 }

    pub fn paddr(&self) -> HostPhysAddr {
        HostPhysAddr::from((self.0 & LaPteAttr::PHYS_ADDR_MASK) as usize)
    }
}

impl GenericPTE for LaPTE {
    fn bits(self) -> usize { self.0 as usize }

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
        self.0 = (self.0 & !LaPteAttr::PHYS_ADDR_MASK) | (paddr.as_usize() as u64 & LaPteAttr::PHYS_ADDR_MASK);
    }

    fn set_flags(&mut self, flags: MappingFlags, _is_huge: bool) {
        let mut attr = LaPteAttr::from(flags);
        if flags.contains(MappingFlags::READ) { attr |= LaPteAttr::V; }
        if flags.contains(MappingFlags::WRITE) { attr |= LaPteAttr::D | LaPteAttr::W; }
        // preserve physical address
        self.0 = (self.0 & LaPteAttr::PHYS_ADDR_MASK) | attr.bits();
    }

    fn is_unused(&self) -> bool { self.0 == 0 }

    fn is_present(&self) -> bool { LaPteAttr::from_bits_truncate(self.0).contains(LaPteAttr::V) }

    fn is_huge(&self) -> bool {
        // For LoongArch, huge page is indicated by directory H bit or alignment; here we conservatively:
        // if PPN encodes a large mapping, detection depends on table level. We return false => not huge.
        false
    }

    fn clear(&mut self) { self.0 = 0; }
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
    type VirtAddr = usize; // or GuestVirtAddr as appropriate

    fn flush_tlb(vaddr: Option<Self::VirtAddr>) {
        unsafe {
            if let Some(va) = vaddr {
                // LoongArch provides INVTLB / TLBFLUSH / TLBFILL family.
                // The exact assembly template depends on CPU and toolchain syntax.
                // Example pseudocode (fill in correct operands for your implementation):
                //
                // asm!("invtlb {}, {}", in(reg) va, in(reg) 0); // <-- replace with correct INVTLB usage
                //
                // For a portable approach, you may call a platform-specific crate function here.
                core::arch::asm!("/* INVTLB/TLB invalidate for VA={} (platform-specific) */", in(reg) va);
            } else {
                // global flush (example placeholder)
                core::arch::asm!("/* Global TLB flush (platform-specific INVTLB/TLBFLUSH) */");
            }
        }
    }
}

/// Convenience alias: a page table type for LoongArch using our PTE & metadata.
pub type LoongNestedPageTable<H> = PageTable64<LaPagingMetaData, LaPTE, H>;
