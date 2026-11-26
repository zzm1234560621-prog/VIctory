//! Title memory at raw GPU addresses (decision 0005, the unified-memory
//! item; the contract is `victory_shader::emit`'s "Title memory").
//! Translated shaders that read or write memory at addresses they compute
//! look each address up in a page table: per 64 KiB page below 2^40, the
//! heap index of a pair of raw views (SRV, UAV) of the host copy holding
//! the page (`crate::shadow`) and the low dword of that copy's title
//! address. The copies are the ones draws and dispatches use through views,
//! so what one shader writes through a view another reads at its address,
//! within a submission too.
//!
//! Each view heap has one page table, which every list using title memory
//! with that heap bound reads, and each host copy has a pair of views in
//! slots of the heap kept for it while it lives ([`HostHeap::pair`], past
//! the per-draw dynamic slots). At each submission with such a list the
//! table is rewritten if copies came or went since, and every copy it names
//! is brought up to date (page by page, as every refresh; once per
//! submission until a step writes title memory). Shaders mark the pages
//! they write in their list's bitmap; after the submission each marked page
//! goes back into title memory ([`crate::shadow::Shadow::copy_back_page`])
//! and the bitmap is cleared.
//!
//! Cost per submission with such a list: one change check of every host
//! copy (the refresh), copies of the pages the title wrote, a compare of
//! each 64 KiB page shaders wrote, and the table's rewrite when copies came
//! or went (two views per new copy); per list a bitmap of 256 KiB.
//!
//! Memory no copy holds reads 0 and takes no writes (null views,
//! bounds-checked); shaders mark such pages (or 256 MiB regions with no
//! block) in the bitmap and [`log_frame`] counts them.
//! shortcut: a page two copies share without either holding all of it is
//! the larger one's.
//! shortcut: a table names copies in at most [`CAPACITY`] 256 MiB regions;
//! past that, the smallest copies read as missing (logged once).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::descriptor::host::HostHeap;
use crate::host::HostList;
use crate::shadow::Shadow;

/// Pages per level-2 block: 4096 pages of 64 KiB, 256 MiB.
const BLOCK: usize = 4096;
/// Level-1 entries: 2^40 bytes in 256 MiB blocks.
const LEVEL1: usize = 4096;
/// Level-2 blocks a table has room for besides the null one: copies in 512
/// distinct 256 MiB regions (the world's ~4,100 copies use a few dozen).
const CAPACITY: usize = 512;
/// Sentinel in [`Layout::owner`] for a page with no host copy.
const NO_OWNER: u32 = u32::MAX;
/// Total dwords in a page table buffer (level-1 + `CAPACITY + 1` blocks of
/// `BLOCK` 2-dword entries).
const TABLE_DWORDS: usize = LEVEL1 + (CAPACITY + 1) * BLOCK * 2;
/// Bits in a list's dirty-page bitmap (one per level-2 entry across all blocks).
const BITMAP_BITS: usize = (CAPACITY + 1) * BLOCK;
/// 32-bit words in a dirty-page bitmap.
const BITMAP_WORDS: usize = BITMAP_BITS / 32;
/// Byte size of a dirty-page bitmap buffer (256-byte aligned).
const BITMAP_BYTES: u64 = (BITMAP_BITS / 8).next_multiple_of(256) as u64;
/// Maximum number of dirty-page bitmaps kept in [`BITMAP_POOL`].
const BITMAP_POOL_CAP: usize = 64;
/// The key of our own view heap ([`heap`]) among the title's.
pub(crate) const OURS: usize = usize::MAX;

/// A list's use of title memory with a view heap bound.
pub(crate) struct Map {
    /// The heap's page table's host GPU address (the root SRV).
    pub(crate) table: u64,
    /// The list's page bitmap's host GPU address (the root UAV).
    pub(crate) dirty: u64,
    /// Ranges that go back byte for byte instead (root UAVs, see
    /// [`Map::precise`]).
    precise: Arc<Mutex<Vec<(u64, u64)>>>,
}

impl Map {
    /// Shaders write `[address, address + len)` through a root UAV, which
    /// goes back by its bytes before and after the run
    /// ([`Shadow::gpu_writes`]): the marked pages it touches are left to
    /// that. A whole page from the copy would put back what it held at
    /// the submission over what the title (or DirectStorage, through
    /// physical pages no watch sees) wrote there since.
    /// shortcut: a marked page partly in such a range loses what shaders
    /// wrote in its other part.
    pub(crate) fn precise(&self, address: u64, len: u64) {
        self.precise
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, len));
    }
}

/// What a heap's page table names: the copies, the level-1 index of each
/// block (block 0 is the null one), and the copy each entry names (or [`NO_OWNER`]).
#[derive(Default)]
struct Layout {
    copies: Vec<Arc<Shadow>>,
    /// Each copy's bit in the change tracker (`crate::watch::clean_bit`).
    bits: Vec<Option<u32>>,
    blocks: Vec<usize>,
    owner: Vec<u32>,
}

/// A 256 KiB dirty-page bitmap buffer kept zeroed while in [`BITMAP_POOL`].
struct Bitmap {
    _resource: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: `data` is the persistent mapping of `_resource`, owned exclusively by
// one `PooledBitmap` while bound or by `BITMAP_POOL` while idle; D3D12
// resources are free-threaded.
unsafe impl Send for Bitmap {}
// SAFETY: as above; `defer_missed` and `defer_after` access the mapping
// sequentially on the submitting thread after the submission's fence waits.
unsafe impl Sync for Bitmap {}

/// Reusable zeroed dirty-page bitmap buffers, capped at [`BITMAP_POOL_CAP`].
static BITMAP_POOL: Mutex<Vec<Bitmap>> = Mutex::new(Vec::new());

impl Bitmap {
    /// Pops a zeroed bitmap from [`BITMAP_POOL`] or allocates and zeroes a new one.
    fn acquire() -> Option<Self> {
        if let Some(bitmap) = BITMAP_POOL.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            return Some(bitmap);
        }
        let (resource, data) = crate::host::host()?.shared_buffer(BITMAP_BYTES)?;
        crate::host::name(&resource, || "title memory written pages".into());
        // SAFETY: `data` is a newly mapped shared buffer of at least `BITMAP_WORDS` dwords.
        unsafe { std::ptr::write_bytes(data.cast::<u32>(), 0, BITMAP_WORDS) };
        // SAFETY: `resource` is a live D3D12 buffer.
        let gpu = unsafe { resource.GetGPUVirtualAddress() };
        Some(Self {
            _resource: resource,
            data: data as usize,
            gpu,
        })
    }
}

/// RAII guard keeping a [`Bitmap`] owned by a `HostList` until `HostList::reset`
/// or `Drop`, ensuring it is zeroed before returning to [`BITMAP_POOL`].
struct PooledBitmap {
    bitmap: Option<Bitmap>,
}

impl PooledBitmap {
    fn new(bitmap: Bitmap) -> Self {
        Self {
            bitmap: Some(bitmap),
        }
    }

    fn gpu(&self) -> u64 {
        self.bitmap.as_ref().expect("bitmap present until drop").gpu
    }

    fn data(&self) -> usize {
        self.bitmap
            .as_ref()
            .expect("bitmap present until drop")
            .data
    }
}

impl Drop for PooledBitmap {
    fn drop(&mut self) {
        let Some(bitmap) = self.bitmap.take() else {
            return;
        };
        // SAFETY: `PooledBitmap` drops only when `HostList::reset` or `Drop`
        // releases its deferred closures after any GPU submission has finished;
        // `bitmap.data` is mapped for `BITMAP_WORDS` dwords.
        let marks =
            unsafe { std::slice::from_raw_parts_mut(bitmap.data as *mut u32, BITMAP_WORDS) };
        // `defer_after` clears non-zero words after execution and un-executed
        // lists start clean; verify and clear any stray non-zero words before pooling.
        if let Some(first) = marks.iter().position(|&m| m != 0) {
            for mark in &mut marks[first..] {
                if *mark != 0 {
                    *mark = 0;
                }
            }
        }
        let overflow = {
            let mut pool = BITMAP_POOL.lock().unwrap_or_else(|e| e.into_inner());
            if pool.len() < BITMAP_POOL_CAP {
                pool.push(bitmap);
                None
            } else {
                Some(bitmap)
            }
        };
        drop(overflow);
    }
}

/// A view heap's page table, shared by every list using title memory with
/// the heap bound, and the view pairs of the copies it names.
struct Shared {
    /// The host heap it was made for (a heap made later at a released one's
    /// address is another).
    host: usize,
    table: ID3D12Resource,
    data: usize,
    gpu: u64,
    /// The pair of null views entries without a copy name.
    null: u32,
    /// Each copy's view pair (its first slot), by copy.
    pairs: HashMap<usize, u32>,
    /// The copies' generation it names ([`crate::shadow::generation`]).
    generation: u64,
    layout: Arc<Layout>,
}

// SAFETY: `data` is the persistent mapping of `table`, written only under
// the lock; D3D12 resources are free-threaded.
unsafe impl Send for Shared {}

/// Page tables by view heap (the title heap's address, or [`OURS`]).
static SHARED: Mutex<Vec<(usize, Shared)>> = Mutex::new(Vec::new());

fn shared() -> std::sync::MutexGuard<'static, Vec<(usize, Shared)>> {
    SHARED.lock().unwrap_or_else(|e| e.into_inner())
}

/// View heap `key` was released: its page table goes with it (lists that
/// name it hold the table), and a heap made later at its address starts
/// with none of its state.
#[cfg(windows)]
pub(crate) fn release(key: usize) {
    shared().retain(|(k, _)| *k != key);
}

/// Whether view heap `key` has a page table.
#[cfg(test)]
pub(crate) fn has_table(key: usize) -> bool {
    shared().iter().any(|(k, _)| *k == key)
}

/// Runs `f` on the host heap behind view heap `key` while it cannot go
/// away; None if it is gone or has no host heap.
fn with_heap<R>(key: usize, f: impl FnOnce(&HostHeap) -> R) -> Option<R> {
    if key == OURS {
        return heap().map(f);
    }
    crate::descriptor::with_host(key, f)
}

/// Since the last frame line: draws and dispatches with shaders using title
/// memory (by pipeline), page tables written, pages copied back, pages and
/// 256 MiB regions missed (no copy), and the first missed addresses.
#[derive(Default)]
struct Stats {
    uses: std::collections::BTreeMap<String, u64>,
    tables: u64,
    written: u64,
    /// Pages shaders wrote that were not copied back: their list read
    /// memory with no copy, so what they computed is not the console's.
    withheld: u64,
    /// The same for ranges written through views (bytes).
    withheld_bytes: u64,
    missed_pages: u64,
    missed_regions: u64,
    first_missed: Vec<u64>,
}

static STATS: std::sync::Mutex<Option<Stats>> = std::sync::Mutex::new(None);

fn stats<R>(f: impl FnOnce(&mut Stats) -> R) -> R {
    f(STATS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

/// `bytes` of a range shaders wrote through a view did not go back into
/// title memory: their list read memory with no copy.
pub(crate) fn note_withheld(bytes: u64) {
    stats(|s| s.withheld_bytes += bytes);
}

/// A draw or dispatch of `pipeline` uses title memory.
pub(crate) fn note_use(pipeline: &str) {
    stats(|s| *s.uses.entry(pipeline.to_owned()).or_default() += 1);
}

/// Frames presented, and the last one in which a list using title memory
/// read or wrote memory with no copy ([`MISS_NEVER`] if none has).
static FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MISSED_IN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(MISS_NEVER);
const MISS_NEVER: u64 = u64::MAX;

/// Whether a list using title memory in the submission being completed
/// read or wrote memory with no copy (completions run one at a time).
static SUBMISSION_MISSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A submission's lists ran: whether one using title memory read or wrote
/// memory with no copy.
pub(crate) fn note_submission(missed: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    SUBMISSION_MISSED.store(missed, Relaxed);
    if missed {
        MISSED_IN.store(FRAMES.load(Relaxed), Relaxed);
    }
}

/// Whether what lists using title memory wrote stays out of title memory
/// (in the host copies, for the GPU): some such list missed memory this
/// frame or the one before. A list that missed computed from zeros, and
/// the lists after it compute from its results in the host copies, the
/// BVH builder's write addresses among them, so withholding only the lists
/// that missed broke the title's own data (run drops1: an AABB's +inf/-inf
/// in a pointer the title followed). The frame before counts too: its
/// lists' results are read in this one, before this frame's first miss.
pub(crate) fn withholding() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    withheld_in(FRAMES.load(Relaxed), MISSED_IN.load(Relaxed))
}

fn withheld_in(frame: u64, missed: u64) -> bool {
    missed != MISS_NEVER && frame <= missed.saturating_add(1)
}

/// Title memory whose host copies hold what shaders wrote this frame and
/// the frame withheld ([`withholding`]; ranges).
static DROPPED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Shaders wrote `[address, address + len)` of title memory in host copies
/// and the submission withheld it: the copies take title memory's bytes
/// again at their next refresh, from right after the submission if a list
/// of it missed (its results would feed the frame's later submissions:
/// garbage BVH nodes over meshlet buffers, run shards20), else from the
/// frame's end ([`retake_dropped`]).
pub(crate) fn drop_withheld(address: usize, len: usize) {
    if SUBMISSION_MISSED.load(std::sync::atomic::Ordering::Relaxed) {
        crate::watch::retake(address, len);
    } else {
        DROPPED.lock().unwrap_or_else(|e| e.into_inner()).push((address, len));
    }
}

/// At a frame's end: what the frame withheld leaves the host copies, so
/// that results computed from memory with no copy (garbage record arrays
/// and node pointers of the BVH builder) do not feed later frames whose
/// writes go back (a box16 node's f16 bounds over a title object, run
/// textures9). Each copy holding such memory takes it from title memory at
/// its next refresh, through its tracker.
fn retake_dropped() {
    let dropped = std::mem::take(&mut *DROPPED.lock().unwrap_or_else(|e| e.into_inner()));
    for (address, len) in dropped {
        crate::watch::retake(address, len);
    }
}

/// Logs what title memory did since the last call, if anything: a line per
/// presented frame while shaders use it.
pub(crate) fn log_frame() {
    FRAMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    retake_dropped();
    let s = stats(std::mem::take);
    if s.uses.is_empty() && s.written + s.withheld + s.withheld_bytes + s.missed_pages + s.missed_regions == 0 {
        return;
    }
    let mut uses: Vec<(&String, &u64)> = s.uses.iter().collect();
    uses.sort_by(|a, b| b.1.cmp(a.1));
    let uses: Vec<String> = uses.iter().take(32).map(|(p, n)| format!("{n} {p}")).collect();
    let missed: Vec<String> = s.first_missed.iter().map(|a| format!("{a:#x}")).collect();
    tracing::info!(
        "title memory this frame: {} draws/dispatches ({}), {} page tables, {} pages \
         copied back, {} withheld and {:.1} MB of views kept (their list read memory with no copy), missed {} pages \
         and {} 256 MiB regions with no copy (first: {})",
        s.uses.values().sum::<u64>(),
        uses.join(", "),
        s.tables,
        s.written,
        s.withheld,
        s.withheld_bytes as f64 / 1e6,
        s.missed_pages,
        s.missed_regions,
        missed.join(" ")
    );
}

/// Our view heap for lists that bind none of the title's (dynamic slots
/// only).
pub(crate) fn heap() -> Option<&'static crate::descriptor::host::HostHeap> {
    static HEAP: std::sync::OnceLock<Option<crate::descriptor::host::HostHeap>> =
        std::sync::OnceLock::new();
    HEAP.get_or_init(|| crate::descriptor::host::HostHeap::new(false, 0))
        .as_ref()
}

/// Writes the page table for host copies `copies` (start, bytes), copy `k`'s
/// view pair at heap index `pairs[k]`, and `null` for unowned pages directly into
/// `out` (which must hold at least [`TABLE_DWORDS`] dwords). Returns
/// `(written_dwords, blocks, owner)` where `owner` holds copy indices or
/// [`NO_OWNER`]. At most [`CAPACITY`] blocks besides the null one.
fn write_table(
    copies: &[(u64, u64)],
    pairs: &[u32],
    null: u32,
    out: &mut [u32],
) -> (usize, Vec<usize>, Vec<u32>) {
    assert!(out.len() >= TABLE_DWORDS);
    let mut order: Vec<usize> = (0..copies.len()).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(copies[k].1));
    // Build level-1 in cached stack memory so we never read from write-combined upload memory.
    let mut level1 = [0u32; LEVEL1];
    // Block 0 is the null block (no copy).
    let mut blocks = vec![usize::MAX];
    let mut owner = vec![NO_OWNER; BLOCK];
    let mut full = false;
    for &k in &order {
        let (start, len) = copies[k];
        let Some(end) = start.checked_add(len) else {
            continue;
        };
        if len == 0 || end > 1 << 40 {
            continue;
        }
        for page in start >> 16..=(end - 1) >> 16 {
            let l1 = (page >> 12) as usize;
            if level1[l1] == 0 {
                if blocks.len() > CAPACITY {
                    full = true;
                    continue;
                }
                level1[l1] = blocks.len() as u32;
                blocks.push(l1);
                owner.resize(owner.len() + BLOCK, NO_OWNER);
            }
            let entry = level1[l1] as usize * BLOCK + (page & 0xFFF) as usize;
            // Larger copies come first: a copy holding the whole page takes
            // it from them; one holding part of it only takes it if free.
            let whole = page << 16 >= start && (page + 1) << 16 <= end;
            if whole || owner[entry] == NO_OWNER {
                owner[entry] = k as u32;
            }
        }
    }
    if full {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(CAPACITY, "title memory copies in more 256 MiB regions than a page table has room for: the rest read as missing"));
    }
    out[..LEVEL1].copy_from_slice(&level1);
    let l2_dwords = owner.len() * 2;
    let l2 = &mut out[LEVEL1..LEVEL1 + l2_dwords];
    let (slots, _) = l2.as_chunks_mut::<2>();
    for (slot, &o) in slots.iter_mut().zip(&owner) {
        if o == NO_OWNER {
            slot[0] = null;
            slot[1] = 0;
        } else {
            let k = o as usize;
            slot[0] = pairs[k];
            slot[1] = copies[k].0 as u32;
        }
    }
    (LEVEL1 + l2_dwords, blocks, owner)
}

#[cfg(test)]
fn table(
    copies: &[(u64, u64)],
    pairs: &[u32],
    null: u32,
) -> (Vec<u32>, Vec<usize>, Vec<Option<usize>>) {
    let mut dwords = vec![0u32; TABLE_DWORDS];
    let (len, blocks, owner) = write_table(copies, pairs, null, &mut dwords);
    dwords.truncate(len);
    let owner = owner
        .into_iter()
        .map(|o| (o != NO_OWNER).then_some(o as usize))
        .collect();
    (dwords, blocks, owner)
}

impl Shared {
    /// A page table for `heap`, named nothing yet (written at the first
    /// submission that uses it).
    fn new(heap: &HostHeap) -> Option<Shared> {
        let bytes = TABLE_DWORDS as u64 * 4;
        let (table, data) = crate::host::host()?.upload_buffer(bytes)?;
        crate::host::name(&table, || "title memory page table".into());
        let null = heap.pair()?;
        heap.raw_pair_at(null, None, 0);
        // SAFETY: a live buffer.
        let gpu = unsafe { table.GetGPUVirtualAddress() };
        Some(Shared {
            host: std::ptr::from_ref(heap) as usize,
            table,
            data: data as usize,
            gpu,
            null,
            pairs: HashMap::new(),
            generation: u64::MAX,
            layout: Arc::default(),
        })
    }

    /// Names every host copy there is now (`now`: their generation): view
    /// pairs made for new copies, those of copies gone made null and given
    /// back, the table rewritten. Only while no submission runs (the GPU
    /// reads the table and the views).
    fn rewrite(&mut self, heap: &HostHeap, now: u64) {
        let copies = crate::shadow::all();
        let live: HashSet<usize> = copies.iter().map(|c| Arc::as_ptr(c) as usize).collect();
        self.pairs.retain(|copy, &mut slot| {
            let keep = live.contains(copy);
            if !keep {
                heap.raw_pair_at(slot, None, 0);
                heap.free_pair(slot);
            }
            keep
        });
        let (mut named, mut slots) = (Vec::with_capacity(copies.len()), Vec::with_capacity(copies.len()));
        for c in copies {
            let key = Arc::as_ptr(&c) as usize;
            let slot = match self.pairs.get(&key) {
                Some(&slot) => slot,
                None => {
                    let Some(slot) = heap.pair() else {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| tracing::warn!("no descriptor slots for a title memory copy's views: it reads as missing"));
                        continue;
                    };
                    heap.raw_pair_at(slot, Some(&c.buffer), c.range().1.next_multiple_of(4));
                    self.pairs.insert(key, slot);
                    slot
                }
            };
            named.push(c);
            slots.push(slot);
        }
        let ranges: Vec<(u64, u64)> = named.iter().map(|c| c.range()).collect();
        // SAFETY: `self.data` is the persistent mapping of `self.table`, with room
        // for `TABLE_DWORDS` dwords; no submission runs.
        let out = unsafe { std::slice::from_raw_parts_mut(self.data as *mut u32, TABLE_DWORDS) };
        let (_, blocks, owner) = write_table(&ranges, &slots, self.null, out);
        tracing::debug!(copies = ranges.len(), blocks = blocks.len() - 1, "title memory page table");
        self.layout = Arc::new(Layout {
            bits: named.iter().map(|c| c.dirty_bit()).collect(),
            copies: named,
            blocks,
            owner,
        });
        self.generation = now;
        stats(|s| s.tables += 1);
    }
}

/// The page table of view heap `key`, as the last submission using it
/// left it.
fn layout(key: usize) -> Arc<Layout> {
    shared()
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, s)| s.layout.clone())
        .unwrap_or_default()
}

/// At submission, for a list using title memory with view heap `key`
/// bound: the heap's page table names every host copy there is, each
/// brought up to date (`add` takes a copy's range, once per submission
/// until a step writes title memory).
pub(crate) fn prepare(key: usize, mut add: impl FnMut(&Arc<Shadow>, usize, usize)) {
    let layout = with_heap(key, |heap| {
        let mut shared = shared();
        let host = std::ptr::from_ref(heap) as usize;
        let (_, s) = shared.iter_mut().find(|(k, s)| *k == key && s.host == host)?;
        let now = crate::shadow::generation();
        if s.generation != now {
            s.rewrite(heap, now);
        }
        Some(s.layout.clone())
    })
    .flatten();
    // Copies the title has not written since they took it are passed by
    // without touching them (the table names every copy, ~4,700 in the world).
    for l in layout.iter() {
        for (c, bit) in l.copies.iter().zip(&l.bits) {
            if !crate::watch::clean_bit(*bit) {
                let (start, len) = c.range();
                add(c, start as usize, len as usize);
            }
        }
    }
}

/// Whether any marked page in `marks` fell outside a valid copy in `l`.
fn has_missed(marks: &[u32], l: &Layout) -> bool {
    for (w, &mark) in marks.iter().enumerate() {
        let mut bits = mark;
        while bits != 0 {
            let entry = w * 32 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let block = entry / BLOCK;
            if block == 0
                || l.blocks.get(block).is_none()
                || l.owner.get(entry).is_none_or(|&o| o == NO_OWNER)
            {
                return true;
            }
        }
    }
    false
}

/// `list` uses title memory with view heap `key` bound: the heap's page
/// table (made on first use) and a page bitmap of the list's, the table
/// brought up to date at each submission of the list, its written pages
/// copied back after.
pub(crate) fn bind(list: &HostList, key: usize) -> Option<Map> {
    let table = with_heap(key, |heap| {
        let mut shared = shared();
        let host = std::ptr::from_ref(heap) as usize;
        if let Some((_, s)) = shared.iter().find(|(k, s)| *k == key && s.host == host) {
            list.keep(&s.table);
            return Some(s.gpu);
        }
        // A released heap's table goes with it (lists that name it hold it).
        shared.retain(|(k, _)| *k != key);
        let s = Shared::new(heap)?;
        list.keep(&s.table);
        let gpu = s.gpu;
        shared.push((key, s));
        Some(gpu)
    })
    .flatten()?;
    let pooled = Arc::new(PooledBitmap::new(Bitmap::acquire()?));
    let dirty = pooled.gpu();
    list.defer_memory(key);
    let precise: Arc<Mutex<Vec<(u64, u64)>>> = Arc::default();
    let byte_for_byte = precise.clone();
    let missed_bitmap = Arc::clone(&pooled);
    list.defer_missed(move || {
        let l = layout(key);
        // SAFETY: the bitmap is mapped and the list's run is over.
        let marks =
            unsafe { std::slice::from_raw_parts(missed_bitmap.data() as *const u32, BITMAP_WORDS) };
        has_missed(marks, &l)
    });
    list.defer_after(move || {
        let Layout { copies, blocks, owner, .. } = &*layout(key);
        let precise = byte_for_byte.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the bitmap is mapped and the list's run is over.
        let marks =
            unsafe { std::slice::from_raw_parts_mut(pooled.data() as *mut u32, BITMAP_WORDS) };
        let (mut pages, mut regions, mut missed) = (0, 0, Vec::new());
        let mut writes = Vec::new();
        for (w, mark) in marks.iter_mut().enumerate() {
            if *mark == 0 {
                continue;
            }
            let mut bits = std::mem::take(mark);
            while bits != 0 {
                let entry = w * 32 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let (block, page) = (entry / BLOCK, entry % BLOCK);
                // Block 0's bits are 256 MiB regions with no block.
                let Some(&l1) = blocks.get(block).filter(|_| block != 0) else {
                    regions += 1;
                    missed.push((entry as u64) << 28);
                    continue;
                };
                let at = (l1 << 28 | page << 16) as u64;
                match owner.get(entry) {
                    Some(_) if precise.iter().any(|&(a, n)| a < at + (1 << 16) && at < a + n) => {}
                    Some(&k) if k != NO_OWNER => writes.push((k as usize, at)),
                    _ => {
                        pages += 1;
                        missed.push(at);
                    }
                }
            }
        }
        // What shaders computed from memory with no copy (read as zeros) is
        // not the console's: it does not go into title memory, where it broke
        // the title's own data (garbage pointers in title threads, run c1),
        // nor while any list using title memory misses ([`withholding`]).
        let withheld = if pages + regions > 0 || withholding() { writes.len() as u64 } else { 0 };
        if withheld > 0 {
            for &(_, at) in &writes {
                drop_withheld(at as usize, 1 << 16);
            }
        }
        let mut written = 0;
        for &(k, at) in writes.iter().take_while(|_| withheld == 0) {
            let n = copies[k].copy_back_page(at as usize);
            tracing::debug!(
                page = format_args!("{at:#x}"),
                copy = format_args!("{:#x?}", copies[k].range()),
                bytes = n,
                "title memory page written back"
            );
            written += 1;
        }
        stats(|s| {
            s.written += written;
            s.withheld += withheld;
            s.missed_pages += pages;
            s.missed_regions += regions;
            let room = 4usize.saturating_sub(s.first_missed.len());
            s.first_missed.extend(missed.into_iter().take(room));
        });
    });
    Some(Map {
        table,
        dirty,
        precise,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes of lists using title memory are held back from a frame with a
    /// miss through the next one, then go back again.
    #[test]
    fn a_miss_holds_back_its_frame_and_the_next() {
        assert!(!withheld_in(0, MISS_NEVER));
        assert!(!withheld_in(u64::MAX, MISS_NEVER));
        assert!(withheld_in(7, 7));
        assert!(withheld_in(8, 7), "the next frame reads this one's results");
        assert!(!withheld_in(9, 7));
    }

    /// The shader's lookup (`victory_shader` lowering) over a table's
    /// dwords: the view index and the byte offset in that view.
    fn look_up(dwords: &[u32], address: u64) -> (u32, u32) {
        let (lo, hi) = (address as u32, (address >> 32) as u32);
        let inside = hi < 256;
        let l1 = if inside { hi << 4 | lo >> 28 } else { 0 };
        let block = if inside { dwords[l1 as usize] } else { 0 };
        let entry = block << 12 | (lo >> 16 & 0xFFF);
        let at = 4096 + 2 * entry as usize;
        (dwords[at], lo.wrapping_sub(dwords[at + 1]))
    }

    #[test]
    fn addresses_find_the_copy_holding_them() {
        // A 4 MiB window, a buffer inside it from 0x40_0000+0x1_8000
        // (pages in part), and a buffer above 4 GiB.
        let copies = [
            (0x40_0000, 0x40_0000),
            (0x41_8000, 0x3_0000),
            (0xd_1234_0000, 0x2_0000),
        ];
        let (dwords, blocks, owner) = table(&copies, &[102, 104, 106], 100);
        // Window, whole pages of the inner buffer, the high buffer.
        assert_eq!(look_up(&dwords, 0x40_0010), (102, 0x10));
        assert_eq!(look_up(&dwords, 0x42_0004), (104, 0x8004));
        assert_eq!(look_up(&dwords, 0xd_1235_fffc), (106, 0x1_fffc));
        // Pages the inner buffer holds in part stay the window's.
        assert_eq!(look_up(&dwords, 0x41_8000), (102, 0x1_8000));
        assert_eq!(look_up(&dwords, 0x44_0000), (102, 0x4_0000));
        // No copy, and above 2^40: the null pair, which reads nothing.
        assert_eq!(look_up(&dwords, 0x9000_0000).0, 100);
        assert_eq!(look_up(&dwords, 0x200_0000_0000).0, 100);
        // Entries map back to their copy and page.
        let entry = |a: u64| blocks.iter().position(|&l| l == (a >> 28) as usize).unwrap() * BLOCK
            + (a >> 16 & 0xFFF) as usize;
        assert_eq!(owner[entry(0x42_0000)], Some(1));
        assert_eq!(owner[entry(0xd_1234_0000)], Some(2));
        assert_eq!(blocks.len(), 3, "null, low 256 MiB, the high buffer's");
    }

    /// Copies in more regions than a table has room for: the largest are
    /// named, the rest read as missing; the table fits its buffer.
    #[test]
    fn a_full_table_names_the_largest_copies() {
        let copies: Vec<(u64, u64)> = (0..CAPACITY as u64 + 2).map(|k| (k << 28, 0x1_0000 + k)).collect();
        let pairs: Vec<u32> = (0..copies.len() as u32).map(|k| 10 + 2 * k).collect();
        let (dwords, blocks, _) = table(&copies, &pairs, 4);
        assert_eq!(blocks.len(), CAPACITY + 1);
        assert_eq!(dwords.len(), LEVEL1 + (CAPACITY + 1) * BLOCK * 2);
        assert_eq!(look_up(&dwords, (CAPACITY as u64 + 1) << 28).0, pairs[CAPACITY + 1]);
        assert_eq!(look_up(&dwords, 0).0, 4, "the smallest copy is left out");
    }

    #[test]
    fn table_skips_overflowing_and_out_of_range_copies() {
        let copies = [
            (u64::MAX - 0x100, 0x200),
            (1u64 << 40, 0x1_0000),
            (0x40_0000, 0x1_0000),
        ];
        let (dwords, blocks, owner) = table(&copies, &[102, 104, 106], 100);
        assert_eq!(blocks, vec![usize::MAX, 0]);
        assert_eq!(look_up(&dwords, 0x40_0010), (106, 0x10));
        assert_eq!(look_up(&dwords, 0).0, 100);
        assert_eq!(owner[BLOCK + 0x40], Some(2));
    }

    #[test]
    fn has_missed_checks_block_zero_out_of_range_unowned_and_owned() {
        let mut owner = vec![NO_OWNER; 2 * BLOCK];
        owner[BLOCK + 5] = 0;
        let l = Layout {
            copies: Vec::new(),
            bits: Vec::new(),
            blocks: vec![usize::MAX, 0],
            owner,
        };
        let mut marks = vec![0u32; BITMAP_WORDS];
        assert!(!has_missed(&marks, &l));

        // Owned entry in block 1.
        let owned = BLOCK + 5;
        marks[owned / 32] = 1 << (owned % 32);
        assert!(!has_missed(&marks, &l));

        // Unowned entry in the same word as an owned entry.
        let unowned = BLOCK + 6;
        marks[unowned / 32] |= 1 << (unowned % 32);
        assert!(has_missed(&marks, &l));

        // Block 0 bit (missing 256 MiB region).
        marks.fill(0);
        marks[0] = 1 << 3;
        assert!(has_missed(&marks, &l));

        // Out-of-range block.
        marks.fill(0);
        let oob = 2 * BLOCK + 1;
        marks[oob / 32] = 1 << (oob % 32);
        assert!(has_missed(&marks, &l));
    }

    fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn bitmap_pool_reuses_and_zeroes_across_resets() {
        let _guard = gpu_test_lock();
        let (Some(host), Some(_views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        let a = host.new_list().unwrap();
        let b = host.new_list().unwrap();
        let map_a = bind(&a, OURS).expect("bitmap for list a");
        let map_b = bind(&b, OURS).expect("bitmap for list b");
        let va_a = map_a.dirty;
        let va_b = map_b.dirty;
        assert_ne!(va_a, va_b, "simultaneously open lists get distinct bitmaps");
        drop((map_a, map_b));
        a.close();
        a.reset();
        let map_a2 = bind(&a, OURS).expect("reused bitmap for list a");
        assert_eq!(map_a2.dirty, va_a, "reset returns bitmap to pool for reuse");
        assert_ne!(map_a2.dirty, va_b, "list b still holds its bitmap");
        drop(map_a2);
        a.close();
        b.close();
        a.reset();
        b.reset();

        // Also verify PooledBitmap zeroes dirty words when dropped without executing.
        let pooled = PooledBitmap::new(Bitmap::acquire().unwrap());
        let data = pooled.data();
        // SAFETY: exclusive ownership of this newly acquired bitmap.
        unsafe { (data as *mut u32).add(7).write(0xDEAD_BEEF) };
        drop(pooled);
        let reused = Bitmap::acquire().unwrap();
        // SAFETY: `reused.data` is mapped for `BITMAP_WORDS` dwords.
        let words = unsafe { std::slice::from_raw_parts(reused.data as *const u32, BITMAP_WORDS) };
        assert!(
            words.iter().all(|&w| w == 0),
            "pooled bitmap must be zeroed on return"
        );
        drop(PooledBitmap::new(reused));
    }

    /// Reads 4 dwords of title memory at (`lo`, `hi`) through the page
    /// table and the heap's views, as translated shaders do, into `dst`
    /// from byte `at`.
    const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=unbounded)), SRV(t0, space=2), UAV(u0), RootConstants(num32BitConstants=3, b0)"
ByteAddressBuffer views[] : register(t0);
ByteAddressBuffer table : register(t0, space2);
RWByteAddressBuffer dst : register(u0);
cbuffer c : register(b0) { uint lo; uint hi; uint at; }
[RootSignature(RS)]
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint a = lo + id.x * 4;
    bool inside = hi < 256;
    uint l1 = inside ? (hi << 4 | a >> 28) : 0;
    uint block = inside ? table.Load(l1 * 4) : 0;
    uint entry = block << 12 | (a >> 16 & 0xFFF);
    uint2 e = table.Load2((4096 + 2 * entry) * 4);
    dst.Store(at + id.x * 4, views[NonUniformResourceIndex(e.x)].Load(a - e.y));
}
"#;

    fn pipeline(source: &str) -> (
        windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
        windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    ) {
        use windows::Win32::Graphics::Direct3D::Fxc::{D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES, D3DCompile};
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::core::s;
        let host = crate::host::host().unwrap();
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names outlive the call; the errors are text.
        unsafe {
            let made = D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("memory"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES,
                0,
                &mut code,
                Some(&mut errors),
            );
            let text = errors.map(|e| String::from_utf8_lossy(std::slice::from_raw_parts(e.GetBufferPointer().cast::<u8>(), e.GetBufferSize())).into_owned());
            assert!(made.is_ok(), "{text:?}");
        }
        let code = code.unwrap();
        // SAFETY: the blob's own bytes; a compiled shader with its root
        // signature inside.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        }
    }

    /// A copy's view pair, in view heap `key`'s table.
    fn pair_of(key: usize, copy: &Arc<Shadow>) -> Option<u32> {
        shared()
            .iter()
            .find(|(k, _)| *k == key)?
            .1
            .pairs
            .get(&(Arc::as_ptr(copy) as usize))
            .copied()
    }

    /// Two lists of a submission read title memory through one page table;
    /// the copies' views stay where they are for later lists and
    /// submissions, also when another copy comes; none takes a per-draw
    /// dynamic slot.
    #[test]
    fn lists_share_a_table_and_copies_keep_their_views() {
        let _guard = gpu_test_lock();
        use crate::watch::PAGE;
        let (Some(host), Some(views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        let (root, pso) = pipeline(SOURCE);
        let (out, at) = host.shared_buffer(256).unwrap();
        // SAFETY: a live buffer.
        let out_gpu = unsafe { out.GetGPUVirtualAddress() };
        // Title memory (below 2^40, as the runtime places it): 1, 2 and
        // (from the second submission on) 1 page, write-watched as its pools.
        let pool = |pages: usize| -> usize {
            use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
            (0..256u64)
                .map(|k| 0x40_0000_0000 + (k << 28))
                // SAFETY: a fresh allocation where nothing is mapped, or none.
                .map(|hint| unsafe { VirtualAlloc(hint as *const _, pages * PAGE, MEM_RESERVE | MEM_COMMIT | crate::watch::tests::MEM_WRITE_WATCH, PAGE_READWRITE) } as usize)
                .find(|&p| p != 0)
                .expect("memory below 2^40")
        };
        let memory = [pool(1), pool(2), pool(1)];
        let copy = |p: usize, pages: usize| {
            crate::shadow::register(p, p as u64, (pages * PAGE) as u64);
            crate::shadow::find(p as u64, (pages * PAGE) as u64).unwrap()
        };
        let (a, b) = (copy(memory[0], 1), copy(memory[1], 2));
        let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
        // Each list reads 4 dwords at its address into its 16 bytes of `out`.
        let record = |reads: [usize; 2]| -> Vec<Map> {
            let maps: Vec<Map> = lists.iter().map(|l| bind(l, OURS).expect("a page table")).collect();
            for (k, (list, &address)) in lists.iter().zip(&reads).enumerate() {
                let constants = [address as u32, (address >> 32) as u32, 16 * k as u32];
                // SAFETY: an open list, live objects of ours.
                unsafe {
                    list.list.SetDescriptorHeaps(&[Some(views.heap.clone())]);
                    list.list.SetComputeRootSignature(&root);
                    list.list.SetComputeRootDescriptorTable(0, views.gpu(0));
                    list.list.SetComputeRootShaderResourceView(1, maps[k].table);
                    list.list.SetComputeRootUnorderedAccessView(2, out_gpu);
                    list.list.SetComputeRoot32BitConstants(3, 3, constants.as_ptr().cast(), 0);
                    list.list.SetPipelineState(&pso);
                    list.list.Dispatch(1, 1, 1);
                }
                list.close();
            }
            maps
        };
        // The title writes after recording, before submitting.
        let fill = |p: usize, base: u32| {
            for k in 0..4 {
                // SAFETY: our pool (a guarded page faults into the handler).
                unsafe { (p as *mut u32).add(k).write_volatile(base + k as u32) };
            }
        };
        let read = || -> Vec<u32> {
            // SAFETY: the mapped output, the GPU done with it.
            unsafe { std::slice::from_raw_parts(at.cast::<u32>(), 8).to_vec() }
        };
        let maps = record([memory[0] + 16, memory[1] + PAGE]);
        assert_eq!(maps[0].table, maps[1].table, "one table for the heap");
        assert_ne!(maps[0].dirty, maps[1].dirty, "a bitmap per list");
        fill(memory[0] + 16, 0x100);
        fill(memory[1] + PAGE, 0x200);
        host.execute(&[&lists[0], &lists[1]]);
        assert_eq!(read(), [0x100, 0x101, 0x102, 0x103, 0x200, 0x201, 0x202, 0x203]);
        let pairs = (pair_of(OURS, &a).unwrap(), pair_of(OURS, &b).unwrap());
        assert_ne!(pairs.0, pairs.1, "a pair per copy");
        // Where the pairs live is the host's business: with the pair region
        // above the dynamic slots they come from there, and on a heap capped
        // below what the region needs it is empty and `pair` takes dynamic
        // slots instead (#3178). Only the first layout says anything about
        // the indices.
        if views.has_pair_region() {
            let first = views.first_pair_slot();
            assert!(pairs.0 >= first && pairs.1 >= first, "past the dynamic slots");
        }
        // A second submission, after a third copy came.
        let c = copy(memory[2], 1);
        for list in &lists {
            list.reset();
        }
        record([memory[2] + 32, memory[0] + 16]);
        fill(memory[2] + 32, 0x300);
        fill(memory[0] + 16, 0x400);
        host.execute(&[&lists[0], &lists[1]]);
        assert_eq!(read(), [0x300, 0x301, 0x302, 0x303, 0x400, 0x401, 0x402, 0x403]);
        assert_eq!((pair_of(OURS, &a).unwrap(), pair_of(OURS, &b).unwrap()), pairs, "views kept");
        assert!(pair_of(OURS, &c).is_some());
        for (&p, copy) in memory.iter().zip([a, b, c]) {
            crate::shadow::unregister(p, p as u64);
            drop(copy);
        }
    }

    /// Writes `value` at (`lo`, `hi`) through the page table, marking its
    /// page as translated shaders do, and with `missing_hi` non-zero reads
    /// (`missing_lo`, `missing_hi`), marking it if no copy holds it.
    const WRITE: &str = r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=unbounded)), SRV(t0, space=2), UAV(u0, space=2), RootConstants(num32BitConstants=5, b0)"
RWByteAddressBuffer views[] : register(u0);
ByteAddressBuffer table : register(t0, space2);
RWByteAddressBuffer dirty : register(u0, space2);
cbuffer c : register(b0) { uint lo; uint hi; uint value; uint missing_lo; uint missing_hi; }
void access(uint a, uint h, bool write) {
    bool inside = h < 256;
    uint l1 = inside ? (h << 4 | a >> 28) : 0;
    uint block = inside ? table.Load(l1 * 4) : 0;
    uint entry = block << 12 | (a >> 16 & 0xFFF);
    uint2 e = table.Load2((4096 + 2 * entry) * 4);
    bool miss = e.x == table.Load(4096 * 4);
    uint n = miss && block == 0 ? l1 : entry;
    if (write || miss) {
        dirty.InterlockedOr(n >> 5 << 2, 1u << (n & 31));
    }
    if (write && !miss) {
        views[NonUniformResourceIndex(e.x + 1)].Store(a - e.y, value);
    }
}
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    access(lo, hi, true);
    if (missing_hi != 0) {
        access(missing_lo, missing_hi, false);
    }
}
"#;

    /// What a submission that missed memory wrote through the page table
    /// leaves the host copies right after it, so the next submission reads
    /// title memory's bytes (run shards20: garbage BVH nodes over meshlet
    /// buffers until the frame's end); one that missed nothing leaves its
    /// writes in the copies.
    #[test]
    fn a_submission_that_missed_leaves_the_copies_at_once() {
        let _guard = gpu_test_lock();
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let (Some(host), Some(views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        let ((write_root, writer), (read_root, reader)) = (pipeline(WRITE), pipeline(SOURCE));
        let (out, out_cpu) = host.shared_buffer(256).unwrap();
        // SAFETY: a live buffer.
        let out_gpu = unsafe { out.GetGPUVirtualAddress() };
        // A page of title memory below 2^40, write-watched as the pools.
        let p = (0..256u64)
            .map(|k| 0x50_0000_0000 + (k << 28))
            // SAFETY: a fresh allocation where nothing is mapped, or none.
            .map(|hint| unsafe { VirtualAlloc(hint as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | crate::watch::tests::MEM_WRITE_WATCH, PAGE_READWRITE) } as usize)
            .find(|&p| p != 0)
            .expect("memory below 2^40");
        let at = p + 16;
        // SAFETY: our pool, not guarded yet.
        unsafe { (at as *mut u32).write_volatile(0x11) };
        crate::shadow::register(p, p as u64, PAGE as u64);
        let copy = crate::shadow::find(p as u64, PAGE as u64).unwrap();
        let list = host.new_list().unwrap();
        // One submission of one dispatch; the list is open again after it.
        let submit = |root: &windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
                      pso: &windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
                      uav: Option<u64>,
                      constants: &[u32]| {
            let map = bind(&list, OURS).expect("a page table");
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(views.heap.clone())]);
                list.list.SetComputeRootSignature(root);
                list.list.SetComputeRootDescriptorTable(0, views.gpu(0));
                list.list.SetComputeRootShaderResourceView(1, map.table);
                list.list.SetComputeRootUnorderedAccessView(2, uav.unwrap_or(map.dirty));
                list.list.SetComputeRoot32BitConstants(3, constants.len() as u32, constants.as_ptr().cast(), 0);
                list.list.SetPipelineState(pso);
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            host.execute(&[&list]);
            list.reset();
        };
        let write = |value: u32, miss: bool| {
            // Memory no copy holds, in a 256 MiB region with no block.
            let missing: u64 = if miss { 0xff_f000_0000 } else { 0 };
            let constants = [at as u32, (at >> 32) as u32, value, missing as u32, (missing >> 32) as u32];
            submit(&write_root, &writer, None, &constants);
        };
        let read = || -> u32 {
            submit(&read_root, &reader, Some(out_gpu), &[at as u32, (at >> 32) as u32, 0]);
            // SAFETY: the mapped output, the GPU done with it.
            unsafe { out_cpu.cast::<u32>().read_volatile() }
        };
        // SAFETY: our pool.
        let title = || unsafe { (at as *const u32).read_volatile() };
        write(0x33, true);
        // Another test's submission may have refreshed the copy already.
        let held = copy.read_host(at as u64, 1)[0];
        assert!(!copy.clean() || held == 0x11, "the written page counts as changed at once");
        assert_eq!(title(), 0x11, "nothing went back");
        assert_eq!(read(), 0x11, "the next submission reads title memory's bytes");
        // Unless another test ends a frame meanwhile (its retake is right).
        let frame = FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        write(0x44, false);
        let kept = read();
        if FRAMES.load(std::sync::atomic::Ordering::Relaxed) == frame {
            assert_eq!(kept, 0x44, "a submission that missed nothing keeps its writes");
        }
        crate::shadow::unregister(p, p as u64);
        drop(copy);
    }
}
