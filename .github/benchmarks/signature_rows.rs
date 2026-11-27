
#[allow(dead_code)] mod baseline;
#[allow(dead_code)] mod optimized;
use std::hint::black_box;
#[cfg(not(count_alloc))] use std::{io::Write, time::Instant};

#[cfg(count_alloc)]
mod alloc {
    use std::{alloc::{GlobalAlloc, Layout, System}, sync::atomic::{AtomicUsize, Ordering::Relaxed}};
    struct Counter;
    #[global_allocator] static GLOBAL: Counter = Counter;
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static BYTES: AtomicUsize = AtomicUsize::new(0);
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    fn raised(n: usize) {
        CALLS.fetch_add(1, Relaxed);
        BYTES.fetch_add(n, Relaxed);
        PEAK.fetch_max(LIVE.fetch_add(n, Relaxed)+n, Relaxed);
    }
    unsafe impl GlobalAlloc for Counter {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let p = unsafe { System.alloc(layout) };
            if !p.is_null() { raised(layout.size()); }
            p
        }
        unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
            LIVE.fetch_sub(layout.size(), Relaxed);
            unsafe { System.dealloc(p,layout) };
        }
        unsafe fn realloc(&self,p:*mut u8,old:Layout,n:usize)->*mut u8 {
            let q=unsafe {System.realloc(p,old,n)};
            if !q.is_null(){LIVE.fetch_sub(old.size(),Relaxed);raised(n);}
            q
        }
    }
    pub fn start()->usize {
        let live=LIVE.load(Relaxed);
        CALLS.store(0,Relaxed);BYTES.store(0,Relaxed);PEAK.store(live,Relaxed);
        live
    }
    pub fn finish(live:usize)->[usize;3] {
        assert_eq!(LIVE.load(Relaxed),live);
        [CALLS.load(Relaxed),BYTES.load(Relaxed),PEAK.load(Relaxed)-live]
    }
}

#[derive(Clone)]
struct Element { row:Option<u32>, comp:u8, kind:u8, mode:u8, usage:u8, index:u32, col:u8 }
macro_rules! signature {
    ($mod:ident,$s:expr) => {
        $s.iter().map(|e|$mod::container::SigElement{
            row:e.row, comp:e.comp,kind:e.kind,interp:e.mode,usage:e.usage,
            index:e.index,col:e.col,cols:1,half:false,name:format!("SEM{}",e.index),
        }).collect::<Vec<_>>()
    };
}
fn rng(s:&mut u64)->u32 {
    *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; (*s>>12) as u32
}
fn fixture(rows:u32,per:u8,sparse:bool,shuffled:bool,mode:u8)->Vec<Element>{
    let mut sig=Vec::new();
    for row in 0..rows {
        for col in 0..per {
            let decided=mode<2 && (col==0 || (mode==1 && col==1));
            sig.push(Element {
                row:Some(if sparse { 100+17*row } else { row }),
                comp:if decided {5} else {9},kind:0,
                mode:if decided {1+col} else {0},usage:0,index:row*per as u32+col as u32,col,
            });
        }
    }
    if shuffled {
        let mut seed=47563;
        for i in (1..sig.len()).rev() {
            sig.swap(i,rng(&mut seed) as usize%(i+1));
        }
    }
    sig
}
fn differential() {
    let mut seed=475637602;
    for i in 0..10_000 {
        let n=(rng(&mut seed)%129) as usize;
        let width=1+((i as u32)%36);
        let s=(0..n).map(|j| {
            let r=rng(&mut seed);
            let normal=(r>>3)%width;
            Element {
                row:match r%8 {0=>None,1=>Some(u32::MAX-normal),_=>Some(normal)},
                comp:[4,5,9,9,9][((r>>8)%5)as usize],kind:u8::from(r%11==0),
                mode:((r>>12)%8)as u8,usage:u8::from(r%7==0),index:j as u32,col:0,
            }
        }).collect::<Vec<_>>();
        for read in [false,true] {
            let mut a=signature!(baseline,s);
            let mut b=signature!(optimized,s);
            baseline::container::same_mode_per_row(&mut a,read);
            optimized::container::same_mode_per_row(&mut b,read);
            for (a,b) in a.iter().zip(&b) {
                assert_eq!((a.interp,a.row,a.index,a.name.as_str(),a.kind,a.comp,a.usage,a.col,a.cols,a.half),(b.interp,b.row,b.index,b.name.as_str(),b.kind,b.comp,b.usage,b.col,b.cols,b.half),"case {i} read {read}");
            }
        }
    }
    eprintln!("20,000 differential cases matched actual upstream output");
}
macro_rules! emit {
    ($mod:ident,$sig:expr,$pixel:expr) => {{
        let mut s=$mod::shader::Shader::new($pixel);
        if $pixel { s.inputs=signature!($mod,$sig); } else { s.outputs=signature!($mod,$sig); }
        s.m.terminate(s.entry,$mod::bitcode::Term::Ret);
        s.finish()
    }};
}
fn metadata() {
    for (rows,per,sparse,shuffle) in [(1,2,false,false),(4,4,true,true),(16,2,false,true),(32,4,false,true)]{
        for mode in 0..3 {
            for pixel in [false,true] {
                let mut s=fixture(rows,per,sparse,shuffle,mode);
                if pixel && rows>1 {s[0].usage=1;}
                let a=emit!(baseline,s,pixel);
                let b=emit!(optimized,s,pixel);
                assert_eq!(a,b,"container mismatch {rows} {per} {sparse} {shuffle} {mode} {pixel}");
                assert!(a.starts_with(b"DXBC"));
                assert!(a.windows(4).any(|w|w==b"DXIL"));
                assert!(a.windows(4).any(|w|w==b"PSV0"));
            }
        }
    }
    eprintln!("24 complete synthetic DXBC containers matched, including DXIL bitcode and PSV0");
}
#[inline(never)]
fn run_old(s:&mut [baseline::container::SigElement]){
    baseline::container::same_mode_per_row(black_box(s),black_box(true));
}
#[inline(never)]
fn run_new(s:&mut [optimized::container::SigElement]){
    optimized::container::same_mode_per_row(black_box(s),black_box(true));
}
const WARM:usize=512;
const ITER:usize=4096;
const BATCH:usize=15;
fn main() {
    differential();
    metadata();
    #[cfg(not(count_alloc))]
    let mut samples = {
        let path = std::env::args_os().nth(1).expect("raw timing CSV path");
        let mut file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        writeln!(file, "rows,elements_per_row,layout,order,mode,batch,baseline_first,baseline_ns,optimized_ns").unwrap();
        file
    };
    #[cfg(count_alloc)]
    println!("rows,elements_per_row,layout,order,mode,baseline_calls,baseline_requested_bytes,baseline_peak_bytes,optimized_calls,optimized_requested_bytes,optimized_peak_bytes");
    #[cfg(not(count_alloc))]
    println!("rows,elements_per_row,layout,order,mode,baseline_median_ns,baseline_min_ns,baseline_max_ns,optimized_median_ns,optimized_min_ns,optimized_max_ns,change_percent");
    for rows in [1,2,4,8,16,32] { for per in [2,4] { for sparse in [false,true] {
       for shuffled in [false,true] { for mode in 0..3 {
        let s=fixture(rows,per,sparse,shuffled,mode);
        let a=signature!(baseline,s);
        let b=signature!(optimized,s);
        for _ in 0..WARM {run_old(&mut a.clone());run_new(&mut b.clone());}
        let layout=if sparse{"sparse"}else{"dense"};
        let order=if shuffled{"shuffled"}else{"sorted"};
        let name=["agreed","conflicting","undecided"][mode as usize];
        #[cfg(count_alloc)] {
            let mut x=a.clone();
            let mut y=b.clone();
            let start=alloc::start();run_old(&mut x);let old=alloc::finish(start);
            let start=alloc::start();run_new(&mut y);let new=alloc::finish(start);
            println!("{rows},{per},{layout},{order},{name},{},{},{},{},{},{}",old[0],old[1],old[2],new[0],new[1],new[2]);
        }
        #[cfg(not(count_alloc))] {
            let mut elapsed = [[0.0; BATCH]; 2];
            let mut old_inputs = vec![a.clone(); ITER];
            let mut new_inputs = vec![b.clone(); ITER];
            for k in 0..BATCH {
                for which in [k % 2, 1 - k % 2] {
                    elapsed[which][k] = if which == 0 {
                        for input in &mut old_inputs {
                            for (e, original) in input.iter_mut().zip(&a) {
                                e.interp = original.interp;
                            }
                        }
                        let start = Instant::now();
                        for input in &mut old_inputs { run_old(input); }
                        start.elapsed().as_secs_f64() * 1e9 / ITER as f64
                    } else {
                        for input in &mut new_inputs {
                            for (e, original) in input.iter_mut().zip(&b) {
                                e.interp = original.interp;
                            }
                        }
                        let start = Instant::now();
                        for input in &mut new_inputs { run_new(input); }
                        start.elapsed().as_secs_f64() * 1e9 / ITER as f64
                    };
                }
                writeln!(
                    samples,
                    "{rows},{per},{layout},{order},{name},{},{},{:.6},{:.6}",
                    k + 1, k % 2 == 0, elapsed[0][k], elapsed[1][k],
                ).unwrap();
            }
            for v in &mut elapsed {v.sort_unstable_by(f64::total_cmp);}
            let [a,b]=elapsed;
            println!("{rows},{per},{layout},{order},{name},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
                a[BATCH/2],a[0],a[BATCH-1],b[BATCH/2],b[0],b[BATCH-1],100.0*(b[BATCH/2]/a[BATCH/2]-1.0));
        }
    }}}}}
}
