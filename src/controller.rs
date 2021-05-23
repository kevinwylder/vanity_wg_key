pub use super::{ KeyBuffer, Scorer, Match, MultiMatch };
use std::{
    sync::{
        Mutex,
        RwLock,
        Arc,
        atomic::{ AtomicBool, AtomicU64, Ordering},
    },
    time::{ Instant, Duration },
};
use crossbeam_channel::{ Sender, Receiver };
use crossbeam_utils::thread::Scope;

pub struct ComputeController<'a> {
    threads: usize,
    reseed_rate: u32,
    terms: &'a Scorer,
    sender: Mutex<Sender<MultiMatch>>,
    loop_durations: Mutex<Vec<Arc<RwLock<Duration>>>>,
    running: AtomicBool,
    total_hashes: AtomicU64,
}

impl ComputeController<'_> {

    pub fn new<'a, 'running>(
        terms: &'a Scorer,
        scope: &Scope<'running>,
        threads: usize,
    ) -> (Arc<ComputeController<'a>>, Receiver<MultiMatch>) where 'a : 'running{
        let (send, receive) = crossbeam_channel::unbounded();

        let mut loop_durations = vec![];
        for _ in 0..threads {
            loop_durations.push(Arc::new(RwLock::new(Duration::from_secs(1))));
        }
        let controller = Arc::new(ComputeController{ 
            threads,
            reseed_rate: 1024, 
            terms, 
            sender: Mutex::new(send),
            loop_durations: Mutex::new(loop_durations),
            total_hashes: AtomicU64::new(0),
            running: AtomicBool::new(true),
        });

        {
            let loop_durations = controller.loop_durations.lock().unwrap();
            for duration in &*loop_durations {
                {
                    let duration = duration.clone();
                    let controller = controller.clone();
                    scope.spawn(move |_| { controller.run(duration) });
                }
            }
        }

        return (controller, receive);
    }

    pub fn run(&self, loop_duration: Arc<RwLock<Duration>>) {
        loop {
            let start = Instant::now();
            let mut key = KeyBuffer::random();
            for _ in 0..self.reseed_rate {
                key.next();
                if let Some(m) = self.terms.score(&key) {
                    self.sender
                        .lock()
                        .unwrap()
                        .send(m)
                        .unwrap();
                }
            }

            self.total_hashes.fetch_add(self.reseed_rate as u64, Ordering::Relaxed);

            if !self.running.load(Ordering::Relaxed) {
                return
            }

            {
                let mut writable_duration = loop_duration.write().unwrap();
                *writable_duration = start.elapsed();
            }
        }
    }

    // speed_stats returns a Tuple of (total_hashes, hashes_per_second);
    pub fn speed_stats(&self) -> (u64, u32) {
        let total_hashes = self.total_hashes.load(Ordering::Relaxed);

        let guard = self.loop_durations.lock().unwrap();
        let duration_slots = &*guard;
        let mut tot_micros: u128 = 0;
        for loop_duration in duration_slots {
            {
                let duration = loop_duration.read().unwrap();
                tot_micros += duration.as_nanos();
            }
        }
        let hashes_per_second = 
            Duration::from_secs(1).as_nanos() * 
            (self.threads * self.threads * self.reseed_rate as usize) as u128  
            / tot_micros;

        return (total_hashes, hashes_per_second as u32);
    }
}
