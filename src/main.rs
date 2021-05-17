mod curve25519;

use std::{
    fmt,
    sync::{
        Arc, 
        Mutex, 
        atomic::{ AtomicBool, Ordering },
    },
    time::{ Instant, Duration },
};
use rand::{ thread_rng, Rng };
use base64;
use crossbeam_channel::{ Sender, Receiver };
use crossbeam_utils::thread;
use clap::{ Arg, App };
use curve25519::curve25519_generate_public;

#[derive(Default, Clone)]
struct KeyBuffer {
    primary: [u8; 32],
    secondary: [u8; 32],
    is_first: bool,
}

impl KeyBuffer {
    
    fn random() -> Self {
        let mut key = KeyBuffer::default();
        thread_rng().fill(&mut key.secondary);
        key
    }

    fn next(&mut self) {
        unsafe {
            curve25519_generate_public(self.pubkey_mut_ptr(), self.privkey().as_ptr());
        }
        self.is_first = !self.is_first;
    }

    fn privkey(&self) -> &[u8; 32] {
        if self.is_first {
            return &self.primary;
        } else {
            return &self.secondary;
        }
    }

    fn pubkey(&self) -> &[u8; 32] {
        if self.is_first {
            return &self.secondary;
        } else {
            return &self.primary;
        }
    }

    fn pubkey_mut_ptr(&mut self) -> *mut u8 {
        if self.is_first {
            return self.secondary.as_mut_ptr()
        } else {
            return self.primary.as_mut_ptr()
        }

    }

    fn search(&self, term: &str) -> bool {
        base64::encode(self.pubkey()).contains(term)
    }

}


impl fmt::Display for KeyBuffer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return write!(f, "{} -> {}", base64::encode(self.privkey()), base64::encode(self.pubkey()))
    }
}

const HASH_INSTANT_CACHE_SIZE: usize = 10;
#[derive(Default)]
struct HashCounter {
    seed_counter: usize,
    recent_seeds: [ Duration; HASH_INSTANT_CACHE_SIZE ],
}

impl HashCounter {

    fn note_reseed(&mut self, duration: Duration) {
        self.recent_seeds[self.seed_counter % HASH_INSTANT_CACHE_SIZE] = duration;
        self.seed_counter += 1;
    }

    fn total_and_rate(&self, hashes_per_seed: u32) -> (u32, u32) { 
        let total_hashes = hashes_per_seed * self.seed_counter as u32;
        let mut avg_duration_per_seed = Duration::default();
        for i in 0..HASH_INSTANT_CACHE_SIZE {
            avg_duration_per_seed += self.recent_seeds[i];
        }
        if self.seed_counter < HASH_INSTANT_CACHE_SIZE {
            if self.seed_counter != 0 {
                avg_duration_per_seed /= self.seed_counter as u32;
            }
        } else {
            avg_duration_per_seed /= HASH_INSTANT_CACHE_SIZE as u32;
        }
        return (
            total_hashes, 
            ((1000000 * hashes_per_seed as u128) / (avg_duration_per_seed.as_micros() + 1)) as u32,
        )
    }

}

struct ComputeController<'a> {
    reseed_rate: u32,
    search_term: &'a str,
    running: AtomicBool,
    sender: Mutex<Sender<ComputeEvent>>
}

impl ComputeController<'_> {

    fn new<'a>(reseed_rate: u32, search_term: &'a str) -> (Arc<ComputeController<'a>>, Receiver<ComputeEvent>) {
        let (send, receive) = crossbeam_channel::unbounded();
        let controller = Arc::new(ComputeController{ 
            reseed_rate, 
            search_term, 
            running: AtomicBool::new(true),
            sender: Mutex::new(send),
        });

        return (controller, receive);
    }

    fn run(&self) {
        loop {
            let mut key = KeyBuffer::random();
            let start = Instant::now();
            for _ in 0..self.reseed_rate {
                key.next();
                if key.search(self.search_term) {
                    self.sender.lock().unwrap().send(ComputeEvent::Match(key.clone())).unwrap();
                }
            }
            if !self.running.load(Ordering::Relaxed) {
                return
            }
            self.sender.lock().unwrap().send(ComputeEvent::Reseed(start.elapsed())).unwrap();
        }
    }
}

enum ComputeEvent {
    Reseed(Duration),
    Match(KeyBuffer),
}

fn main() {

    let args = App::new("vanity wg key worker")
        .arg(Arg::with_name("threads")
            .default_value("2")
            .short("t")
            .long("threads")
            .help("Number of threads to search with")
        )
        .arg(Arg::with_name("reseed_rate")
            .default_value("512")
            .short("r")
            .long("reseed")
            .help("Number of sequential keys to compute before randomizing")
        )
        .arg(Arg::with_name("debug")
            .short("v")
            .long("debug")
            .help("If specified, run in debug mode")
            .takes_value(false)
        )
        .arg(Arg::with_name("term")
            .required(true)
        )
        .get_matches();

    let threads = args.value_of("threads").unwrap()
        .parse::<usize>().expect(args.usage());
    let reseed_rate = args.value_of("reseed_rate").unwrap()
        .parse::<u32>().expect(args.usage());
    let search_term = args.value_of("term").unwrap();

    let (controller, receiver) = ComputeController::new(reseed_rate, search_term);
    let mut hash_counter = HashCounter::default();

    thread::scope(|s| {

        for _ in 0..threads {
            {
                let controller = controller.clone();
                s.spawn(move |_| { controller.run(); });
            }
        }
        
        let start = Instant::now();
        let flush_rate = Duration::from_millis(300);
        let mut flush = start;
        println!("total hashes: 0");
        loop {
            match receiver.recv().unwrap() {
                ComputeEvent::Reseed(duration) => { 
                    hash_counter.note_reseed(duration);
                    if Instant::now() > flush {
                        let (total_hashes, hash_rate) = hash_counter.total_and_rate(reseed_rate);
                        println!("\x1b[A\r\x1b[K{} hashes per second, total hashes: {}", hash_rate * threads as u32, total_hashes);
                        flush = Instant::now() + flush_rate;
                    }
                },
                ComputeEvent::Match(key) => {
                    println!("{}", key);
                    break
                },
            }
        }

        controller.running.store(false, Ordering::Relaxed);
    }).unwrap();
}
