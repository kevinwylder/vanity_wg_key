mod curve25519;

use std::{
    fmt,
    sync::{
        Arc, 
        Mutex, 
        atomic::{ AtomicBool, Ordering },
    },
    fs::File,
    io::{self, BufRead},
    path::Path,
    str,
    time::{ Instant, Duration },
};
use rand::{ thread_rng, Rng };
use base64;
use crossbeam_channel::{ Sender, Receiver };
use crossbeam_utils::thread;
use clap::{ Arg, App };
use curve25519::curve25519_generate_public;

struct Scorer {
    index: [Option<Box<Self>>; 37],
    val: u32,
}

struct Match {
    start: usize,
    len: usize,
    score: u32,
}

impl Default for Scorer {

    fn default() -> Scorer {
        const DEFAULT: Option<Box<Scorer>> = None;
        Scorer{
            index: [DEFAULT; 37],
            val: 0,
        }
    }
    
}

fn alphabet_37(c: char) -> usize {
    if c >= 'a' && c <= 'z' {
        return (c as u8 - 'a' as u8) as usize;
    }
    if c >= 'A' && c <= 'Z' {
        return (c as u8 - 'A' as u8) as usize;
    }
    if c >= '0' && c <= '9' {
        return 26 + (c as u8 - '0' as u8) as usize;
    }
    36
}

impl Scorer {

    pub fn from_file(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut root = Self::default();
        for line in io::BufReader::new(file).lines() {
            if let Ok(s) = line {
                root.add(s.as_str().chars(), (s.len() * s.len()) as u32);
            }
        }
        Ok(root)
    }

    pub fn add(&mut self, mut term: str::Chars, val: u32) {
        match term.next() {
            Some(c) => {
                let pos = alphabet_37(c);
                match self.index[pos] {
                    Some(ref mut child) => {
                        child.add(term, val);
                    }
                    None => {
                        self.index[pos] = Some(Box::new(Scorer::default()));
                        self.index[pos].as_mut().unwrap().add(term, val);
                    }
                }
            },
            None => {
                self.val = val;
            }
        }
    }

    pub fn score(&self, term: &str) -> (Vec<Match>, u32) {
        let mut matches = vec![];
        let mut total_score = 0;
        let mut start = 0;
        let mut iter = term.chars();
        loop {
            let substr = iter.clone();
            let (score, len) = self.search(0, substr);
            if score > 0 {
                matches.push(Match{ start, len, score });
                total_score += score;
            }
            start += 1;
            match iter.next() {
                None => return (matches, total_score),
                _ => (),
            }
        }
    }

    fn search(&self, i: usize, mut term: str::Chars) -> (u32, usize) {
        match term.next() {
            Some(c) => {
                let pos = alphabet_37(c);
                match self.index[pos] {
                    Some(ref child) => {
                        return child.search(i+1, term);
                    },
                    None => {
                        return (self.val, i);
                    }
                }
            },
            None => {
                return (0, 0);
            }
        }
    }

}

#[derive(Default, Clone)]
struct KeyBuffer {
    primary: [u8; 32],
    secondary: [u8; 32],
    is_first: bool,
}

impl KeyBuffer {
    
    pub fn random() -> Self {
        let mut key = KeyBuffer::default();
        thread_rng().fill(&mut key.primary);
        key
    }

    pub fn next(&mut self) {
        let next_pubkey: *mut u8;
        if self.is_first {
            next_pubkey = self.primary.as_mut_ptr();
        } else {
            next_pubkey = self.secondary.as_mut_ptr();
        }
        let next_privkey = self.pubkey().as_ptr();
        unsafe {
            curve25519_generate_public(next_pubkey, next_privkey);
        }
        self.is_first = !self.is_first;
    }

    pub fn privkey(&self) -> &[u8; 32] {
        if self.is_first {
            return &self.primary;
        } else {
            return &self.secondary;
        }
    }

    pub fn pubkey(&self) -> &[u8; 32] {
        if self.is_first {
            return &self.secondary;
        } else {
            return &self.primary;
        }
    }

}


impl fmt::Display for KeyBuffer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return write!(f, "{} -> {}", 
            base64::encode(self.privkey()), 
            base64::encode(self.pubkey())
        )
    }
}

const HASH_INSTANT_CACHE_SIZE: usize = 32;
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
    min_score: u32,
    searcher: &'a Scorer,
    running: AtomicBool,
    sender: Mutex<Sender<ComputeEvent>>
}

impl ComputeController<'_> {

    fn new<'a>(
        reseed_rate: u32,
        min_score: u32,
        searcher: &'a Scorer
    ) -> (Arc<ComputeController<'a>>, Receiver<ComputeEvent>) {
        let (send, receive) = crossbeam_channel::unbounded();
        let controller = Arc::new(ComputeController{ 
            reseed_rate, 
            min_score,
            searcher, 
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
                let encoded = base64::encode(key.pubkey());
                let (matches, score) = self.searcher.score(encoded.as_str());
                if score >= self.min_score {
                    self.sender
                        .lock()
                        .unwrap()
                        .send(
                            ComputeEvent::Match((key.clone(), matches))
                        ).unwrap();
                }
            }
            if !self.running.load(Ordering::Relaxed) {
                return
            }
            self.sender
                .lock()
                .unwrap()
                .send(
                    ComputeEvent::Reseed(start.elapsed())
                ).unwrap();
        }
    }
}

enum ComputeEvent {
    Reseed(Duration),
    Match((KeyBuffer, Vec<Match>)),
}

fn main() {

    let args = App::new("vanity wg key worker")
        .arg(Arg::with_name("threads")
            .default_value("2")
            .short("t")
            .long("threads")
            .help("Number of threads to search with")
        )
        .arg(Arg::with_name("min_score")
            .default_value("5")
            .short("s")
            .long("score")
            .help("Minimum score that warrents getting printed")
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
        .arg(Arg::with_name("terms")
            .required(true)
        )
        .get_matches();

    let threads = args.value_of("threads").unwrap()
        .parse::<usize>().expect(args.usage());
    let reseed_rate = args.value_of("reseed_rate").unwrap()
        .parse::<u32>().expect(args.usage());
    let min_score = args.value_of("min_score").unwrap()
        .parse::<u32>().expect(args.usage());
    let searcher = Scorer::from_file(
            Path::new(args.value_of("terms").unwrap())
        ).unwrap();

    let (controller, receiver) = ComputeController::new(reseed_rate, min_score, &searcher);
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
        println!("");
        loop {
            match receiver.recv() {
                Ok(ComputeEvent::Reseed(duration)) => { 
                    hash_counter.note_reseed(duration);
                    if Instant::now() > flush {
                        let (total_hashes, hash_rate) = hash_counter.total_and_rate(reseed_rate);
                        println!("\x1b[A\r\x1b[K{} hashes per second, total hashes: {}", 
                            hash_rate * threads as u32, 
                            total_hashes
                        );
                        flush = Instant::now() + flush_rate;
                    }
                },
                Ok(ComputeEvent::Match((key, matches))) => {
                    let private_encoded = base64::encode(key.privkey());
                    let public_encoded = base64::encode(key.pubkey());
                    print!("{} -> ", private_encoded);
                    let mut max_index = 0;
                    let mut total_score = 0;
                    for m in matches {
                        if m.start >= max_index {
                            print!("{}\x1b[41m{}\x1b[0m",
                                &public_encoded[max_index..m.start],
                                &public_encoded[m.start..m.start+m.len]
                            );
                            max_index = m.start + m.len;
                        }
                        total_score += m.score;
                    }
                    println!("{} ({})\n", 
                        &public_encoded[max_index..],
                        total_score
                    );
                },
                Err(_) => break,
            }
        }

        controller.running.store(false, Ordering::Relaxed);
    }).unwrap();
}
