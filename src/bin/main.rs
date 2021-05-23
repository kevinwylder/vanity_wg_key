use vanity_wg_key::{ Scorer, ComputeController };
use std::{
    path::Path,
    time::Duration,
};
use base64;
use crossbeam_utils::thread;
use crossbeam_channel::{tick, select};
use clap::{ Arg, App };

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
    let min_score = args.value_of("min_score").unwrap()
        .parse::<u32>().expect(args.usage());
    let searcher = Scorer::from_file(
            Path::new(args.value_of("terms").unwrap()),
            min_score,
        ).unwrap();


    thread::scope(|s| {

        let (controller, receiver) = ComputeController::new(&searcher, s, threads);
        let refresh_hash_rate = tick(Duration::from_millis(200));
        
        println!("");
        loop {
            select! {
                recv(receiver) -> msg => {
                    let multimatch = msg.unwrap();
                    let key = multimatch.key;
                    let public_encoded = base64::encode(key.pubkey());
                    let private_encoded = base64::encode(key.privkey());
                    print!("{} -> ", private_encoded);
                    let mut max_index = 0;
                    for m in multimatch.matches {
                        if m.start >= max_index {
                            print!("{}\x1b[41m{}\x1b[0m",
                                &public_encoded[max_index..m.start],
                                &public_encoded[m.start..m.start+m.len]
                            );
                            max_index = m.start + m.len;
                        }
                    }
                    println!("{} ({})\n", 
                        &public_encoded[max_index..],
                        multimatch.total_score
                    );
                },
                recv(refresh_hash_rate) -> _ => {
                    let (total_hashes, hashes_per_second) = controller.speed_stats();
                    println!("\x1b[A\r\x1b[K{} hashes per second, total hashes: {}", 
                        hashes_per_second,
                        total_hashes,
                    );
                },
            }
        }
    }).unwrap();
}
