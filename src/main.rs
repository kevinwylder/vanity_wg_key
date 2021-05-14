mod curve25519;

use curve25519::curve25519_generate_public;
use base64::{STANDARD, decode_config_slice, encode, DecodeError};
use std::fmt;
use std::time::SystemTime;

#[derive(Default)]
struct KeyBuffer {
    primary: [u8; 32],
    secondary: [u8; 32],
    is_first: bool,
}

impl KeyBuffer {
    
    fn from_base64<T: AsRef<[u8]>>(b64: T) -> Result<Self, DecodeError> {
        let mut key = KeyBuffer::default();
        match decode_config_slice(b64, STANDARD, &mut key.primary) {
            Err(e) => return Err(e),
            Ok(_) => return Ok(key),
        }
    }

    fn next(&mut self) {
        let pubkey: *mut u8;
        let privkey: *const u8;
        if self.is_first {
            pubkey = self.primary.as_mut_ptr();
            privkey = self.secondary.as_ptr();
        } else {
            pubkey = self.secondary.as_mut_ptr();
            privkey = self.primary.as_ptr();
        }
        unsafe {
            curve25519_generate_public(pubkey, privkey);
        }
        self.is_first = !self.is_first;
    }

}

impl fmt::Display for KeyBuffer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_first {
            return write!(f, "{} -> {}", encode(self.primary), encode(self.secondary))
        } else {
            return write!(f, "{} -> {}", encode(self.secondary), encode(self.primary))
        }
    }
}

fn main() {
    let mut key = KeyBuffer::from_base64("yOjeXTXa+anM9MSj6qj7YYP6ScQRUkGwEEfYoC0pnVU=").unwrap();

    let start = SystemTime::now();
    let num_iterations = 1024 * 1024;
    for _ in 0..num_iterations {
        key.next();
    }
    println!("{}", start.elapsed().unwrap().as_millis())
}
