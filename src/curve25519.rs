use std::fmt;
use rand::{ thread_rng, Rng };

#[derive(Default, Clone)]
pub struct KeyBuffer {
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

extern "C" {
    pub fn curve25519_generate_public(pubkey: *mut u8, privke: *const u8);
}
