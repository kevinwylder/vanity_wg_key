use std::{
    fs::File,
    io::{self, BufRead},
    path::Path,
    str,
};
use super::KeyBuffer;

pub struct Scorer {
    root: ScoreDict,
    min_score: u32,
}

struct ScoreDict {
    index: [Option<Box<Self>>; 37],
    val: u32,
}

pub struct Match {
    pub start: usize,
    pub len: usize,
    pub score: u32,
}

pub struct MultiMatch {
    pub matches: Vec<Match>,
    pub total_score: u32,
    pub key: KeyBuffer,
}

impl Default for ScoreDict {

    fn default() -> ScoreDict {
        const DEFAULT: Option<Box<ScoreDict>> = None;
        ScoreDict{
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

impl ScoreDict {

    fn add(&mut self, mut term: str::Chars, val: u32) {
        match term.next() {
            Some(c) => {
                let pos = alphabet_37(c);
                match self.index[pos] {
                    Some(ref mut child) => {
                        child.add(term, val);
                    }
                    None => {
                        self.index[pos] = Some(Box::new(ScoreDict::default()));
                        self.index[pos].as_mut().unwrap().add(term, val);
                    }
                }
            },
            None => {
                self.val = val;
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

impl Scorer {

    pub fn from_file(path: &Path, min_score: u32) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut root = ScoreDict::default();
        for line in io::BufReader::new(file).lines() {
            if let Ok(s) = line {
                root.add(s.as_str().chars(), (s.len() * s.len()) as u32);
            }
        }
        Ok(Scorer{
            root,
            min_score,
        })
    }

    pub fn score(&self, key: &KeyBuffer) -> Option<MultiMatch> {
        let term = base64::encode(key.pubkey());
        let mut matches = vec![];
        let mut total_score = 0;
        let mut start = 0;
        let mut iter = term.chars();
        loop {
            let substr = iter.clone();
            let (score, len) = self.root.search(0, substr);
            if score > 0 {
                matches.push(Match{ start, len, score });
                total_score += score;
            }
            start += 1;
            if let None = iter.next() {
                break;
            }
        }
        if total_score > self.min_score {
            return Some(MultiMatch{
                matches,
                total_score,
                key: key.clone(),
            });
        }
        return None;
    }

}
