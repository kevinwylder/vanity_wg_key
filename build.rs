extern crate cc;

fn main() {
    cc::Build::new()
        .file("./src/curve25519.c")
        .compile("curve25519");
}
