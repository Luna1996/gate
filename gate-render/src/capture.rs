use std::io;

pub fn write_png(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) -> io::Result<()> {
  assert_eq!(rgba.len(), (width as usize) * (height as usize) * 4, "像素字节数与尺寸不符");

  let mut raw = Vec::with_capacity(rgba.len() + height as usize);
  for y in 0..height as usize {
    raw.push(0u8);
    raw.extend_from_slice(&rgba[y * width as usize * 4..(y + 1) * width as usize * 4]);
  }

  let mut z = vec![0x78u8, 0x01];
  let mut off = 0usize;
  while off < raw.len() {
    let n = (raw.len() - off).min(65_535);
    let last = off + n >= raw.len();
    z.push(u8::from(last));
    z.extend_from_slice(&(n as u16).to_le_bytes());
    z.extend_from_slice(&(!(n as u16)).to_le_bytes());
    z.extend_from_slice(&raw[off..off + n]);
    off += n;
  }
  let (mut a, mut b) = (1u32, 0u32);
  for &byte in &raw {
    a = (a + byte as u32) % 65_521;
    b = (b + a) % 65_521;
  }
  z.extend_from_slice(&(((b << 16) | a) ^ 0xFFFF_FFFF).wrapping_add(1).to_be_bytes());

  let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
  let mut ihdr = Vec::new();
  ihdr.extend_from_slice(&width.to_be_bytes());
  ihdr.extend_from_slice(&height.to_be_bytes());
  ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
  chunk(&mut png, b"IHDR", &ihdr);
  chunk(&mut png, b"IDAT", &z);
  chunk(&mut png, b"IEND", &[]);

  if let Some(dir) = path.parent() {
    std::fs::create_dir_all(dir).ok();
  }
  std::fs::write(path, png)
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
  out.extend_from_slice(&(data.len() as u32).to_be_bytes());
  let start = out.len();
  out.extend_from_slice(kind);
  out.extend_from_slice(data);
  let crc = crc32(&out[start..]);
  out.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
  let mut table = [0u32; 256];
  for (i, slot) in table.iter_mut().enumerate() {
    let mut c = i as u32;
    for _ in 0..8 {
      c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
    }
    *slot = c;
  }
  let mut c = 0xFFFF_FFFFu32;
  for &b in data {
    c = table[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
  }
  c ^ 0xFFFF_FFFF
}