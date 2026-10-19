use std::io;
use std::net::TcpListener;

pub const PORT: u16 = 47311;

pub fn try_acquire() -> io::Result<TcpListener> {
  TcpListener::bind(("127.0.0.1", PORT))
}

pub fn acquire(app_name: &str) -> TcpListener {
  match try_acquire() {
    Ok(listener) => listener,
    Err(e) => {
      eprintln!(
        "[gate] 启动中止：已有一个 {app_name} 实例在运行（127.0.0.1:{PORT} 被占用：{e}）。\n\
         [gate] 两个实例会争抢同一块 GPU，帧时间不可比——请先关掉旧实例再启动。"
      );
      std::process::exit(1);
    }
  }
}
