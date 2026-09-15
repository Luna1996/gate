//! tracing-subscriber Layer → Tracy zone 桥接（cargo feature = "profile" 时编译）。
//!
//! span enter → `span_alloc` 开 zone，span exit → drop 关 zone。bevy 开 "trace" feature
//! 后 schedule/system 自动成为 tracing span，作为 CPU zone 与 wgpu-profiler 的 GPU zone
//! 在同一 Tracy 时间线对齐。
//! tracy `Span` 非 Send（zone context 线程局部），enter/exit 同线程配对 → 每线程一张表。

use std::cell::RefCell;
use std::collections::HashMap;

use tracing::{Id, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

/// tracing span → Tracy CPU zone 的 [`Layer`]（无状态单元体，可安全跨线程共享）。
pub(crate) struct TracyLayer;

thread_local! {
  /// 本线程存活中的 span → tracy zone（span exit 时 drop 关 zone）
  static SPANS: RefCell<HashMap<Id, tracy_client::Span>> = RefCell::new(HashMap::new());
}

impl<S: Subscriber + for<'lookup> LookupSpan<'lookup>> Layer<S> for TracyLayer {
  fn enabled(&self, _metadata: &tracing::Metadata<'_>, _ctx: Context<'_, S>) -> bool {
    // Tracy 客户端未 start 时全层跳过
    tracy_client::Client::is_running()
  }

  fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
    let Some(client) = tracy_client::Client::running() else {
      return;
    };
    let Some(meta) = ctx.metadata(id) else {
      return;
    };
    // name = span 名；function/file/line 取 callsite（Tracy 可定位到源文件）
    let span = client.span_alloc(
      Some(meta.name()),
      meta.target(),
      meta.file().unwrap_or(""),
      meta.line().unwrap_or(0),
      0,
    );
    SPANS.with(|spans| {
      spans.borrow_mut().insert(id.clone(), span);
    });
  }

  fn on_exit(&self, id: &Id, _ctx: Context<'_, S>) {
    SPANS.with(|spans| {
      spans.borrow_mut().remove(id);
    });
  }
}
