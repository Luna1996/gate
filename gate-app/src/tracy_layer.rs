use std::cell::RefCell;
use std::collections::HashMap;

use tracing::{Id, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

pub(crate) struct TracyLayer;

thread_local! {
    static SPANS: RefCell<HashMap<Id, tracy_client::Span>> = RefCell::new(HashMap::new());
}

impl<S: Subscriber + for<'lookup> LookupSpan<'lookup>> Layer<S> for TracyLayer {
  fn enabled(&self, _metadata: &tracing::Metadata<'_>, _ctx: Context<'_, S>) -> bool {
    tracy_client::Client::is_running()
  }

  fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
    let Some(client) = tracy_client::Client::running() else {
      return;
    };
    let Some(meta) = ctx.metadata(id) else {
      return;
    };
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
