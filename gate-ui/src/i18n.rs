//! i18n 接线：gate-ui **不依赖任何 i18n 库**，只保存「key → 当前语言文案」的解析闭包。
//!
//! 分工：
//! 1. 调用方启动时插入 [`UiTranslator`]（例如 gate-app 注入 `|k| t!(k).to_string()`），
//!    spawn 时经 [`UiCtx::with_translate`](crate::widgets::UiCtx::with_translate) 传给控件；
//! 2. 切换语言后调用 [`UiTranslator::bump`]；
//! 3. [`i18n_refresh_system`] 检测到版本变化，就重解析所有带 [`I18nKey`] 的文本。
//!
//! 未注入解析器时 key 原样当文案 —— TOML 里直接写字面量（如 `label = "视频"`）也完全成立。

use std::sync::Arc;

use bevy::prelude::*;

use crate::widgets::Tooltip;

/// 文案解析器句柄（[`UiTranslator`] → `UiCtx` 的传递形式）。
///
/// 用 `Arc` 而非借用：spawn 菜单页时 `UiCtx` 要跨 `&mut World` 存活，借用资源会和
/// 可变世界相冲。
pub type TranslatorFn = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// key → 文案的解析器资源（版本号用于驱动语言切换后的重解析）
#[derive(Resource, Default)]
pub struct UiTranslator {
  translate: Option<TranslatorFn>,
  version: u64,
}

impl UiTranslator {
  /// 用解析闭包构造
  pub fn new(f: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
    Self { translate: Some(Arc::new(f)), version: 0 }
  }

  /// 解析 key；未注入解析器 → 原样返回 key
  pub fn resolve(&self, key: &str) -> String {
    match &self.translate {
      Some(f) => f(key),
      None => key.to_string(),
    }
  }

  /// 当前版本号
  pub fn version(&self) -> u64 {
    self.version
  }

  /// 语言切换后调用：UI 下一帧重解析全部 keyed 文本
  pub fn bump(&mut self) {
    self.version = self.version.wrapping_add(1);
  }

  /// 换解析器（同时 bump 触发重解析）
  pub fn set(&mut self, f: impl Fn(&str) -> String + Send + Sync + 'static) {
    self.translate = Some(Arc::new(f));
    self.bump();
  }

  /// 取解析器句柄（spawn 时挂到 `UiCtx`）
  pub fn handle(&self) -> Option<TranslatorFn> {
    self.translate.clone()
  }
}

/// 可翻译文本标记：挂在文本来源是 i18n key 的实体上（`Text` / [`Tooltip`]）。
///
/// 语言切换（[`UiTranslator::bump`]）后由 [`i18n_refresh_system`] 重解析。
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct I18nKey(pub String);

impl I18nKey {
  pub fn new(key: impl Into<String>) -> Self {
    Self(key.into())
  }
}

/// 语言切换后重解析所有 keyed 文本（版本未变 → 一帧零开销）
pub fn i18n_refresh_system(
  i18n: Option<Res<UiTranslator>>,
  mut seen: Local<Option<u64>>,
  mut q: Query<(&I18nKey, Option<&mut Text>, Option<&mut Tooltip>)>,
) {
  let Some(i18n) = i18n else { return };
  if *seen == Some(i18n.version()) {
    return;
  }
  *seen = Some(i18n.version());
  for (key, text, tip) in &mut q {
    let value = i18n.resolve(&key.0);
    if let Some(mut t) = text
      && t.0 != value
    {
      t.0 = value.clone();
    }
    if let Some(mut tp) = tip
      && tp.text != value
    {
      tp.text = value;
    }
  }
}
