use std::sync::Arc;

use bevy::prelude::*;

use crate::widgets::Tooltip;

pub type TranslatorFn = Arc<dyn Fn(&str) -> String + Send + Sync>;

#[derive(Resource, Default)]
pub struct UiTranslator {
  translate: Option<TranslatorFn>,
  version: u64,
}

impl UiTranslator {
  pub fn new(f: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
    Self { translate: Some(Arc::new(f)), version: 0 }
  }

  pub fn resolve(&self, key: &str) -> String {
    match &self.translate {
      Some(f) => f(key),
      None => key.to_string(),
    }
  }

  pub fn version(&self) -> u64 {
    self.version
  }

  pub fn bump(&mut self) {
    self.version = self.version.wrapping_add(1);
  }

  pub fn set(&mut self, f: impl Fn(&str) -> String + Send + Sync + 'static) {
    self.translate = Some(Arc::new(f));
    self.bump();
  }

  pub fn handle(&self) -> Option<TranslatorFn> {
    self.translate.clone()
  }
}

#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct I18nKey(pub String);

impl I18nKey {
  pub fn new(key: impl Into<String>) -> Self {
    Self(key.into())
  }
}

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
