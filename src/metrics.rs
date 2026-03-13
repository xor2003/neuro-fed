// src/metrics.rs
// Централизованный модуль для управления метриками и наблюдаемости.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use once_cell::sync::Lazy;

// --- Ключи Метрик (константы для избежания "магических строк") ---

// Технические/Производительность
pub const HTTP_REQUESTS_TOTAL: &str = "http.requests_total";
pub const HTTP_REQUEST_LATENCY_SECONDS: &str = "http.request_latency_seconds";

// Когнитивные/Архитектурные
pub const PC_FREE_ENERGY: &str = "pc.free_energy";
pub const PC_WEIGHT_ROLLBACKS_TOTAL: &str = "pc.weight_rollbacks_total";
pub const PC_GOSSIP_TRIGGERS_TOTAL: &str = "pc.gossip_triggers_total";
pub const AGENT_VERIFICATION_SUCCESS_TOTAL: &str = "agent.verification_success_total";
pub const AGENT_VERIFICATION_FAILURE_TOTAL: &str = "agent.verification_failure_total";
pub const SLEEP_EPISODES_CONSOLIDATED_TOTAL: &str = "sleep.episodes_consolidated_total";
pub const COGNITIVE_CHUNKS_DISCOVERED_TOTAL: &str = "cognitive.chunks_discovered_total";

// Документная обработка/Обучение
pub const DOCUMENT_PARAGRAPHS_PROCESSED_TOTAL: &str = "document.paragraphs_processed_total";
pub const DOCUMENT_PROCESSING_PERCENT: &str = "document.processing_percent";
pub const DOCUMENT_PARAGRAPHS_PER_SECOND: &str = "document.paragraphs_per_second";
pub const DOCUMENT_LEARNING_TIME_SECONDS: &str = "document.learning_time_seconds";
pub const DOCUMENT_FILES_PROCESSED_TOTAL: &str = "document.files_processed_total";

/// Тип значения метрики
#[derive(Debug, Clone)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
}

/// Простое хранилище метрик в памяти
#[derive(Clone, Default)]
pub struct MetricsStore {
    storage: Arc<RwLock<HashMap<String, MetricValue>>>,
}

impl MetricsStore {
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Увеличивает счетчик на указанное значение
    pub fn increment_counter(&self, key: &str, value: u64) {
        let mut storage = self.storage.write().unwrap();
        let entry = storage.entry(key.to_string()).or_insert(MetricValue::Counter(0));
        match entry {
            MetricValue::Counter(c) => *c += value,
            _ => *entry = MetricValue::Counter(value), // сброс, если тип не совпадает
        }
    }

    /// Устанавливает значение gauge
    pub fn set_gauge(&self, key: &str, value: f64) {
        let mut storage = self.storage.write().unwrap();
        storage.insert(key.to_string(), MetricValue::Gauge(value));
    }

    /// Получает снимок всех метрик
    pub fn get_snapshot(&self) -> HashMap<String, String> {
        let storage = self.storage.read().unwrap();
        storage.iter()
            .map(|(k, v)| {
                let value_str = match v {
                    MetricValue::Counter(c) => c.to_string(),
                    MetricValue::Gauge(g) => format!("{:.4}", g),
                };
                (k.clone(), value_str)
            })
            .collect()
    }
}

pub static METRICS: Lazy<MetricsStore> = Lazy::new(MetricsStore::new);

/// Глобальная инициализация системы метрик (ничего не делает, но оставлена для совместимости)
pub fn init_metrics() {
    tracing::info!("✅ Metrics system initialized (in‑memory store).");
}

/// Макрос для увеличения счетчика (удобный синтаксис)
#[macro_export]
macro_rules! increment_counter {
    ($key:expr, $value:expr) => {
        $crate::metrics::METRICS.increment_counter($key, $value)
    };
    ($key:expr) => {
        $crate::metrics::METRICS.increment_counter($key, 1)
    };
}

/// Макрос для установки gauge
#[macro_export]
macro_rules! gauge {
    ($key:expr, $value:expr) => {
        $crate::metrics::METRICS.set_gauge($key, $value)
    };
}

/// Макрос для записи гистограммы (упрощённо — как gauge)
#[macro_export]
macro_rules! histogram {
    ($key:expr, $value:expr) => {
        $crate::metrics::METRICS.set_gauge($key, $value)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_store_works() {
        init_metrics();
        
        // Увеличиваем счетчик
        increment_counter!("test.counter", 10);
        increment_counter!("test.counter", 5);
        
        // Устанавливаем gauge
        gauge!("test.gauge", 42.5);

        let snapshot = METRICS.get_snapshot();

        assert_eq!(snapshot.get("test.counter").unwrap(), "15");
        assert_eq!(snapshot.get("test.gauge").unwrap(), "42.5000");
    }
}