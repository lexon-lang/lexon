// src/runtime/scheduler.rs
//
// Enhanced async scheduler with cancellation, timeout, and telemetry support

use crate::telemetry::{trace_task_cancellation, trace_task_execution, trace_task_scheduling};
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::{oneshot, Mutex};
use tokio::time::{sleep, timeout};
use uuid::Uuid;

/// Represents a cancellation token that can be used to cancel async operations
#[derive(Debug, Clone)]
pub struct CancellationToken {
    inner: Arc<Mutex<CancellationState>>,
}

#[derive(Debug)]
struct CancellationState {
    is_cancelled: bool,
    reason: Option<String>,
    cancelled_at: Option<Instant>,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CancellationState {
                is_cancelled: false,
                reason: None,
                cancelled_at: None,
            })),
        }
    }

    /// Cancel the token with an optional reason
    pub async fn cancel(&self, reason: Option<String>) {
        let mut state = self.inner.lock().await;
        if !state.is_cancelled {
            state.is_cancelled = true;
            state.reason = reason;
            state.cancelled_at = Some(Instant::now());
            println!("Cancellation token cancelled");
        }
    }

    /// Check if the token is cancelled
    pub async fn is_cancelled(&self) -> bool {
        let state = self.inner.lock().await;
        state.is_cancelled
    }

    /// Wait for the token to be cancelled
    pub async fn cancelled(&self) {
        loop {
            {
                let state = self.inner.lock().await;
                if state.is_cancelled {
                    return;
                }
            }
            // Small delay to avoid busy waiting
            sleep(Duration::from_millis(10)).await;
        }
    }

    /// Get the cancellation reason if available
    pub async fn reason(&self) -> Option<String> {
        let state = self.inner.lock().await;
        state.reason.clone()
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for controlling a running task
#[derive(Debug, Clone)]
pub struct TaskHandle {
    id: String,
    cancellation_token: CancellationToken,
    result_receiver: Arc<Mutex<Option<oneshot::Receiver<TaskResult>>>>,
}

impl TaskHandle {
    /// Create a new task handle
    pub fn new(
        id: String,
        cancellation_token: CancellationToken,
        result_receiver: oneshot::Receiver<TaskResult>,
    ) -> Self {
        Self {
            id,
            cancellation_token,
            result_receiver: Arc::new(Mutex::new(Some(result_receiver))),
        }
    }

    /// Get the task ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Cancel the task
    pub async fn cancel(&self) -> Result<(), String> {
        let span = trace_task_cancellation(&self.id, "user_requested");
        span.record_event("Cancelling task");

        self.cancellation_token
            .cancel(Some("User requested cancellation".to_string()))
            .await;

        span.record_event("Task cancellation requested");
        Ok(())
    }

    /// Check if the task is cancelled
    pub async fn is_cancelled(&self) -> bool {
        self.cancellation_token.is_cancelled().await
    }

    /// Wait for the task to complete and get the result
    pub async fn await_result(&self) -> Result<TaskResult, String> {
        let mut receiver_guard = self.result_receiver.lock().await;
        if let Some(receiver) = receiver_guard.take() {
            receiver
                .await
                .map_err(|e| format!("Failed to receive task result: {}", e))
        } else {
            Err("Result already consumed".to_string())
        }
    }

    /// Wait for the task to complete with a timeout
    pub async fn await_result_timeout(
        &self,
        timeout_duration: Duration,
    ) -> Result<TaskResult, String> {
        let mut receiver_guard = self.result_receiver.lock().await;
        if let Some(receiver) = receiver_guard.take() {
            match timeout(timeout_duration, receiver).await {
                Ok(result) => result.map_err(|e| format!("Failed to receive task result: {}", e)),
                Err(_) => {
                    // Timeout occurred, cancel the task
                    self.cancel().await?;
                    Err("Task timed out".to_string())
                }
            }
        } else {
            Err("Result already consumed".to_string())
        }
    }
}

/// Result of a task execution
#[derive(Debug, Clone)]
pub enum TaskResult {
    Success(String),
    Error(String),
    Cancelled(String),
    Timeout,
}

impl fmt::Display for TaskResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskResult::Success(msg) => write!(f, "Success: {}", msg),
            TaskResult::Error(msg) => write!(f, "Error: {}", msg),
            TaskResult::Cancelled(msg) => write!(f, "Cancelled: {}", msg),
            TaskResult::Timeout => write!(f, "Timeout"),
        }
    }
}

/// Priority levels for task scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskPriority::Low => write!(f, "low"),
            TaskPriority::Normal => write!(f, "normal"),
            TaskPriority::High => write!(f, "high"),
            TaskPriority::Critical => write!(f, "critical"),
        }
    }
}

/// Internal task representation
struct LexonTask {
    id: String,
    priority: TaskPriority,
    cancellation_token: CancellationToken,
    future: Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'static>>,
    result_sender: oneshot::Sender<TaskResult>,
    #[allow(dead_code)]
    created_at: Instant,
    timeout: Option<Duration>,
}

/// Enhanced async scheduler with cancellation and timeout support
pub struct AsyncScheduler {
    task_queue: Arc<Mutex<Vec<LexonTask>>>,
    running_tasks: Arc<Mutex<HashMap<String, TaskHandle>>>,
    max_concurrent_tasks: usize,
    default_timeout: Option<Duration>,
    shutdown_token: CancellationToken,
}

impl AsyncScheduler {
    /// Create a new async scheduler
    pub fn new(max_concurrent_tasks: usize) -> Self {
        Self {
            task_queue: Arc::new(Mutex::new(Vec::new())),
            running_tasks: Arc::new(Mutex::new(HashMap::new())),
            max_concurrent_tasks,
            default_timeout: Some(Duration::from_secs(300)), // 5 minutes default
            shutdown_token: CancellationToken::new(),
        }
    }

    /// Set the default timeout for tasks
    pub fn with_default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = Some(timeout);
        self
    }

    /// Schedule a task for execution
    pub async fn schedule_task<F, Fut>(
        &self,
        future: F,
        priority: TaskPriority,
        timeout: Option<Duration>,
    ) -> Result<TaskHandle, String>
    where
        F: FnOnce(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = Result<String, String>> + Send + 'static,
    {
        let task_id = Uuid::new_v4().to_string();
        let cancellation_token = CancellationToken::new();
        let (result_sender, result_receiver) = oneshot::channel();

        let span = trace_task_scheduling(&task_id, &priority.to_string());
        span.record_event("Scheduling new task");

        // Create the future with cancellation support
        let future_with_cancellation = LexonFuture::new(
            future(cancellation_token.clone()),
            cancellation_token.clone(),
        );

        let task = LexonTask {
            id: task_id.clone(),
            priority,
            cancellation_token: cancellation_token.clone(),
            future: Box::pin(future_with_cancellation),
            result_sender,
            created_at: Instant::now(),
            timeout: timeout.or(self.default_timeout),
        };

        // Add task to queue
        {
            let mut queue = self.task_queue.lock().await;
            queue.push(task);
            // Sort by priority (highest first)
            queue.sort_by(|a, b| b.priority.cmp(&a.priority));
        }

        let handle = TaskHandle::new(task_id.clone(), cancellation_token, result_receiver);

        span.record_event("Task scheduled successfully");
        span.set_attribute("task.id", &task_id);
        span.set_attribute("task.priority", &priority.to_string());

        // Try to start execution immediately if possible
        self.try_execute_next_task().await;

        Ok(handle)
    }

    /// Try to execute the next task in the queue
    async fn try_execute_next_task(&self) {
        let running_count = {
            let running = self.running_tasks.lock().await;
            running.len()
        };

        if running_count >= self.max_concurrent_tasks {
            println!("Max concurrent tasks reached, waiting for slots");
            return;
        }

        let task = {
            let mut queue = self.task_queue.lock().await;
            queue.pop()
        };

        if let Some(task) = task {
            let task_id = task.id.clone();
            let handle = TaskHandle::new(
                task_id.clone(),
                task.cancellation_token.clone(),
                // We need to create a dummy receiver here since the real one is in the task
                oneshot::channel().1,
            );

            {
                let mut running = self.running_tasks.lock().await;
                running.insert(task_id.clone(), handle);
            }

            let scheduler = self.clone();
            let task_id_clone = task_id.clone();

            tokio::spawn(async move {
                scheduler.execute_task(task).await;

                // Remove from running tasks
                {
                    let mut running = scheduler.running_tasks.lock().await;
                    running.remove(&task_id_clone);
                }

                // Note: We don't recursively call try_execute_next_task here to avoid Send issues
                // The scheduler will pick up new tasks when schedule_task is called
            });
        }
    }

    /// Execute a single task
    async fn execute_task(&self, task: LexonTask) {
        let span = trace_task_execution(&task.id);
        span.record_event("Starting task execution");

        let start_time = Instant::now();

        // Check if task is already cancelled
        if task.cancellation_token.is_cancelled().await {
            let reason = task
                .cancellation_token
                .reason()
                .await
                .unwrap_or_else(|| "Unknown".to_string());
            let result = TaskResult::Cancelled(reason);
            let _ = task.result_sender.send(result);
            span.record_event("Task was already cancelled");
            return;
        }

        // Execute the task with timeout if specified
        let result = if let Some(timeout_duration) = task.timeout {
            match timeout(timeout_duration, task.future).await {
                Ok(task_result) => match task_result {
                    Ok(success_msg) => TaskResult::Success(success_msg),
                    Err(error_msg) => {
                        if error_msg == "Task was cancelled" {
                            TaskResult::Cancelled(error_msg)
                        } else {
                            TaskResult::Error(error_msg)
                        }
                    }
                },
                Err(_) => {
                    // Timeout occurred
                    task.cancellation_token
                        .cancel(Some("Timeout".to_string()))
                        .await;
                    TaskResult::Timeout
                }
            }
        } else {
            match task.future.await {
                Ok(success_msg) => TaskResult::Success(success_msg),
                Err(error_msg) => {
                    if error_msg == "Task was cancelled" {
                        TaskResult::Cancelled(error_msg)
                    } else {
                        TaskResult::Error(error_msg)
                    }
                }
            }
        };

        let execution_time = start_time.elapsed();

        // Record execution metrics
        span.set_attribute(
            "task.execution_time_ms",
            &execution_time.as_millis().to_string(),
        );
        span.set_attribute("task.result", &format!("{:?}", result));

        match &result {
            TaskResult::Success(_) => {
                span.record_event("Task completed successfully");
                println!(
                    "Task {} completed successfully in {:?}",
                    task.id, execution_time
                );
            }
            TaskResult::Error(err) => {
                span.record_error(&format!("Task failed: {}", err));
                println!(
                    "Task {} failed: {} (execution time: {:?})",
                    task.id, err, execution_time
                );
            }
            TaskResult::Cancelled(reason) => {
                span.record_warning(&format!("Task cancelled: {}", reason));
                println!(
                    "Task {} cancelled: {} (execution time: {:?})",
                    task.id, reason, execution_time
                );
            }
            TaskResult::Timeout => {
                span.record_warning("Task timed out");
                println!("Task {} timed out after {:?}", task.id, execution_time);
            }
        }

        // Send result
        let _ = task.result_sender.send(result);
    }

    /// Get the number of queued tasks
    pub async fn queued_tasks_count(&self) -> usize {
        let queue = self.task_queue.lock().await;
        queue.len()
    }

    /// Get the number of running tasks
    pub async fn running_tasks_count(&self) -> usize {
        let running = self.running_tasks.lock().await;
        running.len()
    }

    /// Cancel all running tasks
    pub async fn cancel_all_tasks(&self) -> Result<(), String> {
        let span = trace_task_cancellation("all", "shutdown");
        span.record_event("Cancelling all tasks");

        let running_tasks = {
            let running = self.running_tasks.lock().await;
            running.values().cloned().collect::<Vec<_>>()
        };

        for handle in running_tasks {
            handle.cancel().await?;
        }

        span.record_event("All tasks cancelled");
        Ok(())
    }

    /// Shutdown the scheduler
    pub async fn shutdown(&self) -> Result<(), String> {
        println!("Shutting down async scheduler");

        self.shutdown_token
            .cancel(Some("Scheduler shutdown".to_string()))
            .await;
        self.cancel_all_tasks().await?;

        // Wait for all tasks to complete
        let max_wait = Duration::from_secs(10);
        let start = Instant::now();

        while start.elapsed() < max_wait {
            if self.running_tasks_count().await == 0 {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }

        println!("Async scheduler shutdown complete");
        Ok(())
    }
}

impl Clone for AsyncScheduler {
    fn clone(&self) -> Self {
        Self {
            task_queue: Arc::clone(&self.task_queue),
            running_tasks: Arc::clone(&self.running_tasks),
            max_concurrent_tasks: self.max_concurrent_tasks,
            default_timeout: self.default_timeout,
            shutdown_token: self.shutdown_token.clone(),
        }
    }
}

/// A future wrapper that supports cancellation
pub struct LexonFuture<F> {
    future: Pin<Box<F>>,
    cancellation_token: CancellationToken,
}

impl<F> LexonFuture<F>
where
    F: Future<Output = Result<String, String>>,
{
    pub fn new(future: F, cancellation_token: CancellationToken) -> Self {
        Self {
            future: Box::pin(future),
            cancellation_token,
        }
    }
}

impl<F> Future for LexonFuture<F>
where
    F: Future<Output = Result<String, String>>,
{
    type Output = Result<String, String>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Check for cancellation first
        let cancellation_token = self.cancellation_token.clone();
        let mut cancel_future = Box::pin(async move {
            cancellation_token.cancelled().await;
        });

        // Poll cancellation
        if let Poll::Ready(()) = cancel_future.as_mut().poll(cx) {
            return Poll::Ready(Err("Task was cancelled".to_string()));
        }

        // Poll the actual future
        self.future.as_mut().poll(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_cancellation_token() {
        let token = CancellationToken::new();

        assert!(!token.is_cancelled().await);

        token.cancel(Some("Test cancellation".to_string())).await;

        assert!(token.is_cancelled().await);
        assert_eq!(token.reason().await, Some("Test cancellation".to_string()));
    }

    #[tokio::test]
    async fn test_task_scheduling() {
        let scheduler = AsyncScheduler::new(2);

        let handle = scheduler
            .schedule_task(
                |_token| async move {
                    sleep(Duration::from_millis(100)).await;
                    Ok("Task completed".to_string())
                },
                TaskPriority::Normal,
                None,
            )
            .await
            .unwrap();

        let result = handle.await_result().await.unwrap();
        match result {
            TaskResult::Success(msg) => assert_eq!(msg, "Task completed"),
            _ => panic!("Expected success result"),
        }
    }

    #[tokio::test]
    async fn test_task_cancellation() {
        let scheduler = AsyncScheduler::new(2);

        let handle = scheduler
            .schedule_task(
                |token| async move {
                    // Simulate long-running task
                    for _i in 0..100 {
                        if token.is_cancelled().await {
                            return Err("Task was cancelled".to_string());
                        }
                        sleep(Duration::from_millis(10)).await;
                    }
                    Ok("Task completed".to_string())
                },
                TaskPriority::Normal,
                None,
            )
            .await
            .unwrap();

        // Cancel the task after a short delay
        sleep(Duration::from_millis(50)).await;
        handle.cancel().await.unwrap();

        let result = handle.await_result().await.unwrap();
        match result {
            TaskResult::Cancelled(_) => {} // Expected
            _ => panic!("Expected cancelled result, got: {:?}", result),
        }
    }

    #[tokio::test]
    async fn test_task_timeout() {
        let scheduler = AsyncScheduler::new(2);

        let handle = scheduler
            .schedule_task(
                |_token| async move {
                    // Task that takes longer than timeout
                    sleep(Duration::from_millis(200)).await;
                    Ok("Task completed".to_string())
                },
                TaskPriority::Normal,
                Some(Duration::from_millis(100)), // 100ms timeout
            )
            .await
            .unwrap();

        let result = handle.await_result().await.unwrap();
        match result {
            TaskResult::Timeout => {} // Expected
            _ => panic!("Expected timeout result, got: {:?}", result),
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_priority_scheduling() {
        let scheduler = AsyncScheduler::new(1); // Only one concurrent task

        let mut handles = Vec::new();

        // Schedule tasks with different priorities
        for (i, priority) in [TaskPriority::Low, TaskPriority::High, TaskPriority::Normal]
            .iter()
            .enumerate()
        {
            let handle = scheduler
                .schedule_task(
                    move |_token| async move {
                        sleep(Duration::from_millis(50)).await;
                        Ok(format!("Task {} completed", i))
                    },
                    *priority,
                    None,
                )
                .await
                .unwrap();
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            let result = handle.await_result().await.unwrap();
            match result {
                TaskResult::Success(_) => {} // Expected
                _ => panic!("Expected success result, got: {:?}", result),
            }
        }
    }

    #[tokio::test]
    async fn test_scheduler_shutdown() {
        let scheduler = AsyncScheduler::new(2);

        // Schedule some tasks
        let _handle1 = scheduler
            .schedule_task(
                |_token| async move {
                    sleep(Duration::from_millis(100)).await;
                    Ok("Task 1 completed".to_string())
                },
                TaskPriority::Normal,
                None,
            )
            .await
            .unwrap();

        let _handle2 = scheduler
            .schedule_task(
                |_token| async move {
                    sleep(Duration::from_millis(150)).await;
                    Ok("Task 2 completed".to_string())
                },
                TaskPriority::Normal,
                None,
            )
            .await
            .unwrap();

        // Shutdown should cancel all tasks
        scheduler.shutdown().await.unwrap();

        assert_eq!(scheduler.running_tasks_count().await, 0);
    }
}
