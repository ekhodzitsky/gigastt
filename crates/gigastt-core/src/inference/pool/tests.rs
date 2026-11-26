use super::*;

#[test]
fn test_pool_checkout_blocking_fast_path() {
    let pool = Pool::new(vec![42u32]);
    let guard = pool.checkout_blocking().expect("checkout_blocking");
    assert_eq!(*guard, 42);
    drop(guard);
    assert_eq!(pool.available(), 1);
}

#[test]
fn test_pool_checkout_blocking_closed() {
    let pool = Pool::<u32>::new(vec![]);
    pool.close();
    assert!(matches!(pool.checkout_blocking(), Err(PoolError::Closed)));
}

#[test]
fn test_pool_checkout_blocking_slow_path() {
    let pool = std::sync::Arc::new(Pool::new(vec![42u32]));
    let primary = pool.checkout_blocking().unwrap();

    let handle = std::thread::spawn({
        let pool = pool.clone();
        move || pool.checkout_blocking()
    });

    std::thread::sleep(std::time::Duration::from_millis(50));
    drop(primary);

    let guard = handle.join().expect("join").expect("checkout");
    assert_eq!(*guard, 42);
    drop(guard);
    assert_eq!(pool.available(), 1);
}

#[test]
fn test_pool_error_display() {
    assert_eq!(format!("{}", PoolError::Closed), "session pool is closed");
}

#[test]
fn test_pool_try_checkout_takes_available_without_waiters() {
    let pool = Pool::new(vec![1u32, 2]);
    let a = pool.try_checkout().expect("open").expect("slot");
    assert_eq!(*a, 1);
    assert_eq!(pool.available(), 1);
    assert_eq!(pool.waiters(), 0);
    let b = pool.try_checkout().expect("open").expect("slot");
    assert_eq!(*b, 2);
    assert_eq!(pool.available(), 0);
    assert_eq!(pool.waiters(), 0);
    assert!(pool.try_checkout().expect("open").is_none());
    assert_eq!(pool.waiters(), 0);
    drop(a);
    drop(b);
    assert_eq!(pool.available(), 2);
}

#[test]
fn test_pool_try_checkout_empty_does_not_block() {
    let pool = std::sync::Arc::new(Pool::new(vec![1u32]));
    let held = pool.checkout_blocking().unwrap();
    let started = std::time::Instant::now();
    let got = pool.try_checkout().expect("open");
    assert!(got.is_none());
    assert!(
        started.elapsed() < std::time::Duration::from_millis(50),
        "try_checkout must not wait for a checkin"
    );
    assert_eq!(pool.waiters(), 0);
    drop(held);
}

#[test]
fn test_pool_try_checkout_closed() {
    let pool = Pool::<u32>::new(vec![1]);
    pool.close();
    assert!(matches!(pool.try_checkout(), Err(PoolError::Closed)));
}

#[test]
fn test_pool_try_checkout_n_takes_up_to_n_without_waiters() {
    let pool = Pool::new(vec![1u32, 2, 3]);
    let got = pool.try_checkout_n(2).expect("open");
    assert_eq!(got.len(), 2);
    assert_eq!(*got[0], 1);
    assert_eq!(*got[1], 2);
    assert_eq!(pool.available(), 1);
    assert_eq!(pool.waiters(), 0);
    drop(got);
    assert_eq!(pool.available(), 3);
}

#[test]
fn test_pool_try_checkout_n_stops_when_empty() {
    let pool = Pool::new(vec![1u32]);
    let got = pool.try_checkout_n(5).expect("open");
    assert_eq!(got.len(), 1);
    assert_eq!(*got[0], 1);
    assert!(pool.try_checkout_n(1).expect("open").is_empty());
    assert_eq!(pool.waiters(), 0);
}

#[test]
fn test_pool_try_checkout_n_zero_skips_closed() {
    let pool = Pool::new(vec![1u32]);
    pool.close();
    assert!(pool.try_checkout_n(0).expect("open").is_empty());
    assert!(matches!(pool.try_checkout_n(1), Err(PoolError::Closed)));
}

#[test]
fn test_pool_try_checkout_n_closed() {
    let pool = Pool::<u32>::new(vec![1, 2]);
    pool.close();
    assert!(matches!(pool.try_checkout_n(2), Err(PoolError::Closed)));
}

#[test]
fn test_pool_try_checkout_n_empty_does_not_block() {
    let pool = std::sync::Arc::new(Pool::new(vec![1u32]));
    let held = pool.checkout_blocking().unwrap();
    let started = std::time::Instant::now();
    let got = pool.try_checkout_n(3).expect("open");
    assert!(got.is_empty());
    assert!(
        started.elapsed() < std::time::Duration::from_millis(50),
        "try_checkout_n must not wait for a checkin"
    );
    assert_eq!(pool.waiters(), 0);
    drop(held);
}

// ---- Pool tests (B.7) ---------------------------------------------------
//
// These exercise `Pool<T>` with synthetic items so the contract is
// observable without loading ONNX models. `SessionPool = Pool<SessionTriplet>`
// is just an alias, so any property proven here also holds for the real
// pool.

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_pool_guard_returns_triplet_on_normal_drop() {
    let pool = Pool::new(vec![1u32, 2, 3]);
    assert_eq!(pool.available(), 3);
    {
        let _guard = pool.checkout().await.expect("checkout");
        assert_eq!(pool.available(), 2);
    }
    // Dropping the guard returns the item.
    assert_eq!(pool.available(), 3);
}

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_pool_guard_returns_triplet_on_panic_unwind() {
    // The guard's Drop impl runs during unwind, so a panic between
    // checkout and the natural end of scope still restores capacity.
    let pool = std::sync::Arc::new(Pool::new(vec![1u32]));
    assert_eq!(pool.available(), 1);

    let pool_clone = pool.clone();
    let result = tokio::spawn(async move {
        let _guard = pool_clone.checkout().await.expect("checkout");
        assert_eq!(pool_clone.available(), 0);
        panic!("synthetic inference panic");
    })
    .await;
    assert!(result.is_err(), "spawned task must report the panic");

    // Capacity is restored thanks to PoolGuard::drop running on unwind.
    assert_eq!(pool.available(), 1);
}

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_pool_close_wakes_waiters_with_closed() {
    // A waiter blocked in `checkout` after the inventory is exhausted
    // must resolve to PoolError::Closed when `close()` fires. Map the
    // borrowed guard to the `()` success path so the spawn doesn't
    // need to carry the pool's lifetime.
    let pool = std::sync::Arc::new(Pool::<u32>::new(vec![]));
    let waiter = tokio::spawn({
        let pool = pool.clone();
        async move { pool.checkout().await.map(|_g| ()) }
    });

    // Give the waiter a moment to park on the channel.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    pool.close();

    let res = waiter.await.expect("join");
    assert!(matches!(res, Err(PoolError::Closed)));
}

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_pool_fifo_under_contention() {
    // With a single-slot pool and three queued waiters, the order of
    // wake-ups must match the order in which `checkout` was called.
    // The mpsc channel itself is FIFO; the Mutex serializes waiters
    // so ordering is preserved under normal contention.
    let pool = std::sync::Arc::new(Pool::new(vec![0u32]));

    let primary = pool.checkout().await.expect("primary checkout");
    assert_eq!(pool.available(), 0);

    let waker_log = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    for id in 0u32..3 {
        let pool = pool.clone();
        let log = waker_log.clone();
        handles.push(tokio::spawn(async move {
            let g = pool.checkout().await.expect("checkout");
            log.lock().await.push(id);
            drop(g);
        }));
        // Stagger spawns so each waiter is parked before the next one
        // is registered with the channel.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    // Release the only inventory slot so the queued waiters can run.
    drop(primary);
    for h in handles {
        h.await.expect("join");
    }

    let log = waker_log.lock().await.clone();
    assert_eq!(log, vec![0, 1, 2], "waiters must wake in FIFO order");
}

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_into_owned_for_spawn_blocking() {
    // `into_owned` strips the lifetime so the item can be moved into a
    // blocking thread, then `OwnedReservation::checkin` returns it.
    let pool = std::sync::Arc::new(Pool::new(vec![String::from("triplet")]));
    let guard = pool.checkout().await.expect("checkout");
    let reservation = guard.into_owned();

    let result = tokio::task::spawn_blocking(move || {
        // Pretend we're running blocking inference.
        assert_eq!(*reservation, "triplet");
        reservation.checkin();
        "done"
    })
    .await
    .expect("join");

    // After the blocking task returns the item, the pool is full again.
    assert_eq!(pool.available(), 1);
    assert_eq!(result, "done");
}

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_owned_reservation_returns_on_spawn_blocking_panic() {
    // If the blocking task panics, the reservation's Drop must still
    // return the item so the pool does not leak capacity.
    let pool = std::sync::Arc::new(Pool::new(vec![String::from("triplet")]));
    let guard = pool.checkout().await.expect("checkout");
    let reservation = guard.into_owned();

    let result = tokio::task::spawn_blocking(move || {
        let _reservation = reservation;
        panic!("simulated inference panic");
    })
    .await;

    assert!(result.is_err(), "spawn_blocking must report the panic");
    assert_eq!(
        pool.available(),
        1,
        "reservation must be returned after panic"
    );
}

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_owned_reservation_drop_returns_item() {
    // Dropping an unchecked-in reservation still returns the item.
    let pool = std::sync::Arc::new(Pool::new(vec![String::from("triplet")]));
    let guard = pool.checkout().await.expect("checkout");
    let reservation = guard.into_owned();

    tokio::task::spawn_blocking(move || {
        let _reservation = reservation;
        // reservation dropped here
    })
    .await
    .expect("join");

    assert_eq!(pool.available(), 1);
}

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_pool_close_is_idempotent() {
    // `pool.close()` is wired into the shutdown hook; calling it twice
    // (e.g. shutdown signal + Drop) must not panic.
    let pool = Pool::<u32>::new(vec![]);
    pool.close();
    pool.close();
}

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_pool_waiters_count() {
    let pool = std::sync::Arc::new(Pool::<u32>::new(vec![]));
    let w1 = tokio::spawn({
        let p = pool.clone();
        async move { p.checkout().await.map(|_| ()) }
    });
    let w2 = tokio::spawn({
        let p = pool.clone();
        async move { p.checkout().await.map(|_| ()) }
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(pool.waiters(), 2, "both blocked tasks must be waiters");
    pool.close();
    let _ = w1.await;
    let _ = w2.await;
}

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_owned_reservation_round_trip_through_option() {
    // Mirrors the pattern used by `handle_binary_frame` in ws.rs:
    // the reservation is temporarily moved out of an Option into
    // spawn_blocking and then placed back on success.
    let pool = std::sync::Arc::new(Pool::new(vec![42u32]));
    let guard = pool.checkout().await.expect("checkout");
    let mut reservation: Option<OwnedReservation<u32>> = Some(guard.into_owned());

    let (res_back, val) = tokio::task::spawn_blocking(move || {
        let mut r = reservation.take().unwrap();
        *r += 1;
        let v = *r;
        (r, v)
    })
    .await
    .expect("join");

    reservation = Some(res_back);
    assert_eq!(val, 43);
    drop(reservation);
    assert_eq!(pool.available(), 1);
}

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_pool_slot_not_leaked_on_cancelled_checkout() {
    // If a checkout future is cancelled after registering a waiter but
    // before receiving an item, the oneshot receiver is dropped while the
    // sender remains in the waiters queue.  When another item is checked
    // in, the dead waiter must be skipped and the item returned to the
    // pool — otherwise the slot is leaked forever.
    let pool = std::sync::Arc::new(Pool::new(vec![42u32]));
    let primary = pool.checkout().await.expect("checkout");

    let aborted = tokio::spawn({
        let pool = pool.clone();
        async move { pool.checkout().await }
    });
    // Let the spawned task register as a waiter.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    aborted.abort();
    let _ = aborted.await;

    // The abandoned waiter is still queued.
    assert_eq!(pool.waiters(), 1);

    // Return the primary item.  Without the retry loop in checkin this
    // would silently drop the item because tx.send fails.
    drop(primary);

    assert_eq!(pool.available(), 1, "item must return to pool, not leak");
    assert_eq!(pool.waiters(), 0, "dead waiter must be removed");
}

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_pool_slot_not_leaked_on_timeout_checkout() {
    // Same scenario as above, but using tokio::time::timeout instead of
    // abort — this is the exact path hit by the REST and WS handlers.
    let pool = std::sync::Arc::new(Pool::new(vec![42u32]));
    let primary = pool.checkout().await.expect("checkout");

    let result = tokio::time::timeout(std::time::Duration::from_millis(10), pool.checkout()).await;
    assert!(result.is_err(), "checkout must time out");

    assert_eq!(pool.waiters(), 1);

    drop(primary);

    assert_eq!(
        pool.available(),
        1,
        "item must return to pool after timeout"
    );
    assert_eq!(pool.waiters(), 0, "dead waiter must be removed");
}

#[cfg(feature = "async-pool")]
#[tokio::test]
#[cfg_attr(miri, ignore = "tokio runtime is unsupported under Miri")]
async fn test_pool_multiple_dead_waiters_are_skipped() {
    // Several cancelled waiters in a row should all be skipped in one
    // checkin pass.
    let pool = std::sync::Arc::new(Pool::new(vec![0u32]));
    let primary = pool.checkout().await.expect("checkout");

    let mut handles = Vec::new();
    for _ in 0..3 {
        handles.push(tokio::spawn({
            let pool = pool.clone();
            async move { pool.checkout().await }
        }));
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    for h in handles {
        h.abort();
        let _ = h.await;
    }

    assert_eq!(pool.waiters(), 3);

    drop(primary);

    assert_eq!(
        pool.available(),
        1,
        "item returned after skipping 3 dead waiters"
    );
    assert_eq!(pool.waiters(), 0);
}

#[test]
fn test_pool_sequential_checkouts_visit_every_item() {
    // Engine::warmup relies on this FIFO property: `total()` sequential
    // checkout/checkin cycles touch every pooled item exactly once.
    let pool = Pool::new(vec![1u32, 2, 3]);
    let mut seen = Vec::new();
    for _ in 0..pool.total() {
        let guard = pool.checkout_blocking().expect("checkout");
        seen.push(*guard);
        // guard drops here — the item returns to the back of the queue
    }
    seen.sort_unstable();
    assert_eq!(seen, vec![1, 2, 3]);
}
