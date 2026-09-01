use std::process::Command;

use super::BufStorage;

#[test]
fn private_pool_contract() {
    if std::env::var_os("FURIOSA_OPT_STD_POOL_TEST_CHILD").is_none() {
        // A fresh process isolates Rayon's one-shot global pool from the private pool under test.
        // The parent selects the private pool size and only checks the child result.
        let status = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("storage::pool_tests::private_pool_contract")
            .env("FURIOSA_OPT_STD_POOL_TEST_CHILD", "1")
            .env("RAYON_NUM_THREADS", "3")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }

    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
    let values = BufStorage::<_, Vec<u8>>::from_vec(0..(1 << 18));
    let mapped = values.map(|value| {
        assert!(
            std::thread::current()
                .name()
                .is_some_and(|name| name.starts_with("furiosa-opt-buf-"))
        );
        value + 1
    });

    assert_eq!(mapped.get(0), 1);
    assert_eq!(super::BUF_POOL.current_num_threads(), 3);
    assert_eq!(rayon::current_num_threads(), 1);
}
