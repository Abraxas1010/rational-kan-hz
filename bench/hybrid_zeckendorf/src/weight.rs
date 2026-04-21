use rug::Integer;
use std::cell::RefCell;

thread_local! {
    static WEIGHT_CACHE: RefCell<Vec<Integer>> = RefCell::new(vec![
        Integer::from(1),
        Integer::from(1000),
    ]);
}

// Mirrors: HeytingLean.Bridge.Veselov.HybridZeckendorf.weight
pub fn weight(i: u32) -> Integer {
    WEIGHT_CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        while c.len() <= i as usize {
            let prev = c.last().expect("weight cache is never empty").clone();
            c.push(prev.clone() * prev);
        }
        c[i as usize].clone()
    })
}
