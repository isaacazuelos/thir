//! Utility functions to make up for the fact I'm using `Vec`.
//!
//! Since I shouldn't be using `Vec` but instead some kind of iterator adapters,
//! these aren't exactly worried about efficiency.

pub fn lookup<'a, K, V>(key: &K, vec: &'a Vec<(K, V)>) -> Option<&'a V>
where
    K: PartialEq,
{
    for (k, v) in vec.iter() {
        if k == key {
            return Some(v);
        }
    }

    None
}

pub fn union<T>(left: &Vec<T>, right: &Vec<T>) -> Vec<T>
where
    T: PartialEq + Clone,
{
    let mut buf = left.clone();

    for v in right {
        buf.push(v.clone());
    }

    buf.dedup();
    buf
}

pub fn intersection<T>(left: &Vec<T>, right: &Vec<T>) -> Vec<T>
where
    T: PartialEq + Clone,
{
    let mut buf = Vec::new();

    for l in left.iter() {
        if right.contains(l) {
            buf.push(l.clone())
        }
    }

    buf
}
