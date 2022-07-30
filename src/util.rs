//! Utility functions to make up for the fact I'm using `Vec`.
//!
//! Since I shouldn't be using `Vec` but instead some kind of iterator adapters,
//! these aren't exactly worried about efficiency.

pub fn lookup<'a, K, V>(key: &K, vec: &'a [(K, V)]) -> Option<&'a V>
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

pub fn union<T>(left: &[T], right: &[T]) -> Vec<T>
where
    T: PartialEq + Clone,
{
    let mut buf = left.to_vec();

    for v in right {
        buf.push(v.clone());
    }

    buf.dedup();
    buf
}

pub fn intersection<T>(left: &[T], right: &[T]) -> Vec<T>
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

pub fn append<T>(mut left: Vec<T>, right: Vec<T>) -> Vec<T> {
    left.extend(right);
    left
}

pub fn minus<T>(elements: Vec<T>, to_remove: &[T]) -> Vec<T>
where
    T: PartialEq,
{
    elements
        .into_iter()
        .filter(|element| !to_remove.contains(element))
        .collect()
}

pub fn partition<T, F>(predicate: F, elements: Vec<T>) -> (Vec<T>, Vec<T>)
where
    F: Fn(&T) -> bool,
{
    let mut ts = vec![];
    let mut fs = vec![];

    for element in elements {
        if predicate(&element) {
            ts.push(element);
        } else {
            fs.push(element);
        }
    }

    (ts, fs)
}

pub fn zip_with<F, A, B, R>(f: F, lhs: Vec<A>, rhs: Vec<B>) -> Vec<R>
where
    F: Fn(A, B) -> R,
{
    let mut buf = Vec::new();

    for (a, b) in lhs.into_iter().zip(rhs.into_iter()) {
        buf.push(f(a, b));
    }

    buf
}

pub fn zip_with_try<F, A, B, E, R>(mut f: F, lhs: Vec<A>, rhs: Vec<B>) -> Result<Vec<R>, E>
where
    F: FnMut(A, B) -> Result<R, E>,
{
    let mut buf = Vec::new();

    for (a, b) in lhs.into_iter().zip(rhs.into_iter()) {
        buf.push(f(a, b)?)
    }

    Ok(buf)
}
