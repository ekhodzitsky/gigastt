//! Pin the Objective-C ownership contract used by the Core ML bridge.
//!
//! Adapted from objc2's MIT-licensed regression tests for
//! https://github.com/madsmtm/objc2/issues/861. Run in release mode too:
//! optimizer instruction ordering caused the original ownership violation.

/*
Copyright 2026 Mads Marquart

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/

use std::mem::ManuallyDrop;

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{NSObject, NSObjectProtocol};

#[inline(never)]
fn retain_other(object: &NSObject) -> Retained<NSObject> {
    // SAFETY: the caller keeps a strong reference to this live, non-null
    // NSObject, so retaining it cannot return null. Avoid an extra null branch
    // here: it would prevent the tail-call optimization this test exercises.
    unsafe {
        Retained::retain_autoreleased((object as *const NSObject).cast_mut()).unwrap_unchecked()
    }
}

#[inline(never)]
fn nested_retain(first: Retained<NSObject>, other: &NSObject) {
    let _ = Retained::autorelease_ptr(first);
    drop(retain_other(other));
}

#[inline(never)]
fn direct_retain(first: Retained<NSObject>, other: &NSObject) {
    let _ = Retained::autorelease_ptr(first);
    // SAFETY: the caller keeps a strong reference to other throughout this
    // call. Retaining that non-null object cannot return null. Autoreleasing
    // first must not change ownership of a different object.
    let retained = unsafe {
        Retained::retain_autoreleased((other as *const NSObject).cast_mut()).unwrap_unchecked()
    };
    drop(retained);
}

fn check_separate_ownership(trigger: fn(Retained<NSObject>, &NSObject)) {
    let first = NSObject::new();
    let other = NSObject::new();
    let first_clone = first.clone();
    // Preserve an extra reference on a failing implementation so unwinding
    // does not obscure the assertion with a second invalid release.
    let other_clone = ManuallyDrop::new(other.clone());
    assert_eq!(first.retainCount(), 2);
    assert_eq!(other.retainCount(), 2);
    autoreleasepool(|_| {
        trigger(first_clone, &other);
        assert_eq!(first.retainCount(), 2);
        assert_eq!(other.retainCount(), 2);
    });
    assert_eq!(first.retainCount(), 1);
    assert_eq!(other.retainCount(), 2);
    drop(ManuallyDrop::into_inner(other_clone));
}

#[test]
fn test_objc_autorelease_does_not_retain_a_different_object() {
    check_separate_ownership(direct_retain);
}

#[test]
fn test_objc_autorelease_nested_retain_preserves_ownership() {
    check_separate_ownership(nested_retain);
}
