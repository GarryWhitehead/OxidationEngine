use std::marker::PhantomData;

/// A strongly typed handle used for safely passing
/// a resource around rather than a pointer/reference.
/// The id usually refers to an index into a container.
///
/// # Safety
/// It is up to the user to ensure the id is valid
/// and that it is within range of the associated container.
///
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Handle<T> {
    id: usize,
    phantom_data: PhantomData<T>,
}

impl<T> Default for Handle<T> {
    fn default() -> Self {
        Self {
            id: usize::MAX,
            phantom_data: PhantomData,
        }
    }
}

impl<T> Handle<T> {
    /// Create a new handle for the specified type.
    pub fn new(id: usize) -> Handle<T> {
        Self {
            id: id,
            phantom_data: PhantomData,
        }
    }

    /// Get the id of the handle.
    pub fn get_id(&self) -> usize {
        self.id
    }

    /// Get whether this handle has a valid id.
    pub fn is_valid(&self) -> bool {
        self.id != usize::MAX
    }

    /// Invalidate the handle.
    pub fn invalidate(&mut self) {
        self.id = usize::MAX;
    }
}
