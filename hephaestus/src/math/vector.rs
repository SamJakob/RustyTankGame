use hephaestus_macros::{vector_field, VectorBase, VectorBaseImpl};
use std::fmt::Display;
use std::marker::Sized;
use std::ops::{AddAssign, Div, DivAssign, Mul, MulAssign, Sub};

/// VectorField is a trait that defines the set of methods sufficient to implement a vector field.
/// To automatically implement this for a type/struct where the methods have already been defined
/// (e.g., the built-in numeric types), use [vector_field!].
pub trait VectorField:
    Sized
    + PartialEq
    + PartialOrd
    + AddAssign
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Div<Output = Self>
    + DivAssign<Self>
    + Copy
    + Default
    + Display
{
    /// is_nan returns true if the value is Not a Number.
    fn is_nan(self) -> bool;

    /// Compute the absolute value of the [VectorField].
    fn abs(self) -> Self;

    /// Compute the square root of the [VectorField].
    fn sqrt(self) -> Self;
}

// Built-in definitions for common numeric types.
// https://doc.rust-lang.org/reference/types/numeric.html

vector_field!(u8);
vector_field!(u16);
vector_field!(u32);
vector_field!(u64);
vector_field!(u128);
vector_field!(usize);

vector_field!(i8);
vector_field!(i16);
vector_field!(i32);
vector_field!(i64);
vector_field!(i128);
vector_field!(isize);

vector_field!(f32);
vector_field!(f64);

/// VectorBase defines a base set of operations over a [Vector], where each of the fields conforming
/// to [VectorField]. [Vector]s with a [VectorBase] impl benefit from an automatic blanket
/// implementation of [Vector].
///
/// To allow for vector operations to be re-used over [Vector]s with any number of dimensions, the
/// common operations in [VectorBase] are defined in terms of fixed-length arrays, where the fixed
/// length is the [DIMENSION_COUNT] of the vector.
///
/// The ordering of these fields must be deterministic and must follow mathematical convention
/// where appropriate (e.g., the ordering for a 2D vector with dimensions x and y should always
/// be `[0]` = `x`, `[1]` = `y`, etc.,).
pub trait VectorBase<T: VectorField, const N: usize>: Vector<T, N> + Clone + Copy {
    /// The number of dimensions for this [Vector].
    const DIMENSION_COUNT: usize = N;

    /// get_vector_fields returns an array containing the [N] elements that make up the vector.
    fn get_vector_fields(self) -> [T; N];

    /// for_each_vector_field performs the [callback] on each field in the vector.
    fn for_each_vector_field(&mut self, callback: impl Fn(usize, &mut T));

    /// construct_from_vector_fields constructs a new instance of the specified [VectorBase] with
    /// the given fields.
    fn construct_from_vector_fields(fields: [T; N]) -> Self;
}

pub trait Vector<T: VectorField, const N: usize> {
    /// normalize the vector - the values are updated in-place. Alternatively, see [normalized] to
    /// have the normalized form returned as a separate struct, leaving the original intact.
    ///
    /// https://en.wikipedia.org/wiki/Normal_(geometry)
    fn normalize(&mut self);

    /// normalize the vector coordinates, returning the normalized form as a separate struct.
    /// See also [normalize] to update the vector in-place.
    ///
    /// https://en.wikipedia.org/wiki/Normal_(geometry)
    #[must_use]
    fn normalized(self) -> Self;

    /// magnitude computes the Euclidean magnitude/length of the vector.
    ///
    /// https://en.wikipedia.org/wiki/Magnitude_(mathematics)#Euclidean_vector_space
    #[must_use]
    fn magnitude(self) -> T;

    /// dot computes the Dot Product (also known as the Scalar Product) of two vectors.
    ///
    /// https://en.wikipedia.org/wiki/Dot_product
    ///
    /// ```rust
    /// use hephaestus::math::vector::{Vector, Vector2};
    /// assert_eq!(Vector2 { x: -4.0, y: -9.0 }.dot(Vector2 { x: -1.0, y: 2.0 }), -14.0)
    /// ```
    #[must_use]
    fn dot(self, rhs: Self) -> T;
}

impl<T: VectorField, const N: usize, B: VectorBase<T, N> + DivAssign<T>> Vector<T, N> for B {
    fn normalize(&mut self) {
        let magnitude = self.magnitude();
        if !magnitude.is_nan() && (magnitude > T::default() || magnitude < T::default()) {
            *self /= magnitude;
        }
    }

    fn normalized(self) -> Self {
        let magnitude = self.magnitude();
        if !magnitude.is_nan() && (magnitude > T::default() || magnitude < T::default()) {
            Self::construct_from_vector_fields(
                self.get_vector_fields().map(|field| field / magnitude),
            )
        } else {
            self
        }
    }

    fn magnitude(self) -> T {
        self.dot(self).abs().sqrt()
    }

    fn dot(self, rhs: Self) -> T {
        let lhs_fields = self.get_vector_fields();
        let rhs_fields = rhs.get_vector_fields();

        let mut sum: T = T::default();
        for i in 0..N {
            sum += lhs_fields[i] * rhs_fields[i]
        }
        sum
    }
}

/// VectorCrossProduct must be optionally implemented on vectors for which it is mathematically
/// defined.
pub trait VectorCrossProduct {
    /// cross computes the Cross Product (also known as the Vector Product) of two vectors.
    /// It then returns a new vector with the result of the cross product. See also: [apply_cross].
    ///
    /// https://en.wikipedia.org/wiki/Cross_product
    #[must_use]
    fn cross(self, rhs: Self) -> Self;

    /// apply_cross computes the Cross Product (also known as the Vector Product) of two vectors.
    /// It then updates the vector in-place with the result of the cross product. See also: [cross].
    ///
    /// https://en.wikipedia.org/wiki/Cross_product
    fn apply_cross(&mut self, rhs: Self);
}

/// Vector2 is a two-dimensional vector with fields of type [T].
#[derive(Debug, PartialEq, Clone, Copy, VectorBase, VectorBaseImpl)]
pub struct Vector2<T: VectorField> {
    pub x: T,
    pub y: T,
}

/// Vector3 is a three-dimensional vector with fields of type [T].
#[derive(Debug, PartialEq, Clone, Copy, VectorBase, VectorBaseImpl)]
pub struct Vector3<T: VectorField> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: VectorField> VectorCrossProduct for Vector3<T> {
    fn cross(self, rhs: Self) -> Self {
        Self {
            x: (self.y * rhs.z) - (rhs.y * self.z),
            y: (self.z * rhs.x) - (rhs.z * self.x),
            z: (self.x * rhs.y) - (rhs.x * self.y),
        }
    }

    fn apply_cross(&mut self, rhs: Self) {
        let (x, y, z) = (self.x, self.y, self.z);
        self.x = (y * rhs.z) - (rhs.y * z);
        self.y = (z * rhs.x) - (rhs.z * x);
        self.z = (x * rhs.y) - (rhs.x * y);
    }
}

impl<T: VectorField> Mul for Vector3<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self.cross(rhs)
    }
}

impl<T: VectorField> MulAssign for Vector3<T> {
    fn mul_assign(&mut self, rhs: Self) {
        self.apply_cross(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector2_test_add() {
        assert_eq!(
            Vector2 { x: 1.0, y: 5.0 } + Vector2 { x: 2.0, y: 1.0 },
            Vector2 { x: 3.0, y: 6.0 }
        );
    }

    #[test]
    fn vector2_test_add_assign() {
        let mut vector = Vector2 { x: 1.0, y: 5.0 };
        let other_vector = Vector2 { x: 2.0, y: 1.0 };
        vector += other_vector;

        assert_eq!(vector, Vector2 { x: 3.0, y: 6.0 });
    }

    #[test]
    fn vector2_test_sub() {
        assert_eq!(
            Vector2 { x: 3.0, y: 6.0 } - Vector2 { x: 1.0, y: 5.0 },
            Vector2 { x: 2.0, y: 1.0 }
        );
    }

    #[test]
    fn vector2_test_sub_assign() {
        let mut vector = Vector2 { x: 3.0, y: 6.0 };
        let other_vector = Vector2 { x: 2.0, y: 1.0 };
        vector -= other_vector;

        assert_eq!(vector, Vector2 { x: 1.0, y: 5.0 });
    }

    #[test]
    fn vector2_test_mul() {
        assert_eq!(
            Vector2 { x: 3.0, y: 6.0 } * 2.0,
            Vector2 { x: 6.0, y: 12.0 }
        )
    }

    #[test]
    fn vector2_test_mul_assign() {
        let mut vector = Vector2 { x: 3.0, y: 6.0 };
        let scalar = 2.0;
        vector *= scalar;

        assert_eq!(vector, Vector2 { x: 6.0, y: 12.0 })
    }

    #[test]
    fn vector2_test_div() {
        assert_eq!(Vector2 { x: 3.0, y: 6.0 } / 2.0, Vector2 { x: 1.5, y: 3.0 })
    }

    #[test]
    fn vector2_test_div_assign() {
        let mut vector = Vector2 { x: 3.0, y: 6.0 };
        let scalar = 2.0;
        vector /= scalar;

        assert_eq!(vector, Vector2 { x: 1.5, y: 3.0 })
    }

    #[test]
    fn vector2_test_normalize_zero() {
        let mut vector = Vector2 { x: 0.0, y: 0.0 };
        vector.normalize();

        assert_eq!(vector, Vector2 { x: 0.0, y: 0.0 })
    }

    #[test]
    fn vector2_test_normalize() {
        let mut vector = Vector2 { x: 3.0, y: 4.0 };
        vector.normalize();

        assert_eq!(
            vector,
            Vector2 {
                x: 3.0 / 5.0,
                y: 4.0 / 5.0,
            }
        )
    }

    #[test]
    fn vector2_test_normalize_assign_zero() {
        let vector = Vector2 { x: 0.0, y: 0.0 };

        assert_eq!(vector.normalized(), Vector2 { x: 0.0, y: 0.0 })
    }

    #[test]
    fn vector2_test_normalize_assign() {
        let vector = Vector2 { x: 3.0, y: 4.0 };

        assert_eq!(
            vector.normalized(),
            Vector2 {
                x: 3.0 / 5.0,
                y: 4.0 / 5.0,
            }
        )
    }

    #[test]
    fn vector3_test_magnitude() {
        assert_eq!(
            Vector3 {
                x: 3.0,
                y: 4.0,
                z: 12.0
            }
            .magnitude(),
            13.0
        );

        assert_eq!(
            Vector3 {
                x: 3.0,
                y: 4.0,
                z: -12.0
            }
            .magnitude(),
            13.0
        );
    }

    #[test]
    fn vector2_test_dot() {
        assert_eq!(
            Vector2 { x: -4.0, y: -9.0 }.dot(Vector2 { x: -1.0, y: 2.0 }),
            -14.0
        );
    }

    #[test]
    fn vector2_test_cross() {
        assert_eq!(
            Vector3 {
                x: 3.0,
                y: -3.0,
                z: 1.0
            } * Vector3 {
                x: 4.0,
                y: 9.0,
                z: 2.0
            },
            Vector3 {
                x: -15.0,
                y: -2.0,
                z: 39.0
            }
        );
    }

    #[test]
    fn vector2_test_cross_assign() {
        let mut vector = Vector3 {
            x: 3.0,
            y: -3.0,
            z: 1.0,
        };
        let other_vector = Vector3 {
            x: 4.0,
            y: 9.0,
            z: 2.0,
        };

        vector *= other_vector;

        assert_eq!(
            vector,
            Vector3 {
                x: -15.0,
                y: -2.0,
                z: 39.0
            }
        );
    }
}
