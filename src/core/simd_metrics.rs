use crate::core::calc::same_dimension;
#[cfg(feature = "simd")]
use packed_simd::{f32x16, f64x8};

pub trait SIMDOptmized<T = Self> {
    fn dot_product(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn manhattan_distance(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn euclidean_distance(a: &[T], b: &[T]) -> Result<T, &'static str>;
}

macro_rules! simd_optimized_impl {
    ( $type_id:ident, $simd_type:ident , $size:expr ) => {
        impl SIMDOptmized for $type_id {
            fn dot_product(a: &[$type_id], b: &[$type_id]) -> Result<$type_id, &'static str> {
                assert_eq!(a.len(), b.len());
                #[cfg(feature = "simd")]
                {
                    let size = a.len() - (a.len() % $size);
                    let c = a.chunks_exact($size)
                        .zip(b.chunks_exact($size))
                        .map(|(a_chunk, b_chunk)| {
                            let av = $simd_type::from_slice_unaligned(a_chunk);
                            let bv = $simd_type::from_slice_unaligned(b_chunk);
                            av * bv
                        })
                        .fold($simd_type::splat(0.0), |acc, x| acc + x)
                        .sum();
                    let d: $type_id = a[size..].iter().zip(&b[size..]).map(|(p, q)| p * q).sum();
                    Ok(c + d)
                }
                #[cfg(not(feature = "simd"))]
                {
                    Ok(a.iter().zip(b).map(|(p, q)| p * q).sum())
                }
            }

            fn manhattan_distance(a: &[$type_id], b: &[$type_id]) -> Result<$type_id, &'static str> {
                assert_eq!(a.len(), b.len());
                #[cfg(feature = "simd")]
                {
                    let size = a.len() - (a.len() % $size);
                    let c = a.chunks_exact($size)
                        .zip(b.chunks_exact($size))
                        .map(|(a_chunk, b_chunk)| {
                            let av = $simd_type::from_slice_unaligned(a_chunk);
                            let bv = $simd_type::from_slice_unaligned(b_chunk);
                            (av - bv).abs()
                        })
                        .fold($simd_type::splat(0.0), |acc, x| acc + x)
                        .sum();
                    let d: $type_id = a[size..].iter().zip(&b[size..])
                        .map(|(p, q)| (p - q).abs())
                        .sum();
                    Ok(c + d)
                }
                #[cfg(not(feature = "simd"))]
                {
                    Ok(a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum())
                }
            }

            fn euclidean_distance(a: &[$type_id], b: &[$type_id]) -> Result<$type_id, &'static str> {
                same_dimension(a, b)?;
                #[cfg(feature = "simd")]
                {
                    let size = a.len() - (a.len() % $size);
                    let c = a.chunks_exact($size)
                        .zip(b.chunks_exact($size))
                        .map(|(a_chunk, b_chunk)| {
                            let av = $simd_type::from_slice_unaligned(a_chunk);
                            let bv = $simd_type::from_slice_unaligned(b_chunk);
                            let diff = av - bv;
                            diff * diff
                        })
                        .fold($simd_type::splat(0.0), |acc, x| acc + x)
                        .sum();
                    let d: $type_id = a[size..].iter().zip(&b[size..])
                        .map(|(p, q)| (p - q).powi(2))
                        .sum();
                    Ok(c + d)
                }
                #[cfg(not(feature = "simd"))]
                {
                    Ok(a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum())
                }
            }
        }
    };
}

// Implement for f32 and f64 using the corrected macro
simd_optimized_impl!(f32, f32x16, 16);
simd_optimized_impl!(f64, f64x8, 8);
