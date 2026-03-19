# Examples

This directory contains small runnable programs that mirror the `hem` package
workflow used in the tests.

- `go run ./examples/encrypt_vector` shows the single-vector
  encrypt/decrypt round trip. The current public API exposes single-vector
  encryption through `BatchEncrypt`, so the example passes a one-element slice.
- `go run ./examples/encrypted_similarity` shows cosine similarity on
  encrypted vectors. The vectors are normalized before encryption, and the
  resulting similarity score is read from slot `0` after the homomorphic
  inner-sum step.
