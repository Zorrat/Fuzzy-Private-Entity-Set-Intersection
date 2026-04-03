# Fuzzy Private Entity Set Intersection

Privacy-preserving fuzzy entity matching in Go using [Lattigo CKKS](https://github.com/tuneinsight/lattigo). This repository focuses on matching names and company-like entities without exposing raw vectors during similarity computation.

The current codebase is best understood as a research-style prototype for encrypted entity resolution:

- Text inputs are cleaned and normalized.
- Names are converted into character n-gram TF-IDF vectors.
- Vectors can optionally be compressed in the frequency domain.
- Query vectors are encrypted with CKKS.
- Cosine-style similarity is computed homomorphically as a dot product between encrypted and plaintext-normalized vectors.
- Only the final similarity scores are decrypted.

## What This Project Does

This repository explores a practical pipeline for fuzzy matching between two datasets when one side wants to keep its feature vectors encrypted during computation.

It is useful for experiments around:

- privacy-preserving entity resolution
- encrypted approximate string matching
- CKKS-based similarity search
- compression tradeoffs before homomorphic evaluation

The implementation is centered on entity names, but the same vector pipeline can be adapted to other short text identifiers.

## Core Idea

At a high level, the workflow is:

1. Clean entity names with normalization rules such as lowercasing, punctuation removal, accent stripping, and company suffix standardization.
2. Fit a TF-IDF vectorizer on a shared corpus of names.
3. Transform the query and candidate names into numeric vectors.
4. Normalize vectors so the dot product approximates cosine similarity.
5. Optionally compress vectors using FFT-based filtering.
6. Encrypt the query vector with CKKS.
7. Evaluate encrypted dot products against plaintext candidate vectors.
8. Decrypt slot `0` of each result ciphertext to recover the similarity score.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `hem/` | Homomorphic encryption contexts, batch encryption/decryption, and encrypted dot product evaluation |
| `data/` | Name loading, cleaning, suffix normalization, TF-IDF vectorization, and n-gram generation |
| `compression/` | FFT helpers plus low/high/band pass filtering experiments |
| `clustering/` | Vector clustering and centroid ordering helpers |
| `utils/` | Cosine distance, normalization, dot products, and test vector helpers |
| `examples/` | Small runnable examples for encryption/decryption and encrypted similarity |
| `cmd/server/` | Gin-based demo API for encrypted entity matching |
| `client/` | Static demo UI plus a small FastAPI mock used during UI prototyping |
| `tests/` | End-to-end experimental tests for matching and compression |
| `serialization/` | Placeholder package for future serialization work |

## Homomorphic Encryption Stack

The project uses:

- `github.com/tuneinsight/lattigo/v6`
- the CKKS scheme for approximate arithmetic on real-valued vectors
- batched encoding so multiple slots are available per ciphertext

`hem.GenerateContexts(ln int)` creates the encryption, decryption, and evaluation contexts used throughout the repo. In the current implementation:

- `ln=8` corresponds to roughly 256 usable slots
- `ln=9` corresponds to roughly 512 usable slots
- `ln=10` corresponds to roughly 1024 usable slots

The code currently generates one keypair locally and uses it for both encryption and decryption in tests and examples.

## Quick Start

### Prerequisites

- Go `1.24.1` or a compatible `1.24.x` toolchain

The module path includes a `.git` suffix, so imports in local code look like:

```go
import "github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/hem"
```

### Install Dependencies

```bash
go mod download
```

## Run The Examples

Two small examples live in [`examples/`](examples/):

### 1. Encrypt and decrypt a single vector

```bash
go run ./examples/encrypt_vector
```

This shows the basic round-trip through CKKS:

- create contexts
- encrypt one vector through `BatchEncrypt`
- decrypt with `BatchDecrypt`
- slice the decoded slots back to the original vector length

### 2. Compute encrypted cosine similarity

```bash
go run ./examples/encrypted_similarity
```

This example:

- normalizes two vectors
- encrypts them
- evaluates the encrypted dot product
- decrypts the result
- compares the encrypted score against the plaintext dot product

Additional example notes are documented in [`examples/README.md`](examples/README.md).

## Example Library Workflow

The public API used throughout the repo looks like this:

```go
package main

import (
	"log"

	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/hem"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/utils"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

func main() {
	query := []float64{1, 1, 0, 0}
	candidate := []float64{1, 0, 1, 0}

	utils.NormalizeVector(&query)
	utils.NormalizeVector(&candidate)

	encCtx, decCtx, evalCtx := hem.GenerateContexts(8)

	encrypted := encCtx.BatchEncrypt([][]float64{query})
	if len(encrypted) != 1 || encrypted[0] == nil {
		log.Fatal("encryption failed")
	}

	result := encrypted[0].CopyNew()
	if err := evalCtx.DotProduct(encrypted[0], candidate, result); err != nil {
		log.Fatal(err)
	}

	decrypted := decCtx.BatchDecrypt([]*rlwe.Ciphertext{result})
	score := decrypted[0][0]

	log.Printf("encrypted cosine similarity: %.6f", score)
}
```

## Data Pipeline

### Cleaning and normalization

`data.CleanCompanyName` applies several preprocessing steps:

- Unicode normalization with accent stripping
- lowercasing
- punctuation removal
- whitespace collapsing
- replacement of company suffix variants with standardized forms

This is important because the similarity model relies on consistent text structure before vectorization.

### TF-IDF vectorization

`data.TfidfVectorizer` builds character n-gram features and transforms each string into a dense `[]float64` vector. The tests and demo server currently use:

- n-gram length `2`
- minimum document frequency `1`

### Vector normalization

`utils.NormalizeVector` scales vectors to unit norm so the homomorphic dot product corresponds to cosine similarity.

## Compression Experiments

The `compression` package contains FFT-based utilities for reducing or reshaping vectors before encryption:

- `Prepare`
- `FFT`
- `LowPassFilter`
- `HighPassFilter`
- `BandPassFilter`
- `BandStopFilter`
- `ToFloat64`
- `FromFloat64`

The experimental tests in [`tests/all_test.go`](tests/all_test.go) and [`hem/hem_test.go`](hem/hem_test.go) compare compressed and uncompressed similarity quality using metrics such as:

- mean absolute error
- RMSE
- thresholded classification accuracy
- top-k matching behavior

This makes the repo useful not only as a CKKS demo, but also as a sandbox for evaluating compression-vs-accuracy tradeoffs.

## Run The Demo Server

The Go backend is in [`cmd/server/`](cmd/server/). Because it loads `global.json` from its current working directory, run it from inside that folder:

```bash
cd cmd/server
go run .
```

The server:

- fits a TF-IDF vectorizer on `cmd/server/global.json`
- accepts a query string and a list of candidate strings
- cleans and vectorizes the input
- encrypts the query vector
- computes encrypted similarities against plaintext candidate vectors
- returns similarity scores as JSON

### API

`POST /`

Request body:

```json
{
  "query": "Acme Corporation",
  "data": ["Acme Corp", "Zenith Solutions", "Global Industries"]
}
```

Response shape:

```json
{
  "cosine_sims": [0.98, 0.12, 0.07],
  "query_enc": [0.31, -0.55, 0.04]
}
```

`cosine_sims` is the meaningful output for matching. The current `query_enc` field is a demo-friendly placeholder vector rather than a serialized ciphertext.

## Run The Client Demo

The static UI lives in [`client/index.html`](client/index.html) and sends requests to `http://localhost:8080/`.

One simple way to view it locally is:

```bash
cd client
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

Notes:

- The page expects the Go server above to be running on port `8080`.
- `client/main.py` is a separate FastAPI mock that returns random similarity values and was used for UI prototyping. It is not the main homomorphic backend.

## Testing

Run the full Go test suite from the repository root:

```bash
go test ./... -run=. -v
```

To run tests plus benchmarks and capture output:

```bash
go test ./... -bench=. -run=. -v -benchmem > tests.txt
```

Coverage in the repository includes:

- unit tests for cleaning, vectorization, clustering, compression, and utility functions
- HE round-trip encryption/decryption tests
- encrypted cosine similarity tests
- end-to-end name matching experiments with and without compression

The checked-in [`tests.txt`](tests.txt) file contains a historical benchmark run for reference.

## Notable Tests To Read First

- [`hem/hem_test.go`](hem/hem_test.go): core encrypted workflow, similarity checks, and name matching demos
- [`tests/all_test.go`](tests/all_test.go): compression quality experiments and larger matching analyses
- [`compression/compressor_test.go`](compression/compressor_test.go): FFT correctness and benchmark comparisons
- [`data/data_test.go`](data/data_test.go): normalization and vectorization behavior

## Known Limitations

This repo is promising, but it is still prototype code. A few important caveats:

- Despite the project name, the current implementation is closer to encrypted fuzzy matching than a full production PSI protocol.
- The code uses a single locally generated CKKS keypair in tests and examples, not distributed threshold decryption.
- Candidate vectors are plaintext during evaluation; the main protected asset is the encrypted query side.
- `serialization/` is currently a placeholder.
- The server response does not yet expose real ciphertext serialization for frontend use.
- Several datasets such as `global.json` are duplicated across folders for convenience in tests and demos.

## Development Notes

- Use Go `1.24.x` to match `go.mod`.
- If you are extending the HE layer, start with [`hem/keygeneration.go`](hem/keygeneration.go) and [`hem/evaluator.go`](hem/evaluator.go).
- If you are improving matching quality, start with [`data/cleaner.go`](data/cleaner.go), [`data/vectorizer.go`](data/vectorizer.go), and the experiments in [`tests/all_test.go`](tests/all_test.go).
- If you want to tweak the demo experience, look at [`cmd/server/server.go`](cmd/server/server.go) and [`client/index.html`](client/index.html).

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
