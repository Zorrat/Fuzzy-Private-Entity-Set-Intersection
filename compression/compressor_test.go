package fpsi

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
	"time"
	"log"

	"github.com/stretchr/testify/assert"
)

func generateSparseSignal(rows int, feats int, sparsity float64) [][]complex128 {
	signal := make([][]complex128, rows)
	for i := 0; i < rows; i++ {
		signal[i] = make([]complex128, feats)
	}
	numNonZero := int(float64(rows) * float64(feats) * sparsity)
	for i := 0; i < numNonZero; i++ {
		row_idx := rand.Intn(rows)
		feat_idx := rand.Intn(feats)
		val := rand.Float64()
		signal[row_idx][feat_idx] = complex(val, 0)
	}
	return signal
}

func FT(x []complex128) []complex128 {
	N := len(x)
	y := make([]complex128, N)
	for k := 0; k < N; k++ {
		for n := 0; n < N; n++ {
			phi := -2.0 * math.Pi * float64(k*n) / float64(N)
			s, c := math.Sincos(phi)
			y[k] += x[n] * complex(c, s)
		}
	}
	return y
}

func TestBatchFFT(t *testing.T) {
	var rows = 128
	var feats = 128
	signal := generateSparseSignal(rows, feats, 0.01)
	FT_signal := make([][]complex128, rows)
	for i := 0; i < rows; i++ {
		FT_signal[i] = make([]complex128, feats)
		FT_signal[i] = FT(signal[i])
	}
	BatchFFT(signal)
	var mse float64
	for i := 0; i < rows; i++ {
		for j := 0; j < feats; j++ {
			diff := signal[i][j] - FT_signal[i][j]
			mse += real(diff)*real(diff) + imag(diff)*imag(diff)
		}
	}
	mse /= float64(rows * feats)
	tolerance := 1e-9
	assert.LessOrEqual(t, mse, tolerance, "MSE is too high!")
}

func BenchmarkBatchFFT(b *testing.B) {
	runtime.GOMAXPROCS(runtime.NumCPU() - 2) // to not forkbomb the system
	var rows int = 1_000
	var feats int = 1024
	signal := generateSparseSignal(rows, feats, 0.01)

	for i := 0; i < b.N; i++ {
		start := time.Now()
		BatchFFT(signal)
		elapsed := time.Since(start)
		b.Logf("Batch FFT took: %v", elapsed)
	}
}

func BenchmarkPlainFFT(b *testing.B) {
	runtime.GOMAXPROCS(runtime.NumCPU() - 2) // to not forkbomb the system
	var rows int = 1_000
	var feats int = 1024
	signal := generateSparseSignal(rows, feats, 0.01)

	for i := 0; i < b.N; i++ {
		start := time.Now()
		for i := range signal {
			err := fft(signal[i])
			if err != nil {
				log.Fatal(err)
			}
		}
		elapsed := time.Since(start)
		b.Logf("Batch FFT took: %v", elapsed)
	}
}
