package compression

import (
	"math"
	"testing"
	"time"

	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/utils"
	"github.com/stretchr/testify/assert"
)

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
	signal := utils.GenerateSparseSignal(rows, feats, 0.01)
	FT_signal := make([][]complex128, rows)
	for i := 0; i < rows; i++ {
		FT_signal[i] = make([]complex128, feats)
		FT_signal[i] = FT(signal[i])
	}
	utils.BatchApply(signal, FFT)
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

// tests for filters by length
func TestLowPassFilter(t *testing.T) {
	var signal = []complex128{complex(1, 0), complex(2, 0), complex(3, 0), complex(4, 0)}
	var cutoff = 2
	var expected = []complex128{complex(1, 0), complex(2, 0)}
	assert.Equal(t, LowPassFilter(signal, cutoff), expected)
}

func TestHighPassFilter(t *testing.T) {
	var signal = []complex128{complex(1, 0), complex(2, 0), complex(3, 0), complex(4, 0)}
	var cutoff = 2
	var expected = []complex128{complex(3, 0), complex(4, 0)}
	assert.Equal(t, HighPassFilter(signal, cutoff), expected)
}

func TestBandPassFilter(t *testing.T) {
	var signal = []complex128{complex(1, 0), complex(2, 0), complex(3, 0), complex(4, 0)}
	var low = 1
	var high = 3
	var expected = []complex128{complex(2, 0), complex(3, 0)}
	assert.Equal(t, BandPassFilter(signal, low, high), expected)
}

func TestBandStopFilter(t *testing.T) {
	var signal = []complex128{complex(1, 0), complex(2, 0), complex(3, 0), complex(4, 0)}
	var low = 1
	var high = 3
	var expected = []complex128{complex(1, 0), complex(4, 0)}
	assert.Equal(t, BandStopFilter(signal, low, high), expected)
}

func BenchmarkBatchFFT(b *testing.B) {
	var rows int = 1_000_0
	var feats int = 1024
	signal := utils.GenerateSparseSignal(rows, feats, 0.01)

	for i := 0; i < b.N; i++ {
		start := time.Now()
		utils.BatchApply(signal, FFT)
		elapsed := time.Since(start)
		b.Logf("Batch FFT took: %v", elapsed)
	}
}

func BenchmarkPlainFFT(b *testing.B) {
	var rows int = 1_000_0
	var feats int = 1024
	signal := utils.GenerateSparseSignal(rows, feats, 0.01)

	for i := 0; i < b.N; i++ {
		start := time.Now()
		for i := range signal {
			FFT(signal[i])
		}
		elapsed := time.Since(start)
		b.Logf("Batch FFT took: %v", elapsed)
	}
}
