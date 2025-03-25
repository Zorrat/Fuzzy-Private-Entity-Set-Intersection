package fpsi

import (
	"log"
	"math/bits"
	"math/cmplx"
)

// inplace FFT to save ram
//   - warning: this function will clear the memory of x
//   - note: only works when len(x) is power of 2
func fft(x []complex128) {
	N := len(x)
	if N&(N-1) != 0 {
		log.Fatal("Length of input vector must be a power of 2")
		return
	}
	permute(x)
	// Butterfly
	// First 2 steps
	for i := 0; i < N; i += 4 {
		f := complex(imag(x[i+2])-imag(x[i+3]), real(x[i+3])-real(x[i+2]))
		x[i], x[i+1], x[i+2], x[i+3] = x[i]+x[i+1]+x[i+2]+x[i+3], x[i]-x[i+1]+f, x[i]-x[i+2]+x[i+1]-x[i+3], x[i]-x[i+1]-f
	}
	// Remaining steps
	w := complex(0, -1)
	for n := 4; n < N; n <<= 1 {
		w = cmplx.Sqrt(w)
		for o := 0; o < N; o += (n << 1) {
			wj := complex(1, 0)
			for k := 0; k < n; k++ {
				i := k + o
				f := wj * x[i+n]
				x[i], x[i+n] = x[i]+f, x[i]-f
				wj *= w
			}
		}
	}
}

// permutate permutes the input vector using bit reversal.
//
// Uses an in-place algorithm that runs in O(N) time and O(1)
// additional space.
func permute(x []complex128) {
	N := len(x)
	shift := 64 - uint64(bits.Len64(uint64(N-1)))
	N2 := N >> 1
	for i := 0; i < N; i += 2 {
		ind := int(bits.Reverse64(uint64(i)) >> shift)
		// Skip cases where low bit isn't set while high bit is
		// This eliminates 25% of iterations
		if i < N2 {
			if ind > i {
				x[i], x[ind] = x[ind], x[i]
			}
		}
		ind |= N2 // Fast way to get int(bits.Reverse64(uint64(i+1)) >> shift) here
		if ind > i+1 {
			x[i+1], x[ind] = x[ind], x[i+1]
		}
	}
}

// low pass filter for []complex128
func LowPassFilter(x []complex128, cutoff int) []complex128 {
	return x[:cutoff]
}

// high pass filter for []complex128
func HighPassFilter(x []complex128, cutoff int) []complex128 {
	return x[cutoff:]
}

// band pass filter for []complex128
func BandPassFilter(x []complex128, low, high int) []complex128 {
	return x[low:high]
}

// band stop filter for []complex128
func BandStopFilter(x []complex128, low, high int) []complex128 {
	return append(x[:low], x[high:]...)
}

// hstack to float64 from complex128 of [][]complex128
//
// warning: this function will clear the memory of x
func ToFloat64(x [][]complex128) [][]float64 {

	N := len(x)
	M := len(x[0])
	res := make([][]float64, N)
	for i := range res {
		res[i] = make([]float64, 2*M)
		for j := range x[i] {
			res[i][2*j] = real(x[i][j])
			res[i][2*j+1] = imag(x[i][j])
		}
		// clear memory
		x[i] = nil
	}

	return res
}

// use [][]float64 as real part to convert to [][]complex128
//
// warning: this function will clear the memory of x
func FromFloat64(x [][]float64) [][]complex128 {
	N := len(x)
	M := len(x[0]) / 2
	res := make([][]complex128, N)
	for i := range res {
		res[i] = make([]complex128, M)
		for j := range res[i] {
			res[i][j] = complex(x[i][2*j], x[i][2*j+1])
		}
		// clear memory
		x[i] = nil
	}
	return res
}
