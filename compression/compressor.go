package fpsi

import (
	"errors"
	"log"
	"math/bits"
	"math/cmplx"
	"sync"
)

func BatchFFT(x [][]complex128) {
	var wg sync.WaitGroup
	for i := range x {
		wg.Add(1)
		go func(signal []complex128) {
			defer wg.Done()
			err := fft(signal)
			if err != nil {
				log.Fatal(err)
			}
		}(x[i])
	}
	wg.Wait()
}

// note only works when len(x) is power of 2
// inplace FFT to save ram
func fft(x []complex128) error {
	N := len(x)
	if N&(N-1) != 0 {
		return errors.New("fft: length of input must be a power of 2")
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
	// no error
	return nil
}

// permutate permutes the input vector using bit reversal.
// Uses an in-place algorithm that runs in O(N) time and O(1) additional space.
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
