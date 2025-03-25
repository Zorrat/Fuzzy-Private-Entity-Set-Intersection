package utils

import (
	"sync"
)

// Batch apply a function to a slice of complex128
func BatchApply(x [][]complex128, f func([]complex128)) {
	var wg sync.WaitGroup
	for i := range x {
		wg.Add(1)
		go func(signal []complex128) {
			defer wg.Done()
			f(signal)
		}(x[i])
	}
	wg.Wait()
}
