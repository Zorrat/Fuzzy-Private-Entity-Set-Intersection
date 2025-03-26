package utils

import (
	"math"
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

// Cosine distance between two vectors
func CosineDistance(v1, v2 []float64) float64 {
	dot, norm1, norm2 := 0.0, 0.0, 0.0
	for i := range v1 {
		dot += v1[i] * v2[i]
		norm1 += v1[i] * v1[i]
		norm2 += v2[i] * v2[i]
	}
	if norm1 == 0 || norm2 == 0 {
		return 1.0 // Max distance if either vector is zero
	}
	return 1.0 - (dot / (math.Sqrt(norm1) * math.Sqrt(norm2))) // Cosine distance
}

func AverageCosineDistance(vector []float64, vectors [][]float64) float64 {
	var totalSimilarity float64
	for j := 0; j < len(vectors); j++ {
		totalSimilarity += CosineDistance(vector, vectors[j])
	}
	return totalSimilarity / float64(len(vectors))
}
