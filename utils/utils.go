package utils

import (
	"math"
	"math/rand"
	"reflect"
	"sync"
)

func BatchApply[T any](x []T, f interface{}, extras ...interface{}) {
	var wg sync.WaitGroup
	funcValue := reflect.ValueOf(f)
	for i := range x {
		wg.Add(1)
		go func(item T) {
			defer wg.Done()
			args := []reflect.Value{reflect.ValueOf(item)}
			for _, extra := range extras {
				args = append(args, reflect.ValueOf(extra))
			}
			funcValue.Call(args)
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

func CosineDistanceAll(v1, v2 [][]float64) [][]float64 {
	result := make([][]float64, len(v1))
	for i := range v1 {
		result[i] = make([]float64, len(v2))
		for j := range v2 {
			result[i][j] = CosineDistance(v1[i], v2[j])
		}
	}
	return result
}

func MeanAverageError(v1, v2 [][]float64) float64 {
	if len(v1) != len(v2) {
		panic("Vectors must be of the same length")
	}
	var totalError float64
	for i := range v1 {
		for j := range v1[i] {
			totalError += math.Abs(v1[i][j] - v2[i][j])
		}
	}
	return totalError / float64(len(v1)*len(v1[0]))
}

func AverageCosineDistance(vector []float64, vectors [][]float64) float64 {
	var totalSimilarity float64
	for j := 0; j < len(vectors); j++ {
		totalSimilarity += CosineDistance(vector, vectors[j])
	}
	return totalSimilarity / float64(len(vectors))
}

func GenerateSparseSignal(rows int, feats int, sparsity float64) [][]complex128 {
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

// Helper functions
func DotProduct(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func GenerateTestVector(length int) []float64 {
	vec := make([]float64, length)
	for i := range vec {
		vec[i] = rand.Float64()*2 - 1 // Values between -1 and 1
	}
	return vec
}

func norm(vec []float64) float64 {
	var sum float64
	for _, v := range vec {
		sum += v * v
	}
	return math.Sqrt(sum)
}

func NormalizeVector(vec *[]float64) {
	n := norm(*vec)
	for i := range *vec {
		(*vec)[i] /= n
	}
}
