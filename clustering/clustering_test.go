package clustering

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestComputeCentroid(t *testing.T) {
	vectors := [][]float64{
		{1, 0},
		{0, 1},
		{0.7, 0.7},
	}
	centroid := computeCentroid(vectors)
	assert.Contains(t, vectors, centroid, "Centroid should be one of the input vectors")
}

func TestCluster(t *testing.T) {
	vectors := [][]float64{
		{1, 0},
		{0, 1},
		{0.9, 0.1},
		{0.2, 0.8},
		{0.7, 0.7},
	}
	k := 2
	iterations := 10

	clusters := Cluster(vectors, k, iterations)
	assert.Equal(t, len(vectors), len(clusters), "Number of output vectors should match input")
}

func TestToVector(t *testing.T) {
	centroids := [][]float64{
		{1, 0},
		{0, 1},
	}
	clusters := [][][]float64{
		{{1, 0}, {0.9, 0.1}},
		{{0, 1}, {0.2, 0.8}},
	}

	sortedVectors := ToVector(centroids, 2, clusters, true)
	assert.Equal(t, len(sortedVectors), 4, "Sorted vectors should have the same count as input")
}
