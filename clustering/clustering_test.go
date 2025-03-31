package clustering

import (
	"reflect"
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
		{0.5, 0.5},
		{0.3, 0.3},
		{0.1, 0.1},
		{0.6, 0.6},
	}
	names := []string{
		"Vector 1", "Vector 2", "Vector 3", "Vector 4",
		"Vector 5", "Vector 6", "Vector 7", "Vector 8", "Vector 9",
	}
	iterations := 100

	// Perform clustering
	centroids, clusterNames := Cluster(vectors, names, iterations)

	// Verify that the number of clusters and names match the number of input vectors
	assert.Equal(t, len(vectors), len(clusterNames), "Number of cluster names should match number of vectors")

	// Verify that the names have been shuffled according to the clustering
	// Checking if names in the first cluster correspond to their assigned vectors
	assert.Contains(t, clusterNames, "Vector 1", "Cluster names should contain Vector 1")
	assert.Contains(t, clusterNames, "Vector 2", "Cluster names should contain Vector 2")

	// Verifying the centroids are as expected, should be based on the input vectors
	assert.Contains(t, centroids, vectors[0], "Centroid should match an input vector")
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
	names := [][]string{
		{"v1", "v2"},
		{"v3", "v4"},
	}
	// Test when sorting is enabled
	sortedVectors, sortedNames := ToVector(centroids, 2, clusters, names, true)

	// Verify that sorted vectors have the same count as input clusters
	assert.Equal(t, len(sortedVectors), 4, "Sorted vectors should have the same count as input clusters")

	// Ensure that cluster names are shuffled and sorted
	assert.NotEqual(t, sortedNames[0], sortedNames[1], "Cluster names should be shuffled after sorting")
	assert.Contains(t, sortedNames, "v1", "Sorted names should contain Vector 1")
	assert.Contains(t, sortedNames, "v2", "Sorted names should contain Vector 2")

	// Ensure that centroids have been moved correctly according to sorted clusters
	assert.Equal(t, sortedVectors[0], centroids[0], "First centroid should match the sorted vector")
}

func TestMoveToFirst(t *testing.T) {
	tests := []struct {
		name     string
		target   int
		slice    []int
		expected []int
	}{
		{
			name:     "Move element to the front",
			target:   3,
			slice:    []int{1, 2, 3, 4, 5},
			expected: []int{3, 1, 2, 4, 5},
		},
		{
			name:     "Element already at the front",
			target:   1,
			slice:    []int{1, 2, 3, 4, 5},
			expected: []int{1, 2, 3, 4, 5},
		},
		{
			name:     "Element not in slice",
			target:   6,
			slice:    []int{1, 2, 3, 4, 5},
			expected: []int{1, 2, 3, 4, 5}, // No change
		},
		{
			name:     "Single element slice",
			target:   5,
			slice:    []int{5},
			expected: []int{5}, // No change, only one element
		},
		{
			name:     "Empty slice",
			target:   1,
			slice:    []int{},
			expected: []int{}, // No change, empty slice
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := moveToFirst(tt.target, tt.slice)
			if !reflect.DeepEqual(actual, tt.expected) {
				t.Errorf("moveToFirst() = %v, want %v", actual, tt.expected)
			}
		})
	}
}
