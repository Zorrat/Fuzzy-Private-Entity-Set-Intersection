package clustering

import (
	"math"
	"math/rand"
	"reflect"
	"slices"
	"sort"

	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/utils"
)

// Compute centroid by selecting the vector with the highest average cosine similarity
func computeCentroid(vectors [][]float64) []float64 {
	bestCandidate := vectors[0]
	bestSimilarity := -1.0

	// Iterate through all vectors and compute their average similarity to others
	for i := 0; i < len(vectors); i++ {
		averageSimilarity := utils.AverageCosineDistance(vectors[i], vectors)

		// Update the best candidate if the current vector has higher average similarity
		if averageSimilarity > bestSimilarity {
			bestCandidate = vectors[i]
			bestSimilarity = averageSimilarity
		}
	}
	return bestCandidate
}

// K-Means clustering (without structs)
func Cluster(vectors [][]float64, k, iterations int) [][]float64 {
	n := len(vectors)

	// Initialize centroids randomly
	centroids := make([][]float64, k)
	for i := range centroids {
		centroids[i] = vectors[rand.Intn(n)]
	}
	clusters := make([][][]float64, k)
	for iter := 0; iter < iterations; iter++ {
		// Assign vectors to closest centroid
		clusters = make([][][]float64, k)
		for _, vec := range vectors {
			minDist, bestCluster := math.MaxFloat64, 0
			for j, cent := range centroids {
				dist := utils.CosineDistance(vec, cent)
				if dist < minDist {
					minDist = dist
					bestCluster = j
				}
			}
			clusters[bestCluster] = append(clusters[bestCluster], vec)
		}

		// Update centroids
		for i := range centroids {
			centroids[i] = computeCentroid(clusters[i])
		}
	}
	return ToVector(centroids, k, clusters, true)
}

func moveToFirst(target []float64, matrix [][]float64) [][]float64 {
	for i, row := range matrix {
		if reflect.DeepEqual(row, target) {
			// Move found row to the front
			matrix = append([][]float64{row}, append(matrix[:i], matrix[i+1:]...)...)
			break
		}
	}
	return matrix
}

func ToVector(centroid [][]float64, k int, clusters [][][]float64, sorted bool) [][]float64 {
	if sorted {
		var within_cluster_sims []float64
		for i := 0; i < k; i++ {
			within_cluster_sims = append(within_cluster_sims, utils.AverageCosineDistance(centroid[i], clusters[i]))
			moveToFirst(centroid[i], clusters[i])
		}

		// sort clusters by similarity to centroid
		sort.Slice(clusters, func(i, j int) bool {
			return within_cluster_sims[i] < within_cluster_sims[j]
		})
	}
	return slices.Concat(clusters...)
}
