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

// K-Means clustering (ensures each cluster contains exactly K vectors)
func Cluster(vectors [][]float64, iterations int) [][]float64 {
	n := len(vectors)
	k := int(math.Sqrt(float64(n)))
	

	// Ensure that N is divisible by K
	if n%k != 0 {
		panic("The number of vectors must be divisible by K")
	}

	// Initialize centroids randomly (can be optimized with K-means++)
	centroids := make([][]float64, k)
	for i := range centroids {
		centroids[i] = vectors[rand.Intn(n)]
	}

	// Create the initial empty clusters
	clusters := make([][][]float64, k)

	for iter := 0; iter < iterations; iter++ {
		// Reset clusters for this iteration
		clusters = make([][][]float64, k)
		clusterSizes := make([]int, k) // Track the number of vectors in each cluster

		// Assign vectors to the closest centroid
		for _, vec := range vectors {
			// Find the nearest centroid
			minDist, bestCluster := math.MaxFloat64, 0
			for j, cent := range centroids {
				dist := utils.CosineDistance(vec, cent)
				if dist < minDist {
					minDist = dist
					bestCluster = j
				}
			}

			// If the chosen cluster has less than K vectors, assign the vector
			if clusterSizes[bestCluster] < k {
				clusters[bestCluster] = append(clusters[bestCluster], vec)
				clusterSizes[bestCluster]++
			} else {
				// If the cluster already has K vectors, find the next best cluster
				// and assign the vector there.
				for i := 0; i < k; i++ {
					if clusterSizes[i] < k {
						clusters[i] = append(clusters[i], vec)
						clusterSizes[i]++
						break
					}
				}
			}
		}

		// Update centroids (compute centroid for non-empty clusters)
		for i := range centroids {
			if len(clusters[i]) > 0 {
				centroids[i] = computeCentroid(clusters[i])
			}
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
