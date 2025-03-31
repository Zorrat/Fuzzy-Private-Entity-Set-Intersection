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
func Cluster(vectors [][]float64, names []string, iterations int) ([][]float64, []string) {
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
	clusterNames := make([][]string, k) 

	for iter := 0; iter < iterations; iter++ {
		// Reset clusters for this iteration
		clusters = make([][][]float64, k)
		clusterNames = make([][]string, k) // Reset names for each cluster
		clusterSizes := make([]int, k)     // Track the number of vectors in each cluster

		// Assign vectors to the closest centroid
		for idx, vec := range vectors {
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
				clusterNames[bestCluster] = append(clusterNames[bestCluster], names[idx])
				clusterSizes[bestCluster]++
			} else {
				// If the cluster already has K vectors, find the next best cluster
				// and assign the vector there.
				for i := 0; i < k; i++ {
					if clusterSizes[i] < k {
						clusters[i] = append(clusters[i], vec)
						clusterNames[i] = append(clusterNames[i], names[idx])
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

	return ToVector(centroids, k, clusters, clusterNames, true)
}

// Move the target element (of type T) to the first position in the slice
func moveToFirst[T any](target T, slice []T) []T {
	for i, element := range slice {
		if reflect.DeepEqual(element, target) {
			// Move found element to the front
			return append([]T{element}, append(slice[:i], slice[i+1:]...)...)
		}
	}
	return slice
}

// ToVector processes centroids, clusters, and names, and sorts them by similarity to centroids
func ToVector(centroids [][]float64, k int, clusters [][][]float64, clusterNames [][]string, sorted bool) ([][]float64, []string) {
	if sorted {
		var withinClusterSims []float64
		for i := 0; i < k; i++ {
			withinClusterSims = append(withinClusterSims, utils.AverageCosineDistance(centroids[i], clusters[i]))
			clusters[i] = moveToFirst(centroids[i], clusters[i])
			clusterNames[i] = moveToFirst(clusterNames[i][0], clusterNames[i])
		}

		// Sort clusters by similarity to centroid
		sort.Slice(clusters, func(i, j int) bool {
			return withinClusterSims[i] < withinClusterSims[j]
		})
	}
	return slices.Concat(clusters...), slices.Concat(clusterNames...)
}
