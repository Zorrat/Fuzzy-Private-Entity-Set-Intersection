package utils

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCosineDistance(t *testing.T) {
	vec1 := []float64{1, 0}
	vec2 := []float64{0, 1}
	vec3 := []float64{1, 1}
	dist1 := CosineDistance(vec1, vec2)
	dist2 := CosineDistance(vec1, vec3)

	assert.Equal(t, math.Round(dist1*100)/100, 1.0, "Orthogonal vectors should have cosine distance of 1")
	assert.True(t, dist2 < 1, "Closer vectors should have smaller cosine distance")
}
