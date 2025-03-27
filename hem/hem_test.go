package hem

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestEncryptDecryptCycle(t *testing.T) {
	// Initialize contexts for n=2
	encCtx, decCtx, _ := generateContexts(2)
	
	// Test vector with values that can be precisely represented
	testVector := []float64{0.5, 0.25, -0.75, 1.0}
	
	// Encode and encrypt
	plaintext := ckks.NewPlaintext(*encCtx.params, encCtx.params.MaxLevel())
	encCtx.encoder.Encode(testVector, plaintext)
	ciphertext, err := encCtx.encryptor.EncryptNew(plaintext)
	assert.NoError(t, err, "Encryption failed")
	
	// Decrypt and decode
	decrypted_pt := decCtx.decryptor.DecryptNew(ciphertext)
	decoded := make([]float64, len(testVector))
	if err = decCtx.encoder.Decode(decrypted_pt, decoded); err != nil {
		panic(err)
	}

	// Verify parameters used
	assert.Equal(t, 10, encCtx.params.LogN(), "Should use LogN=10 for n=2")
	
	// Compare with tolerance for floating point errors
	assert.InDeltaSlice(t, testVector, decoded[:len(testVector)], 1e-5, "Decrypted values should match original")
}

func TestCosineSimilarity(t *testing.T) {
	encCtx, decCtx, evalCtx := generateContexts(2)
	
	// Create test vectors
	a := []float64{1.0, 2.0, 3.0}
	b := []float64{4.0, 5.0, 6.0}

	normalized_a := normalizeVector(a)
	normalized_b := normalizeVector(b)
	
	// Encrypt vectors
	ptA := ckks.NewPlaintext(*encCtx.params, encCtx.params.MaxLevel())
	encCtx.encoder.Encode(normalized_a, ptA)
	ctA, err := encCtx.encryptor.EncryptNew(ptA)
	assert.NoError(t, err, "Encryption failed")
	
	
	// Compute encrypted dot product
	encDot, err := evalCtx.dotProduct(ctA, normalized_b)
	assert.NoError(t, err, "Dot product calculation failed")
	
	// Decrypt result
	decrypted := decCtx.decryptor.DecryptNew(encDot)
	decoded := make([]float64, len(normalized_a))
	decCtx.encoder.Decode(decrypted, decoded)
	fmt.Println(decoded)
	decodedDotProduct := decoded[0]
	
	// Calculate plaintext values

	expectedDotProduct := dotProduct(normalized_a, normalized_b)
	// Compare with decrypted result (first element contains the sum)
	assert.InDelta(t, expectedDotProduct, decodedDotProduct, 1e-5, "Cosine similarity mismatch")
}


// Helper functions for plaintext calculations
func dotProduct(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func norm(vec []float64) float64 {
	var sum float64
	for _, v := range vec {
		sum += v * v
	}
	return math.Sqrt(sum)
}
func normalizeVector(vec []float64) []float64 {
	norm := norm(vec)
	for i := range vec {
		vec[i] /= norm
	}
	return vec
}
