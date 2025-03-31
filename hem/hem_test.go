package hem

import (
	"math"
	"runtime"
	"testing"

	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/utils"
	"github.com/stretchr/testify/assert"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)
func TestEncryptDecryptCycle(t *testing.T) {
	encCtx, decCtx, _ := GenerateContexts(8)
	testVector := []float64{0.5, 0.25, -0.75, 1.0}
	
	plaintext := ckks.NewPlaintext(*encCtx.params, encCtx.params.MaxLevel())
	encCtx.encoder.Encode(testVector, plaintext)
	ciphertext, err := encCtx.encryptor.EncryptNew(plaintext)
	assert.NoError(t, err, "Encryption failed")
	
	decrypted_pt := decCtx.decryptor.DecryptNew(ciphertext)
	decoded := make([]float64, len(testVector))
	if err = decCtx.encoder.Decode(decrypted_pt, decoded); err != nil {
		panic(err)
	}

	assert.Equal(t, 10, encCtx.params.LogN(), "Should use LogN=10 for n=2")
	assert.InDeltaSlice(t, testVector, decoded[:len(testVector)], 1e-5, "Decrypted values should match original")
}

func TestBatchEncryptDecryptCycle(t *testing.T) {
    // Initialize contexts with reasonable parameters
    encCtx, decCtx, _ := GenerateContexts(8)
    
    // Prepare multiple test vectors of different sizes
    numVectors := 100
    testVectors := make([][]float64, numVectors)

    for i := range testVectors {
       testVectors[i] = utils.GenerateTestVector(800)
    }
    
    // Encrypt all vectors in batch
    encryptedVectors := encCtx.BatchEncrypt(testVectors)
    
    // Verify the number of encrypted vectors matches the input
    assert.Equal(t, len(testVectors), len(encryptedVectors), 
        "Number of encrypted vectors should match input count")
    
    // Decrypt all vectors in batch
    decryptedVectors := decCtx.BatchDecrypt(encryptedVectors)
    
    // Verify the number of decrypted vectors matches
    assert.Equal(t, len(testVectors), len(decryptedVectors), 
        "Number of decrypted vectors should match input count")

    // Verify each vector was correctly processed through the encrypt-decrypt cycle
    for i, originalVector := range testVectors {
        // Skip any nil results
        if decryptedVectors[i] == nil {
            t.Errorf("Decrypted vector at index %d is nil", i)
            continue
        }
        // Check if values match (within tolerance)
        // Note: We only check up to the original vector's length
        // as the decrypted vectors might be padded to MaxSlots
        assert.InDeltaSlice(t, originalVector, decryptedVectors[i][:len(originalVector)], 
            1e-4, "Vector %d: decrypted values should match original within tolerance", i)
    }
    
    // Test edge cases
    
    // Empty batch case
    emptyVectors := [][]float64{}
    emptyEncrypted := encCtx.BatchEncrypt(emptyVectors)
    assert.Empty(t, emptyEncrypted, "Empty input should produce empty encrypted result")
    
    emptyDecrypted := decCtx.BatchDecrypt(emptyEncrypted)
    assert.Empty(t, emptyDecrypted, "Empty encrypted input should produce empty decrypted result")
    
    // Mixed nil case
    mixedVectors := make([]*rlwe.Ciphertext, 3)
    mixedVectors[0] = encryptedVectors[0]  // Valid
    mixedVectors[1] = nil                  // Nil
    mixedVectors[2] = encryptedVectors[1]  // Valid
    
    mixedDecrypted := decCtx.BatchDecrypt(mixedVectors)
    assert.Equal(t, 3, len(mixedDecrypted), "Should have 3 results for 3 inputs")
    assert.NotNil(t, mixedDecrypted[0], "First result should not be nil")
    assert.Nil(t, mixedDecrypted[1], "Second result should be nil")
    assert.NotNil(t, mixedDecrypted[2], "Third result should not be nil")
}


func TestCosineSimilarity(t *testing.T) {
	encCtx, decCtx, evalCtx := GenerateContexts(8)
	
	
	normalized_a := []float64{1.0, 2.0, 3.0}
	normalized_b := []float64{4.0, 5.0, 6.0}

	utils.NormalizeVector(&normalized_a)
	utils.NormalizeVector(&normalized_b)
	
	ptA := ckks.NewPlaintext(*encCtx.params, encCtx.params.MaxLevel())
	encCtx.encoder.Encode(normalized_a, ptA)
	ctA, err := encCtx.encryptor.EncryptNew(ptA)
	assert.NoError(t, err, "Encryption failed")
	
	output_ct := ckks.NewCiphertext(*encCtx.params, encCtx.params.MaxSlots(), encCtx.params.MaxLevel())

	err = evalCtx.DotProduct(ctA,normalized_b,output_ct)
	assert.NoError(t, err, "Dot product calculation failed")
	
	decrypted := decCtx.decryptor.DecryptNew(output_ct)
	decoded := make([]float64, len(normalized_a))
	decCtx.encoder.Decode(decrypted, decoded)
	
	expected := utils.DotProduct(normalized_a, normalized_b)
	assert.InDelta(t, expected, decoded[0], 1e-4, "Cosine similarity mismatch")
}

func TestBatchCosineSimilarity(t *testing.T) {
    // Initialize contexts
    encCtx, decCtx, evalCtx := GenerateContexts(8)
    runtime.GOMAXPROCS(6)
    // Create test vectors for ciphertexts
    numCtVectors := 200
    vectorSize := 235 
    ctVectors := make([][]float64, numCtVectors)
    for i := range ctVectors {
        ctVectors[i] = utils.GenerateTestVector(vectorSize)
		utils.NormalizeVector(&ctVectors[i])
    }

	// Create plaintext vectors for comparison
	numPtVectors := 500
	ptVectors := make([][]float64, numPtVectors)
	for i := range ptVectors {
		ptVectors[i] = utils.GenerateTestVector(vectorSize)
		utils.NormalizeVector(&ptVectors[i])
	}

	// Compute expected cosine similarities
	expectedSims := make([][]float64, numCtVectors)
	for i := range ctVectors {
		expectedSims[i] = make([]float64, numPtVectors)
		for j := range ptVectors {
			expectedSims[i][j] = utils.DotProduct(ctVectors[i], ptVectors[j])
		}
	}
	
	encryptedVectors := encCtx.BatchEncrypt(ctVectors)
    
	resultMatrix, err := evalCtx.BatchDotProduct(encryptedVectors, ptVectors)
    assert.NoError(t, err, "Batch dot product failed")

	// Verify each result
    for i := range resultMatrix {
        // Decrypt a batch of results (an entire row of the result matrix)
        decryptedBatch := decCtx.BatchDecrypt(resultMatrix[i])
        
        // Verify each cosine similarity value
        for j := range decryptedBatch {
            if decryptedBatch[j] == nil {
                t.Errorf("Decrypted result at index (%d,%d) is nil", i, j)
                continue
            }
            // The cosine similarity value is stored in the first element
            // due to the InnerSum operation in dotProduct
            actualCosineSim := decryptedBatch[j][0]
            
            assert.InDelta(t, expectedSims[i][j], actualCosineSim, 1e-4,
                "Cosine similarity mismatch at position (%d,%d): expected %f, got %f", 
                i, j, expectedSims[i][j], actualCosineSim)
            
            maxSlots := len(decryptedBatch[j])
            for k := 1; k < maxSlots; k++ {
                if math.Abs(decryptedBatch[j][k]) > 1e-4 {
                    t.Logf("Note: Non-zero value at position (%d,%d) slot %d: %f", 
                          i, j, k, decryptedBatch[j][k])
                }
            }
        }
    }
	
}