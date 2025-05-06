package hem

import (
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/compression"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/data"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/utils"
	"github.com/mjibson/go-dsp/fft"
	"github.com/stretchr/testify/assert"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestEncryptDecryptCycle(t *testing.T) {
	encCtx, decCtx, _ := GenerateContexts(8)
	testVector := []float64{0.5, 0.25, -0.75, 1.0}

	plaintext := ckks.NewPlaintext(*encCtx.params, encCtx.params.MaxLevel())
	// log plaintest in print while testing
	// log.Println("Plaintext before encoding:", plaintext)
	encCtx.encoder.Encode(testVector, plaintext)
	ciphertext, err := encCtx.encryptor.EncryptNew(plaintext)
	assert.NoError(t, err, "Encryption failed")
	// log.Println("Ciphertext after encryption:", ciphertext)
	decrypted_pt := decCtx.decryptor.DecryptNew(ciphertext)
	decoded := make([]float64, len(testVector))
	if err = decCtx.encoder.Decode(decrypted_pt, decoded); err != nil {
		panic(err)
	}
	log.Println("Decoded values after decryption:", decoded[:len(testVector)], testVector)

	assert.InDeltaSlice(t, testVector, decoded[:len(testVector)], 1e-5, "Decrypted values should match original")
}

func TestBatchEncryptDecryptCycle(t *testing.T) {
	// Initialize contexts with reasonable parameters
	encCtx, decCtx, _ := GenerateContexts(8)

	// Prepare multiple test vectors of different sizes
	numVectors := 100
	testVectors := make([][]float64, numVectors)

	for i := range testVectors {
		testVectors[i] = utils.GenerateTestVector(245)
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
	mixedVectors[0] = encryptedVectors[0] // Valid
	mixedVectors[1] = nil                 // Nil
	mixedVectors[2] = encryptedVectors[1] // Valid

	mixedDecrypted := decCtx.BatchDecrypt(mixedVectors)
	assert.Equal(t, 3, len(mixedDecrypted), "Should have 3 results for 3 inputs")
	assert.NotNil(t, mixedDecrypted[0], "First result should not be nil")
	assert.Nil(t, mixedDecrypted[1], "Second result should be nil")
	assert.NotNil(t, mixedDecrypted[2], "Third result should not be nil")
}

func TestCosineSimilarity(t *testing.T) {
	encCtx, decCtx, evalCtx := GenerateContexts(8)

	normalized_a := utils.GenerateTestVector(5)
	normalized_b := utils.GenerateTestVector(5)

	utils.NormalizeVector(&normalized_a)
	utils.NormalizeVector(&normalized_b)
	log.Println("Normalized vector a:", normalized_a)
	log.Println("Normalized vector b:", normalized_b)

	ptA := ckks.NewPlaintext(*encCtx.params, encCtx.params.MaxLevel())
	encCtx.encoder.Encode(normalized_a, ptA)
	ctA, err := encCtx.encryptor.EncryptNew(ptA)
	log.Println("Ciphertext A:", ctA)
	assert.NoError(t, err, "Encryption failed")

	output_ct := ckks.NewCiphertext(*encCtx.params, encCtx.params.MaxSlots(), encCtx.params.MaxLevel())

	err = evalCtx.DotProduct(ctA, normalized_b, output_ct)
	assert.NoError(t, err, "Dot product calculation failed")
	log.Println("Ciphertext after dot product:", output_ct)

	decrypted := decCtx.decryptor.DecryptNew(output_ct)
	decoded := make([]float64, len(normalized_a))
	decCtx.encoder.Decode(decrypted, decoded)

	log.Println("Decoded values after decryption:", decoded[0])
	expected := utils.DotProduct(normalized_a, normalized_b)
	log.Println("Expected cosine similarity:", expected)
	assert.InDelta(t, expected, decoded[0], 1e-4, "Cosine similarity mismatch")
}

func TestBatchCosineSimilarity(t *testing.T) {
	// Initialize contexts
	encCtx, decCtx, evalCtx := GenerateContexts(8)
	// Create test vectors for ciphertexts
	numCtVectors := 1
	vectorSize := 100
	ctVectors := make([][]float64, numCtVectors)
	for i := range ctVectors {
		ctVectors[i] = utils.GenerateTestVector(vectorSize)
		utils.NormalizeVector(&ctVectors[i])
	}

	// Create plaintext vectors for comparison
	numPtVectors := 10
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
	log.Println("Expected cosine similarities:", expectedSims)

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
					//t.Logf("Note: Non-zero value at position (%d,%d) slot %d: %f", i, j, k, decryptedBatch[j][k])
				}
			}
		}
	}
}

func TestWithNamesWithoutCompression(t *testing.T) {
	query := "Mohan Tej"
	store := []string{"Bindu", "Sudheer", "Rohan", "Sahiti", "Kartik", "Phani", "Keyur", "Aditya", "Priya", "Mohan Teja"}
	path, _ := os.Getwd()
	loader := data.NewLoader(path)

	globalNames, err := loader.LoadNames("global.json")
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("=== Entity Matching with Homomorphic Encryption ===")
	log.Print("Loaded names, Vectorizing...")
	vectorizer := data.NewTfidfVectorizer(2, 1)
	vectorizer.Fit(globalNames)

	// Transform data
	queryVector := vectorizer.Transform(query)
	storeVectors := vectorizer.BatchTransform(store)

	utils.NormalizeVector(&queryVector)
	for i := range storeVectors {
		utils.NormalizeVector(&storeVectors[i])
	}
	queryVectors := make([][]float64, 1)
	queryVectors[0] = queryVector

	// Calculate expected plaintext similarities for verification
	log.Println("Expected plaintext similarities:")
	for i, name := range store {
		sim := utils.DotProduct(queryVector, storeVectors[i])
		log.Printf("  %s: %.6f", name, sim)
	}

	// Encrypt the query vector
	encCtx, decCtx, evalCtx := GenerateContexts(10)

	// Batch encrypt the query vector
	encryptedQuery := encCtx.BatchEncrypt(queryVectors)

	// Compute cosine similarities using HE
	resultMatrix, err := evalCtx.BatchDotProduct(encryptedQuery, storeVectors)
	if err != nil {
		t.Fatalf("Error computing batch dot product: %v", err)
	}

	// Create a matrix to store all similarity values
	similarityMatrix := make([][]float64, 1) // 1 query x len(store) items
	similarityMatrix[0] = make([]float64, len(store))

	// Decrypt the results
	log.Println("\nHE-computed Cosine Similarity Matrix:")
	for i := range resultMatrix {
		// Each row contains similarities between query and all store vectors
		decryptedBatch := decCtx.BatchDecrypt(resultMatrix[i])

		log.Printf("Similarities for query '%s':", query)
		for j, storeName := range store {
			if decryptedBatch[j] == nil {
				log.Printf("  %s: <nil>", storeName)
				similarityMatrix[i][j] = -1 // Use -1 to indicate null values
				continue
			}

			// The cosine similarity value is stored in the first element
			// due to the InnerSum operation in DotProduct
			similarity := decryptedBatch[j][0]

			// Store the similarity value in our matrix
			similarityMatrix[i][j] = similarity

			log.Printf("  %s: %.6f", storeName, similarity)

			// Verify the result matches the expected plaintext calculation
			expectedSim := utils.DotProduct(queryVector, storeVectors[j])
			assert.InDelta(t, expectedSim, similarity, 1e-5,
				"Cosine similarity mismatch for '%s': expected %.6f, got %.6f",
				storeName, expectedSim, similarity)
		}
	}

	// Log the complete similarity matrix
	log.Println("\nDecrypted Cosine Similarity Matrix:")
	for i := range similarityMatrix {
		rowValues := make([]string, len(similarityMatrix[i]))
		for j, val := range similarityMatrix[i] {
			if val == -1 {
				rowValues[j] = "nil"
			} else {
				rowValues[j] = fmt.Sprintf("%.6f", val)
			}
		}
		log.Printf("Row %d: [%s]", i, strings.Join(rowValues, ", "))
	}

	// You can also print it as a table format with entity names
	log.Println("\nSimilarity Matrix with Entity Names:")
	log.Printf("%-15s | %s", "Query \\ Store", strings.Join(store, " | "))
	log.Printf("%s", strings.Repeat("-", 16+len(store)*15))
	for i := range similarityMatrix {
		values := make([]string, len(similarityMatrix[i]))
		for j, val := range similarityMatrix[i] {
			if val == -1 {
				values[j] = "nil      "
			} else {
				values[j] = fmt.Sprintf("%-9.6f", val)
			}
		}
		log.Printf("%-15s | %s", query, strings.Join(values, " | "))
	}

	// Find and print the best match
	bestMatch := ""
	bestScore := -1.0

	decryptedBatch := decCtx.BatchDecrypt(resultMatrix[0])
	for j, storeName := range store {
		if decryptedBatch[j] != nil && decryptedBatch[j][0] > bestScore {
			bestScore = decryptedBatch[j][0]
			bestMatch = storeName
		}
	}

	log.Printf("\nBest match for '%s': '%s' with similarity %.6f",
		query, bestMatch, bestScore)
}

func TestWithNamesWithCompression(t *testing.T) {
    query := "Mohan Tej"
    store := []string{"Bindu", "Sudheer", "Rohan", "Sahiti", "Kartik", "Phani", "Keyur", "Aditya", "Priya", "Mohan Teja"}
    path, _ := os.Getwd()
    loader := data.NewLoader(path)

    globalNames, err := loader.LoadNames("global.json")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("=== Entity Matching with Compression + Homomorphic Encryption ===")
    log.Print("Loaded names, Vectorizing...")
    vectorizer := data.NewTfidfVectorizer(2, 1)
    vectorizer.Fit(globalNames)

    // Transform data
    queryVector := vectorizer.Transform(query)
    storeVectors := vectorizer.BatchTransform(store)
    log.Printf("Original vector size: %d", len(queryVector))

    // Calculate plaintext similarities before compression for reference
    log.Println("Original plaintext similarities (before compression):")
    utils.NormalizeVector(&queryVector)
    for i := range storeVectors {
        utils.NormalizeVector(&storeVectors[i])
    }
    for i, name := range store {
        sim := utils.DotProduct(queryVector, storeVectors[i])
        log.Printf("  %s: %.6f", name, sim)
    }

    // Apply FFT + High Pass Filter compression
    cutoff := 512
    log.Printf("=== Applying High Pass Filter Compression (cutoff=%d) ===", cutoff)
    
    // Apply FFT to query vector
    queryFft := fft.FFTReal(queryVector)
    
    // Apply High Pass Filter to query vector
    queryHp := compression.HighPassFilter(queryFft, cutoff)
    
    // Convert back to float64
    queryCompressed := compression.ToFloat64([][]complex128{queryHp})[0]
    
    // Apply FFT and compression to store vectors
    storeCompressed := make([][]float64, len(storeVectors))
    for i, vector := range storeVectors {
        fftVector := fft.FFTReal(vector)
        hpVector := compression.HighPassFilter(fftVector, cutoff)
        storeCompressed[i] = compression.ToFloat64([][]complex128{hpVector})[0]
    }
    
    log.Printf("Compressed vector size: %d (%.2f%% reduction)", 
        len(queryCompressed), 100*(1-float64(len(queryCompressed))/float64(len(queryVector))))
    
    // Calculate plaintext similarities after compression for reference
    log.Println("Plaintext similarities after compression:")
    utils.NormalizeVector(&queryCompressed)
    for i := range storeCompressed {
        utils.NormalizeVector(&storeCompressed[i])
    }
    
    for i, name := range store {
        sim := utils.DotProduct(queryCompressed, storeCompressed[i])
        log.Printf("  %s: %.6f", name, sim)
    }

    // Prepare for homomorphic encryption
    queryVectors := make([][]float64, 1)
    queryVectors[0] = queryCompressed
    
    // Initialize encryption contexts
    encCtx, decCtx, evalCtx := GenerateContexts(10)
    
    // Batch encrypt the compressed query vector
    encryptedQuery := encCtx.BatchEncrypt(queryVectors)
    
    // Compute cosine similarities using HE
    resultMatrix, err := evalCtx.BatchDotProduct(encryptedQuery, storeCompressed)
    if err != nil {
        t.Fatalf("Error computing batch dot product: %v", err)
    }
    
    // Create a matrix to store all similarity values
    similarityMatrix := make([][]float64, 1) // 1 query x len(store) items
    similarityMatrix[0] = make([]float64, len(store))
    
    // Decrypt the results
    log.Println("\nHE-computed Cosine Similarity Matrix (with compression):")
    decryptedBatch := decCtx.BatchDecrypt(resultMatrix[0])
    
    log.Printf("Similarities for query '%s':", query)
    for j, storeName := range store {
        if decryptedBatch[j] == nil {
            log.Printf("  %s: <nil>", storeName)
            similarityMatrix[0][j] = -1 // Use -1 to indicate null values
            continue
        }
        
        // The cosine similarity value is stored in the first element
        similarity := decryptedBatch[j][0]
        
        // Store the similarity value in our matrix
        similarityMatrix[0][j] = similarity
        
        log.Printf("  %s: %.6f", storeName, similarity)
        
        // Verify the result matches the expected plaintext calculation
        expectedSim := utils.DotProduct(queryCompressed, storeCompressed[j])
        assert.InDelta(t, expectedSim, similarity, 1e-5,
            "Cosine similarity mismatch for '%s': expected %.6f, got %.6f",
            storeName, expectedSim, similarity)
    }
    
    // Find and print the best match
    bestMatch := ""
    bestScore := -1.0
    
    for j, storeName := range store {
        if decryptedBatch[j] != nil && decryptedBatch[j][0] > bestScore {
            bestScore = decryptedBatch[j][0]
            bestMatch = storeName
        }
    }
    
    log.Printf("\nBest match for '%s' (with compression): '%s' with similarity %.6f",
        query, bestMatch, bestScore)
        
    // Compare with uncompressed results (optional)
    log.Println("\nComparing compressed vs uncompressed results:")
    log.Printf("%-15s | %-12s | %-15s | %-15s", "Store Name", "Original Sim", "Compressed Sim", "Difference")
    log.Printf("%s", strings.Repeat("-", 65))
    
    utils.NormalizeVector(&queryVector) // Re-normalize the original query vector
    for i, name := range store {
        utils.NormalizeVector(&storeVectors[i]) // Re-normalize original store vectors
        originalSim := utils.DotProduct(queryVector, storeVectors[i])
        compressedSim := similarityMatrix[0][i]
        diff := math.Abs(originalSim - compressedSim)
        
        log.Printf("%-15s | %-12.6f | %-15.6f | %-15.6f", 
            name, originalSim, compressedSim, diff)
    }
}