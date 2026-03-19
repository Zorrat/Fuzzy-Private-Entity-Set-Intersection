package main

import (
	"fmt"
	"log"

	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/hem"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/utils"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

func main() {
	vectorA := []float64{1, 1, 0, 0}
	vectorB := []float64{1, 0, 1, 0}

	utils.NormalizeVector(&vectorA)
	utils.NormalizeVector(&vectorB)

	encCtx, decCtx, evalCtx := hem.GenerateContexts(8)

	encryptedBatch := encCtx.BatchEncrypt([][]float64{vectorA, vectorB})
	if len(encryptedBatch) != 2 || encryptedBatch[0] == nil || encryptedBatch[1] == nil {
		log.Fatal("vector encryption failed")
	}

	// Reuse the ciphertext shape for the output so no internal parameters leak into the example.
	similarityCiphertext := encryptedBatch[0].CopyNew()
	if err := evalCtx.DotProduct(encryptedBatch[0], encryptedBatch[1], similarityCiphertext); err != nil {
		log.Fatalf("homomorphic dot product failed: %v", err)
	}

	decryptedBatch := decCtx.BatchDecrypt([]*rlwe.Ciphertext{similarityCiphertext})
	if len(decryptedBatch) != 1 || decryptedBatch[0] == nil {
		log.Fatal("similarity decryption failed")
	}

	// After the inner-sum in DotProduct, slot 0 contains the cosine similarity.
	encryptedSimilarity := decryptedBatch[0][0]
	plaintextSimilarity := utils.DotProduct(vectorA, vectorB)

	fmt.Printf("encrypted similarity: %.6f\n", encryptedSimilarity)
	fmt.Printf("plaintext similarity: %.6f\n", plaintextSimilarity)
}
