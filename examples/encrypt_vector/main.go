package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/hem"
)

func main() {
	vector := []float64{0.5, 0.25, -0.75, 1.0}

	encCtx, decCtx, _ := hem.GenerateContexts(8)

	// The current public API encrypts one vector through the batch entry point.
	encryptedBatch := encCtx.BatchEncrypt([][]float64{vector})
	if len(encryptedBatch) != 1 || encryptedBatch[0] == nil {
		log.Fatal("vector encryption failed")
	}

	decryptedBatch := decCtx.BatchDecrypt(encryptedBatch)
	if len(decryptedBatch) != 1 || decryptedBatch[0] == nil {
		log.Fatal("vector decryption failed")
	}

	// CKKS returns the full slot array, so slice back to the original length.
	decodedVector := decryptedBatch[0][:len(vector)]

	fmt.Printf("original vector: %s\n", formatVector(vector))
	fmt.Printf("decoded vector:  %s\n", formatVector(decodedVector))
}

func formatVector(values []float64) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = fmt.Sprintf("%.6f", value)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}
