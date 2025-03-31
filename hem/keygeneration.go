package hem

import (
	"fmt"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type encryptorContext struct {
	pk *rlwe.PublicKey
	rlk *rlwe.RelinearizationKey
	gks []*rlwe.GaloisKey
	params *ckks.Parameters
	encoder *ckks.Encoder
	encryptor *rlwe.Encryptor
}

type decryptorContext struct {
	sk *rlwe.SecretKey
	params *ckks.Parameters
	encoder *ckks.Encoder
	decryptor *rlwe.Decryptor
}



func getHEParameters(ln int) ckks.Parameters {
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            ln, // With the Bigram Model we will be under 1024 slots
		LogQ:            []int{45, 35, 35}, 
		LogP:            []int{40},
		LogDefaultScale: 35,
		RingType:        ring.ConjugateInvariant, // real numbers.
	})
	if err != nil {
		panic(err)
	}
	return params
}

// ln is the log of the number of slots 8 = 256, 9 = 512, 10 = 1024.
// Set this value based on expected
func GenerateContexts(ln int) (*encryptorContext, *decryptorContext, *evaluatorContext) {
	params := getHEParameters(ln)
	kgen := ckks.NewKeyGenerator(params)
	sk,pk := kgen.GenKeyPairNew()
	rlk := kgen.GenRelinearizationKeyNew(sk)
	gks := kgen.GenGaloisKeysNew(params.GaloisElementsForInnerSum(1, params.MaxSlots()), sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(params.GaloisElementsForInnerSum(1, params.MaxSlots()), sk)...)
	
	evaluator := ckks.NewEvaluator(params,evk)
	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, sk)
	decryptor := rlwe.NewDecryptor(params, sk)


	encryptorCtx := &encryptorContext{
		params:    &params,
		pk:        pk,
		rlk:       rlk,
		gks:       gks,
		encoder:   encoder,
		encryptor: encryptor,
	}

	decryptorCtx := &decryptorContext{
		params:    &params,
		sk:        sk,
		encoder:   encoder,
		decryptor: decryptor,
	}

	evaluatorCtx := &evaluatorContext{
		params:    &params,
		encoder:   encoder,
		evaluator:  evaluator,
	}
	return encryptorCtx, decryptorCtx, evaluatorCtx

}

func (ec *encryptorContext) BatchEncrypt(vectors [][]float64) []*rlwe.Ciphertext {
    numVectors := len(vectors)
    results := make([]*rlwe.Ciphertext, numVectors)
    var wg sync.WaitGroup

    for i := range vectors {
        wg.Add(1)
        go func(index int) {
            defer wg.Done()
            vector := vectors[index]

            // Create thread-local shallow copies.
            localEncoder := ec.encoder.ShallowCopy()
            localEncryptor := ec.encryptor.ShallowCopy()

            plaintext := ckks.NewPlaintext(*ec.params, ec.params.MaxLevel())
            if err := localEncoder.Encode(vector, plaintext); err != nil {
                // Depending on your error handling strategy, you might log the error or handle it differently.
                fmt.Printf("Encode error at index %d: %v\n", index, err)
                return
            }
            var err error
            results[index], err = localEncryptor.EncryptNew(plaintext)
            if err != nil {
                fmt.Printf("Encryption error at index %d: %v\n", index, err)
            }
        }(i)
    }
    wg.Wait()
    return results
}

func (dc *decryptorContext) BatchDecrypt(ciphertexts []*rlwe.Ciphertext) [][]float64 {
    numCiphertexts := len(ciphertexts)
    results := make([][]float64, numCiphertexts)
    var wg sync.WaitGroup

    for i := range ciphertexts {
        wg.Add(1)
        go func(index int) {
            defer wg.Done()
            // Skip nil ciphertexts
            if ciphertexts[index] == nil {
                results[index] = nil
                return
            }
			localEncoder := dc.encoder.ShallowCopy()
			localDecryptor := dc.decryptor.ShallowCopy()
            // Decrypt the ciphertext
            decryptedPlaintext := localDecryptor.DecryptNew(ciphertexts[index])
            // Determine the vector size - use MaxSlots for safety
            vectorSize := dc.params.MaxSlots()
            // Allocate and decode the result vector
            decoded := make([]float64, vectorSize)
            err := localEncoder.Decode(decryptedPlaintext, decoded)
            if err != nil {
                results[index] = nil
                return
            }
            results[index] = decoded
        }(i)
    }
    wg.Wait()
    return results
}

func (dc *decryptorContext) CosineSimMatrixDecrypt(cosineSimMatrix [][]*rlwe.Ciphertext) [][]float64 {
	// allocate space for the results
	numRows := len(cosineSimMatrix)
	numCols := len(cosineSimMatrix[0])
	results := make([][]float64, numRows)
	for i := range results {
		results[i] = make([]float64, numCols)
		decryptedBatch := dc.BatchDecrypt(cosineSimMatrix[i])
		for j := range results[i] {
			results[i][j] = decryptedBatch[j][0]
		}
	}
	return results
	
}