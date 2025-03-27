package hem

import (
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


// getHEParameters generates CKKS encryption parameters based on the given n-gram size.
// If the number of chosen n-grams (n) is 2, the logarithmic degree (LogN) is set to 10.
// If n is 3, the logarithmic degree (LogN) is set to 11.
// If n is 4, the logarithmic degree (LogN) is set to 12.
// The function initializes CKKS parameters with predefined logarithmic values for Q, P, and default scale.
// It also sets the ring type to ConjugateInvariant for real number computations.
// If an error occurs during parameter creation, the function will panic.
func getHEParameters(n int16) ckks.Parameters {
	
	ln :=0
	if n == 2{
		ln = 10
	} else if n == 3{
		ln = 11
	} else if n >= 4{
		ln = 12
	}

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


func generateContexts(n int16) (*encryptorContext, *decryptorContext, *evaluatorContext) {
	params := getHEParameters(n)
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
