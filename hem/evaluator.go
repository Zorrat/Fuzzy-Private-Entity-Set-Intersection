package hem

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type evaluatorContext struct {
	params *ckks.Parameters
	encoder *ckks.Encoder
	evaluator *ckks.Evaluator
}


func (ec *evaluatorContext) dotProduct(ct *rlwe.Ciphertext,pt_vector rlwe.Operand)  (*rlwe.Ciphertext,error) {
	elementWiseMult,err := ec.evaluator.MulRelinNew(ct, pt_vector)
	if err != nil {
		return &rlwe.Ciphertext{},err
	}
	ec.evaluator.InnerSum(elementWiseMult,1,ec.params.MaxSlots(),elementWiseMult)

	return elementWiseMult,err
}

