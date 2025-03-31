package hem

import (
	"fmt"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type evaluatorContext struct {
	params *ckks.Parameters
	encoder *ckks.Encoder
	evaluator *ckks.Evaluator
}

func (ec *evaluatorContext) ShallowCopy() *evaluatorContext {
    return &evaluatorContext{
        params:    ec.params,            // Params can be shared safely
        encoder:   ec.encoder.ShallowCopy(),
        evaluator: ec.evaluator.ShallowCopy(),
    }
}

func (ec *evaluatorContext) DotProduct(ct *rlwe.Ciphertext,pt_vector rlwe.Operand, output *rlwe.Ciphertext ) (error) {
	err := ec.evaluator.MulRelin(ct, pt_vector, output)
	ec.evaluator.InnerSum(output,1,ec.params.MaxSlots(),output)
	return err
}

func (ec *evaluatorContext)BatchDotProduct(ct_matrix []*rlwe.Ciphertext, pt_matrix [][]float64) ([][]*rlwe.Ciphertext, error) {
    // Initialize the result matrix
    numRows := len(ct_matrix)
    numCols := len(pt_matrix)
    results := make([][]*rlwe.Ciphertext, numRows)
    
    // Proper initialization of the results matrix with pre-allocated ciphertexts
    for i := 0; i < numRows; i++ {
        results[i] = make([]*rlwe.Ciphertext, numCols)
        for j := 0; j < numCols; j++ {
            results[i][j] = ckks.NewCiphertext(*ec.params, ec.params.MaxSlots(), ec.params.MaxLevel())
        }
    }
    // Use a WaitGroup to synchronize goroutines
    var wg sync.WaitGroup
    // Channel for collecting errors
    errChan := make(chan error, numRows*numCols)
    
    // Process each ciphertext against each plaintext vector
    for i := range ct_matrix {
        for j := range pt_matrix {
            wg.Add(1)
            go func(rowIdx int, colIdx int) {
                defer wg.Done()
                // Skip if ciphertext is nil
                if ct_matrix[rowIdx] == nil {
                    results[rowIdx][colIdx] = nil
                    return
                }
                localEC := ec.ShallowCopy()
                // Compute the dot product
                err := localEC.DotProduct(ct_matrix[rowIdx], pt_matrix[colIdx],results[rowIdx][colIdx])
                if err != nil {
                    errChan <- fmt.Errorf("dot product error at (%d,%d): %v", rowIdx, colIdx, err)
                    return
                }
            }(i, j)
        }
    }
    // Wait for all goroutines to complete
    wg.Wait()
    close(errChan)
    // Check for errors
    if len(errChan) > 0 {
        // Return the first error encountered
        return results, <-errChan
    }
    return results, nil
}
