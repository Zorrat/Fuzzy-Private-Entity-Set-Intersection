package tests

import (
	"log"
	"math"
	"os"
	"sort"
	"testing"

	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/compression"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/data"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/utils"
	"github.com/mjibson/go-dsp/fft"
)

func TestCompression(t *testing.T) {
	path, _ := os.Getwd()
	loader := data.NewLoader(path)

	// Load data
	names1, err := loader.LoadNames("senderList.json")
	if err != nil {
		log.Fatal(err)
	}
	names2, err := loader.LoadNames("receiverList.json")
	if err != nil {
		log.Fatal(err)
	}
	totalNames, err := loader.LoadNames("global.json")
	if err != nil {
		log.Fatal(err)
	}

	log.Print("Loaded names, Vectorizing...")
	vectorizer := data.NewTfidfVectorizer(2, 1)
	vectorizer.Fit(totalNames)

	// Transform data
	tfidf1 := vectorizer.BatchTransform(names1)
	tfidf2 := vectorizer.BatchTransform(names2)
	log.Printf("Original sizes - tfidf1: %dx%d, tfidf2: %dx%d", 
		len(tfidf1), len(tfidf1[0]), len(tfidf2), len(tfidf2[0]))

	// Baseline cosine distance
	d1 := utils.CosineDistanceAll(tfidf1, tfidf2)
	log.Printf("Original cosine distance matrix: %dx%d", len(d1), len(d1[0]))

	// Prepare FFT inputs
	tfidf1_fft := make([][]complex128, len(tfidf1))
	for i, slice := range tfidf1 {
		tfidf1_fft[i] = fft.FFTReal(slice)
	}
	tfidf2_fft := make([][]complex128, len(tfidf2))
	for i, slice := range tfidf2 {
		tfidf2_fft[i] = fft.FFTReal(slice)
	}
	log.Printf("FFT transformed sizes - tfidf1: %dx%d, tfidf2: %dx%d", 
		len(tfidf1_fft), len(tfidf1_fft[0]), len(tfidf2_fft), len(tfidf2_fft[0]))

	// Test different compression techniques
	
	// 1. High Pass Filter with different cutoffs
	highPassCutoffs := []int{128, 256, 384, 512,550}
	for _, cutoff := range highPassCutoffs {
		log.Printf("------ Testing High Pass Filter (cutoff=%d) ------", cutoff)
		
		// Apply filter
		tfidf1_hp := make([][]complex128, len(tfidf1_fft))
		tfidf2_hp := make([][]complex128, len(tfidf2_fft))
		
		for i := range tfidf1_fft {
			tfidf1_hp[i] = compression.HighPassFilter(tfidf1_fft[i], cutoff)
		}
		for i := range tfidf2_fft {
			tfidf2_hp[i] = compression.HighPassFilter(tfidf2_fft[i], cutoff)
		}
		
		log.Printf("After HighPassFilter - tfidf1: %dx%d, tfidf2: %dx%d", 
			len(tfidf1_hp), len(tfidf1_hp[0]), len(tfidf2_hp), len(tfidf2_hp[0]))
		
		// Convert to float64
		tfidfCompressed1 := compression.ToFloat64(tfidf1_hp)
		tfidfCompressed2 := compression.ToFloat64(tfidf2_hp)
		log.Printf("After ToFloat64 - tfidf1: %dx%d, tfidf2: %dx%d", 
			len(tfidfCompressed1), len(tfidfCompressed1[0]), 
			len(tfidfCompressed2), len(tfidfCompressed2[0]))
		
		// Calculate cosine distance
		d2 := utils.CosineDistanceAll(tfidfCompressed1, tfidfCompressed2)
		
		// Ensure dimensions match for MAE calculation
		if len(d1) == len(d2) && len(d1[0]) == len(d2[0]) {
			d1D2Mae := utils.MeanAverageError(d1, d2)
			log.Printf("High Pass (cutoff=%d) MAE: %f", cutoff, d1D2Mae)
		} else {
			log.Printf("ERROR: Matrix dimensions don't match for MAE calculation: d1(%dx%d) vs d2(%dx%d)",
				len(d1), len(d1[0]), len(d2), len(d2[0]))
		}
	}
	
	// 2. Low Pass Filter with different cutoffs
	lowPassCutoffs := []int{128, 256, 384, 512,530}
	for _, cutoff := range lowPassCutoffs {
		log.Printf("------ Testing Low Pass Filter (cutoff=%d) ------", cutoff)
		
		// Apply filter
		tfidf1_lp := make([][]complex128, len(tfidf1_fft))
		tfidf2_lp := make([][]complex128, len(tfidf2_fft))
		
		for i := range tfidf1_fft {
			tfidf1_lp[i] = compression.LowPassFilter(tfidf1_fft[i], cutoff)
		}
		for i := range tfidf2_fft {
			tfidf2_lp[i] = compression.LowPassFilter(tfidf2_fft[i], cutoff)
		}
		
		log.Printf("After LowPassFilter - tfidf1: %dx%d, tfidf2: %dx%d", 
			len(tfidf1_lp), len(tfidf1_lp[0]), len(tfidf2_lp), len(tfidf2_lp[0]))
		
		// Convert to float64
		tfidfCompressed1 := compression.ToFloat64(tfidf1_lp)
		tfidfCompressed2 := compression.ToFloat64(tfidf2_lp)
		log.Printf("After ToFloat64 - tfidf1: %dx%d, tfidf2: %dx%d", 
			len(tfidfCompressed1), len(tfidfCompressed1[0]), 
			len(tfidfCompressed2), len(tfidfCompressed2[0]))
		
		// Calculate cosine distance
		d2 := utils.CosineDistanceAll(tfidfCompressed1, tfidfCompressed2)
		
		// Ensure dimensions match for MAE calculation
		if len(d1) == len(d2) && len(d1[0]) == len(d2[0]) {
			d1D2Mae := utils.MeanAverageError(d1, d2)
			log.Printf("Low Pass (cutoff=%d) MAE: %f", cutoff, d1D2Mae)
		} else {
			log.Printf("ERROR: Matrix dimensions don't match for MAE calculation: d1(%dx%d) vs d2(%dx%d)",
				len(d1), len(d1[0]), len(d2), len(d2[0]))
		}
	}
	
	// 3. Band Pass Filter with different ranges
	bandRanges := []struct{ low, high int }{
		{64, 192}, {128, 256}, {192, 320}, {256, 384},
	}
	for _, band := range bandRanges {
		log.Printf("------ Testing Band Pass Filter (range=%d-%d) ------", band.low, band.high)
		
		// Apply filter
		tfidf1_bp := make([][]complex128, len(tfidf1_fft))
		tfidf2_bp := make([][]complex128, len(tfidf2_fft))
		
		for i := range tfidf1_fft {
			tfidf1_bp[i] = compression.BandPassFilter(tfidf1_fft[i], band.low, band.high)
		}
		for i := range tfidf2_fft {
			tfidf2_bp[i] = compression.BandPassFilter(tfidf2_fft[i], band.low, band.high)
		}
		
		log.Printf("After BandPassFilter - tfidf1: %dx%d, tfidf2: %dx%d", 
			len(tfidf1_bp), len(tfidf1_bp[0]), len(tfidf2_bp), len(tfidf2_bp[0]))
		
		// Convert to float64
		tfidfCompressed1 := compression.ToFloat64(tfidf1_bp)
		tfidfCompressed2 := compression.ToFloat64(tfidf2_bp)
		log.Printf("After ToFloat64 - tfidf1: %dx%d, tfidf2: %dx%d", 
			len(tfidfCompressed1), len(tfidfCompressed1[0]), 
			len(tfidfCompressed2), len(tfidfCompressed2[0]))
		
		// Calculate cosine distance
		d2 := utils.CosineDistanceAll(tfidfCompressed1, tfidfCompressed2)
		
		// Ensure dimensions match for MAE calculation
		if len(d1) == len(d2) && len(d1[0]) == len(d2[0]) {
			d1D2Mae := utils.MeanAverageError(d1, d2)
			log.Printf("Band Pass (range=%d-%d) MAE: %f", band.low, band.high, d1D2Mae)
		} else {
			log.Printf("ERROR: Matrix dimensions don't match for MAE calculation: d1(%dx%d) vs d2(%dx%d)",
				len(d1), len(d1[0]), len(d2), len(d2[0]))
		}
	}
	
	// 4. Band Stop Filter with different ranges
	for _, band := range bandRanges {
		log.Printf("------ Testing Band Stop Filter (range=%d-%d) ------", band.low, band.high)
		
		// Apply filter
		tfidf1_bs := make([][]complex128, len(tfidf1_fft))
		tfidf2_bs := make([][]complex128, len(tfidf2_fft))
		
		for i := range tfidf1_fft {
			tfidf1_bs[i] = compression.BandStopFilter(tfidf1_fft[i], band.low, band.high)
		}
		for i := range tfidf2_fft {
			tfidf2_bs[i] = compression.BandStopFilter(tfidf2_fft[i], band.low, band.high)
		}
		
		log.Printf("After BandStopFilter - tfidf1: %dx%d, tfidf2: %dx%d", 
			len(tfidf1_bs), len(tfidf1_bs[0]), len(tfidf2_bs), len(tfidf2_bs[0]))
		
		// Convert to float64
		tfidfCompressed1 := compression.ToFloat64(tfidf1_bs)
		tfidfCompressed2 := compression.ToFloat64(tfidf2_bs)
		log.Printf("After ToFloat64 - tfidf1: %dx%d, tfidf2: %dx%d", 
			len(tfidfCompressed1), len(tfidfCompressed1[0]), 
			len(tfidfCompressed2), len(tfidfCompressed2[0]))
		
		// Calculate cosine distance
		d2 := utils.CosineDistanceAll(tfidfCompressed1, tfidfCompressed2)
		
		// Ensure dimensions match for MAE calculation
		if len(d1) == len(d2) && len(d1[0]) == len(d2[0]) {
			d1D2Mae := utils.MeanAverageError(d1, d2)
			log.Printf("Band Stop (range=%d-%d) MAE: %f", band.low, band.high, d1D2Mae)
		} else {
			log.Printf("ERROR: Matrix dimensions don't match for MAE calculation: d1(%dx%d) vs d2(%dx%d)",
				len(d1), len(d1[0]), len(d2), len(d2[0]))
		}
		
	}
	// Summary
	log.Printf("Completed testing all compression techniques")
}

func TestMatching(t *testing.T) {
	path, _ := os.Getwd()
	loader := data.NewLoader(path)

	// Load data
	names1, err := loader.LoadNames("senderList.json")
	if err != nil {
		log.Fatal(err)
	}
	names2, err := loader.LoadNames("receiverList.json")
	if err != nil {
		log.Fatal(err)
	}
	totalNames, err := loader.LoadNames("global.json")
	if err != nil {
		log.Fatal(err)
	}

	log.Print("Loaded names, Vectorizing...")
	vectorizer := data.NewTfidfVectorizer(2, 1)
	vectorizer.Fit(totalNames)

	// Transform data
	tfidf1 := vectorizer.BatchTransform(names1)
	tfidf2 := vectorizer.BatchTransform(names2)
	log.Printf("Original sizes - tfidf1: %dx%d, tfidf2: %dx%d", 
		len(tfidf1), len(tfidf1[0]), len(tfidf2), len(tfidf2[0]))

	// Baseline cosine distance
	d1 := utils.CosineDistanceAll(tfidf1, tfidf2)
	log.Printf("Original cosine distance matrix: %dx%d", len(d1), len(d1[0]))
	// top k matches
	k := 5
	topKMatches := make([][]int, len(d1))
	for i := range d1 {
		topKMatches[i] = make([]int, k)
		for j := 0; j < k; j++ {
			topKMatches[i][j] = -1
		}
	}
	// Find top k matches
	for i := range d1 {
		for j := range d1[i] {
			for k := 0; k < len(topKMatches[i]); k++ {
				if d1[i][j] < d1[i][topKMatches[i][k]] || topKMatches[i][k] == -1 {
					topKMatches[i][k] = j
					break
				}
			}
		}
	}
	log.Printf("Top %d matches: %v", k, topKMatches)
	// Print top k matches
	for i := range topKMatches {
		log.Printf("Top %d matches for %d: %v", k, i, topKMatches[i])
	}
	// Print top k matches for each name
	for i := range topKMatches {
		log.Printf("Top %d matches for %d: ", k, i)
		for j := range topKMatches[i] {
			log.Printf("%s ", names2[topKMatches[i][j]])
		}
		log.Println()
	}
	// Print top k matches for each name
	for i := range topKMatches {
		log.Printf("Top %d matches for %d: ", k, i)
		for j := range topKMatches[i] {
			log.Printf("%s ", names2[topKMatches[i][j]])
		}
		log.Println()
	}

}

func TestCompressionAccuracy(t *testing.T) {
    path, _ := os.Getwd()
    loader := data.NewLoader(path)

    // Load data
    names1, err := loader.LoadNames("senderList.json")
    if err != nil {
        log.Fatal(err)
    }
    names2, err := loader.LoadNames("receiverList.json")
    if err != nil {
        log.Fatal(err)
    }
    totalNames, err := loader.LoadNames("global.json")
    if err != nil {
        log.Fatal(err)
    }

    // Define similarity threshold for matching
    similarityThreshold := 0.88
    
    log.Printf("=== Entity Matching Classification Test with Threshold %.2f ===", similarityThreshold)
    log.Print("Loaded names, Vectorizing...")
    vectorizer := data.NewTfidfVectorizer(2, 1)
    vectorizer.Fit(totalNames)

    // Transform data
    tfidf1 := vectorizer.BatchTransform(names1)
    tfidf2 := vectorizer.BatchTransform(names2)
    log.Printf("Original sizes - tfidf1: %dx%d, tfidf2: %dx%d", 
        len(tfidf1), len(tfidf1[0]), len(tfidf2), len(tfidf2[0]))

    // Calculate original cosine similarities (not distances)
    originalDistances := utils.CosineDistanceAll(tfidf1, tfidf2)
    originalSimilarities := make([][]float64, len(originalDistances))
    for i := range originalDistances {
        originalSimilarities[i] = make([]float64, len(originalDistances[i]))
        for j := range originalDistances[i] {
            // Convert distance to similarity
            originalSimilarities[i][j] = 1.0 - originalDistances[i][j]
        }
    }
    
    // Create ground truth classification matrix based on threshold
    groundTruth := make([][]bool, len(originalSimilarities))
    numPositives := 0
    for i := range originalSimilarities {
        groundTruth[i] = make([]bool, len(originalSimilarities[i]))
        for j := range originalSimilarities[i] {
            groundTruth[i][j] = originalSimilarities[i][j] >= similarityThreshold
            if groundTruth[i][j] {
                numPositives++
            }
        }
    }
    log.Printf("Ground truth: %d matches out of %d pairs (%.2f%%)", 
        numPositives, len(names1)*len(names2), 
        float64(numPositives)*100/float64(len(names1)*len(names2)))
    
    // Print some sample matches for verification
    log.Println("=== Sample Original Matches ===")
    printCount := 0
    for i := range groundTruth {
        for j := range groundTruth[i] {
            if groundTruth[i][j] && printCount < 10 {
                log.Printf("Match: %s <-> %s (similarity: %.4f)", 
                    names1[i], names2[j], originalSimilarities[i][j])
                printCount++
            }
        }
    }
    
    // Apply FFT + High Pass Filter compression
    cutoff := 530
    log.Printf("=== Testing High Pass Filter (cutoff=%d) ===", cutoff)
    
    // Prepare FFT inputs
    tfidf1Fft := make([][]complex128, len(tfidf1))
    for i, slice := range tfidf1 {
        tfidf1Fft[i] = fft.FFTReal(slice)
    }
    tfidf2Fft := make([][]complex128, len(tfidf2))
    for i, slice := range tfidf2 {
        tfidf2Fft[i] = fft.FFTReal(slice)
    }
    
    // Apply High Pass Filter
    tfidf1Hp := make([][]complex128, len(tfidf1Fft))
    tfidf2Hp := make([][]complex128, len(tfidf2Fft))
    for i := range tfidf1Fft {
        tfidf1Hp[i] = compression.HighPassFilter(tfidf1Fft[i], cutoff)
    }
    for i := range tfidf2Fft {
        tfidf2Hp[i] = compression.HighPassFilter(tfidf2Fft[i], cutoff)
    }
    
    // Convert back to float64
    tfidf1Compressed := compression.ToFloat64(tfidf1Hp)
    tfidf2Compressed := compression.ToFloat64(tfidf2Hp)
    
    // Calculate cosine similarities for compressed vectors
    compressedDistances := utils.CosineDistanceAll(tfidf1Compressed, tfidf2Compressed)
    compressedSimilarities := make([][]float64, len(compressedDistances))
    for i := range compressedDistances {
        compressedSimilarities[i] = make([]float64, len(compressedDistances[i]))
        for j := range compressedDistances[i] {
            compressedSimilarities[i][j] = 1.0 - compressedDistances[i][j]
        }
    }
    
    // Create predicted classification matrix based on same threshold
    predicted := make([][]bool, len(compressedSimilarities))
    for i := range compressedSimilarities {
        predicted[i] = make([]bool, len(compressedSimilarities[i]))
        for j := range compressedSimilarities[i] {
            predicted[i][j] = compressedSimilarities[i][j] >= similarityThreshold
        }
    }
    
    // Calculate confusion matrix
    var truePositives, falsePositives, trueNegatives, falseNegatives int
    
    for i := range groundTruth {
        for j := range groundTruth[i] {
            if groundTruth[i][j] && predicted[i][j] {
                truePositives++
            } else if !groundTruth[i][j] && predicted[i][j] {
                falsePositives++
            } else if !groundTruth[i][j] && !predicted[i][j] {
                trueNegatives++
            } else if groundTruth[i][j] && !predicted[i][j] {
                falseNegatives++
            }
        }
    }
    
    // Calculate metrics
    total := truePositives + falsePositives + trueNegatives + falseNegatives
    accuracy := float64(truePositives+trueNegatives) / float64(total)
    precision := float64(truePositives) / float64(truePositives+falsePositives)
    if truePositives+falsePositives == 0 {
        precision = 0 // Handle division by zero
    }
    recall := float64(truePositives) / float64(truePositives+falseNegatives)
    if truePositives+falseNegatives == 0 {
        recall = 0 // Handle division by zero
    }
    f1 := 2 * precision * recall / (precision + recall)
    if precision+recall == 0 {
        f1 = 0 // Handle division by zero
    }
    
    // Output results
    log.Println("\n=== Classification Results ===")
    log.Printf("Confusion Matrix:")
    log.Printf("                | Predicted Positive | Predicted Negative |")
    log.Printf("----------------|--------------------|-------------------|")
    log.Printf("Actual Positive |       %6d        |       %6d       |", truePositives, falseNegatives)
    log.Printf("Actual Negative |       %6d        |       %6d       |", falsePositives, trueNegatives)
    log.Printf("\nMetrics:")
    log.Printf("Accuracy:  %.4f", accuracy)
    log.Printf("Precision: %.4f", precision)
    log.Printf("Recall:    %.4f", recall)
    log.Printf("F1 Score:  %.4f", f1)
    
    // Print some misclassifications for analysis
    log.Println("\n=== Sample Misclassifications ===")
    
    // False positives (Predicted match but actually not a match)
    printCount = 0
    log.Println("False Positives (Incorrectly predicted as matches):")
    for i := range groundTruth {
        for j := range groundTruth[i] {
            if !groundTruth[i][j] && predicted[i][j] && printCount < 5 {
                log.Printf("  %s <-> %s (Original: %.4f, Compressed: %.4f)", 
                    names1[i], names2[j], 
                    originalSimilarities[i][j], compressedSimilarities[i][j])
                printCount++
            }
        }
    }
    
    // False negatives (Predicted not match but actually a match)
    printCount = 0
    log.Println("\nFalse Negatives (Incorrectly predicted as non-matches):")
    for i := range groundTruth {
        for j := range groundTruth[i] {
            if groundTruth[i][j] && !predicted[i][j] && printCount < 5 {
                log.Printf("  %s <-> %s (Original: %.4f, Compressed: %.4f)", 
                    names1[i], names2[j], 
                    originalSimilarities[i][j], compressedSimilarities[i][j])
                printCount++
            }
        }
    }
    
    // Calculate compression ratio
    originalSize := len(tfidf1) * len(tfidf1[0]) * 8 // Assuming float64 = 8 bytes
    compressedSize := len(tfidf1Compressed) * len(tfidf1Compressed[0]) * 8
    compressionRatio := float64(originalSize) / float64(compressedSize)
    
    log.Printf("\nCompression Stats:")
    log.Printf("Original vector dimensions: %dx%d", len(tfidf1), len(tfidf1[0]))
    log.Printf("Compressed vector dimensions: %dx%d", len(tfidf1Compressed), len(tfidf1Compressed[0]))
    log.Printf("Compression ratio: %.2fx", compressionRatio)
    
    log.Printf("\n=== Test Complete ===")
}

func TestCompressionRMSE(t *testing.T) {
    path, _ := os.Getwd()
    loader := data.NewLoader(path)

    // Load data
    names1, err := loader.LoadNames("senderList.json")
    if err != nil {
        log.Fatal(err)
    }
    names2, err := loader.LoadNames("receiverList.json")
    if err != nil {
        log.Fatal(err)
    }
    totalNames, err := loader.LoadNames("global.json")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("=== RMSE Test: Original vs Compressed Similarities ===")
    log.Print("Loaded names, Vectorizing...")
    vectorizer := data.NewTfidfVectorizer(2, 1)
    vectorizer.Fit(totalNames)

    // Transform data
    tfidf1 := vectorizer.BatchTransform(names1)
    tfidf2 := vectorizer.BatchTransform(names2)
    log.Printf("Original sizes - tfidf1: %dx%d, tfidf2: %dx%d", 
        len(tfidf1), len(tfidf1[0]), len(tfidf2), len(tfidf2[0]))

    // Calculate original cosine similarities (ground truth)
    originalDistances := utils.CosineDistanceAll(tfidf1, tfidf2)
    groundTruth := make([][]float64, len(originalDistances))
    for i := range originalDistances {
        groundTruth[i] = make([]float64, len(originalDistances[i]))
        for j := range originalDistances[i] {
            // Convert distance to similarity
            groundTruth[i][j] = 1.0 - originalDistances[i][j]
        }
    }
    
    // Test different high-pass cutoffs to find optimal
    cutoffs := []int{128, 256, 384, 512, 530, 550, 600}
    
    for _, cutoff := range cutoffs {
        log.Printf("\n=== Testing High Pass Filter (cutoff=%d) ===", cutoff)
        
        // Prepare FFT inputs
        tfidf1Fft := make([][]complex128, len(tfidf1))
        for i, slice := range tfidf1 {
            tfidf1Fft[i] = fft.FFTReal(slice)
        }
        tfidf2Fft := make([][]complex128, len(tfidf2))
        for i, slice := range tfidf2 {
            tfidf2Fft[i] = fft.FFTReal(slice)
        }
        
        // Apply High Pass Filter
        tfidf1Hp := make([][]complex128, len(tfidf1Fft))
        tfidf2Hp := make([][]complex128, len(tfidf2Fft))
        for i := range tfidf1Fft {
            tfidf1Hp[i] = compression.HighPassFilter(tfidf1Fft[i], cutoff)
        }
        for i := range tfidf2Fft {
            tfidf2Hp[i] = compression.HighPassFilter(tfidf2Fft[i], cutoff)
        }
        
        // Convert back to float64
        tfidf1Compressed := compression.ToFloat64(tfidf1Hp)
        tfidf2Compressed := compression.ToFloat64(tfidf2Hp)
        
        // Calculate cosine similarities for compressed vectors
        compressedDistances := utils.CosineDistanceAll(tfidf1Compressed, tfidf2Compressed)
        predictions := make([][]float64, len(compressedDistances))
        for i := range compressedDistances {
            predictions[i] = make([]float64, len(compressedDistances[i]))
            for j := range compressedDistances[i] {
                predictions[i][j] = 1.0 - compressedDistances[i][j]
            }
        }
        
        // Calculate RMSE
        var sumSquaredErrors float64
        var sumAbsErrors float64
        var maxError float64
        var totalElements int
        var numSimilarityDiffsGreaterThan01 int
        var numSimilarityDiffsGreaterThan05 int
        
        for i := range groundTruth {
            for j := range groundTruth[i] {
                error := groundTruth[i][j] - predictions[i][j]
                sumSquaredErrors += error * error
                absError := math.Abs(error)
                sumAbsErrors += absError
                if absError > maxError {
                    maxError = absError
                }
                
                if absError > 0.1 {
                    numSimilarityDiffsGreaterThan01++
                }
                if absError > 0.5 {
                    numSimilarityDiffsGreaterThan05++
                }
                
                totalElements++
            }
        }
        
        rmse := math.Sqrt(sumSquaredErrors / float64(totalElements))
        mae := sumAbsErrors / float64(totalElements)
        
        // Output error metrics
        log.Printf("Error Metrics:")
        log.Printf("RMSE: %.6f", rmse)
        log.Printf("MAE: %.6f", mae)
        log.Printf("Max Absolute Error: %.6f", maxError)
        log.Printf("Errors > 0.1: %d (%.2f%%)", 
            numSimilarityDiffsGreaterThan01, 
            float64(numSimilarityDiffsGreaterThan01)*100/float64(totalElements))
        log.Printf("Errors > 0.5: %d (%.2f%%)", 
            numSimilarityDiffsGreaterThan05, 
            float64(numSimilarityDiffsGreaterThan05)*100/float64(totalElements))
        
        // Show error distribution
        errors := make([]float64, 0, totalElements)
        for i := range groundTruth {
            for j := range groundTruth[i] {
                errors = append(errors, math.Abs(groundTruth[i][j] - predictions[i][j]))
            }
        }
        sort.Float64s(errors)
        
        log.Printf("Error Distribution:")
        log.Printf("Min Error: %.6f", errors[0])
        log.Printf("25th Percentile: %.6f", errors[len(errors)/4])
        log.Printf("Median Error: %.6f", errors[len(errors)/2])
        log.Printf("75th Percentile: %.6f", errors[3*len(errors)/4])
        log.Printf("95th Percentile: %.6f", errors[95*len(errors)/100])
        log.Printf("99th Percentile: %.6f", errors[99*len(errors)/100])
        log.Printf("Max Error: %.6f", errors[len(errors)-1])
        
        // Print a few sample comparisons
        log.Println("\nSample Similarity Comparisons:")
        printedSamples := 0
        maxSamples := 5
        
        // Find some high-error examples
        highErrorThreshold := rmse * 2 // Examples with error at least 2x the RMSE
        for i := range groundTruth {
            for j := range groundTruth[i] {
                error := math.Abs(groundTruth[i][j] - predictions[i][j])
                if error > highErrorThreshold && printedSamples < maxSamples {
                    log.Printf("High Error Example: %s <-> %s", names1[i], names2[j])
                    log.Printf("  Original: %.4f, Compressed: %.4f, Error: %.4f", 
                        groundTruth[i][j], predictions[i][j], error)
                    printedSamples++
                }
            }
        }
        
        // Calculate compression ratio
        originalSize := len(tfidf1) * len(tfidf1[0]) * 8 // Assuming float64 = 8 bytes
        compressedSize := len(tfidf1Compressed) * len(tfidf1Compressed[0]) * 8
        compressionRatio := float64(originalSize) / float64(compressedSize)
        
        log.Printf("\nCompression Stats:")
        log.Printf("Original vector dimensions: %dx%d", len(tfidf1), len(tfidf1[0]))
        log.Printf("Compressed vector dimensions: %dx%d", len(tfidf1Compressed), len(tfidf1Compressed[0]))
        log.Printf("Compression ratio: %.2fx", compressionRatio)
        
        // Record results for comparison
        log.Printf("\nCutoff %d Summary:", cutoff)
        log.Printf("RMSE: %.6f, MAE: %.6f, Comp.Ratio: %.2fx", rmse, mae, compressionRatio)
    }
    
    log.Printf("\n=== RMSE Test Complete ===")
}