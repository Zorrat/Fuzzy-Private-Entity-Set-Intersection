package fpsi

import (
	"math"
	"regexp"
)


type TfidfVectorizer struct {
	Vocabulary  map[string]int
	NgramFunc   func(string, int) []string
	NgramLength int
	MinDF       int
}

func NewTfidfVectorizer( ngramLength, minDF int) *TfidfVectorizer {
	return &TfidfVectorizer{
		Vocabulary:  make(map[string]int),
		NgramFunc:   CalculateNGrams,
		NgramLength: ngramLength,
		MinDF:       minDF,
	}
}

func (v *TfidfVectorizer) FitTransform(data []string) [][]float64 {
	// Build vocabulary
	documentFreq := make(map[string]int)
	for _, text := range data {
		uniqueNgrams := make(map[string]bool)
		for _, ngram := range v.NgramFunc(text, v.NgramLength) {
			uniqueNgrams[ngram] = true
		}
		for ngram := range uniqueNgrams {
			if _, ok := v.Vocabulary[ngram]; !ok {
				v.Vocabulary[ngram] = len(v.Vocabulary)
				documentFreq[ngram]++
			}
		}
	}

	// Filter terms based on min_df
	for ngram, freq := range documentFreq {
		if freq < v.MinDF {
			delete(v.Vocabulary, ngram)
		}
	}

	// Create TF-IDF matrix
	tfidfMatrix := make([][]float64, len(data))
	for i, text := range data {
		tfidfMatrix[i] = v.Transform(text)
	}

	return tfidfMatrix
}

func (v *TfidfVectorizer) Transform(text string) []float64 {
	ngrams := v.NgramFunc(text, v.NgramLength) // Adjust n-gram length as needed
	tfidfVector := make([]float64, len(v.Vocabulary))
	maxFrequency := 0

	// Calculate term frequency (TF)
	wordFreq := make(map[string]int)
	for _, ngram := range ngrams {
		if _, ok := v.Vocabulary[ngram]; ok {
			wordFreq[ngram]++
			if wordFreq[ngram] > maxFrequency {
				maxFrequency = wordFreq[ngram]
			}
		}
	}

	// Calculate TF-IDF
	for ngram, freq := range wordFreq {
		if idx, ok := v.Vocabulary[ngram]; ok {
			tf := float64(freq) / float64(len(ngram))
			idf := math.Log(float64(len(v.Vocabulary)) / float64(v.MinDF))
			tfidfVector[idx] = tf * idf
		}
	}

	return tfidfVector
}

func CalculateNGrams(str string, n int) []string {
	// Remove punctuation from the string
	// This function will create all ngram sets from 1 to n. ex: n=3, will create 1-gram, 2-gram, 3-gram

	reg, _ := regexp.Compile(`[,-./]|\sBD`)
	str = reg.ReplaceAllString(str, "")

	// Generate zip of ngrams (n defined in function argument)
	var result []string
	for r:=1; r<=n; r++ {

		for i := 0; i < len(str)-r+1; i++ {
			result = append(result, str[i:i+r])
		}

	}

	return result
}
