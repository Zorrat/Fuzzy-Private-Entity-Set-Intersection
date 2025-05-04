package data

import (
	"math"
	"regexp"
)

// OrderedMap maintains both a map for fast lookups and a slice for order preservation
type OrderedMap struct {
	Map   map[string]int
	Keys  []string
	Count int
}

// NewOrderedMap creates a new OrderedMap
func NewOrderedMap() *OrderedMap {
	return &OrderedMap{
		Map:   make(map[string]int),
		Keys:  []string{},
		Count: 0,
	}
}

// Set adds a key to the OrderedMap if it doesn't exist
func (om *OrderedMap) Set(key string) int {
	if idx, exists := om.Map[key]; exists {
		return idx
	}

	// Add new key
	om.Map[key] = om.Count
	om.Keys = append(om.Keys, key)
	om.Count++
	return om.Count - 1
}

// Get returns the index of a key
func (om *OrderedMap) Get(key string) (int, bool) {
	idx, exists := om.Map[key]
	return idx, exists
}

// Delete removes a key from the OrderedMap
func (om *OrderedMap) Delete(key string) {
	if _, exists := om.Map[key]; !exists {
		return
	}

	// Find and remove from Keys slice
	for i, k := range om.Keys {
		if k == key {
			om.Keys = append(om.Keys[:i], om.Keys[i+1:]...)
			break
		}
	}

	// Remove from map
	delete(om.Map, key)

	// Rebuild indices to maintain consistency
	om.Count = 0
	for _, k := range om.Keys {
		om.Map[k] = om.Count
		om.Count++
	}
}

// Size returns the number of elements in the OrderedMap
func (om *OrderedMap) Size() int {
	return om.Count
}

// TfidfVectorizer implements TF-IDF feature extraction
type TfidfVectorizer struct {
	Vocabulary  *OrderedMap
	NgramFunc   func(string, int) []string
	NgramLength int
	MinDF       int
}

// NewTfidfVectorizer creates a new TF-IDF vectorizer
func NewTfidfVectorizer(ngramLength, minDF int) *TfidfVectorizer {
	return &TfidfVectorizer{
		Vocabulary:  NewOrderedMap(),
		NgramFunc:   CalculateNGrams,
		NgramLength: ngramLength,
		MinDF:       minDF,
	}
}

// Fit builds the vocabulary from training data
func (v *TfidfVectorizer) Fit(data []string) {
	// Build vocabulary
	documentFreq := make(map[string]int)
	for _, text := range data {
		uniqueNgrams := make(map[string]bool)
		for _, ngram := range v.NgramFunc(text, v.NgramLength) {
			uniqueNgrams[ngram] = true
		}
		for ngram := range uniqueNgrams {
			v.Vocabulary.Set(ngram)
			documentFreq[ngram]++
		}
	}

	// Filter terms based on min_df and rebuild vocabulary
	tempVocab := NewOrderedMap()
	for _, ngram := range v.Vocabulary.Keys {
		if documentFreq[ngram] >= v.MinDF {
			tempVocab.Set(ngram)
		}
	}
	v.Vocabulary = tempVocab
}

// BatchTransform converts a batch of text into TF-IDF vectors
func (v *TfidfVectorizer) BatchTransform(data []string) [][]float64 {
	// Create TF-IDF matrix
	tfidfMatrix := make([][]float64, len(data))
	for i, text := range data {
		tfidfMatrix[i] = v.Transform(text)
	}

	return tfidfMatrix
}

// Transform converts a single text into a TF-IDF vector
func (v *TfidfVectorizer) Transform(text string) []float64 {
	ngrams := v.NgramFunc(text, v.NgramLength)
	tfidfVector := make([]float64, v.Vocabulary.Size())

	// Calculate term frequency (TF)
	wordFreq := make(map[string]int)
	for _, ngram := range ngrams {
		if _, ok := v.Vocabulary.Get(ngram); ok {
			wordFreq[ngram]++
		}
	}

	// Calculate TF-IDF
	for ngram, freq := range wordFreq {
		if idx, ok := v.Vocabulary.Get(ngram); ok {
			tf := float64(freq) / float64(len(ngram))
			idf := math.Log(float64(v.Vocabulary.Size()) / float64(v.MinDF))
			tfidfVector[idx] = tf * idf
		}
	}

	return tfidfVector
}

// CalculateNGrams generates n-grams from a string
func CalculateNGrams(str string, n int) []string {
	// Remove punctuation from the string
	reg, _ := regexp.Compile(`[,-./]|\sBD`)
	str = reg.ReplaceAllString(str, "")

	// Generate n-grams of length n
	var result []string
	for i := 0; i < len(str)-n+1; i++ {
		result = append(result, str[i:i+n])
	}

	return result
}
