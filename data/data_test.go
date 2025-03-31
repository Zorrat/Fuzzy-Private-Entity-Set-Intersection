package data

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func init() {
	// Mock suffix standards for testing
	suffixStandards = map[string][]string{
		"inc": {"incorporated", "incorp", "inc"},
		"llc": {"limited liability company", "l l c"},
	}
}


func TestCleanCompanyName(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "normalize unicode",
			input:    "Café Inc.",
			expected: "cafe inc",
		},
		{
			name:     "remove punctuation",
			input:    "A.B.C. Inc.",
			expected: "abc inc",
		},
		{
			name:     "replace suffixes",
			input:    "Foo Incorporated",
			expected: "foo inc",
		},
		{
			name:     "combined cleaning",
			input:    "Bär & Hönig L.L.C.",
			expected: "bar honig llc",
		},
		{
			name:     "combined cleaning",
			input:    "Bär & Hönig l.L.C.",
			expected: "bar honig llc",
		},
		{
			name:     "combined cleaning",
			input:    "Bär-Hönig L.L.C.",
			expected: "barhonig llc",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CleanCompanyName(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestNormalizeString(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"Café", "cafe"},
		{"Hönig", "honig"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := normalizeString(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestRemovePunctuation(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"A.B.C.", "ABC"},
		{"Foo & Bar", "Foo Bar"},
		{"Test-Company", "TestCompany"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := removePunctuation(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestReplaceSuffixes(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"Foo Incorporated", "Foo inc"},
		{"Bar Incorp", "Bar inc"},
		{"Baz L L C", "Baz llc"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := replaceSuffixes(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestJsonLoader(t *testing.T) {
	path, _ := os.Getwd()
	loader := NewLoader(path)
	names,err := loader.load_names("names_train.json")
	if err != nil {
		fmt.Println(err)
	}
	
	fmt.Printf("Loaded names count: %d\n", len(names))
	assert.Equal(t, 8000, len(names), "Expected 1000 names")
}
func TestTxtLoader(t *testing.T) {
	path, _ := os.Getwd()
	loader := NewLoader(path)
	names,err := loader.load_names("names_train.txt")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Printf("Loaded names count: %d\n", len(names))
	assert.Equal(t, 6, len(names), "Expected 1000 names")
}

func TestVectorizer(t *testing.T) {
	path, _ := os.Getwd()
	loader := NewLoader(path)
	names,err := loader.load_names("names_train.json")
	if err != nil {
		fmt.Println(err)
	}
	
	v := NewTfidfVectorizer(2, 1)
	tfidfMatrix := v.FitTransform(names)
	fmt.Println(len(tfidfMatrix[0]))
	assert.Equal(t, len(names), len(tfidfMatrix), "Expected 8000 names")

}
