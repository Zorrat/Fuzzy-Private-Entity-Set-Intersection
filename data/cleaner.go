package data

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

var suffixStandards map[string][]string

func init() {
	err := loadSuffixStandards("suffix_standards.json")
	if err != nil {
		panic(err)
	}
}

func loadSuffixStandards(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	return decoder.Decode(&suffixStandards)
}

func CleanCompanyName(name string) string {
	name = normalizeString(name)
	name = removePunctuation(name)
	name = replaceSuffixes(name)
	return strings.TrimSpace(name)
}

func normalizeString(s string) string {
	t := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)
	result, _, _ := transform.String(t, s)
	return strings.ToLower(result)
}

func removePunctuation(s string) string {
	// First remove all punctuation
	reg := regexp.MustCompile(`[^a-zA-Z0-9\s]`)
	s = reg.ReplaceAllString(s, "")
	
	// Then collapse multiple spaces to single space
	spaceReg := regexp.MustCompile(`\s+`)
	return spaceReg.ReplaceAllString(s, " ")
}

func replaceSuffixes(name string) string {
	for standard, variations := range suffixStandards {
		for _, variant := range variations {
			pattern := `(?i)\b` + regexp.QuoteMeta(variant) + `\.?\b`
			re := regexp.MustCompile(pattern)
			name = re.ReplaceAllString(name, standard)
		}
	}
	return name
}
