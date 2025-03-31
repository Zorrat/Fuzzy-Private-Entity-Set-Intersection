package data

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
)

// LoadNames loads names from names_train.json into a string slice

type Loader struct {
	basePath string
}

// NewLoader creates a new Loader instance
func NewLoader(basePath string) *Loader {
	return &Loader{basePath: basePath}
}

func _load_json_file(path string) ([]string, error) {

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var names []string
	if err := json.NewDecoder(file).Decode(&names); err != nil {
		return nil, err
	}

	return names, nil
}

func _load_txt_file(path string) ([]string, error) {

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var names []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		names = append(names, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return names, nil
}

func (l *Loader) LoadNames(fileName string) ([]string, error) {
	// if path ends in .json, load json file
	filePath := filepath.Join(l.basePath, fileName)
	if filepath.Ext(filePath) == ".json" {
		return _load_json_file(filePath)
	} else {
		return _load_txt_file(filePath)
	}
}
