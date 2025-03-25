package fpsi

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
)

// LoadNames loads names from names_train.json into a string slice



func _load_json_file(fileName string) ([]string, error) {
	basePath, err := os.Getwd()
	if err != nil {
		return nil, err
	}

	path := filepath.Join(basePath, fileName)
	
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

func _load_txt_file(fileName string) ([]string, error) {
	basePath, err := os.Getwd()
	if err != nil {
		return nil, err
	}

	path := filepath.Join(basePath, fileName)

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


func load_names(filePath string) ([]string, error) {
	// if path ends in .json, load json file
	if filepath.Ext(filePath) == ".json" {
		return _load_json_file(filePath)
	} else {
		return _load_txt_file(filePath)
	}
}
