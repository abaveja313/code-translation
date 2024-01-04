#!/bin/bash

cd $(mktemp -d)

# Function to compile Java program
compile_java() {
    echo "$1" > Solution.java
    javac Solution.java
    return $?
}

# Function to run test cases
run_test_cases() {
    local test_cases="$1"
    local failed=0

    # Loop through each test case
    echo "$test_cases" | jq -c '.[]' | while IFS= read -r test_case; do
        local test_case_name=$(echo "$test_case" | jq -r '.test_case_name')
        local input=$(echo "$test_case" | jq -r '.input')
        local expected=$(echo "$test_case" | jq -r '.expected')

        # Run the program with the input and capture the output and error
        output=$(echo "$input" | java Program 2>&1)

        # Check for runtime exceptions or different output
        if [[ "$output" != "$expected" ]]; then
            echo "$test_case_name" >&2
            failed=1
        fi
    done

    return $failed
}

# Main script execution starts here

# Read JSON from stdin
json_input=$(cat)

# Extract Java program string
java_program=$(echo "$json_input" | jq -r '.java_program')

# Compile Java program
compile_java "$java_program"
compile_status=$?

# Check if compilation was successful
if [ $compile_status -ne 0 ]; then
    exit 2
fi

# Extract test cases
test_cases=$(echo "$json_input" | jq '.test_cases')

# Run test cases
run_test_cases "$test_cases"
test_cases_status=$?

# Exit with the appropriate status
exit $test_cases_status
