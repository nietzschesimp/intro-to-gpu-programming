#!/bin/bash

# This code builds the project

if [[ -d "build" ]]
then
	echo "Removing previous build..."
	rm -rf build
fi

mkdir build
cd build
cmake ..
make
cd ..

echo "Successfully built project!"
