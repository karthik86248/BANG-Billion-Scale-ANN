#!/usr/bin/env bash

#Copying the code from the 'code' folder

rm parANN.h

cp code/parANN_skeleton.h parANN.h


#Replacing the place holders with actual parameters
sed -i -e "s/ DATABASE_PLACE_HOLDER/ $1/g" *parANN.h
sed -i -e "s/ L_PLACE_HOLDER/ $2/g" *parANN.h
sed -i -e "s/ CHUNKS_PLACE_HOLDER/ $3/g" *parANN.h

#Compile
make clean
make

