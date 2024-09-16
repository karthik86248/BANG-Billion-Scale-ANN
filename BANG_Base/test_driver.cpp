/* Copyright 2024 Indian Institute Of Technology Hyderbad, India. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Authors: Karthik V., Saim Khan, Somesh Singh
//
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <assert.h>
#include "bang.h"
using namespace std;
int main(int argc, char** argv)
{

	if (argc < 6) {
		cerr << "Too few parameters! " << argv[0] << " " << "<path with file prefix to the director with index files > <query file> <GroundTruth File> <NumQueries> <recall parameter k>"  << endl;
		exit(1);
	}
	/*

	1. Directory of INdex Files
	2. Qery file
	3. GroundTruth File
	4. Number of Queries
	5. 
	*/ 
	bang_load<uint8_t>(argv[1]);
	bang_query<uint8_t>(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]) );
}



