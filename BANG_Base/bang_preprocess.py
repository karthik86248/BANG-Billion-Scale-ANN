import struct
import sys
import numpy
import numpy as np
# this script converts the *disk.index (o/p of the DiskANN graph construction) into *disk.bin *metadata.bin files


print("*** Ensure to update the SECTOR LENGTH in the Script (if non-default sector lengths are used during DiskANN graph build)")

if len(sys.argv) != 6:
    print("Usage :",sys.argv[0],"<path to DiskANN graph index file (.index)> <path to store the o/p pre-process file for use by BANG (.bin)> \
<dataset dimension> <dataset datatype: 0 -> int8, 1 -> uint8, 2 -> float> <degree (i.e. R) of the DiskANN graph index>")
    exit()

file_to_read = sys.argv[1]  # INDEX FILE
file_to_write = sys.argv[2]  # BIN FILE
DIM = int(sys.argv[3])         #No of dimensions, 
DATATYPE = int(sys.argv[4])         #Data Type of base dataset, 
DEGREE = int(sys.argv[5])
SECTORLEN = 4096    #Mentioned in pq_flash_index.h
DATATYPESIZE = 4 if DATATYPE == 2 else 1
w = open(file_to_write,"wb")
file1_to_write = file_to_write[:len(file_to_write)-4] + "_metadata" + file_to_write[len(file_to_write)-4:] 
w1 = open(file1_to_write,"wb")
with open(file_to_read, "rb") as f:
        a=f.read(8)
        filesize =struct.unpack('<Q',a)[0]
        print(filesize)    #filesize

        a=f.read(8)
        total_nodes =struct.unpack('<Q',a)[0]
        print(total_nodes)    #No of nodes

        a=f.read(8)
        d =struct.unpack('<Q',a)[0]
        print("Medoid is : ",d)    #Medoid ID
        w1.write(a)
        
        a=f.read(8)
        maxNodeLen =struct.unpack('<Q',a)[0]
        print(maxNodeLen)    #max_node_len in bytes
        w1.write(a)

        a=f.read(8)
        nodesPerSec =struct.unpack('<Q',a)[0]
        print(nodesPerSec)    #nnodes_per_sector


        w1.write(struct.pack('<I',int(DATATYPE)))
        w1.write(struct.pack('<I',int(DIM)))
        w1.write(struct.pack('<I',int(DEGREE)))
        print("Datatype = ", DATATYPE, "Datatype size =", DATATYPESIZE)
        NodesRead = 0
        # Sectores in file
        print(int(filesize/SECTORLEN)-1)
        i=0
        offset=0
        for i in range(int(filesize/SECTORLEN)-1):
            f.seek((i+1)*SECTORLEN,0)
            # Nodes in a Sector
            for j in range(nodesPerSec):
                if (NodesRead == total_nodes) :
                     continue
                for dim in range(DIM):
                     b=f.read(int(DATATYPESIZE))
                     w.write(b) 
                a=f.read(4)
                w.write(a)   
                d=struct.unpack('<I',a)[0]
                #strr=str(d)
                #print("dim=",strr)
                if(d>DEGREE or d == 0) :
                    print("crap")
                    print(d)
                    exit()
                arr = numpy.zeros(d,  dtype='<u4')
                for k in range(d):
                    a=f.read(4)
                    #w.write(a)
                    neighbour = struct.unpack("<I",a)[0] 
                    #print("neighbour=", str(neighbour))
                    arr[k] = neighbour                  
                #print(arr)
                arr_sorted = np.sort(arr)
                for l in range(d):
                   w.write(struct.pack("<I",arr_sorted[l]))
                #print(arr_sorted)
                for kk in range(k+1,DEGREE):
                    #print(kk)
                    a=f.read(4)
                    w.write(a)
                NodesRead=NodesRead+1
                
                if(NodesRead % 10000 == 0):
                   print(NodesRead)
        print("Total # of Nodes Discovered =",NodesRead)

        w1.write(struct.pack('<I',int(NodesRead)))

w.close()
f.close()
w1.close()
