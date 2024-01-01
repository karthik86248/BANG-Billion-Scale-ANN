import struct
import sys
import numpy
import numpy as np
# this script converts the *disk.index (o/p of the DiskANN graph construction) into *disk.bin and *disk_offset.bin files
# the *disk.bin files format is :  <neighbours of node1, neighbours of node 2 ...>. There are no holes or zeros.
# inserted to maintain a fixed degree for each node's nighbour list
# Its an array of uint32s

# the *disk_offset.bin will have the 0-based starting index (on the *disk.bin) indicating the position of the adj list for a
# given node. *disk_offset.bin has N+1 entries
# This is an array of uint64s
file_to_read = "/mnt/hdd_volume1/diskANN_graphs/chunk74/sift1b_index_disk.index"
file_to_write = "/mnt/ssd_volume/diskANN-working/build/tests/sift1b_index_disk.bin"
w = open(file_to_write,"wb")

with open(file_to_read, "rb") as f:
        a=f.read(8)
        filesize =struct.unpack('<Q',a)[0]
        print(filesize)    #filesize

        a=f.read(8)
        d =struct.unpack('<Q',a)[0]
        print(d)    #No of nodes

        a=f.read(8)
        d =struct.unpack('<Q',a)[0]
        print("Medoid is : ",d)    #Medoid ID

        a=f.read(8)
        maxNodeLen =struct.unpack('<Q',a)[0]
        print(maxNodeLen)    #max_node_len in bytes

        a=f.read(8)
        nodesPerSec =struct.unpack('<Q',a)[0]
        print(nodesPerSec)    #nnodes_per_sector


        SECTORLEN = 4096    #Mentioned in pq_flash_index.h
        DIM = 128         #No of dimensions, 
        DATATYPESIZE = 1    # 1 or 4
        DEGREE = 64

        NodesRead = 0
        # Sectores in file
        print(int(filesize/SECTORLEN)-1)
        i=0
        offset=0
        for i in range(int(filesize/SECTORLEN)-1):
            f.seek((i+1)*SECTORLEN,0)
            # Nodes in a Sector
            for j in range(nodesPerSec):
                for dim in range(DIM):
                     b=f.read(DATATYPESIZE)
                     w.write(b) 
                a=f.read(4)
                w.write(a)   
                d=struct.unpack('<I',a)[0]
                #strr=str(d)
                #print("dim=",strr)
                if(d>DEGREE):
                    print("crap")
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

w.close()
f.close()

