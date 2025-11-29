#!/bin/sh

rm output.txt

for L in 10 #10 20 30 40 60 80 120 160
do
./compileKVGPUANN.sh DEEP100M $L 96
./parANN /mnt/hdd_volume2/deep100m96/deep100m_pq_pivots.bin  /mnt/hdd_volume2/deep100m96/deep100m_pq_compressed.bin /mnt/hdd_volume2/deep100m96/deep100m_graph.bin  /mnt/hdd_volume2/deep100m96/deep100m_query.bin /mnt/hdd_volume2/deep100m96/deep100m_chunk_offsets.bin /mnt/hdd_volume2/deep100m96/deep100m_centroid.bin  /mnt/hdd_volume2/deep100m96/deep100m_gndtruth.bin 10000 1 256 512 256 100 64 0 << EOM >> output.txt
y
y
y
y
y
EOM
done
