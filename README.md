# maraton2024
MaraTON Challenge 1

Contest page: https://codeforces.com/contest/2054

Contest discussion: https://codeforces.com/blog/entry/137533

Here were what I had done during the contest:
- Tried gzip from ton's lib. Then I tried tiny LZMA which has similar performance compared to LZMA level 4 but still better than ton's gzip.
- After that, I explored light weight compressors in [Large Text Compression Benchmark](https://mattmahoney.net/dc/text.html). Among the compressors with short prog size, `paq9a` (a byte-wise neural network/context mixing compressor) and `tangelo` (bit-wise) are two of the strongest ones. Bit-wise compressor seems to work better on this kind of data.
- First I used paq9a since tangelo was a bit slow. After realizing that the SHA256 hashes in exotic cells doesn't repeat and is almost impossible to compress, I left these hashes uncompressed and only try to compress the rest. At this point, tangelo was then fast enough to compress the rest in under 2s.
- I also tried to re-order the serialized BoC. The layout that I used was:
    - ref_bs, off_bs, cell count, data_size.
    - List of d1 of all cells.
    - List of d2 of ordinary cells.
    - Relative value of ref (ref_i - i) of all cells.
    - Cell data of ordinary cells. Cells are sorted by (length, cell index). Some cells have identical data. Since d1's 5th bit (with_hash=true/false) is always 0 for mode=0, I used it to cached=true/false instead.
    - Depth information in exotic cells.
    - Hashes in exotic cells.

For the context mixing compressor, longest text matching model was the most important part. I also added some other contexts to increase prediction accuracy:
- The first byte of current cell.
- The current byte, for the pattern 0000... FFFF... in the data.
- The byte at index i - current cell's length (cells of type `72xxxH1H2...yyyy`, `72xxxxH3H4..zzzz` have lots of repeating byte at the same index).
- The first half (16 bit) and second half of the current byte (there are lots of cases that the last byte of each cell is `x8` or `x0` in hex value).

I saw some other kinds of pattern when skimming the data cell but could not go further since I did not have much time to explore the details of BoC structure.

The tangelo 1.0 compressor has other models like DMC model, sparse model, indirect model but I needed to exclude them to comfortably fit the time limit of provisional data. Indeed, data in final test cases are shorter than in provisonal ones and my run time was less than 1.5s. Though, anyway, even if I had added these models, I would still stay far behind top 3 since seems removing redundant data, and re-ordering the rest is the most important part of this contest.

One final improvement that I made was to use pre-trained data. We can compress original data into a compressed text. When we decompress the text, the model loads all the weights. Then when we compress the original data again, it only costs a dozen of bytes since the model predict the next bit with near 100% accuracy. Not only that, it also learned important features from the context. For example, when I trained my model with test case 1-006.txt, scores of 1-009.txt, 1-017.txt, and 1-022.txt jumped from 129x to 133x since their contexts are highly relevant. Other test cases also have score increased by 2-5 points. In the end, I loaded pre-trained weights of two test cases 1-007.txt and 1-020.txt. Seems my score jumped from 1284 to 1296 thanks to the pre-trained data.

Ideas that I tried but failed:
- Since there are lots of pair of bytes missing from the compressible part, I tried to map highly repetitive substrings into 2 bytes each. I did it by using Suffix+LCP Array to find substring occurences and greedily choose the substring with the highest score = #occurences*(length-2) - 2 - length, which is the estimated number of bytes saved if we map it into a pair of bytes. To avoid complex logic, I only chose non-overlapped substrings. This idea reduced the size of the data by around 35%, but after that the data became almost incompressible xD. I still believe some softs of dictionary transform would work well on this data.
- I also tried to compress the SHA256 hashes since they all have a fixed length of 32 bytes/256 bits. I toggled the hashes so that 0 has more count than 1 at every position. It helped to save only a few bytes, which was not worth doing.
- I briefly tried to use 2-opt to find a permutation of data cells that maximise the sum of longest common substring between two neighbor cells. It was too slow and did not bring any improvement.

I had almost no knowledge about data compression but after this contest, I learned lots of things. I particularly find Matt Mahoney's [data compression document](https://mattmahoney.net/dc/dce.html) and Byron Knoll's [thesis](https://www.byronknoll.com/papers.html) really interesting to read and easy to understand.