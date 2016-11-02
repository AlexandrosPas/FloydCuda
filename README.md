# FloydCuda
**Floyd - Warshall Algorithm developed using CUDA**

Implementation of the directed-graph Floyd Warshall algorithm in CUDA. calculates the best route between all pairs 
of vertices (All Pairs Shortest Path) of a directed graph with nonnegative length edges. Input for a graph with set 
of vertices `{1,2, ..., n}` is an `nxn` matrix `A` float type where `A [i] [j]` is `0` for `i == j`, or equal to the 
cost of the edge `(i, j)` or `Inf` for non existing edges. The output is:

__(a)__ a float `nxn` matrix `D` where index `D [i] [j]` will contain the shortest distance to the 
transition from node `i` to node `j` or `Inf` if there is no such route, and

__(b)__ an `nxn` table int `Q` where index `Q [i] [j]` will contain the intermediate node of the path 
from `i` to `j` or `0` if the edge `(i, j)` is the shortest path from `i` to `j` or `undetermined` 
if there is no such route. 

The pair `D` and `Q` tables should allow the reconstruction of the sequence of edges that form 
the optimal route for each pair `(i, j)`.
