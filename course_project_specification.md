Parallel and distributed computing AID323: course projects
Students are required to work in teams of no more than two members to complete
a comprehensive project that addresses a scientific and computationally intensive
problem that is inherently time-consuming and suitable for parallelization. Each
team must select a distinct project topic—such as those provided in the suggested
list  (e.g.,  N-body  simulation,  distributed  graph  processing,  DNA  sequence
alignment,  or  PDE solvers)—with  no  duplication  allowed  across teams, and all
selections are subject to instructor approval.
For  the  selected  project,  each  team  is  expected  to  complete  the  following
components:
- Problem Presentation: Prepare and deliver a professional presentation that
clearly    explains    the    chosen    problem,    its    real-world    relevance,
computational  challenges,  and  why  it  requires  parallel  processing.  The
presentation  should  also  outline  the  proposed  approach  and  justify  the
choice of parallel techniques.
- Sequential  Implementation: Develop  a  correct  and  efficient sequential
version of  the  solution.  This  implementation  will  serve  as  a  baseline  for
comparison with the parallel version and must be clearly documented and
tested.
- Parallel  Implementation  (Hybrid  Model): Implement  a hybrid  parallel
solution that combines:
o MPI for distributed memory parallelism (across multiple nodes), and
o Multithreading (e.g.,  OpenMP   or   threads)  for   shared   memory
parallelism in each node.
The    solution    should    demonstrate    effective    workload    distribution,
synchronization, and communication between processes and threads.
- Performance Analysis: Conduct  a detailed  performance  evaluation  of the
parallel implementation compared to the sequential version. This analysis
must include:
o Speedup and Efficiency calculations
o Scalability   analysis,   including   graphs   showing   the   effect   of
increasing the number of processes/nodes on execution time
o Evaluation    of communication    overhead and   its   impact   on
performance

o Discussion of bottlenecks and possible optimizations
Each team must submit:
o Source code (well-documented)
o A written report summarizing methodology, implementation, and results
o Presentation slides
 All projects must be original and non-overlapping between teams
 Proper comparison between sequential and parallel versions is mandatory