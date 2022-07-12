 ##########################################################
# Universidade de São Paulo (USP)                          #
# Instituto de Ciências Matemáticas e de Computação (ICMC) #
# SSC0903 - Computação de Alto Desempenho                  #
#                                                          #
# Trabalho: Método Iterativo de Jacobi-Richardson          #
#                                                          #
# Matheus Yasuo Ribeiro Utino - 11233689                   #
# Pedro Ribas Serras - 11234328                            #
# Vinícius Silva Montanari - 11233709                      #
 ##########################################################

all: par-omp par-mpi seq te

par-mpi:
	mpicc -o jacobipar jacobipar-MPI_e_OpenMP.c -fopenmp 

te:
	mpicc -o teste jacobipar-MPI-teste.c -fopenmp 

par-omp:
	gcc -o jacobipar-OpenMP jacobipar-OpenMP.c -fopenmp 

seq:
	gcc -o jacobiseq jacobiseq.c -fopenmp 

run:
	@echo "------- Algoritimo Sequencial -------"
	@echo ""
	./jacobiseq $(N)
	@echo "-------------------------------------"
	@echo "-------- Algoritimo Paralelo OMP --------"
	@echo ""
	./jacobipar-OpenMP $(N) $(T)
	@echo "-------------------------------------"
	@echo "-------- Algoritimo Paralelo MPI e OMP --------"
	@echo ""
	mpirun -np $(P) -host hal02,hal03,hal04,hal05,hal06 jacobipar $(N) $(P) $(T)
	@echo "-------------------------------------"
	
run-seq: 
	./jacobiseq $(N)

run-parOMP:
	./jacobipar-OpenMP $(N) $(T)

run-par:
	mpirun -np $(P) -host hal02,hal03,hal04,hal05,hal06 jacobipar $(N) $(P) $(T)

run-teste:
	mpirun -np 4 -host hal02,hal03,hal04,hal05,hal06,hal07,hal08,hal09 teste 46000 4 4
	mpirun -np 6 -host hal02,hal03,hal04,hal05,hal06,hal07,hal08,hal09 teste 46000 6 4
	mpirun -np 8 -host hal02,hal03,hal04,hal05,hal06,hal07,hal08,hal09 teste 46000 8 4

resto-testes:
	@echo "------- Algoritimo Sequencial -------"
	@echo ""
	./jacobiseq $(N)
	@echo "-------------------------------------"
	@echo "-------- Algoritimo Paralelo OMP --------"
	@echo ""
	./jacobipar-OpenMP $(N) $(T)
	@echo "-------------------------------------"
	@echo "-------- Algoritimo Paralelo MPI e OMP --------"
	@echo ""
	mpirun -np $(P) -host hal02,hal03,hal04,hal05,hal06,hal07,hal08,hal09 teste $(N) $(P) $(T)
	@echo "-------------------------------------"
