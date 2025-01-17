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

all: par-mpi seq

par-mpi:
	mpicc -o jacobipar jacobi-mpi.c -fopenmp 

seq:
	gcc -o jacobiseq jacobiseq.c -fopenmp 

run:
	@echo "------- Algoritimo Sequencial -------"
	@echo ""
	./jacobiseq $(N)
	@echo "-------------------------------------"
	@echo "-------- Algoritimo Paralelo MPI e OMP --------"
	@echo ""
	mpirun -np $(P) -host hal02,hal03,hal04,hal05,hal06,hal07,hal08,hal09 jacobipar $(N) $(P) $(T)
	@echo "-------------------------------------"
	
run-seq: 
	./jacobiseq $(N)

run-par:
	mpirun -np $(P) -host hal02,hal03,hal04,hal05,hal06,hal07,hal08,hal09 jacobipar $(N) $(P) $(T)
