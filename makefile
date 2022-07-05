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

all: par seq

par:
	mpicc -o jacobipar jacobipar-MPI_e_OpenMP.c -fopenmp 

seq:
	gcc -o jacobiseq jacobiseq.c -fopenmp 

run:
	@echo "------- Algoritimo Sequencial -------"
	@echo ""
	./jacobiseq $(N)
	@echo "-------------------------------------"
	@echo "-------- Algoritimo Paralelo --------"
	@echo ""
	./jacobipar $(N) $(T)
	@echo "-------------------------------------"
	
run-seq: 
	./jacobiseq $(N)

run-par:
	mpirun -np $(P) jacobipar $(N) $(P) $(T)

