//########################################################
// Universidade de São Paulo (USP)
// Instituto de Ciências Matemáticas e de Computação (ICMC)
// SSC0903 - Computação de Alto Desempenho

// Trabalho: Método Iterativo de Jacobi-Richardson

// Matheus Yasuo Ribeiro Utino - 11233689
// Pedro Ribas Serras - 11234328
// Vinícius Silva Montanari - 11233709
//########################################################

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define SEED 131

int P = 2; // Número de processos MPI
int T = 8; // Número de threads

void iniciandoMatrizes(double *A, double *b, double *xanterior, int n, int maxInter)
{
    /*
         Gerando uma matriz que sempre convirja para testar o algoritmo
         No caso, serão gerados valores no intervalo de [0, maxInter] e posteriormente os valores de uma linha serão somados e todos os elementos
         dessa linha, menos o elemento da diagonal principal, serão divididos por esse somatório.
         */

    for (int i = 0; i < n; i++)
    {
        double linha = 0.0;
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = rand() % maxInter + 1;
            linha += A[i * n + j];
        }

        for (int j = 0; j < n; j++)
        {
            if (i != j)
                A[i * n + j] = A[i * n + j] / linha;
        }
    }

    // Randomizando os valores do vetor b
    for (int i = 0; i < n; i++)
    {
        b[i] = rand();
    }

    // Iniciando o vetor xanterior inicialmente com valores nulos
    for (int i = 0; i < n; i++)
    {
        xanterior[i] = 0;
    }
}

// Função para verificar a condição de parada do algoritmo
int verificaCondicaoDeParada(double *x, double *xa, double limiar, int n)
{

    double maxN = fabs(x[0] - xa[0]);
    double maxD = fabs(x[0]);
#pragma omp parallel for num_threads(T) reduction(max \
                                                  : maxN, maxD)
    for (int i = 1; i < n; i++)
    {
        double temp = fabs(x[i] - xa[i]);
        maxN = MAX(maxN, temp);
        temp = fabs(x[i]);
        maxD = MAX(maxD, temp);
    }

    if (maxN / maxD <= limiar)
        return 1;

    return 0;
}

// Função para verificar se a matriz analisada converge para o algoritmo
int linhaConverge(double *m, int n, int diagonalPrincipal, int inicioDaLinha)
{
    int res = 1;

    double sum = 0.0;

    // A diagonal principal não pode ter elementos nulos
    if (m[inicioDaLinha + diagonalPrincipal] == 0)
        res = 0;
    else
    {
        // Feito dois fors para evitar o uso de if dentro do for
        for (int j = 0; j < diagonalPrincipal; j++)
        {
            sum += fabs(m[inicioDaLinha + j]);
        }
        for (int j = diagonalPrincipal + 1; j < n; j++)
        {
            sum += fabs(m[inicioDaLinha + j]);
        }

        // caso maior que 1, não converge
        if (sum / fabs(m[inicioDaLinha + diagonalPrincipal]) > 1)
            res = 0;
    }
    // Se passar por todos os teste é porque a linha converge
    return res;
}

// Função para verificar se a matriz analisada converge para o algoritmo
int converge(double *m, int n)
{
    int res = 1;

#pragma omp parallel for num_threads(T) reduction(& \
                                                  : res)
    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;

        // A diagonal principal não pode ter elementos nulos
        if (!m[i * n + i])
            res = 0;

        // Feito dois fors para evitar o uso de if dentro do for
        for (int j = 0; j < i; j++)
        {
            sum += fabs(m[i * n + j]);
        }
        for (int j = i + 1; j < n; j++)
        {
            sum += fabs(m[i * n + j]);
        }

        // caso maior que 1, não converge
        if (sum / fabs(m[i * n + i]) > 1)
            res = 0;
    }
    // Se passar por todos os teste é porque converge
    return res;
}

int main(int argc, char *argv[])
{
    srand(SEED);

    int n = 3;            // Tamanho da matriz quadrada
    int maxInter = 100;   // Variável para definir o max que os valores aleatórios serão gerados
    double wtime;         // Variável para captar o tempo de execução
    double *sendB, *recB; // Buffers de envio e recebimento

    double limiar = 1e-14; // Definindo o limiar para a condição de parada

    int numprocs, rank, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int iam = 0, np = 1, provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE /*MPI_THREAD_SINGLE*/, &provided);

    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 4)
    {
        printf("A dimensao n da matriz A não foi passada. Assumindo o valor n = 3 e T = 8.\n");
    }

    else
    {
        n = atoi(argv[1]);
        P = atoi(argv[2]);
        T = atoi(argv[3]);
    }

    // Como a divisão de processos é feita por linhas, vamos calcular quantas linhas cada uma receberá de forma genériaca
    int linhasPorNo = n / P;
    int restoLinhas = n % P;
    // Também são iniciados os vetores que determinam os dados que vão para cada processo por meio de um scatterv
    int scatterDataMap[P];
    int scatterDataCount[P];
    int gatherDataMap[P];
    int gatherDataCount[P];

    double *A;
    double *x;
    double *b;
    double *xanterior;
    xanterior = (double *)malloc((n) * sizeof(double));

    double *eb, *ex;

    if (rank == 0)
    {
        gatherDataMap[0] = 0;
        gatherDataCount[0] = (0 < restoLinhas ? linhasPorNo + 1 : linhasPorNo);
        // Como os vetores scatterData só importam para o sorce do scatter, eles são preenchidos apenas dentro do processo fonte.
        // Nessa separação, as linhas que sobraram da divisão inteira de linhas por nós são disbtribuidas para os primeiros nós.
        // Para isso cada processo receberá uma linha a mais, no máximo, do que a calculáda, sendo que essas linhas a mais são distribuidas
        // a partir do primeiro processo (rank == 0) subindo em direção ao último processo. Como é feito nas linhas abaixo.
        scatterDataMap[0] = 0;
        scatterDataCount[0] = n * gatherDataCount[0];

        for (int i = 1; i < P; i++)
        {
            gatherDataMap[i] = gatherDataCount[i - 1] + gatherDataMap[i - 1];
            gatherDataCount[i] = (i < restoLinhas ? linhasPorNo + 1 : linhasPorNo);
            scatterDataCount[i] = n * gatherDataCount[i];
            scatterDataMap[i] = scatterDataCount[i - 1] + scatterDataMap[i - 1];
        }

        // Alocando buffer de recebimento para o processo 0
        recB = (double *)malloc(scatterDataCount[0] * sizeof(double));

        // Alocando as matrizes do problema
        A = (double *)malloc((n * n) * sizeof(double));
        x = (double *)malloc((n) * sizeof(double));
        b = (double *)malloc((n) * sizeof(double));

        // Incializando os valores das matrizes
        iniciandoMatrizes(A, b, xanterior, n, maxInter);

        // Inicio do Método Iterativo de Jacobi-Richardson, logo começando a contagem do tempo
        wtime = omp_get_wtime();

        // Comunicação scatterV para distribuir as linhas para cada nó do cluster
        MPI_Scatterv(A, scatterDataCount, scatterDataMap, MPI_DOUBLE, recB, scatterDataCount[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int numLinhasDoNo = rank < restoLinhas ? linhasPorNo + 1 : linhasPorNo;
    int numDeDadosDoNo = n * numLinhasDoNo;
    int numDaPrimeiraLinhaDoNo = rank * linhasPorNo + (rank < restoLinhas ? rank : restoLinhas);
    ex = (double *)malloc(numLinhasDoNo * sizeof(double));
    eb = (double *)malloc(numLinhasDoNo * sizeof(double));

    // printf("No: %d | numLinhasDoNo: %d | pl: %d\n", rank, numLinhasDoNo, numDaPrimeiraLinhaDoNo);

    if (rank != 0)
    {
        recB = (double *)malloc(numDeDadosDoNo * sizeof(double));
        MPI_Scatterv(sendB, scatterDataCount, scatterDataMap, MPI_DOUBLE, recB, numDeDadosDoNo, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    // Verificando a convergência do método, caso não convirja imprime a matriz e uma mensagem informando isso e finaliza o algoritmo
    int matrizConverge;
    int noConverge = 1;
#pragma omp parallel for num_threads(T) reduction(& \
                                                  : noConverge)
    for (int i = 0; i < numLinhasDoNo; i++)
    {
        // printf("Linha %d -> %lf %lf %lf %lf %lf \n", numDaPrimeiraLinhaDoNo + i, recB[0 + i * n], recB[1 + i * n], recB[2 + i * n], recB[3 + i * n], recB[4 + i * n]);

        if (!linhaConverge(recB, n, numDaPrimeiraLinhaDoNo + i, i * n))
        {
            noConverge = 0;
        }
    }
    if (noConverge)
        MPI_Reduce(&noConverge, &matrizConverge, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {

        if (!matrizConverge)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    printf("%lf ", A[i * n + j]);
                }
                printf("\n");
            }
            printf("Nao converge... finalizando\n");
        }
    }

    // MPI_Scatterv(sendB, scatterDataCount, scatterDataMap, MPI_DOUBLE, recB, numDeDadosDoNo, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(b, gatherDataCount, gatherDataMap, MPI_DOUBLE, eb, numLinhasDoNo, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // printf("N linhas gc %d | N linhas n %d\n Primeira linha gc %d | Primeira linha n %d\n", gatherDataCount[rank], numLinhasDoNo, gatherDataMap[rank], numDaPrimeiraLinhaDoNo);
    // for(int i = 0; i<numLinhasDoNo; i ++){

    // }

    MPI_Bcast(xanterior, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int continua = 0;
    int k = 0;

    while (1)
    {
// o vetor xantigo é passado do nó master para os outros por um Broadcast
// Calculando o x atual segundo o método
#pragma omp parallel for num_threads(T)
        for (int i = 0; i < numLinhasDoNo; i++)
        {
            int numLinha = numDaPrimeiraLinhaDoNo + i;
            // if (k == 0)
            // {
            //     printf("Linha %d -> b =>%lf\n", numLinha, eb[i]);
            // }
            ex[i] = eb[i];
            for (int j = 0; j < numLinha; j++)
            {
                ex[i] -= recB[i * n + j] * xanterior[j];
            }
            for (int j = numLinha + 1; j < n; j++)
            {
                ex[i] -= recB[i * n + j] * xanterior[j];
            }

            ex[i] /= recB[i * n + numLinha];
        }

        MPI_Gatherv(ex, numLinhasDoNo, MPI_DOUBLE, x, gatherDataCount, gatherDataMap, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // verificação condição de parada
        if (rank == 0)
        {
            // for (int i = 0; i < n; i++)
            // {
            //     printf("x[%d] = %lf\n", i, x[i]);
            // }
            // for (int i = 0; i < n; i++)
            // {
            //     printf("xa[%d] = %lf\n", i, xanterior[i]);
            // }
            continua = !verificaCondicaoDeParada(x, xanterior, limiar, n);
        }

        MPI_Bcast(&continua, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (!continua)
        {
            break;
        }

        // copia x em xanterior
        // #pragma omp parallel for num_threads(T)
        if (rank == 0)
        {
            for (int i = 0; i < n; i++)
            {
                // printf("x[%d] = %lf | xa[%d] = %lf\n", i, x[i], i, xanterior[i]);
                xanterior[i] = x[i];
            }
        }
        MPI_Bcast(xanterior, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        // for (int j = 0; j < n; j++)
        // {
        //     double sum = 0;
        //     for (int i = 0; i < n; i++)
        //     {
        //         sum += A[j * n + i] * x[i];
        //         // printf("%lf*%lf ", A[le*n+i], x[i]);
        //     }
        //     printf("bj[%d] = %lf\n", j, sum);
        // }
        // for (int i = 0; i < n; i++)
        // {
        //     printf("b[%d] = %lf\n", i, b[i]);
        // }
        wtime = omp_get_wtime() - wtime;
        printf("Tempo: %lf\n", wtime);

        free(A);
        free(x);
        free(b);
    }
    free(xanterior);

    MPI_Finalize();

    //     int le;

    //     printf("Escolha uma linha entre 1-%d para verificar o resultado da equacao: ", n);
    //     scanf("%d", &le);
    //     while (le < 1 || le > n)
    //     {
    //         printf("Linha invalida. O numero da linha deve estar 1-%d: ", n);
    //         scanf("%d", &le);
    //     }
    //     le--;

    //     double sum = 0;
    //     for (int i = 0; i < n; i++)
    //     {
    //         sum += A[le * n + i] * x[i];
    //         // printf("%lf*%lf ", A[le*n+i], x[i]);
    //     }
    //     printf("Valor pelo método iterativo de jacobi-richardson: %lf \n", sum);
    //     printf("Valor real: %lf\n", b[le]);

    //     // Desalocando a memória dos vetores

    return 0;
}
