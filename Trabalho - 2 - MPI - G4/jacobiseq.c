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

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define SEED 131

// Função para verificar a condição de parada do algoritmo
int verificaCondicaoDeParada(double *x, double *xa, double limiar, int n)
{
    double maxN = fabs(x[0] - xa[0]);
    double maxD = fabs(x[0]);
    int temp;
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
int converge(double *m, int n)
{
    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;

        // A diagonal principal não pode ter elementos nulos
        if (!m[i * n + i])
            return 0;

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
            return 0;
    }
    // Se passar por todos os teste é porque converge
    return 1;
}

int main(int argc, char *argv[])
{

    srand(SEED);
    int n = 3;          // Tamanho da matriz quadrada
    int maxInter = 100; // Variável para definir o max que os valores aleatórios serão gerados
    double wtime;       // Variável para captar o tempo de execução

    double limiar = 1e-14; // Definindo o limiar para a condição de parada

    if (argc == 1)
    {
        printf("A dimensao n da matriz A não foi passada. Assumindo o valor n = 3.\n");
    }
    else
    {
        n = atoi(argv[1]);
    }

    // Alocando as matrizes
    double *A = (double *)malloc((n * n) * sizeof(double));
    double *x = (double *)malloc((n) * sizeof(double));
    double *b = (double *)malloc((n) * sizeof(double));
    double *xanterior = (double *)malloc((n) * sizeof(double));

    /*
     Gerando uma matriz que sempre convirja para testar o algoritmo
     No caso, serão gerados valores no intervalo de [0, maxInter] e posteriormente os valores de uma linha serão somados e todos os elementos
     dessa linha, menos o elemento da diagonal principal, serão divididos por esse somatório.
     */

    for (int i = 0; i < n; i++)
    {
        srand(SEED + i);
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
        srand(SEED + i);
        b[i] = rand();
    }

    // Iniciando o vetor xanterior inicialmente com valores nulos

    for (int i = 0; i < n; i++)
    {
        xanterior[i] = 0;
    }

    // Inicio do Método Iterativo de Jacobi-Richardson, logo começando a contagem do tempo
    wtime = omp_get_wtime();

    // Verificando a convergência do método, caso não convirja imprime a matriz e uma mensagem informando isso e finaliza o algoritmo
    if (!converge(A, n))
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                printf("%lf ", A[i * n + j]);
            }
            printf("\n");
        }
        printf("Nao converge... finalizando");
        return -1;
    }

    while (1)
    {
        // Calculando o x atual segundo o método
        for (int i = 0; i < n; i++)
        {
            x[i] = b[i];
            for (int j = 0; j < i; j++)
            {
                x[i] -= A[i * n + j] * xanterior[j];
            }
            for (int j = i + 1; j < n; j++)
            {
                x[i] -= A[i * n + j] * xanterior[j];
            }

            x[i] /= A[i * n + i];
        }

        // verificação condição de parada
        if (verificaCondicaoDeParada(x, xanterior, limiar, n))
        {
            break;
        }

        // copia x em xanterior
        for (int i = 0; i < n; i++)
        {
            xanterior[i] = x[i];
        }
    }

    wtime = omp_get_wtime() - wtime; // Fim do Método Iterativo de Jacobi-Richardson, logo fim da contagem do tempo

    printf("Tempo: %lf\n", wtime);

    int le;

    printf("Escolha uma linha entre 1-%d para verificar o resultado da equacao: ", n);
    scanf("%d", &le);
    while (le < 1 || le > n)
    {
        printf("Linha invalida. O numero da linha deve estar 1-%d: ", n);
        scanf("%d", &le);
    }
    le--;

    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += A[le * n + i] * x[i];
        // printf("%lf*%lf ", A[le*n+i], x[i]);
    }
    printf("Valor pelo método iterativo de jacobi-richardson: %lf \n", sum);
    printf("Valor real: %lf\n", b[le]);

    // Desalocando a memória dos vetores
    free(A);
    free(x);
    free(b);
    free(xanterior);

    return 0;
}