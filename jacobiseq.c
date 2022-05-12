#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct
{
    double **M;
    int l, c;
} Matriz;

Matriz criaM(int linha, int coluna)
{
    Matriz m;
    m.l = linha;
    m.c = coluna;

    m.M = (double **)malloc(linha * sizeof(double *));

    for (int i = 0; i < linha; i++)
    {
        m.M[i] = (double *)malloc(coluna * sizeof(double));
    }
    return m;
}

void printM(Matriz m)
{
    for (int i = 0; i < m.l; i++)
    {
        for (int j = 0; j < m.c; j++)
        {
            printf("%lf ", m.M[i][j]);
        }
        printf("\n");
    }
}

void preencheM(Matriz m)
{
    for (int i = 0; i < m.l; i++)
    {
        for (int j = 0; j < m.c; j++)
        {
            m.M[i][j] = rand();
        }
    }
}

void preencheMZero(Matriz m)
{
    for (int i = 0; i < m.l; i++)
    {
        for (int j = 0; j < m.c; j++)
        {
            m.M[i][j] = 0;
        }
    }
}

void copiaM(Matriz m, Matriz copy)
{
    for (int i = 0; i < m.l; i++)
    {
        for (int j = 0; j < m.c; j++)
        {
            copy.M[i][j] = m.M[i][j];
        }
    }
}

int verificaCondicaoDeParada(Matriz x, Matriz xa, double limiar)
{

    float maxN = abs(x.M[0][0] - xa.M[0][0]);
    float maxD = abs(x.M[0][0]);
    int temp;
    for (int i = 1; i < x.l; i++)
    {
        int temp = abs(x.M[i][0] - xa.M[i][0]);
        maxN = maxN < temp ? temp : maxN;
        temp = abs(x.M[i][0]);
        maxD = maxD < temp ? temp : maxD;
    }

    if (maxN / maxD <= limiar)
        return 1;

    return 0;
}

int converge(Matriz m)
{
    for (int i = 0; i < m.l; i++)
    {
        double sum = 0.0;

        // A diagonal principal não pode ter elementos nulos
        if (!m.M[i][i])
            return 0;

        // Feito dois fors para evitar o uso de if dentro do for
        for (int j = 0; j < i; j++)
        {
            sum += m.M[i][j];
        }
        for (int j = i + 1; j < m.c; j++)
        {
            sum += m.M[i][j];
        }

        // caso maior que 1, não converge
        if (sum / m.M[i][i] > 1)
            return 0;
    }
    // Se passar por todos os teste é porque converge
    return 1;
}

int main(int argc, char *argv[])
{

    srand(10);
    int n = 3;
    if (argc == 1)
    {
        printf("A dimensao n da matriz A não foi passada. Assumindo o valor n = 3.\n");
    }
    else
    {
        n = atoi(argv[1]);
    }
    double limiar = 0.01;

    Matriz A = criaM(n, n);
    Matriz x = criaM(n, 1);
    Matriz b = criaM(n, 1);
    Matriz xanterior = criaM(n, 1);

    preencheM(A);

    if (!converge(A))
    {
        printM(A);
        printf("Nao converge... finalizando");
        return -1;
    }

    preencheM(b);

    preencheMZero(xanterior);
    int k = 0;

    while (1)
    {

        for (int i = 0; i < n; i++)
        {
            x.M[i][0] = b.M[i][0];
            for (int j = 0; j < i; j++)
            {
                x.M[i][0] -= A.M[i][j] * xanterior.M[j][0];
            }
            for (int j = i + 1; j < n; j++)
            {
                x.M[i][0] -= A.M[i][j] * xanterior.M[j][0];
            }

            x.M[i][0] /= A.M[i][i];
        }

        // verificação condição de parada
        if (verificaCondicaoDeParada(x, xanterior, limiar))
        {
            break;
        }

        // copia x em xanterior
        copiaM(x, xanterior);
        k++;
    }

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
        sum += A.M[le][i] * x.M[i][0];
        printf("%lf*%lf ", A.M[le][i], x.M[i][0]);
    }
    printf("= %lf ", sum);
    printf("~= %lf\n", b.M[le][0]);

    return 0;
}